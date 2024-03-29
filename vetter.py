import pylab as pyl, numpy as np
import glob, pickle
from trippy import tzscale
from astropy.visualization import interval

pyl.rcParams['keymap.quit'].remove('q')
pyl.rcParams['keymap.save'].remove('s')

class vetter():

    def __init__(self,
                 visit='03447',
                 chip='040',
                 stamps_dir ='/media/fraserw/rocketdata/Projects/kbmod/stamps',
                 stamps_grid = (9, 7),
                 results_dir = '/media/fraserw/rocketdata/Projects/kbmod/kbmod_results/',
                 plantLists_dir = '/media/fraserw/SecondStage/Projects/kbmod/DATA/rerun/diff_warpCompare/deepDiff/03447/HSC-R2/plantLists/',
                 max_assoc_dist = 3.0,
                 contrast = 0.5
                 ):
        # subplot # wide and high
        self.stamps_grid = np.array(stamps_grid)

        # load the ML classes
        if len(visit)==5:
            self.classes_fn = results_dir+f'{visit}/results_{chip}_upper_0/rn_classes.pickle'
        else:
            self.classes_fn = results_dir+f'{visit}/results_{chip}/rn_classes.pickle'
        print('Loading ML results from '+self.classes_fn)
        with open(self.classes_fn, 'rb') as han:
            self.classes = pickle.load(han)


        # load the stamps
        self.stamps_dir = stamps_dir
        if self.stamps_dir[-1]!='!':
            self.stamps_dir += '/'
        if len(visit)==5:
            self.stamps_dir+=visit+'/'

        if len(visit)==5:
            self.stamp_files = glob.glob(self.stamps_dir+f'stamps_tg_{chip}_w_med.pickle')
        else:
            self.stamp_files = glob.glob(self.stamps_dir+f'stamps_tg_{chip}_w_sr_med.pickle')

        # wide median stamps
        with open(self.stamp_files[0], 'rb') as han:
            stamps_med = pickle.load(han)
        w = np.where(np.isnan(stamps_med))
        stamps_med[w] = -1000.
        # wide mean stamps
        with open(self.stamp_files[0].replace('_med.', '.'), 'rb') as han:
            stamps_mean = pickle.load(han)
        w = np.where(np.isnan(stamps_mean))
        stamps_mean[w] = -1000.

        if len(stamps_med.shape)==4:
            self.all_stamps = [stamps_med[:, 5, :, :], stamps_mean[:, 5, :, :]]
        else:
            self.all_stamps = [stamps_med[:, :, :], stamps_mean[:, :, :]]
        self.stamps = [self.all_stamps[0][np.where(self.classes)], self.all_stamps[1][np.where(self.classes)]]


        # load the kbmod results file
        if len(visit) == 5:
            with open(results_dir+f'{visit}/results_{chip}_upper_0/results_MERGED.txt') as han:
                data = han.readlines()
        else:
            with open(results_dir+f'{visit}/results_{chip}/results_.txt') as han:
                data = han.readlines()
        self.all_kb_xy = []
        for i in range(len(data)):
            s = data[i].split()
            l = float(s[1])
            f = float(s[3])
            x = float(s[5])
            y = float(s[7])
            vx = float(s[9])
            vy = float(s[11])
            self.all_kb_xy.append([x, y, l, f, vx, vy])
        self.all_kb_xy = np.array(self.all_kb_xy)
        self.kb_xy = self.all_kb_xy[np.where(self.classes)]

        if len(self.all_stamps[0])!= len(self.all_kb_xy) or len(self.all_stamps[1])!= len(self.all_kb_xy) or len(self.classes)!=len(self.all_kb_xy):
            print('WARNING: Stamps length, kbmod output length, and/or classes length do not match!')
            print(len(self.all_stamps[0]), len(self.all_stamps[1]), len(self.all_kb_xy),len(self.classes))
            exit()

        #sort the kb_xy and the stamps
        self.sort_args = np.argsort(self.kb_xy[:,0])
        self.kb_xy = self.kb_xy[self.sort_args]
        self.stamps = [self.stamps[0][self.sort_args], self.stamps[1][self.sort_args]]


        # load the plantList in case the user decides to display them
        plantLists = glob.glob(f'{plantLists_dir}/{chip}/*plantList')
        plantLists.sort()
        if len(plantLists) > 0:
            self.plantList = plantLists[0]
            print(f'Using planted sources in {self.plantList}')
            self.plant_xy = []
            with open(self.plantList) as han:
                data = han.readlines()
            for i in range(1, len(data)):
                s = data[i].split()
                x,y,m = float(s[3]), float(s[4]), float(s[9])
                ind = int(float(s[0]))
                if m<27:
                    self.plant_xy.append([x, y, m, -1, ind])
                    dist = ((self.kb_xy[:, 0] - x)**2 + (self.kb_xy[:, 1] - y)**2)**0.5
                    arg = np.argmin(dist)
                    if dist[arg]<max_assoc_dist:
                        self.plant_xy[-1][3] = arg
                    #print(self.plant_xy[-1], dist[arg])
            self.plant_xy = np.array(self.plant_xy)
        else:
            self.plant_xy = np.array([[0., 0., 0., 0., 0.]])

        # the vet results array
        self.all_elims = np.ones(self.stamps[0].shape[0], dtype='bool')


        # misc bits and bobs
        self.stamps_shape = self.stamps[0][0].shape

        self.vet_counter = 0
        self.showing = False
        self.reveal_plants = False

        self.saved = False

        self.contrast = contrast


    def get_zscale(self, downscale = 5):
        s = self.stamps[0][::downscale, :, :]
        (n, B, C) = s.shape
        s = s.reshape(n*B, C)
        (z1,z2) = tzscale.zscale(s, contrast = self.contrast)
        self.zscale = (z1, z2)
        self.normer = interval.ManualInterval(z1,z2)


    def save_single_frame_vets(self):
        print('Saving vets in this window')
        b = min(len(self.elims),len(self.all_elims)-self.vet_counter)
        self.all_elims[self.vet_counter:self.vet_counter+b] = self.elims[:b]
        #print(self.all_elims[self.vet_counter:self.vet_counter+b])

    def save_vets(self):
        outhan =  open(self.classes_fn.replace('classes', 'vets'), 'w+')
        print('#  x     y      vx      vy       like     flux Good/Bogus Planted # plant object ind x y mag', file=outhan)
        print('#  x     y      vx      vy       like     flux Good/Bogus Planted # plant object ind x y mag')
        for i in range(len(self.all_elims)):
            (x,y,l,f,vx,vy) = self.kb_xy[i]
            if i in self.plant_xy[:, 3]:
                w = np.where( self.plant_xy[:, 3] == i)
                (px, py, pm, bunk, p_ind) = self.plant_xy[w][0]
                print('{:>5} {:>5} {:8.2f} {:8.2f} {:8.2f} {:8.2f}  {}  {}  # {:>6} {:6.1f} {:6.1f} {:5.2f}'.format(int(x), int(y), vx, vy, l, f, 'G' if ~self.all_elims[i] else 'B', 'T' if i in self.plant_xy[:,3] else 'F', int(p_ind), px, py, pm), file=outhan)
                print('{:>5} {:>5} {:8.2f} {:8.2f} {:8.2f} {:8.2f}  {}  {}  # {:>6} {:6.1f} {:6.1f} {:5.2f}'.format(int(x), int(y), vx, vy, l, f, 'G' if ~self.all_elims[i] else 'B', 'T' if i in self.plant_xy[:,3] else 'F', int(p_ind), px, py, pm))
            else:
                print('{:>5} {:>5} {:8.2f} {:8.2f} {:8.2f} {:8.2f}  {}  {}'.format(int(x), int(y), vx, vy, l, f, 'G' if ~self.all_elims[i] else 'B', 'T' if i in self.plant_xy[:,3] else 'F'), file=outhan)
                print('{:>5} {:>5} {:8.2f} {:8.2f} {:8.2f} {:8.2f}  {}  {}'.format(int(x), int(y), vx, vy, l, f, 'G' if ~self.all_elims[i] else 'B', 'T' if i in self.plant_xy[:,3] else 'F'))
        self.saved = True
        print('Saved all vets.')


    def make_window(self, window_size_scale = 1.3):

        self.window = pyl.figure('Vet this', figsize = self.stamps_grid*np.array([window_size_scale*1.3,window_size_scale]))
        self.subplots = []
        self.elims = []
        for i in range(self.stamps_grid[0]): #up down
            for j in range(self.stamps_grid[1]): #left right
                self.subplots.append(self.window.add_subplot(self.stamps_grid[0],
                                                             self.stamps_grid[1],
                                                             i*self.stamps_grid[1]+j+1,
                                                             xticklabels='', yticklabels=''))
                self.subplots[-1].axis('equal')

                self.elims.append(False)

        self.window.subplots_adjust(hspace=0.03, wspace=0.03)
        self.window.canvas.mpl_connect('button_press_event', self.selector_function)
        self.window.canvas.mpl_connect('key_press_event', self.draw_next)


    def display_stamps(self, start_ind = 0, keep_elims = False):
        for i in range(self.stamps_grid[0]): #up down
            for j in range(self.stamps_grid[1]): #left right
                sp_ind = i*self.stamps_grid[1]+j
                self.subplots[sp_ind].clear()
                if start_ind+sp_ind <len(self.stamps[0]):
                    if start_ind+sp_ind in self.plant_xy[:, 3] and self.reveal_plants:
                        colour = 'r'
                    else:
                        colour='0.8'
                    #text = f'l: {int(self.kb_xy[start_ind+sp_ind][2])} f: {int(self.kb_xy[start_ind+sp_ind][3])} '
                    text = 'l:{:.1f} f:{} '.format(self.kb_xy[start_ind+sp_ind][2], int(self.kb_xy[start_ind+sp_ind][3]))
                    text += f'({int(self.kb_xy[start_ind+sp_ind][0])}, {int(self.kb_xy[start_ind+sp_ind][1])})'
                    self.subplots[sp_ind].text(self.stamps_shape[1], 5.0,
                                               text,
                                               color=colour, horizontalalignment='center',
                                               fontweight = 'bold', fontsize=12)

                    double_stamp = np.concatenate([self.stamps[0][start_ind+sp_ind].T, self.stamps[1][start_ind+sp_ind].T]).T

                    self.subplots[sp_ind].imshow(self.normer(double_stamp))
                    self.subplots[sp_ind].xaxis.set_ticklabels([])
                    self.subplots[sp_ind].yaxis.set_ticklabels([])

                for spine in ['top', 'bottom', 'left', 'right']:
                    self.subplots[sp_ind].spines[spine].set_linewidth('3')
                    self.subplots[sp_ind].spines[spine].set_color('b')
                if not keep_elims:
                    self.elims[sp_ind] = False


        if not self.showing:
            pyl.show()
            self.showing = True
            pyl.draw()

    def selector_function(self, event):
        for i in range(self.stamps_grid[0]): #up down
            for j in range(self.stamps_grid[1]): #left right
                sp_ind = i*self.stamps_grid[1]+j
                if event.inaxes == self.subplots[sp_ind]:
                    if self.elims[sp_ind]: #already selected as bad, changing back to good
                        for spine in ['top', 'bottom', 'left', 'right']:
                            self.subplots[sp_ind].spines[spine].set_linewidth('3')
                            self.subplots[sp_ind].spines[spine].set_color('b')
                        self.elims[sp_ind] = False
                    else:
                        for spine in ['top', 'bottom', 'left', 'right']:
                            self.subplots[sp_ind].spines[spine].set_linewidth('5')
                            self.subplots[sp_ind].spines[spine].set_color('r')
                        self.elims[sp_ind] = True
        pyl.draw()

    def draw_next(self, event):
        if event.key in ['p', 'P']:
            self.reveal_plants = False if self.reveal_plants else True
            self.display_stamps(start_ind = self.vet_counter, keep_elims = True)
            for i in range(self.stamps_grid[0]): #up down
                for j in range(self.stamps_grid[1]): #left right
                    sp_ind = i*self.stamps_grid[1]+j
                    if self.elims[sp_ind]:
                        for spine in ['top', 'bottom', 'left', 'right']:
                            self.subplots[sp_ind].spines[spine].set_linewidth('5')
                            self.subplots[sp_ind].spines[spine].set_color('r')
                        #self.elims[sp_ind] = True
            pyl.draw()

        if event.key in ['a', 'A']:
            print('Setting all to elimated')
            for i in range(self.stamps_grid[0]): #up down
                for j in range(self.stamps_grid[1]): #left right
                    sp_ind = i*self.stamps_grid[1]+j
                    for spine in ['top', 'bottom', 'left', 'right']:
                        self.subplots[sp_ind].spines[spine].set_linewidth('5')
                        self.subplots[sp_ind].spines[spine].set_color('r')
                    self.elims[sp_ind] = True
            pyl.draw()


        if event.key in ['n', 'N']:
            # here we save the interim clicks
            self.save_single_frame_vets()
            print(self.vet_counter)
            if self.vet_counter+self.stamps_grid[0]*self.stamps_grid[1]<len(self.all_elims):
                self.vet_counter +=self.stamps_grid[0]*self.stamps_grid[1]
                self.display_stamps(start_ind = self.vet_counter)
                pyl.draw()
            else:
                print('At end of list')

        if event.key in ['b', 'b']:
            # here we save the interim clicks
            self.save_single_frame_vets()
            self.vet_counter -=self.stamps_grid[0]*self.stamps_grid[1]
            self.display_stamps(start_ind = self.vet_counter)
            pyl.draw()

        if event.key in ['w', 'W']:
            self.save_single_frame_vets()
            self.save_vets()

        if event.key in ['q', 'Q'] and self.saved:
            print('Saved all vets. Quitting')
            pyl.close()



    def activate(self, window_size_scale=2.2):
        vedder.get_zscale()
        vedder.make_window(window_size_scale=window_size_scale)
        vedder.display_stamps()

        vedder.display_stamps(start_ind = vedder.vet_counter)

        return self.saved

if __name__ == "__main__":

    import sys
    visit = '03447'
    chip = '050'
    if len(sys.argv)>1:
        visit = sys.argv[1]
        chip = str(sys.argv[2]).zfill(3)

    contrasts = {'03447': 0.5, '03455': 0.6, '00000':0.5, '03473': 0.5, '03805': 0.5, '03806': 0.5, '03832': 0.5, '03833': 0.5, '2022-08-22': 0.5}

    if len(visit)>5:
        chip = chip[1:]
        vedder = vetter(chip = chip, visit = visit,
                        stamps_dir =f'/media/fraserw/SecondStage/Projects/kbmod/DATA/rerun/diff_warpCompare/deepDiff/{visit}/warps/',
                        stamps_grid = (9, 7),
                        results_dir = '/media/fraserw/rocketdata/Projects/kbmod/kbmod_results/',
                        plantLists_dir = f'/media/fraserw/SecondStage/Projects/kbmod/DATA/rerun/diff_warpCompare/deepDiff/{visit}/HSC-R2/warps/',
                        contrast = contrasts[visit],
                        max_assoc_dist = 5.0)

    else:
        vedder = vetter(chip = chip, visit = visit,
                        plantLists_dir = f'/media/fraserw/SecondStage/Projects/kbmod/DATA/rerun/diff_warpCompare/deepDiff/{visit}/HSC-R2/plantLists/',
                        contrast = contrasts[visit])
    vedder.activate(window_size_scale=2.2)
