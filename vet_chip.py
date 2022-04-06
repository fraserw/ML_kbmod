from keras_filter import *
import sys, glob, pickle
import pylab as pyl, numpy as np
from trippy import tzscale
from astropy.visualization import interval

rots_dict = {'000': 0, '001': 0, '002': 0, '003': 0, '004': 0, '005': 0, '006': 0, '007': 0, '008': 0, '010': 0,
             '011': 0, '012': 0, '013': 0, '014': 0, '015': 0, '016': 2, '017': 2, '018': 2, '019': 0, '020': 0,
             '021': 0, '022': 2, '023': 2, '024': 2, '025': 2, '026': 0, '027': 0, '028': 0, '029': 0, '030': 2,
             '031': 2, '032': 2, '033': 2, '034': 0, '035': 0, '036': 0, '037': 0, '038': 2, '039': 2, '040': 2,
             '041': 2, '042': 0, '043': 0, '044': 0, '045': 0, '046': 2, '047': 2, '048': 2, '049': 2, '050': 0,
             '051': 0, '052': 0, '053': 0, '054': 2, '055': 2, '056': 2, '057': 2, '058': 0, '059': 0, '060': 0,
             '061': 0, '062': 2, '063': 2, '064': 2, '065': 2, '066': 0, '067': 0, '068': 0, '069': 0, '070': 2,
             '071': 2, '072': 2, '073': 2, '074': 0, '075': 0, '076': 0, '077': 0, '078': 2, '079': 2, '080': 2,
             '081': 0, '082': 0, '083': 0, '084': 2, '085': 2, '086': 2, '087': 2, '088': 2, '089': 2, '090': 2,
             '091': 2, '092': 2, '093': 2, '094': 2, '095': 2, '096': 2, '097': 2, '098': 2, '099': 2, '100': 1,
             '101': 1, '102': 3, '103': 3}

models = ['ML_KBmod_modelSave_19_22.8_0',
              'ML_KBmod_modelSave_19_22.8_1',
              'ML_KBmod_modelSave_19_22.8_2',
              'ML_KBmod_modelSave_22.5_23.8_0',
              'ML_KBmod_modelSave_22.5_23.8_1',
              'ML_KBmod_modelSave_22.5_23.8_2',
              'ML_KBmod_modelSave_23.5_24.8_0',
              'ML_KBmod_modelSave_23.5_24.8_1',
              'ML_KBmod_modelSave_23.5_24.8_2',
              'ML_KBmod_modelSave_24.5_25.8_0',
              'ML_KBmod_modelSave_24.5_25.8_1',
              'ML_KBmod_modelSave_24.5_25.8_2',
              'ML_KBmod_modelSave_25.5_26.8_0',
              'ML_KBmod_modelSave_25.5_26.8_1',
              'ML_KBmod_modelSave_25.5_26.8_2',
              'ML_KBmod_modelSave_26.0_26.8_0',
              'ML_KBmod_modelSave_26.0_26.8_1',
              'ML_KBmod_modelSave_26.0_26.8_2',
             ]
conf_thresholds = [[0.99, 0.83, 0.84],
                   [0.96, 0.85, 0.994],
                   [0.98, 0.88, 0.94],
                   [0.97, 0.88, 0.99],
                   [0.93, 0.96, 0.92],
                   [0.8,  0.82,  0.75],
                  ]

# default used in search of 03447
conf_thresholds = np.array(conf_thresholds) * np.repeat(np.array([[0.9], [0.8], [0.95], [0.95], [0.8], [0.95]]), 3, axis=1)
#conf_thresholds = np.array(conf_thresholds) * np.repeat(np.array([[0.9], [0.6], [0.6], [0.8], [0.5], [0.5]]), 3, axis=1)

for i in range(len(models)):
    models[i] = '../ML_SNS/'+models[i]

chip = '000'
visit = '00000'
if len(sys.argv)>2:
    chip = str(sys.argv[1]).zfill(3)
    visit = sys.argv[2]
if '-d' in sys.argv:
    doAllchips = True
else:
    doAllchips = False

if visit == '00000':
    conf_thresholds*=0.5

#load the models
keras_filter = filter_stamps(models)

#load the stamps
if doAllchips:
    n_total = 0
    for c in range(0,104):
        if c == 9:
            continue
        chip = str(c).zfill(3)

        results_path = f'/media/fraserw/rocketdata/Projects/kbmod/warps_results/{visit}/results_{chip}_upper_0/'
        stamps_path = f'/media/fraserw/rocketdata/Projects/kbmod/stamps/{visit}/'
        stamp_file = stamps_path+f'stamps_tg_{chip}.pickle'
        with open(stamp_file, 'rb') as han:
            f = pickle.load(han)[:,5,:,:]

        f = np.clip(f, -3500., np.max(f))
        w = np.where(np.isnan(f))
        f[w] = 0.0


        if rots_dict[chip]!=0:
            f = np.rot90(f, k=-rots_dict[chip], axes=(1, 2))
        f = np.expand_dims(f, axis=-1)

        classes = keras_filter.filter_stamps(f, conf_thresholds, class_w_triplets=True)
        #print(classes)
        print(chip, len(np.where(classes)[0]))
        n_total+=len(np.where(classes)[0])
        with open(results_path+'classes.pickle', 'wb') as han:
            pickle.dump(classes, han)

    print(n_total)
    exit()


results_path = f'/media/fraserw/rocketdata/Projects/kbmod/warps_results/{visit}/results_{chip}_upper_0/'
stamps_path = f'/media/fraserw/rocketdata/Projects/kbmod/stamps/{visit}/'
stamp_file = stamps_path+f'stamps_tg_{chip}.pickle'
with open(stamp_file, 'rb') as han:
    f = pickle.load(han)[:,5,:,:]

f = np.clip(f, -3500., np.max(f))
w = np.where(np.isnan(f))
f[w] = 0.0


if rots_dict[chip]!=0:
    f = np.rot90(f, k=-rots_dict[chip], axes=(1, 2))
f = np.expand_dims(f, axis=-1)

classes = keras_filter.filter_stamps(f, conf_thresholds, class_w_triplets=True)
print(classes)
print(len(np.where(classes)[0]))
with open(results_path+'classes.pickle', 'wb') as han:
    pickle.dump(classes, han)

"""
(n, B, C) = f.shape
s = f.reshape(n*B, C)
(z1,z2) = tzscale.zscale(s)
zscale = (z1, z2)
normer = interval.ManualInterval(z1,z2)

with open(f'/media/fraserw/rocketdata/Projects/kbmod/warps_results/{visit}/results_{chip}_upper_0/results_MERGED.txt') as han:
    kbm = han.readlines()

for i in np.where(classes)[0]:
    print(kbm[i])
    pyl.imshow(normer(f[i, :, :,  0]))
    pyl.show()
"""
