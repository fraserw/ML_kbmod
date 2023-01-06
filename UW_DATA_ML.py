from keras_filter_resnet import *
#from keras_filter import *
import sys, glob, pickle
import pylab as pyl, numpy as np
from trippy import tzscale
from astropy.visualization import interval
from scipy import signal



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

"""
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
## default used in search of 03447
conf_thresholds = np.array(conf_thresholds) * np.repeat(np.array([[0.9], [0.8], [0.95], [0.95], [0.8], [0.95]]), 3, axis=1)
"""

models = ['RNML_KBmod_modelSave_10.0_27.0_wRotAug_4',
'RNML_KBmod_modelSave_10.0_27.0_wRotAug_7',
'RNML_KBmod_modelSave_10.0_27.0_wRotAug_8',
             ]


conf_thresholds = np.array([[0.9, 0.9, 0.9]])




x = np.load('/media/fraserw/rocketdata/Projects/kbmod/stamps/Hayden/stamps_and_ids.npy', allow_pickle=True)

"""
stamps = np.zeros((len(x[0]), 21, 21), dtype=np.float)
for i in range(len(x[0])):
    stamps[i, :, :] = x[0][i][:, :]
"""
stamps = np.zeros((len(x[0]), 43, 43), dtype=np.float)
for i in range(len(x[0])):
    m = np.mean(np.concatenate([x[0][i][0,:], x[0][i][0-1,:], x[0][i][:,0], x[0][i][:,-1]]))
    stamps[i,:,:] = m
    #stamps[i,:,:]=-0.0507
    stamps[i, 11-1:32-1, 11-1:32-1] = x[0][i][:, :]
ids = x[1]

stamps = np.clip(stamps, -3500., np.max(stamps))
w = np.where(np.isnan(stamps))
stamps[w] = 0.0

applyConvolve = False
if applyConvolve:
    kernel = np.zeros((21,21), dtype=np.float)
    a = np.meshgrid(np.arange(21), np.arange(21))
    kernel = np.exp(-((a[0]-10.5)**2 + (a[1]-10.5)**2)/(2*1.6**2) )
    kernel/=np.sum(kernel)

    for i in range(len(stamps)):
        stamps[i,:,:] = signal.fftconvolve(stamps[i,:,:], kernel, mode='same')

showStamps = False
if showStamps:

    (A,B,C) = stamps.shape
    (z1,z2) = tzscale.zscale(stamps.reshape(A*B,C)[::100,:])
    normer = interval.ManualInterval(z1,z2)

    for k in range(1,2):
        n=0
        fig = pyl.figure(1)
        for i in range(5):
            for j in range(5):
                sp = fig.add_subplot(5,5,n+1)
                sp.imshow(normer(stamps[n+k*25]))
                sp.set_title(n+k*25)
                n+=1

        pyl.show()
    exit()


stamps = np.expand_dims(stamps, axis=-1)


for i in range(len(models)):
    models[i] = '../ML_SNS/'+models[i]

#load the models
keras_filter = filter_stamps_rn(models)


classes = keras_filter.filter_stamps(stamps, conf_thresholds, class_w_triplets=True)
classes_half = keras_filter.filter_stamps(stamps, conf_thresholds*0.0+0.6, class_w_triplets=True)
classes_quarter = keras_filter.filter_stamps(stamps, conf_thresholds*0.0+0.1, class_w_triplets=True)
with open('/media/fraserw/rocketdata/Projects/kbmod/stamps/Hayden/source_classes.txt', 'w+') as han:
    for i in range(len(ids)):
        print(ids[i], classes[i], classes_half[i], classes_quarter[i], file=han)
print(np.sum(classes), np.sum(classes_half), np.sum(classes_quarter))
