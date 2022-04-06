from ensemble import *
import pickle as pick

class filter_stamps():
    def __init__(self, models_list = ['ML_KBmod_modelSave_19_22.8_0',
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
                 ],
                 input_shape = (21,21,1)):
        self.models_list = models_list

        self.models = []
        self.means = []
        self.stds = []
        self.all_probs = []
        self.input_shape = input_shape
        for i in range(len(self.models_list)):
            self.models.append(convnet_model(input_shape))
            self.models[-1].compile()
            (mean, std, random_split_seed) = self.models[-1].loadModel(self.models_list[i])
            self.means.append(mean)
            self.stds.append(std)

    def _predict(self, stamps, confidence_thresholds,verbose=0):
        w = np.where(np.isnan(stamps))

        self.all_probs = []
        for i in range(len(self.models)):
            regularized = (stamps-self.means[i])/self.stds[i]
            regularized[w] = 0.0
            if len(regularized.shape[1:]) == len(self.input_shape)-1:
                regularized = np.expand_dims(regularized, axis=-1)
            #print(regularized.shape)
            print(f'Predicting for model {i+1} of {len(self.models)}')
            self.all_probs.append(self.models[i].predict(regularized, verbose=verbose)[:, 1])

        self.all_probs = np.array(self.all_probs)

    def filter_stamps(self, stamps, confidence_thresholds = [0.9], class_w_triplets = True, verbose=0):

        """
        w = np.where(np.isnan(stamps))

        for i in range(len(self.models)):
            regularized = (stamps-self.means[i])/self.stds[i]
            regularized[w] = 0.0
            print(regularized.shape)
            regularized = np.expand_dims(regularized, axis=-1)
            print(regularized.shape)
            print(f'Predicting for model {i+1} of {len(self.models)}')
            self.all_probs.append(self.models[i].predict(regularized)[:, 1])

        self.all_probs = np.array(self.all_probs)
        """
        self._predict(stamps, confidence_thresholds, verbose=verbose)

        if not class_w_triplets:
            classed = np.zeros(self.all_probs.shape, dtype=np.int)
            for i in range(len(self.models)):
                w = np.where(self.all_probs[i]>confidence_thresholds[i])
                classed[i, w] = 1


        else:
            classed = np.zeros(len(self.all_probs[0]), dtype=np.int)
            probs = []
            for i in range(0, int(len(self.models)/3)):
                w = np.where((self.all_probs[i*3]>confidence_thresholds[i][0]) & \
                             (self.all_probs[3*i+1]>confidence_thresholds[i][1]) & \
                             (self.all_probs[3*i+2]>confidence_thresholds[i][2]))
                classed[w] = 1

        return classed

if __name__ == '__main__':


    with open('/media/fraserw/rocketdata/Projects/kbmod/stamps/03148/stamps_tg_045.pickle', 'rb') as han:
        s = pick.load(han)

    stamps = s[:,5,:,:]
    stamps = np.clip(stamps, -3500.0, np.max(stamps))

    models_list = ['ML_KBmod_modelSave_19_22.8_0',
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
]

    conf_thresholds = [[0.99, 0.83, 0.84],
                   [0.96, 0.85, 0.994],
                   [0.98, 0.88, 0.94],
                   [0.97, 0.88, 0.99],
                   [0.93, 0.96, 0.92],
                  ]

    fs = filter_stamps(models_list)
    classed = fs.filter_stamps(stamps, conf_thresholds, class_w_triplets = True)
    w = np.where(classed)
    print(len(classed), len(w[0]))
