from ensemble_resnet import *
import pickle as pick

class filter_stamps_rn():
    def __init__(self, models_list = ['RNML_KBmod_modelSave_10.0_27.0_0',
                  'RNML_KBmod_modelSave_10.0_27.0_1',
                  'RNML_KBmod_modelSave_10.0_27.0_2',
                  'RNML_KBmod_modelSave_10.0_27.0_3',
                 ],
                 input_shape = (43,43,1)):
        self.models_list = models_list

        self.models = []
        self.means = []
        self.stds = []
        self.all_probs = []
        self.input_shape = input_shape
        for i in range(len(self.models_list)):
            self.models.append(resnet_model(input_shape))
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


    with open('/media/fraserw/rocketdata/Projects/kbmod/stamps/03447/stamps_tg_045_w.pickle', 'rb') as han:
        s = pick.load(han)

    stamps = s[:,5,:,:]
    stamps = np.clip(stamps, -3500.0, np.max(stamps))

    models_list = ['RNML_KBmod_modelSave_10.0_27.0_0',
                  'RNML_KBmod_modelSave_10.0_27.0_1',
                  'RNML_KBmod_modelSave_10.0_27.0_2',
                  'RNML_KBmod_modelSave_10.0_27.0_2',
                 ]

    conf_thresholds = [[0.2, 0.2, 0.2, 0.2],
                  ]

    fs = filter_stamps_rn(models_list)
    classed = fs.filter_stamps(stamps, conf_thresholds, class_w_triplets = True)
    w = np.where(classed)
    print(len(classed), len(w[0]))
    print(w)
