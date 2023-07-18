from ensemble_resnet import *
import pickle as pick

class filter_stamps_rn():
    def __init__(self, models_list = ['RNML_KBmod_modelSave_10.0_27.0_0',
                  'RNML_KBmod_modelSave_10.0_27.0_1',
                  'RNML_KBmod_modelSave_10.0_27.0_2',
                  'RNML_KBmod_modelSave_10.0_27.0_3',
                 ],
                 input_shape = (43,43,1),
                 batch_size=1024):
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
        self.batch_size=batch_size

    def _predict(self, stamps, verbose=0, merge_type='mean'):
        w = np.where(np.isnan(stamps))

        self.all_probs = []
        for i in range(len(self.models)):
            regularized = (stamps-self.means[i])/self.stds[i]
            regularized[w] = 0.0
            if len(regularized.shape[1:]) == len(self.input_shape)-1:
                regularized = np.expand_dims(regularized, axis=-1)
            #print(regularized.shape)
            print(f'Predicting for model {i+1} of {len(self.models)}')
            self.all_probs.append(self.models[i].predict(regularized, verbose=verbose, merge_type=merge_type, batch_size=self.batch_size)[:, 1])
        self.all_probs = np.array(self.all_probs)

    def filter_stamps(self, stamps, confidence_thresholds = [0.9], verbose=0, merge_type='mean', force_repredict=False):
        """
        confidence_thresholds should be of length K where K is the number of
        models, or a float.

        prediction step will be run only once if multiple confidence thresholds
        are being tested. Force a rerun with force_repredict = True.
        """

        # this evaluates the probability of each stamp (N) through each network (K)
        # probabilities sorted in self.all_probs in the shape [K, N]
        #
        # recall that each network is itself a number of similarly shaped sub-networks
        # the probabilities are actually ensemble probabilities from each sub-network
        # merge type is used to merge the probabilities of each sub-network

        # check if confidence_thresholds is the correct shape
        if hasattr(confidence_thresholds, '__len__'):
            if len(confidence_thresholds) != self.models:
                raise Exception("Length of confidence_thresholds != number of models")
                exit()

        if len(self.all_probs) == 0 or force_repredict:
            self._predict(stamps, verbose=verbose, merge_type=merge_type)

        ind_model_classed = np.zeros(self.all_probs.shape, dtype=np.int)
        classed = np.zeros(self.all_probs.shape[1], dtype=np.int)

        for i in range(len(self.models)):
            if hasattr(confidence_thresholds, '__len__'):
                w = np.where(self.all_probs[i]>confidence_thresholds[i])
            else:
                w = np.where(self.all_probs[i]>confidence_thresholds)
            ind_model_classed[i, w] = 1
        zum = np.sum(ind_model_classed, axis=0)
        classed[np.where(zum==len(self.models))] = 1

        return classed

    def _filter_stamp_testing(self, stamps, confidence_thresholds = [0.9], verbose=0, merge_type='mean'):

        self._predict(stamps, verbose=verbose, merge_type=merge_type)

        with open('/arc/projects/classy/YTCSandbox/kbmod_results/2022-08-01/results_00/results_.txt') as han:
            data = han.readlines()
        x,y=[],[]
        for ii in range(len(data)):
            s = data[ii].split()
            x.append(float(s[5]))
            y.append(float(s[7]))
        x,y = np.array(x), np.array(y)
        print(y.shape,self.all_probs.shape)
        exit()

        classed = np.zeros(self.all_probs.shape, dtype=np.int)
        for i in range(len(self.models)):
            w = np.where(self.all_probs[i]>confidence_thresholds[i])
            classed[i, w] = 1

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
