"""Import the basics: numpy, pandas, matplotlib, etc."""
import numpy as np
import pickle, os, time
"""Import keras and other ML tools"""
import tensorflow as tf
import keras
import keras.backend as K

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Activation, Add, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPool2D, Concatenate
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.optimizers import adam_v2 as Adam
from tensorflow.keras.optimizers import Adam

###for custom keras loss function
#from keras.layers import Lambda, Multiply,Add
#from tensorflow.python.keras.losses import Loss, LossFunctionWrapper
#from tensorflow.python.keras.utils import losses_utils
#from tensorflow.python.util.tf_export import keras_export

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


"""
The resnet blocks used here are lifted gleefully from
https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/
"""

class resnet_model():

    def __init__(self, input_shape, learning_rate = 0.0007,
                 dropout_rate = 0.2,
                 num_dense_nodes = 64,
                 num_models = 3, unique_labels = 2,
                 block_layers = [2,2], initial_filter_size=16):


        super(resnet_model, self).__init__()

        self.input_shape = input_shape
        self._model_type = 'cnn'
        self.lr = learning_rate
        self.initializer = 'he_normal'
        #self.initializer = 'random_uniform'
        self.activation = 'relu'
        #self.num_dense_layers = num_dense_layers
        self.num_dense_nodes = num_dense_nodes
        self.num_models = num_models

        self.beta_1 = 0.9  # exponential decay rate for the 1st moment estimates for optimization algorithm
        self.beta_2 = 0.999  # exponential decay rate for the 2nd moment estimates for optimization algorithm
        self.optimizer_epsilon = 1e-08  # a small constant for numerical stability for optimization algorithm
        self.clipnorm = 1.0
        self.dropout_rate = dropout_rate
        self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2,
                              epsilon=self.optimizer_epsilon,
                              clipnorm=self.clipnorm)
        
        self.last_layer_activation = 'softmax'

        self.loss_func = 'categorical_crossentropy'

        self.unique_labels = unique_labels

        self.classifiers = {}

        self.block_layers = block_layers #[2, 2] # len==# of resnet layers, index==# of CNN layers in each resnet layer
        self.initial_filter_size = initial_filter_size # the # of filters in the initial layers of the resnet. It doubles for each additional ResNet layer
        
    def compile(self, verbose = 1):

        input_tensor = Input(shape=self.input_shape, name='input')

        checkpointer = ModelCheckpoint('keras_cnn_kbmod_model.h5', verbose=verbose)
        self.callbacks = [checkpointer]

        self.models = {}
        for ii in range(self.num_models):
            model_ = Model(input_tensor, self.mag_resnet_model(input_tensor))

            model_.compile(optimizer=self.optimizer,
                            loss = self.loss_func,
                            metrics=["accuracy"])
            self.models[ii] = model_
        self.model_ = self.models[ii]

    def identity_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = Conv2D(filter, (3,3), padding = 'same')(x)
        #####x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        # Layer 2
        x = Conv2D(filter, (3,3), padding = 'same')(x)
        #####x = BatchNormalization(axis=-1)(x)
        # Add Residue
        x = Add()([x, x_skip])
        x = Activation('relu')(x)
        return x

    def convolutional_block(self, x, filter):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
        #####x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        # Layer 2
        x = Conv2D(filter, (3,3), padding = 'same')(x)
        #####x = BatchNormalization(axis=-1)(x)
        # Processing Residue with conv(1,1)
        x_skip = Conv2D(filter, (1,1), strides = (2,2))(x_skip)
        # Add Residue
        x = Add()([x, x_skip])
        x = Activation('relu')(x)
        return x


    def mag_resnet_model(self, x):
        # Step 2 (Initial Conv layer along with maxPool)
        x = Conv2D(filters=16, kernel_size=(3, 3), input_shape=self.input_shape, padding='same')(x)
        #####x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
        # Define size of sub-blocks and initial filter size
        block_layers = self.block_layers
        filter_size = self.initial_filter_size
        # Step 3 Add the Resnet Blocks
        for i in range(len(block_layers)):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    x = self.identity_block(x, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size*2
                x = self.convolutional_block(x, filter_size)
                for j in range(block_layers[i] - 1):
                    x = self.identity_block(x, filter_size)
        # Step 4 End Dense Network
        #x = AveragePooling2D((2,2), padding = 'same')(x)
        x = MaxPool2D(pool_size=(2,2), padding = 'same')(x)
        x = Flatten()(x)
        x = Dense(self.num_dense_nodes, activation = self.activation)(x)
        output = Dense(self.unique_labels, activation = self.last_layer_activation)(x)
        ####output = Dense(self.unique_labels, activation = 'sigmoid')(x) #### this seems to enhance the peaks of middling probability. softmax seems to minimize them, which is good.
        return output

        """
        cnl_1 = Conv2D(filters=16, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu', padding='same')(x)
        do_1 = Dropout(self.dropout_rate)(cnl_1)
        mp_1 = MaxPool2D(pool_size=(2, 2), padding='valid')(do_1)

        cnl_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(mp_1)
        do_2 = Dropout(self.dropout_rate)(cnl_2)
        mp_2 = MaxPool2D(pool_size=(2, 2), padding='valid')(do_2)

        bn_1 = BatchNormalization()(mp_2)

        cnl_3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(bn_1)
        do_3 = Dropout(self.dropout_rate)(cnl_3)
        mp_3 = MaxPool2D(pool_size=(2, 2), padding='valid')(do_3)

        flattener = Flatten()(mp_3)

        d_1 = Dense(self.num_dense_nodes, activation = self.activation)(flattener)
        d_2 = Dense(self.num_dense_nodes, activation = self.activation)(d_1)

        output = Dense(self.unique_labels, activation = 'softmax')(d_2)

        return output
        """

    def train_models_with_one_trainset(self, X_train, Y_train,
                                       sample_weights = None,
                                       train_epochs = 90,
                                       batch_size=512,
                                       verbose=1
                                       ):
        #checkpointer = ModelCheckpoint('keras_cnn_mag_model.h5', verbose=1)

        start = time.time()
        self.classifiers = {}
        for ii in range(self.num_models):
            print(f'\nTraining model {ii+1} of {self.num_models}.')
            t1 = time.time()
            if sample_weights is not None:
                self.classifiers[ii] = self.models[ii].fit(X_train, Y_train, sample_weight = sample_weights,
                                                           epochs=train_epochs,
                                                           batch_size=batch_size,
                                                           callbacks=[],#callbacks=[ checkpointer],
                                                           verbose=verbose)
            else:
                self.classifiers[ii] = self.models[ii].fit(X_train, Y_train,
                                                           epochs=train_epochs,
                                                           batch_size=batch_size,
                                                           callbacks=[],#callbacks=[ checkpointer],
                                                           verbose=verbose)
            t2 = time.time()
            print(f'...in {t2-t1} seconds.')
        end = time.time()
        print('Training completed in', round(end-start, 2), ' seconds')


    def train_all_models(self, data, labels,
                         test_fraction=0.1,
                         random_state = None,
                         train_epochs = 90,
                         batch_size=512,
                         useSampleWeights = True,
                         nominal_false_weight=0.83):
        for ii in range(self.num_models):
            self.train_one_model(data, labels,
                                 test_fraction=test_fraction,
                                 model_to_train = ii,
                                 random_state = random_state,
                                 train_epochs = train_epochs,
                                 batch_size=batch_size,
                                 useSampleWeights = useSampleWeights,
                                 nominal_false_weight=nominal_false_weight)


    def train_one_model(self, X_train, Y_train,
                        test_fraction=0.1,
                        model_to_train = 0,
                        random_state = None,
                        sample_weights = None,
                        train_epochs = 90,
                        batch_size=512,
                        useSampleWeights = True,
                        nominal_false_weight=0.83,
                        verbose=1):

        """
        print(f'Shuffle splitting for model {model_to_train+1}')
        skf = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
        skf.split(data, labels)


        for train_index, test_index in skf.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            Y_train, Y_test = labels[train_index], labels[test_index]
        """

        checkpointer = ModelCheckpoint('keras_resnet_mag_model.h5', verbose=1)

        start = time.time()
        if len(self.classifiers)<self.num_models:
            for ii in range(self.num_models):
                self.classifiers[ii] = [None]

        ii = model_to_train
        print(f'\nTraining model {ii+1} of {self.num_models}.')
        if sample_weights is not None:
            self.classifiers[ii] = self.models[ii].fit(X_train, Y_train, sample_weight = sample_weights,
                                                       epochs=train_epochs,
                                                       batch_size=batch_size,
                                                       callbacks=[ checkpointer],
                                                       verbose=1)
        else:
            self.classifiers[ii] = self.models[ii].fit(X_train, Y_train,
                                                       epochs=train_epochs,
                                                       batch_size=batch_size,
                                                       callbacks=[ checkpointer],
                                                       verbose =1)
        end = time.time()
        print('Training completed in', round(end-start, 2), ' seconds')


    def predict(self, X, merge_type='mean', verbose = 1, batch_size=4096):
        full_preds = []
        for ii in range(self.num_models):
            p = self.models[ii].predict(X, verbose=verbose, batch_size=batch_size)
            full_preds.append(p)

        full_preds = np.array(full_preds) #shape (# internal_model per model, # sources, # models)

        # for merge_types mean/median,max merges of internal models, and returns array of shape (# of sources, # of models)
        # for merge_type==None it returns the full_probability array
        if merge_type =='mean':
            out = np.mean(full_preds, axis=0)
        elif merge_type =='median':
            out = np.median(full_preds, axis=0)
        elif merge_type =='max':
            out = np.max(full_preds, axis=0)
        elif merge_type == None:
            out = full_preds
        return(out)


    def saveModel(self, means, stds, save_dir = 'RNML_KBmod_modelSave', random_split_seed = None):
        for ii in range(self.num_models):
            dir_name = f'{save_dir}/model_{ii}'
            if os.path.isdir(dir_name):
                os.system(f'rm -r {dir_name}')
            os.system(f'mkdir -p {dir_name}')
            self.models[ii].save_weights(f'{dir_name}/ensemble_weights.h5')
            print(f'   saved to {dir_name}\n')

        with open(f'{save_dir}/median.properties','w+') as han:
            if type(means) is list:
                for ii in range(self.num_models):
                    print(ii, means[ii], stds[ii], file=han)
            else:
                print(means, stds, file=han)
            print('Random split seed:,', random_split_seed, file=han)

    def loadModel(self, save_dir = 'RNML_KBmod_modelSave'):

        for ii in range(self.num_models):
            dir_name = f'{save_dir}/model_{ii}'
            print(f'\n   Loading model {dir_name}\n')
            self.models[ii].load_weights(f'{dir_name}/ensemble_weights.h5')

        with open(f'{save_dir}/median.properties') as han:
            data = han.readlines()
        if len(data)>2:
            means, std = [], []
            for ii in range(self.num_models):
                s = data[ii].split()
                mean = float(s[0])
                std = float(s[1])
                means.append(mean)
                stds.append(std)
            random_split_seed = int(float(data[-1].split()[3]))
            return (means, stds, random_split_seed)
        else:
            s = data[0].split()
            mean = float(s[0])
            std = float(s[1])
            random_split_seed = int(float(data[-1].split()[3]))
            return (mean, std, random_split_seed)

    def summary(self):
        print(f'Ensemble of {self.num_models}, each with the follwing configuration:\n')
        self.models[0].summary()
