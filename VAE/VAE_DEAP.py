from tensorflow.keras.layers import CategoryEncoding
from keras.layers import Lambda, Input, Dense, LeakyReLU, Flatten, Reshape, GlobalAveragePooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D,Conv2DTranspose, MaxPooling2D
from keras.layers.core import Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mean_squared_error
from keras.utils.vis_utils import plot_model
from keras import backend as K
from tensorflow.keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

import sklearn.multioutput as MOR
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import welch
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from mne.decoding import Vectorizer

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import argparse
import os
import h5py
import scipy.io as sio
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()

# importing data
def import_data(path):
    X_data=np.zeros([32,40,40,8064])
    y_data=np.zeros([32,40,3])
    for participant in range(32):
        if participant<9:
            sub_data_file = sio.loadmat(path+'0'+str(participant+1)+'.mat')
        else:
            sub_data_file = sio.loadmat(path+str(participant+1)+'.mat')
        x_train=sub_data_file['data']
        y_data[participant]=sub_data_file['labels'][:,:3]
        
        # data normalization
        mu=np.mean(x_train,axis=0)
        std=np.std(x_train,axis=0,ddof=1)
        for epoch in range(40):
            x_train[epoch]=(x_train[epoch]-mu)/std
        X_data[participant]=x_train
    return X_data, y_data

def reshaping(X_data):
    # reshaping
    channel_dim = 32
    X_data_reshaped=np.zeros([32,40,8064,32,1])
    for participant in range(X_data.shape[0]):
        x_train=tf.expand_dims(X_data[participant],3)

        x_train=x_train[:,:channel_dim,:,:]

        lst=[]
        for epoch in range(40):
            lst.append(tf.transpose(x_train[epoch,:,:,0]))
        x_train=tf.stack(lst)

        x_train=tf.expand_dims(x_train,3)
        X_data_reshaped[participant]=x_train
    x_test=X_data_reshaped

    return X_data_reshaped, x_test

class VAE:
    def __init__(self):
        # Hyperparameters
        self.channel_dim = 32
        self.epochs = 50
        self.batch_size = 2
        self.latent_dim = 15
        self.epoch_dim = 40
        self.subNum = 32
        self.zscore = True
        self.sample_dim = 8064
    
    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def encoder(self):
        # Encoder
        inputs = Input(shape=(self.sample_dim, self.channel_dim, 1), name='encoder_input')
        h = Conv2D(filters=30, kernel_size=(8, 2), strides=(8, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(inputs)
        h = BatchNormalization(axis=-1)(h)
        h = Conv2D(filters=30, kernel_size=(8, 2), strides=(8, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(h)
        h = BatchNormalization(axis=-1)(h)
        h = Conv2D(filters=30, kernel_size=(6, 2), strides=(6, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(h)
        h = BatchNormalization(axis=-1)(h)
        h = Dropout(0.5)(h)
        h = Flatten()(h)
        h = Dense(252, activation=LeakyReLU(alpha=0.3))(h)
        z_mean = Dense(100, name='z_mean')(h)
        z_log_var = Dense(100, name='z_log_var')(h)
        z = Lambda(self.sampling, output_shape=(100,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def decoder(self):
        # Decoder
        latent_inputs = Input(shape=(100,), name='z_sampling')
        h_decoded = Dense(252, activation=LeakyReLU(alpha=0.3))(latent_inputs)
        inputs_decoded = Dense(2520, activation=LeakyReLU(alpha=0.3))(h_decoded)
        inputs_decoded = Reshape((21, 4, 30))(inputs_decoded)
        inputs_decoded = Conv2DTranspose(30, kernel_size=(6, 2), strides=(6, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(inputs_decoded)
        inputs_decoded = BatchNormalization(axis=-1)(inputs_decoded)
        inputs_decoded = Dropout(0.5)(inputs_decoded)
        inputs_decoded = Conv2DTranspose(30, kernel_size=(8, 2), strides=(8, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(inputs_decoded)
        inputs_decoded = BatchNormalization(axis=-1)(inputs_decoded)
        inputs_decoded = Conv2DTranspose(1, kernel_size=(8, 2), strides=(8, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(inputs_decoded)
        inputs_decoded = BatchNormalization(axis=-1)(inputs_decoded)
        inputs_decoded = Reshape((self.sample_dim, self.channel_dim, 1))(inputs_decoded)
        decoder = Model(latent_inputs, inputs_decoded, name='decoder')
        return decoder

    def vae(self):
        # VAE
        inputs = Input(shape=(self.sample_dim, self.channel_dim, 1), name='encoder_input')
        encoder = self.encoder()
        decoder = self.decoder()

        outputs = decoder(encoder.output[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = K.square(inputs) - K.square(outputs)
        reconstruction_loss = K.mean(reconstruction_loss, axis=None)
        reconstruction_loss *= self.channel_dim

        kl_loss = 1 + encoder.output[1] - K.square(encoder.output[0]) - K.exp(encoder.output[1])
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        vae.add_loss(vae_loss)

        return vae
    
    def model_summary(self):
        # plot model summary and architecture
        vae = self.vae()
        vae.summary()
        plot_model(vae,to_file='vae_mlp.png', show_shapes=True)

    def train(self, X_data, x_test):
        # Training
        vae_path = '/content/drive/My Drive/Dissertation/DEAP/checkpoints/vae_deap_'
        encoder_path = '/content/drive/My Drive/Dissertation/DEAP/checkpoints/encoder_deap_'

        for participant in range(32):
            vae = self.vae()
            vae.compile(optimizer='adam')

            # Train the autoencoder
            history = vae.fit(X_data[participant],
                              epochs=self.epochs,
                              steps_per_epoch=self.epoch_dim // self.batch_size,
                              batch_size=self.batch_size,
                              validation_data=(x_test[participant], None),
                              validation_steps=self.epoch_dim // self.batch_size)

            # Save the model
            vae.save_weights(vae_path + str(participant + 1) + '.h5', overwrite=True, save_format='h5')
            encoder = self.encoder()
            encoder.save_weights(encoder_path + str(participant + 1) + '.h5', overwrite=True, save_format='h5')

            # Plot loss function
            plt.clf()

            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                                y=history.history['loss'],
                                name='Train'))
            fig.add_trace(go.Scattergl(
                                y=history.history['val_loss'],
                                name='Valid'))
            fig.update_layout(height=500, 
                                width=700,
                                title='Loss function',
                                xaxis_title='Epoch',
                                yaxis_title='Loss')
            fig.show()

def preprocessing(X_data, encoder):
    X_pred=np.zeros([32,40,100])
    for participant in range(32):
        X_pred[participant]=encoder.predict(X_data[participant])[2]

    return X_pred

# Power Spectral density
def feature_extraction(X_pred):
    psd=np.zeros([32,40,51])
    for participant in range(X_pred.shape[0]):
        psd[participant]=welch(X_pred[participant], fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
    return psd

class RegressionAnalyzer:
    def __init__(self):
        pass

    def svr(self, psd, y_data):
        rmse=[]
        mae=[]
        for participant in range(32):
            # REGRESSION 
            # With SVR
            svr=svm.SVR(kernel='rbf',C=1.0,epsilon=0.2)
            clf=MOR.MultiOutputRegressor(svr)
            y_pred=cross_val_predict(clf,psd[participant],y_data[participant],cv=5)

            # METRICS FOR SVR
            MSE=mean_squared_error(y_data[participant],y_pred)
            MAE=mean_absolute_error(y_data[participant],y_pred)

            rmse.append(MSE)
            mae.append(MAE)

        # Plot results in bar charts
        parts=np.arange(1,33)
        lst=list(range(1,33))
        fig=plt.figure(figsize=(12,6))
        ax=fig.add_subplot()
        ax=plt.gca()
        lns1=ax.bar(parts-0.2,np.sqrt(rmse),0.3,color='green',label='RMSE')
        ax2=ax.twinx()
        lns2=ax2.bar(parts+0.2,mae,0.3,color='red',label='MAE')
        plt.xticks(parts, lst)

        ax.set_xlabel('Participants')
        ax.set_ylabel('RMSE')
        ax2.set_ylabel('MAE')

        ax.legend([lns1,lns2],['RMSE','MAE'],loc=0)
        plt.title('SVMs')
        plt.grid(True)
        plt.show()

    def knn(self, psd, y_data):
        rmse=[]
        mae=[]
        for participant in range(32):
            # REGRESSION 
            # With k-NNs
            knn=KNeighborsRegressor(n_neighbors=23)
            regr_knn=MOR.MultiOutputRegressor(knn)
            y_pred_knn=cross_val_predict(regr_knn,psd[participant],y_data[participant],cv=5)

            # METRICS FOR k-NN
            MSE_knn=mean_squared_error(y_data[participant],y_pred_knn)
            MAE_knn=mean_absolute_error(y_data[participant],y_pred_knn)

            rmse.append(MSE_knn)
            mae.append(MAE_knn)
        
        # Plot results in bar charts
        parts=np.arange(1,33)
        lst=list(range(1,33))
        fig=plt.figure(figsize=(12,6))
        ax=fig.add_subplot()
        ax=plt.gca()
        lns1=ax.bar(parts-0.2,np.sqrt(rmse),0.3,color='green',label='RMSE')
        ax2=ax.twinx()
        lns2=ax2.bar(parts+0.2,mae,0.3,color='red',label='MAE')
        plt.xticks(parts, lst)

        ax.set_xlabel('Participants')
        ax.set_ylabel('RMSE')
        ax2.set_ylabel('MAE')

        ax.legend([lns1,lns2],['RMSE','MAE'],loc=0)
        plt.title('k-NNs')
        plt.grid(True)
        plt.show()

    def rf(self, psd, y_data):
        rmse=[]
        mae=[]
        for participant in range(32):
            # REGRESSION 
            # With RFs
            rf=RandomForestRegressor(n_estimators=58, criterion='absolute_error', max_depth=8)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd[participant],y_data[participant],cv=5)

            # METRICS FOR RF
            MSE_rf=mean_squared_error(y_data[participant],y_pred_rf)
            MAE_rf=mean_absolute_error(y_data[participant],y_pred_rf)

            rmse.append(MSE_rf)
            mae.append(MAE_rf)
        
        # Plot results in bar charts
        parts=np.arange(1,33)
        lst=list(range(1,33))
        fig=plt.figure(figsize=(12,6))
        ax=fig.add_subplot()
        ax=plt.gca()
        lns1=ax.bar(parts-0.2,np.sqrt(rmse),0.3,color='green',label='RMSE')
        ax2=ax.twinx()
        lns2=ax2.bar(parts+0.2,mae,0.3,color='red',label='MAE')
        plt.xticks(parts, lst)

        ax.set_xlabel('Participants')
        ax.set_ylabel('RMSE')
        ax2.set_ylabel('MAE')

        ax.legend([lns1,lns2],['RMSE','MAE'],loc=0)
        plt.title('Random Forests')
        plt.grid(True)
        plt.show()

class TuningAnalyzer:
    def __init__(self):
        pass

    # HYPER-PARAMETER TUNING FOR K-NNS
    # data only with a random participant
    def knn_tuning(self, psd, y_data):
        mse=[]
        for k in range(1,32):
            knn=KNeighborsRegressor(n_neighbors=k)
            regr_knn=MOR.MultiOutputRegressor(knn)
            y_pred_knn=cross_val_predict(regr_knn,psd,y_data[2],cv=5)
            MSE_knn=mean_squared_error(y_data[2],y_pred_knn)
            mse.append(MSE_knn)

        k=np.where(mse==np.min(mse))

        plt.figure(figsize=(7,5))
        plt.plot(range(1,32),mse)
        plt.title('MSE for k-neighbours')
        plt.xlabel('k')
        plt.ylabel('MSE')
        plt.show()
        print('')
        print('Number of k with lowest MSE equal to {} and with MSE equal to {}'.format(k[0][0],np.min(mse)))

    # HYPER-PARAMETER TUNING FOR RANDOM FORESTS
    def rf_tuning(self, psd, y_data):
        criteria=['squared_error','absolute_error', 'poisson']

        for criterion in criteria:
            mse=[]
            for estimator in range(1,50):
                rf=RandomForestRegressor(n_estimators=estimator, criterion=criterion, max_depth=None)
                regr_rf=MOR.MultiOutputRegressor(rf)
                y_pred_rf=cross_val_predict(regr_rf,psd,y_data[2],cv=5)
                MSE_rf=mean_squared_error(y_data[2],y_pred_rf)
                mse.append(MSE_rf)

            _k=np.where(mse==np.min(mse))

        plt.figure(figsize=(7,5))
        plt.plot(range(1,50),mse)
        plt.title('MSE for random forests with {} as criterion'.format(criterion))
        plt.xlabel('estimators')
        plt.ylabel('MSE')
        plt.show()
        print('')
        print('Number of estimators with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))

        # TUNING FOR NUMBER OF ESTIMATORS
        mse=[]
        for estimator in range(1,100):
            rf=RandomForestRegressor(n_estimators=estimator, criterion='absolute_error', max_depth=None)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd,y_data[2],cv=5)
            MSE_rf=mean_squared_error(y_data[2],y_pred_rf)
            mse.append(MSE_rf)

        _k=np.where(mse==np.min(mse))

        plt.figure(figsize=(7,5))
        plt.plot(range(1,100),mse)
        plt.title('MSE for random forests with {} as criterion'.format('absolute_error'))
        plt.xlabel('estimators')
        plt.ylabel('MSE')
        plt.show()
        print('')
        print('Number of estimators with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))

        # TUNING FOR MAX-DEPTH
        mse=[]
        for depth in range(1,15):
            rf=RandomForestRegressor(n_estimators=58, criterion='absolute_error', max_depth=depth)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd,y_data[2],cv=5)
            MSE_rf=mean_squared_error(y_data[2],y_pred_rf)
            mse.append(MSE_rf)

        _k=np.where(mse==np.min(mse))

        plt.figure(figsize=(7,5))
        plt.plot(range(1,15),mse)
        plt.title('MSE for random forests with {} as criterion'.format('absolute_error'))
        plt.xlabel('estimators')
        plt.ylabel('MSE')
        plt.show()
        print('')
        print('Maximum depth with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))

def main():
    # import the data
    path = '/content/drive/My Drive/Dissertation/DEAP/data_preprocessed_matlab/s'
    X_data, y_data = import_data(path)

    # reshaping
    X_data, X_test = reshaping(X_data)

    # VAE
    vae = VAE()
    encoder = vae.encoder()

    # preprocessing
    X_pred = preprocessing(X_data, encoder)

    # feature extraction
    psd = feature_extraction(X_pred)

    # regression and hyperparameter tuning
    regressor = RegressionAnalyzer()
    tuner = TuningAnalyzer()

    # SVR
    regressor.svr(psd, y_data)

    # k-NN
    regressor.knn(psd, y_data)
    tuner.knn_tuning(psd, y_data)

    # random forests
    regressor.rf(psd, y_data)
    tuner.rf_tuning(psd, y_data)

if __name__ == "__main__":
    main()