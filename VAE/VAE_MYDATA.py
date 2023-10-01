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
from keras.callbacks import ModelCheckpoint
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
from mne.preprocessing import ICA
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

def import_data(path, path_00):
    # Import files from mydata
    X_data=np.zeros([9,95,14,512])
    y_data=np.zeros([9,95,3])

    for participant in range(9):
        # '/content/drive/My Drive/Dissertation/MYDATA_NEW/s0'
        _path=path+str(participant+1)+'/'
        X=pd.read_csv(_path+'data_s0'+str(participant+1)+'.csv')
        X=X.iloc[:,1:]
        X=X.to_numpy()
        for epoch in range(95):
            X_data[participant][epoch]=np.transpose(X[512*epoch:512*(epoch+1)])

        y=pd.read_csv(_path+'labels_s0'+str(participant+1)+'.csv')
        y=y.iloc[:,1:]
        y=y.to_numpy()
        y_data[participant]=y

    # path_00='/content/drive/My Drive/Dissertation/MYDATA_NEW/s00/'

    X_00_data=np.zeros([94,14,512])
    X_00=pd.read_csv(path_00+'data_s00.csv')
    X_00=X_00.iloc[:,1:]
    X_00=X_00.to_numpy()
    for epoch in range(94):
        X_00_data[epoch]=np.transpose(X_00[512*epoch:512*(epoch+1)])

    y_00=pd.read_csv(path_00+'labels_s00.csv')
    y_00=y_00.iloc[:,1:]
    y_00=y_00.to_numpy()

    return X_data, y_data, X_00_data, y_00

def normalize(X_data, X_00_data):
    # data normalization
    for participant in range(X_data.shape[0]):
        mu=np.mean(X_data[participant],axis=0)
        std=np.std(X_data[participant],axis=0,ddof=1)
        for epoch in range(X_data.shape[1]):
            X_data[participant][epoch]=(X_data[participant][epoch]-mu)/std

    mu_00=np.mean(X_00_data[participant],axis=0)
    std_00=np.std(X_00_data[participant],axis=0,ddof=1)
    for epoch in range(X_00_data.shape[0]):
        X_00_data[epoch]=(X_00_data[epoch]-mu_00)/std_00

    return X_data, X_00_data

def reshaping(X_data, X_00_data):
    X_data_reshaped=np.zeros([9,95,512,14,1])
    X_data_reshaped_00=np.zeros([94,512,14,1])
    for participant in range(X_data.shape[0]):
        x_train=tf.expand_dims(X_data[participant],3)

        lst=[]
        for epoch in range(X_data.shape[1]):
            lst.append(tf.transpose(x_train[epoch,:,:,0]))
        x_train=tf.stack(lst)
        x_train=tf.expand_dims(x_train,3)
        X_data_reshaped[participant]=x_train

    x_test=X_data_reshaped

    x_train_00=tf.expand_dims(X_00_data,3)
    lst_00=[]
    for epoch in range(X_00_data.shape[0]):
        lst_00.append(tf.transpose(x_train_00[epoch,:,:,0]))
    x_train_00=tf.stack(lst_00)
    x_train_00=tf.expand_dims(x_train_00,3)
    X_data_reshaped_00=x_train_00
    x_test_00=X_data_reshaped_00

    return X_data_reshaped, x_test, X_data_reshaped_00, x_test_00  

def filtering(X_data, X_00_data):
    # FILTERING AND ICA
    # BANDPASS FILTER

    channels=['F3','FC5','AF3','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4']
    info=mne.create_info(ch_names=channels,sfreq=128,ch_types='eeg')

    for participant in range(X_data.shape[0]):
        for epoch in range(X_data.shape[0]):
            raw=mne.io.RawArray(X_data[participant][epoch],info)
            eeg_channels = mne.pick_types(raw.info, eeg=True,emg=True,ecg=True,eog=True)
            cov=mne.compute_raw_covariance(raw, picks=eeg_channels)
            raw.del_proj()
            filt_raw = raw.copy().filter(l_freq=4, h_freq=50,picks=eeg_channels)

            # ICA
            ica = ICA(n_components=14,noise_cov=cov,method='fastica',max_iter='auto',random_state=97)
            ica.fit(filt_raw,picks=eeg_channels,reject_by_annotation=True)
            ica.exclude = [2]
            reconst_raw = filt_raw.copy()
            ica.apply(reconst_raw)

            X_data[participant][epoch]=reconst_raw.get_data()

    for epoch in range(X_00_data.shape[0]):
        raw_00=mne.io.RawArray(X_00_data[epoch],info)
        eeg_channels_00 = mne.pick_types(raw_00.info, eeg=True,emg=True,ecg=True,eog=True)
        cov_00=mne.compute_raw_covariance(raw_00, picks=eeg_channels_00)
        raw_00.del_proj()
        filt_raw_00 = raw_00.copy().filter(l_freq=4, h_freq=50,picks=eeg_channels_00)

        ica_00 = ICA(n_components=14,noise_cov=cov_00,method='fastica',max_iter='auto',random_state=97)
        ica_00.fit(filt_raw_00,picks=eeg_channels_00,reject_by_annotation=True)
        ica_00.exclude = [2]
        reconst_raw_00 = filt_raw_00.copy()
        ica_00.apply(reconst_raw)

        X_00_data[epoch]=reconst_raw_00.get_data()
    
    return X_data, X_00_data

# PSEUDO CLASSIFIER
def pseudo(X,y,estimator):
    "Classifier that is used to generate pseudo-labels for observations in data that are unlabelled"
    # INDICES
    unlabelled_indices=np.where(y==0)
    labelled_indices=np.where(y!=0)

    # LABELLED/UNLABELLED DATA WITH TARGETS
    X_labelled=X[labelled_indices]
    y_labelled=y[labelled_indices]

    X_unlabelled=X[unlabelled_indices]

    # PREDICTIONS FOR PSEUDO-LABELS
    regr_labelled=estimator.fit(X_labelled,y_labelled)
    pred_labelled=regr_labelled.predict(X_unlabelled)

    y[unlabelled_indices]=pred_labelled

    return y

def apply_pseudo(psd, y_data):
    """Apply the pseudo classifier"""
    y_new=np.zeros([9,95,3])
    estimator=svm.SVC(kernel='rbf',C=0.1)

    for participant in range(9):
        y_new[participant,:,0]=pseudo(psd[participant],y_data[participant,:,0],estimator)
        y_new[participant,:,1]=pseudo(psd[participant],y_data[participant,:,1],estimator)
        y_new[participant,:,2]=pseudo(psd[participant],y_data[participant,:,2],estimator)

    return y_new


class VAE:
    def __init__(self):
        # Hyperparameters
        self.channel_dim = 14
        self.epochs = 50
        self.batch_size = 5
        self.latent_dim = 15
        self.epoch_dim = 95
        self.subNum = 32
        self.zscore = True
        self.sample_dim = 512
    
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
        h = Conv2D(filters=30, kernel_size=(3, 2), strides=(3, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(inputs)
        h = BatchNormalization(axis=-1)(h)
        h = Conv2D(filters=30, kernel_size=(3, 2), strides=(3, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(h)
        h = BatchNormalization(axis=-1)(h)
        h = Conv2D(filters=30, kernel_size=(3, 2), strides=(3, 2), padding='valid', activation=LeakyReLU(alpha=0.3))(h)
        h = BatchNormalization(axis=-1)(h)
        h = Dropout(0.5)(h)
        h = Flatten()(h)
        h = Dense(270, activation=LeakyReLU(alpha=0.3))(h)
        z_mean = Dense(100, name='z_mean')(h)
        z_log_var = Dense(100, name='z_log_var')(h)
        z = Lambda(self.sampling, output_shape=(100,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def decoder(self):
        # Decoder
        latent_inputs = Input(shape=(100,), name='z_sampling')
        h_decoded = Dense(270, activation=LeakyReLU(alpha=0.3))(latent_inputs)
        inputs_decoded = Dense(540, activation=LeakyReLU(alpha=0.3))(h_decoded)
        inputs_decoded = Reshape((18, 1, 30))(inputs_decoded)
        inputs_decoded = Conv2DTranspose(30, kernel_size=(3, 2), strides=(3, 2), padding='valid', output_padding=(2,1), activation=LeakyReLU(alpha=0.3))(inputs_decoded)
        inputs_decoded = BatchNormalization(axis=-1)(inputs_decoded)
        inputs_decoded = Dropout(0.5)(inputs_decoded)
        inputs_decoded = Conv2DTranspose(30, kernel_size=(3, 2), strides=(3, 2), padding='valid', output_padding=(2,1), activation=LeakyReLU(alpha=0.3))(inputs_decoded)
        inputs_decoded = BatchNormalization(axis=-1)(inputs_decoded)
        inputs_decoded = Conv2DTranspose(1, kernel_size=(3, 2), strides=(3, 2), padding='valid', output_padding=(2,1), activation=LeakyReLU(alpha=0.3))(inputs_decoded)
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

        # Loss function
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
        vae_path = '/content/drive/My Drive/Dissertation/MYDATA_NEW/checkpoints/vae_mydata_'
        encoder_path = '/content/drive/My Drive/Dissertation/MYDATA_NEW/checkpoints/encoder_mydata_'

        for participant in range(9):
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
    # denoise the signal
    X_pred=np.zeros([9,95,100])
    for participant in range(9):
        X_pred[participant]=encoder.predict(X_data[participant])[2]

    return X_pred

# Power Spectral density
def feature_extraction(X_pred):
    psd=np.zeros([9,95,51])
    for participant in range(X_pred.shape[0]):
        psd[participant]=welch(X_pred[participant], fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
    return psd

class RegressionAnalyzer:
    def __init__(self):
        pass

    def svr(self, psd, y_data):
        rmse=[]
        mae=[]
        for participant in range(9):
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
        parts=np.arange(1,11)
        lst=list(range(1,11))
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
        for participant in range(9):
            # REGRESSION 
            # With k-NNs
            knn=KNeighborsRegressor(n_neighbors=25)
            regr_knn=MOR.MultiOutputRegressor(knn)
            y_pred_knn=cross_val_predict(regr_knn,psd[participant],y_data[participant],cv=5)

            # METRICS FOR k-NN
            MSE_knn=mean_squared_error(y_data[participant],y_pred_knn)
            MAE_knn=mean_absolute_error(y_data[participant],y_pred_knn)

            rmse.append(MSE_knn)
            mae.append(MAE_knn)
        
        # Plot results in bar charts
        parts=np.arange(1,11)
        lst=list(range(1,11))
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
        for participant in range(9):
            # REGRESSION 
            # With RFs
            rf=RandomForestRegressor(n_estimators=55, criterion='squared_error', max_depth=1)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd[participant],y_data[participant],cv=5)

            # METRICS FOR RF
            MSE_rf=mean_squared_error(y_data[participant],y_pred_rf)
            MAE_rf=mean_absolute_error(y_data[participant],y_pred_rf)

            rmse.append(MSE_rf)
            mae.append(MAE_rf)
        
        # Plot results in bar charts
        parts=np.arange(1,11)
        lst=list(range(1,11))
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
    
    def nonlinearity(self, psd):
        # proof of non-linearity of data
        x=np.mean(x,axis=1)

        anger=x[[0,9,18,27,40,49,58,67,75,84,93]]
        disgust=x[[1,10,19,28,42,51,60,68,71,80,89]]

        # PLOT TO SHOW DATA ARE NOT LINEARLY SEPERABLE
        plt.figure(figsize=(7,5))
        plt.scatter(anger,anger,color=['orange'],label='anger',marker='*')
        plt.scatter(disgust,disgust,color='blue',label='disgust',marker='+')
        plt.legend()
        plt.show()

    # HYPER-PARAMETER TUNING FOR K-NNS
    # data only with a random participant
    def knn_tuning(self, psd, y_data):
        mse=[]
        for k in range(1,76):
            knn=KNeighborsRegressor(n_neighbors=k)
            regr_knn=MOR.MultiOutputRegressor(knn)
            y_pred_knn=cross_val_predict(regr_knn,psd,y_data[1],cv=5)
            MSE_knn=mean_squared_error(y_data[1],y_pred_knn)
            mse.append(MSE_knn)

        k=np.where(mse==np.min(mse))

        plt.figure(figsize=(7,5))
        plt.plot(range(1,76),mse)
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
                y_pred_rf=cross_val_predict(regr_rf,psd,y_data[1],cv=5)
                MSE_rf=mean_squared_error(y_data[1],y_pred_rf)
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
            rf=RandomForestRegressor(n_estimators=estimator, criterion='squared_error', max_depth=None)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd,y_data[1],cv=5)
            MSE_rf=mean_squared_error(y_data[1],y_pred_rf)
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
            rf=RandomForestRegressor(n_estimators=55, criterion='squared_error', max_depth=depth)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd,y_data[1],cv=5)
            MSE_rf=mean_squared_error(y_data[1],y_pred_rf)
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
    X_data, y_data, X_00_data, _ = import_data(path)

    # data normalisation
    X_data, X_00_data = normalize(X_data, X_00_data)

    # filtering
    X_data, X_00_data = filtering(X_data, X_00_data)

    # reshaping
    X_data, X_test, X_00_data, X_test_00  = reshaping(X_data, X_00_data)

    # VAE
    vae = VAE()
    encoder = vae.encoder()

    # preprocessing
    X_pred = preprocessing(X_data, encoder)

    # feature extraction
    psd = feature_extraction(X_pred)

    # pseudo-labelling
    y_new = apply_pseudo(psd, y_data)

    # regression and hyperparameter tuning
    regressor = RegressionAnalyzer()
    tuner = TuningAnalyzer()

    # SVR
    regressor.svr(psd, y_new)

    # k-NN
    regressor.knn(psd, y_new)
    tuner.knn_tuning(psd, y_new)

    # random forests
    regressor.rf(psd, y_new)
    tuner.rf_tuning(psd, y_new)

if __name__ == "__main__":
    main()