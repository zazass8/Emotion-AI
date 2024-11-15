import numpy as np
import mne
from mne.preprocessing import ICA
from utils import norm

from keras.layers import Input
from keras.models import Model
import keras.backend as K
from keras.utils.vis_utils import plot_model
from abc import ABC, abstractmethod

def ica(X_data, channels, exclude_lst, n_comps):
    
    info=mne.create_info(ch_names=channels,sfreq=128,ch_types='eeg')
    for participant in range(X_data.shape[0]):
        for epoch in range(X_data.shape[1]):
            raw=mne.io.RawArray(X_data[participant][epoch],info)
            eeg_channels = mne.pick_types(raw.info, eeg=True,emg=True,ecg=True)
            cov=mne.compute_raw_covariance(raw, picks=eeg_channels)
            raw.del_proj()

            # BANDPASS FILTER
            filt_raw = raw.copy().filter(l_freq=4, h_freq=45,picks=eeg_channels)

            # ICA
            ica = ICA(n_components=n_comps,noise_cov=cov,method='fastica',max_iter='auto',random_state=97)
            ica.fit(filt_raw,picks=eeg_channels,reject_by_annotation=True)
            ica.exclude = exclude_lst
            reconst_raw = filt_raw.copy()
            ica.apply(reconst_raw)

            # X_NEW
            X_data[participant][epoch]=reconst_raw.get_data()
        
    # DATA NORMALIZATION
    X_data = norm(X_data)    
        
    return X_data

class BaseVAE(ABC):
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
    def sampling(self, args):
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
    
    @abstractmethod
    def encoder(self):
        """Abstract method for encoder architecture."""
        pass

    @abstractmethod
    def decoder(self):
        """Abstract method for decoder architecture."""
        pass

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
        vae = self.vae()
        vae.summary()
        plot_model(vae,to_file='vae_mlp.png', show_shapes=True)

    @abstractmethod
    def train(self, X_data, x_test):
        """Abstract method for training."""
        pass

    @abstractmethod
    def denoise(self):
        """Abstract method for decoder architecture."""
        pass