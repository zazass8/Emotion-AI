from keras.layers import Lambda, Input, Dense, LeakyReLU, Flatten, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.core import Dropout
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from visuals import loss_function
from utils import import_mydata, norm, reshaping, apply_pseudo, feature_extraction_vae
from analyzer import Regression, Tuning
from common import BaseVAE, ica
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()


class VAE(BaseVAE):
    def __init__(self):
        super().__init__()

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

    def train(self, X_data, x_test):
        # Training
        vae_path = '/content/drive/My Drive/Dissertation/MYDATA_NEW/checkpoints/vae_mydata_'

        for participant in range(9):
            vae = self.vae()
            vae.compile(optimizer='adam')

            # Model checkpoint
            checkpoint = ModelCheckpoint(vae_path + str(participant + 1) + "best_VAE.h5", save_best_only=True, save_weights_only=False)

            # Train the autoencoder
            history = vae.fit(X_data[participant],
                              epochs=self.epochs,
                              steps_per_epoch=self.epoch_dim // self.batch_size,
                              batch_size=self.batch_size,
                              validation_data=(x_test[participant], None),
                              validation_steps=self.epoch_dim // self.batch_size,
                              callbacks=checkpoint)

            # Save the model weights
            vae.save_weights(vae_path + str(participant + 1) + '.h5', overwrite=True, save_format='h5')

            # Plot loss function
            loss_function(history)
    
    def denoise(self, X_data):
        # denoise the signal
        X_pred=np.zeros([9,95,100])
        for participant in range(9):
            X_pred[participant]=self.vae.predict(X_data[participant])[2]

        return X_pred


def main():
    # import the data
    path = "/content/drive/My Drive/Dissertation/MYDATA_NEW/"
    vae_path = '/content/drive/My Drive/Dissertation/MYDATA_NEW/checkpoints/vae_mydata_'
    channels=['F3','FC5','AF3','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4']

    X_data, y_data, X_00_data, _ = import_mydata(path)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, shuffle = False)
    
    # data normalisation
    X_train = norm(X_train)
    X_test = norm(X_test)

    # filtering
    X_train = ica(X_train, channels, [2], 14)
    X_test = ica(X_test, channels, [2], 14)

    # reshaping
    X_train, X_train2 = reshaping(X_train)
    X_test, _ = reshaping(X_test)

    # VAE
    vae = VAE()
    # vae = load_model(vae_path+'01_best_VAE.h5')

    # import model weights
    vae.load_weights(vae_path + '01.h5')

    # denoising
    X_pred_trn = vae.denoise(X_train)
    X_pred_tst = vae.denoise(X_test)

    # feature extraction
    psd_trn = feature_extraction_vae(X_pred_trn)
    psd_tst = feature_extraction_vae(X_pred_tst)

    # pseudo-labelling
    y_new_trn = apply_pseudo(psd_trn, y_train)
    y_new_tst = apply_pseudo(psd_tst, y_test)

    # regression and hyperparameter tuning
    regressor = Regression()
    tuner = Tuning()

    # SVR
    regressor.svr(psd_trn, y_new_trn, psd_tst, y_new_tst, 10+1)
    tuner.nonlinearity(psd_trn)

    # k-NN
    regressor.knn(psd_trn, y_new_trn, psd_tst, y_new_tst, 10+1)
    tuner.knn_tuning(psd_trn, y_new_trn, 76)

    # random forests
    regressor.rf(psd_trn, y_new_trn, psd_tst, y_new_tst, 10+1)
    tuner.rf_tuning(psd_trn, y_new_trn, 50, 100, 15)

if __name__ == "__main__":
    main()