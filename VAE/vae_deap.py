from keras.layers import Lambda, Input, Dense, LeakyReLU, Flatten, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.layers.core import Dropout
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from visuals.plotting import loss_function
from utils import import_data, norm, reshaping, feature_extraction_vae
from analyzer import Regression, Tuning
from common import BaseVAE
# tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()

class VAE(BaseVAE):
    def __init__(self):
        super().__init__()

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

    def train(self, X_data, x_test):
        # Training
        vae_path = '/content/drive/My Drive/Dissertation/DEAP/checkpoints/vae_deap_'
        
        for participant in range(32):
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
        X_pred=np.zeros([32,40,100])
        for participant in range(32):
            X_pred[participant]=self.vae.predict(X_data[participant])[2]

        return X_pred


def main():
    # import the data
    path = '/content/drive/My Drive/Dissertation/DEAP/data_preprocessed_matlab/s'
    vae_path = '/content/drive/My Drive/Dissertation/DEAP/checkpoints/vae_deap_'
    X_data, y_data = import_data(path, channels = 40)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, shuffle = False)

    # normalise data
    X_train = norm(X_train)
    X_test = norm(X_test)

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

    # regression and hyperparameter tuning
    regressor = Regression()
    tuner = Tuning()

    # SVR
    regressor.svr(psd_trn, y_train, psd_tst, y_test, 32+1)

    # k-NN
    regressor.knn(psd_trn, y_train, psd_tst, y_test, 32+1)
    tuner.knn_tuning(psd_trn, y_train, 32)

    # random forests
    regressor.rf(psd_trn, y_train, psd_tst, y_test, 32+1)
    tuner.rf_tuning(psd_trn, y_train, 50, 100, 15)

if __name__ == "__main__":
    main()