from analyzer import Regression, Tuning
from common import ica
from sklearn.model_selection import train_test_split
from utils import import_data, feature_extraction


def main():
    path = '/content/drive/My Drive/Dissertation/DEAP/data_preprocessed_matlab/s'
    X_data, y_data = import_data(path, channels = 32)

    channels=['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1',
              'CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4',
              'P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4',
              'F8','AF4','Fp2','Fz','Cz']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, shuffle = False)

    # preprocessing 
    X_train = ica(X_train, channels, [], 32)
    X_test = ica(X_test, channels, [], 32)
    
    # feature extraction
    psd_trn, psd_tuning_trn = feature_extraction(X_train)
    psd_tst, _ = feature_extraction(X_test)

    # regressor
    regressor = Regression()

    # tuner 
    tuner = Tuning()

    # Support Vector Machines
    regressor.svr(psd_trn, y_train, psd_tst, y_test, 32+1)

    # k-Nearest Neighbours
    regressor.knn(psd_trn, y_train, psd_tst, y_test, 32+1)
    tuner.knn_tuning(psd_tuning_trn, y_train, 32)

    # Random Forests
    regressor.rf(psd_trn, y_train, psd_tst, y_test, 32+1)
    tuner.rf_tuning(psd_tuning_trn, y_train, 50, 100, 15)

if __name__ == "__main__":
    main()
