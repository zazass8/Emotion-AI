from utils import import_mydata, apply_pseudo, feature_extraction
from analyzer import Regression, Tuning
from sklearn.model_selection import train_test_split
from common import ica

def main():
    path = '/content/drive/My Drive/Dissertation/MYDATA_NEW/'
    X_data, y_data, _, __ = import_mydata(path)

    channels=['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1',
              'CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4',
              'P4','P8','CP6','CP2','C4','T8','FC6','FC2','F4',
              'F8','AF4','Fp2','Fz','Cz']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, shuffle = False)

    # preprocessing 
    X_train = ica(X_train, channels, [2], 14)
    X_test = ica(X_test, channels, [2], 14)
    
    # feature extraction
    psd_trn, psd_tuning_trn = feature_extraction(X_train)
    psd_tst, _ = feature_extraction(X_test)

    # pseudo-labelling
    y_new_trn = apply_pseudo(psd_trn, y_train)
    y_new_tst = apply_pseudo(psd_tst, y_test)

    # regressor, tuner
    regressor = Regression()
    tuner = Tuning()

    # Support Vector Machines
    regressor.svr(psd_trn, y_new_trn, psd_tst, y_new_tst, 10+1)

    # k-Nearest Neighbours
    regressor.knn(psd_trn, y_new_trn, psd_tst, y_new_tst, 10+1)
    tuner.knn_tuning(psd_tuning_trn, y_new_trn, 76)

    # Random Forests
    regressor.rf(psd_trn, y_new_trn, psd_tst, y_new_tst, 10+1)
    tuner.rf_tuning(psd_tuning_trn, y_new_trn, 50, 100, 15)

if __name__ == "__main__":
    main()
