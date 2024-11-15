import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn import svm
from scipy.signal import welch
from mne.decoding import Vectorizer

def import_data(path, channels):
    """Importing the DEAP data"""

    X_data=np.zeros([32, 40, channels, 8064])
    y_data=np.zeros([32,40,3])
    for participant in range(32):
        if participant<9:
            sub_data_file = sio.loadmat(path+'0'+str(participant+1)+'.mat')
        else:
            sub_data_file = sio.loadmat(path+str(participant+1)+'.mat')
        x_train=sub_data_file['data']
        X_data[participant]=x_train[:,:channels,:]
        y_data[participant]=sub_data_file['labels'][:,:3]
    return X_data, y_data

def import_mydata(path):
    """Import files from mydata"""

    X_data=np.zeros([9,95,14,512])
    y_data=np.zeros([9,95,3])

    for participant in range(9):
        _path=path+str(participant+1)+'/'
        X=pd.read_csv(_path+'data_s0'+str(participant+1)+'.csv').iloc[:,1:].to_numpy()
        for epoch in range(95):
            X_data[participant][epoch]=np.transpose(X[512*epoch:512*(epoch+1)])

        y=pd.read_csv(_path+'labels_s0'+str(participant+1)+'.csv')
        y=y.iloc[:,1:]
        y=y.to_numpy()
        y_data[participant]=y

    X_00_data=np.zeros([94,14,512])
    X_00=pd.read_csv(path+'data_s00.csv').iloc[:,1:].to_numpy()
    for epoch in range(94):
        X_00_data[epoch]=np.transpose(X_00[512*epoch:512*(epoch+1)])

    y_00=pd.read_csv(path+'labels_s00.csv')
    y_00=y_00.iloc[:,1:]
    y_00=y_00.to_numpy()

    return X_data, y_data, X_00_data, y_00

def norm(X_data, X_00_data=None):
    """data normalization"""
    
    for participant in range(X_data.shape[0]):
        mu=np.mean(X_data[participant],axis=0)
        std=np.std(X_data[participant],axis=0,ddof=1)
        for epoch in range(X_data.shape[1]):
            X_data[participant][epoch]=(X_data[participant][epoch]-mu)/std

    if X_00_data is not None:
        for participant in range(X_00_data.shape[0]):
            mu_00=np.mean(X_00_data[participant],axis=0)
            std_00=np.std(X_00_data[participant],axis=0,ddof=1)
            for epoch in range(X_00_data.shape[1]):
                X_00_data[participant][epoch]=(X_00_data[participant][epoch]-mu_00)/std_00

        return X_data, X_00_data
    
    return X_data

def reshaping(X_data, X_00_data=None):
    """reshaping"""

    X_data_reshaped = X_data.transpose(0, 1, 3, 2)
    X_data_reshaped = np.expand_dims(X_data_reshaped, axis = -1)
    x_test = X_data_reshaped

    if X_00_data is not None:
        X_00_data_reshaped = X_00_data.transpose(0, 2, 1)
        X_00_data_reshaped = np.expand_dims(X_00_data_reshaped, axis = -1)
        x_test_00 = X_00_data_reshaped
        return X_data_reshaped, x_test, X_00_data_reshaped, x_test_00 

    return X_data_reshaped, x_test

def apply_pseudo(psd, y_data):
    """Apply the pseudo classifier"""

    def pseudo(X,y,estimator):
        "Classifier that is used to generate pseudo-labels for observations in data that are unlabelled"

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

    y_new=np.zeros([9,95,3])
    estimator=svm.SVC(kernel='rbf',C=0.1)

    for participant in range(y_new.shape[0]):
        y_new[participant,:,0]=pseudo(psd[participant],y_data[participant,:,0],estimator)
        y_new[participant,:,1]=pseudo(psd[participant],y_data[participant,:,1],estimator)
        y_new[participant,:,2]=pseudo(psd[participant],y_data[participant,:,2],estimator)

    return y_new

def feature_extraction(X_new):
    """Power Spectral Density using the Welch algorithm"""

    vec=Vectorizer()
    X_new_=[]
    psd=[]

    dec=X_new[1]
    psd_knn=welch(dec, fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
    psd_tuning=vec.fit_transform(psd_knn)

    for participant in range(X_new.shape[0]):
        psd.append(welch(X_new[participant], fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1])
        X_new_.append(vec.fit_transform(psd[participant]))
    
    return np.array(X_new_), psd_tuning

def feature_extraction_vae(X_pred):
    """Power Spectral Density using the Welch algorithm"""

    psd=[]
    for participant in range(X_pred.shape[0]):
        psd.append(welch(X_pred[participant], fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1])
    return np.array(psd)