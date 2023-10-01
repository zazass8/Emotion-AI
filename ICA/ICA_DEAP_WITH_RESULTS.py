import mne
import pymatreader
import scipy
import numpy as np
import scipy.io as sio
import os
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
import sklearn.multioutput as MOR
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import welch
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from mne.decoding import Vectorizer
from mne.decoding import Scaler, PSDEstimator, FilterEstimator
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import hfda

from scipy.signal import welch
import scipy.stats as stats
import matplotlib.pyplot as plt


# Importing the DEAP dataset
def import_data(path):
    X_data=np.zeros([32,40,32,8064])
    y_data=np.zeros([32,40,3])
    for i in range(32):
        if i<9:
            sub_data_file = sio.loadmat(path+'0'+str(i+1)+'.mat')
        else:
            sub_data_file = sio.loadmat(path+str(i+1)+'.mat')
        X = sub_data_file['data']
        X_data[i]=X[:,:32,:]
        y = sub_data_file['labels']
        y_data[i]=y[:,:3]
    return X_data, y_data

def preprocessing(X_data, channels):
    X_new=np.zeros([32,40,32,8064])
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
            ica = ICA(n_components=32,noise_cov=cov,method='fastica',max_iter='auto',random_state=97)
            ica.fit(filt_raw,picks=eeg_channels,reject_by_annotation=True)
            ica.exclude = []
            reconst_raw = filt_raw.copy()
            ica.apply(reconst_raw)

            # X_NEW
            X_new[participant][epoch]=reconst_raw.get_data()
        
        # DATA NORMALIZATION
        mu=np.mean(X_new[participant],axis=0)
        std=np.std(X_new[participant],axis=0,ddof=1)
        for epoch in range(X_new.shape[1]):
            X_new[participant][epoch]=(X_new[participant][epoch]-mu)/std
        
    return X_new

def feature_extraction(X_new):
    vec=Vectorizer()
    X_new_=np.zeros([32,40,4128])
    psd=np.zeros([32,40,32,129])

    dec=X_new[2]
    psd_tuning=welch(dec, fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
    psd_tuning=vec.fit_transform(psd)

    for participant in range(X_new.shape[0]):
        psd[participant]=welch(X_new[participant], fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
        X_new_[participant]=vec.fit_transform(psd[participant])
    
    return X_new_, psd_tuning

class RegressionAnalyzer:
    def __init__(self):
        pass

    def SVM(self, X_new, y_data):
        rmse=[]
        mae=[]

        # REGRESSION 
        # With SVR
        for participant in range(X_new.shape[0]):
            svr=svm.SVR(kernel='rbf',C=1.0,epsilon=0.2)
            clf=MOR.MultiOutputRegressor(svr)
            y_pred=cross_val_predict(clf,X_new[participant],y_data[participant],cv=5)


            # METRICS FOR SVR
            MSE=mean_squared_error(y_data[participant],y_pred)
            MAE=mean_absolute_error(y_data[participant],y_pred)
            R2=r2_score(y_data[participant],y_pred)
            # print('RMSE for participant {} = {}'.format(participant,np.sqrt(MSE)))
            # print('MAE for participant {} = {}'.format(participant,MAE))
            # print('R2 for participant {} = {}'.format(participant,R2))
            # print('')
            rmse.append(MSE)
            mae.append(MAE)
        
        # Plot results in bar charts
        parts=np.arange(1,33)
        lst=list(range(1, 33))
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

    def knn(self, X_new, y_data):
        rmse=[]
        mae=[]

        # REGRESSION 
        # With k-NNS
        for participant in range(X_new.shape[0]):
            knn=KNeighborsRegressor(n_neighbors=16)
            regr_knn=MOR.MultiOutputRegressor(knn)
            y_pred_knn=cross_val_predict(regr_knn,X_new[participant],y_data[participant],cv=5)

            # METRICS FOR k-NN
            MSE_knn=mean_squared_error(y_data[participant],y_pred_knn)
            MAE_knn=mean_absolute_error(y_data[participant],y_pred_knn)
            
            rmse.append(MSE_knn)
            mae.append(MAE_knn)
        
        # Plot results in bar charts
        parts=np.arange(1,33)
        lst=list(range(1, 33))
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

    def rf(self, X_new, y_data):
        rmse=[]
        mae=[]
        # REGRESSION
        # With RFs
        for participant in range(X_new.shape[0]):
            rf=RandomForestRegressor(n_estimators=51, criterion='absolute_error', max_depth=8)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,X_new[participant],y_data[participant],cv=5)

            # METRICS FOR RF
            MSE_rf=mean_squared_error(y_data[participant],y_pred_rf)
            MAE_rf=mean_absolute_error(y_data[participant],y_pred_rf)
            
            rmse.append(MSE_rf)
            mae.append(MAE_rf)
        
        # Plot results in bar charts
        parts=np.arange(1,33)
        lst=list(range(1, 33))
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

    # HYPER-PARAMETER TUNING FOR RFs
    def rf_tuning(self, psd, y_data):
        criteria=['squared_error','absolute_error', 'poisson']

        # TUNING FOR CRITERION
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
        plt.title('MSE for random forests with {} as criterion'.format('squared_error'))
        plt.xlabel('estimators')
        plt.ylabel('MSE')
        plt.show()
        print('')
        print('Number of estimators with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))

        # TUNING FOR MAX-DEPTH
        mse=[]
        for depth in range(1,15):
            rf=RandomForestRegressor(n_estimators=51, criterion='absolute_error', max_depth=depth)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd,y_data[2],cv=5)
            MSE_rf=mean_squared_error(y_data[2],y_pred_rf)
            mse.append(MSE_rf)

        _k=np.where(mse==np.min(mse))

        plt.figure(figsize=(7,5))
        plt.plot(range(1,15),mse)
        plt.title('MSE for random forests with {} as criterion'.format('squared_error'))
        plt.xlabel('depth')
        plt.ylabel('MSE')
        plt.show()
        print('')
        print('Maximum depth with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))
    

def main():
    path = '/content/drive/My Drive/Dissertation/DEAP/data_preprocessed_matlab/s'
    X_data, y_data = import_data(path)

    channels=['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4',
          'T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']

    # preprocessing 
    X_new = preprocessing(X_data, channels)
    
    # feature extraction
    psd, psd_tuning = feature_extraction(X_new)

    # regressor
    regressor = RegressionAnalyzer()

    # tuner 
    tuner = TuningAnalyzer()

    # Support Vector Machines
    regressor.SVM(psd, y_data)

    # k-Nearest Neighbours
    regressor.knn(psd, y_data)
    tuner.knn_tuning(psd_tuning, y_data)

    # Random Forests
    regressor.rf(psd, y_data)
    tuner.rf_tuning(psd_tuning, y_data)

if __name__ == "__main__":
    main()
