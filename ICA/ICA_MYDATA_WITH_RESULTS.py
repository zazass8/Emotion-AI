import mne
import pymatreader
import scipy
import numpy as np
import pandas as pd
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

import matplotlib.pyplot as plt
from scipy.signal import welch
import scipy.stats as stats
import matplotlib.pyplot as plt

# Importing the DEAP dataset
def import_data(path, path_00):
    X_data=np.zeros([9,95,14,512])
    y_data=np.zeros([9,95,3])
    for participant in range(9):
        _path=path+str(participant+1)+'/'
        X=pd.read_csv(_path+'data_s0'+str(participant+1)+'.csv')
        X=X.iloc[:,1:]
        X=X.to_numpy()
        for epoch in range(95):
            X_data[participant][epoch]=np.transpose(X[512*epoch:512*(epoch+1)])

        y=pd.read_csv(path+'labels_s0'+str(participant+1)+'.csv')
        y=y.iloc[:,1:]
        y=y.to_numpy()
        y_data[participant]=y

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

def preprocessing(X_data, channels):
    X_new=np.zeros([9,95,14,512])
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
            ica = ICA(n_components=14,noise_cov=cov,method='fastica',max_iter='auto',random_state=97)
            ica.fit(filt_raw,picks=eeg_channels,reject_by_annotation=True)
            ica.exclude = [2]
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

def apply_pseudo(X_data, y_data):
    vec=Vectorizer()
    y_new=np.zeros([9,95,3])
    estimator=svm.SVC(kernel='rbf',C=0.1)
    X_new__=np.zeros([9,95,14,129])
    X_new_transformed=np.zeros([9,95,1806])
    for participant in range(X_data.shape[0]):
        X_new__[participant]=welch(X_data[participant], fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
        X_new_transformed[participant]=vec.fit_transform(X_new__[participant])
        y_new[participant,:,0]=pseudo(X_new_transformed[participant],y_data[participant,:,0],estimator)
        y_new[participant,:,1]=pseudo(X_new_transformed[participant],y_data[participant,:,1],estimator)
        y_new[participant,:,2]=pseudo(X_new_transformed[participant],y_data[participant,:,2],estimator)
    
    return y_new

def feature_extraction(X_new):
    vec=Vectorizer()
    X_new_=np.zeros([9,95,1806])
    psd=np.zeros([9,95,14,129])

    dec=X_new[1]
    psd_knn=welch(dec, fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
    psd_tuning=vec.fit_transform(psd_knn)

    for participant in range(X_new.shape[0]):
        psd[participant]=welch(X_new[participant], fs=128, window='hann', detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')[1]
        X_new_[participant]=vec.fit_transform(psd[participant])
    
    return X_new_, psd_tuning

class RegressionAnalyzer:
    def __init__(self):
        pass

    def SVM(self, X_new, y_new):
        rmse=[]
        mae=[]

        # REGRESSION 
        # With SVR
        for participant in range(X_new.shape[0]):
            svr=svm.SVR(kernel='rbf',C=1.0,epsilon=0.2)
            clf=MOR.MultiOutputRegressor(svr)
            y_pred=cross_val_predict(clf,X_new[participant],y_new[participant],cv=5)


            # METRICS FOR SVR
            MSE=mean_squared_error(y_new[participant],y_pred)
            MAE=mean_absolute_error(y_new[participant],y_pred)

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

    def knn(self, X_new, y_new):
        rmse=[]
        mae=[]

        # REGRESSION 
        # With k-NNS
        for participant in range(X_new.shape[0]):
            knn=KNeighborsRegressor(n_neighbors=71)
            regr_knn=MOR.MultiOutputRegressor(knn)
            y_pred_knn=cross_val_predict(regr_knn,X_new[participant],y_new[participant],cv=5)

            # METRICS FOR k-NN
            MSE_knn=mean_squared_error(y_new[participant],y_pred_knn)
            MAE_knn=mean_absolute_error(y_new[participant],y_pred_knn)
            
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

    def rf(self, X_new, y_new):
        rmse=[]
        mae=[]
        # REGRESSION
        # With RFs
        for participant in range(X_new.shape[0]):
            rf=RandomForestRegressor(n_estimators=59, criterion='squared_error', max_depth=5)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,X_new[participant],y_new[participant],cv=5)

            # METRICS FOR RF
            MSE_rf=mean_squared_error(y_new[participant],y_pred_rf)
            MAE_rf=mean_absolute_error(y_new[participant],y_pred_rf)
            
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

    # HYPER-PARAMETER TUNING FOR K-NNS
    def knn_tuning(self, psd, y_new):
        mse=[]
        for k in range(1,76):
            knn=KNeighborsRegressor(n_neighbors=k)
            regr_knn=MOR.MultiOutputRegressor(knn)
            y_pred_knn=cross_val_predict(regr_knn,psd,y_new[1],cv=5)
            MSE_knn=mean_squared_error(y_new[1],y_pred_knn)
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

    # HYPER-PARAMETER TUNING FOR RFs
    def rf_tuning(self, psd, y_new):
        criteria=['squared_error','absolute_error', 'poisson']

        # TUNING FOR CRITERION
        for criterion in criteria:
            mse=[]
            for estimator in range(1,50):
                rf=RandomForestRegressor(n_estimators=estimator, criterion=criterion, max_depth=None)
                regr_rf=MOR.MultiOutputRegressor(rf)
                y_pred_rf=cross_val_predict(regr_rf,psd,y_new[1],cv=5)
                MSE_rf=mean_squared_error(y_new[1],y_pred_rf)
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
            y_pred_rf=cross_val_predict(regr_rf,psd,y_new[1],cv=5)
            MSE_rf=mean_squared_error(y_new[1],y_pred_rf)
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
            rf=RandomForestRegressor(n_estimators=59, criterion='squared_error', max_depth=depth)
            regr_rf=MOR.MultiOutputRegressor(rf)
            y_pred_rf=cross_val_predict(regr_rf,psd,y_new[1],cv=5)
            MSE_rf=mean_squared_error(y_new[1],y_pred_rf)
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
    path = '/content/drive/My Drive/Dissertation/MYDATA_NEW/'
    X_data, y_data = import_data(path)

    channels=['Fp1','AF3','F7','F3','FC1','FC5','T7','C3','CP1','CP5','P7','P3','Pz','PO3','O1','Oz','O2','PO4','P4','P8','CP6','CP2','C4',
          'T8','FC6','FC2','F4','F8','AF4','Fp2','Fz','Cz']

    # preprocessing 
    X_new = preprocessing(X_data, channels)
    
    # feature extraction
    psd, psd_tuning = feature_extraction(X_new)

    # pseudo-labelling
    y_new = apply_pseudo(X_new, y_data)

    # regressor
    regressor = RegressionAnalyzer()

    # tuner 
    tuner = TuningAnalyzer()

    # Support Vector Machines
    regressor.SVM(psd, y_new)

    # k-Nearest Neighbours
    regressor.knn(psd, y_new)
    tuner.knn_tuning(psd_tuning, y_new)

    # Random Forests
    regressor.rf(psd, y_new)
    tuner.rf_tuning(psd_tuning, y_new)

if __name__ == "__main__":
    main()
