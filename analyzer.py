from sklearn import svm
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.multioutput as MOR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error,mean_squared_error
from visuals.plotting import bar_plot, tuning_plot

class Regression:

    def svr(self, psd_trn, y_train, psd_tst, y_test, n_parts):
        rmse=[]
        mae=[]
        for participant in range(psd_trn.shape[0]):
            # REGRESSION 
            # With SVR
            svr=svm.SVR(kernel='rbf',C=1.0,epsilon=0.2)
            clf=MOR.MultiOutputRegressor(svr)
            clf.fit(psd_trn[participant],y_train[participant])
            y_pred=cross_val_predict(clf,psd_tst[participant],y_test[participant],cv=5)

            # METRICS FOR SVR
            MSE=mean_squared_error(y_test[participant],y_pred)
            MAE=mean_absolute_error(y_test[participant],y_pred)

            rmse.append(MSE)
            mae.append(MAE)

        # Plot results in bar charts
        bar_plot(rmse, mae, "svm", n_parts)

    def knn(self, psd_trn, y_train, psd_tst, y_test, n_parts):
        rmse=[]
        mae=[]
        for participant in range(psd_trn.shape[0]):
            # REGRESSION 
            # With k-NNs
            knn=KNeighborsRegressor(n_neighbors=23)
            regr_knn=MOR.MultiOutputRegressor(knn)
            regr_knn.fit(psd_trn[participant],y_train[participant])
            y_pred_knn=cross_val_predict(regr_knn,psd_tst[participant],y_test[participant],cv=5)

            # METRICS FOR k-NN
            MSE_knn=mean_squared_error(y_test[participant],y_pred_knn)
            MAE_knn=mean_absolute_error(y_test[participant],y_pred_knn)

            rmse.append(MSE_knn)
            mae.append(MAE_knn)
        
        # Plot results in bar charts
        bar_plot(rmse, mae, "k-neighbours", n_parts)

    def rf(self, psd_trn, y_train, psd_tst, y_test, n_parts):
        rmse=[]
        mae=[]
        for participant in range(psd_trn.shape[0]):
            # REGRESSION 
            # With RFs
            rf=RandomForestRegressor(n_estimators=58, criterion='absolute_error', max_depth=8)
            regr_rf=MOR.MultiOutputRegressor(rf)
            regr_rf.fit(psd_trn[participant],y_train[participant])
            y_pred_rf=cross_val_predict(regr_rf,psd_tst[participant],y_test[participant],cv=5)

            # METRICS FOR RF
            MSE_rf=mean_squared_error(y_test[participant],y_pred_rf)
            MAE_rf=mean_absolute_error(y_test[participant],y_pred_rf)

            rmse.append(MSE_rf)
            mae.append(MAE_rf)
        
        # Plot results in bar charts
        bar_plot(rmse, mae, "rf", n_parts)


class Tuning:

    def nonlinearity(self, psd):
        # proof of non-linearity of data
        x=np.mean(psd, axis=1)

        anger=x[[0,9,18,27,40,49,58,67,75,84,93]]
        disgust=x[[1,10,19,28,42,51,60,68,71,80,89]]

        # PLOT TO SHOW DATA ARE NOT LINEARLY SEPERABLE
        tuning_plot("svm", anger, disgust)

    # HYPER-PARAMETER TUNING FOR K-NNS
    # data only with a random participant
    def knn_tuning(self, psd, y_data, n_k):
        mse=[]
        for k in range(1,n_k):
            knn=KNeighborsRegressor(n_neighbors=k)
            regr_knn=MOR.MultiOutputRegressor(knn)
            regr_knn.fit(psd,y_data)
            y_pred_knn=cross_val_predict(regr_knn,psd,y_data[2],cv=5)
            MSE_knn=mean_squared_error(y_data[2],y_pred_knn)
            mse.append(MSE_knn)

        k=np.where(mse==np.min(mse))

        tuning_plot(mse, "k-neighbours", n_k)
        print('')
        print('Number of k with lowest MSE equal to {} and with MSE equal to {}'.format(k[0][0],np.min(mse)))

    # HYPER-PARAMETER TUNING FOR RANDOM FORESTS
    def rf_tuning(self, psd, y_data, n_est_crit, n_est, n_dpt):

        def mse_formula(n):
            mse=[]
            for estimator in range(1,n):
                rf=RandomForestRegressor(n_estimators=estimator, criterion=criterion, max_depth=None)
                regr_rf=MOR.MultiOutputRegressor(rf)
                regr_rf.fit(psd,y_data)
                y_pred_rf=cross_val_predict(regr_rf,psd,y_data[2],cv=5)
                MSE_rf=mean_squared_error(y_data[2],y_pred_rf)
                mse.append(MSE_rf)
            return mse
        
        criteria=['squared_error','absolute_error', 'poisson']

        for criterion in criteria:
            mse=mse_formula(1,n_est_crit)
            _k=np.where(mse==np.min(mse))

        tuning_plot(mse, "rf", n_est_crit, criterion)
        print('')
        print('Number of estimators with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))

        # TUNING FOR NUMBER OF ESTIMATORS
        mse=mse_formula(1,n_est)
        _k=np.where(mse==np.min(mse))

        tuning_plot(mse, "rf", n_est, "absolute_error")
        print('')
        print('Number of estimators with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))

        # TUNING FOR MAX-DEPTH
        mse=mse_formula(1,n_dpt)
        _k=np.where(mse==np.min(mse))

        tuning_plot(mse, "rf", n_dpt, "absolute_error")
        print('')
        print('Maximum depth with lowest MSE equal to {} and with MSE equal to {}'.format(_k[0][0],np.min(mse)))