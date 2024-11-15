import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def bar_plot(rmse, mae, algo, n_parts):
    """ Plot results in bar charts"""

    parts=np.arange(1, n_parts)
    lst=list(range(1, n_parts))
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
    if algo=="svm":
        plt.title('SVMs')
    if algo=="k-neighbours":
        plt.title('k-NNs')
    if algo=="rf":
        plt.title('Random Forests')
    plt.grid(True)
    plt.show()

def tuning_plot(algo, mse=None, n_estimators=None, **kwargs):

    plt.figure(figsize=(7,5))
    if algo=="svm":
        anger = kwargs.get('anger', None)
        disgust = kwargs.get('disgust', None)
        if anger is not None:
            plt.scatter(anger, anger, color='orange', label='anger', marker='*')
        if disgust is not None:
            plt.scatter(disgust, disgust, color='blue', label='disgust', marker='+')
        plt.legend()

    if algo=="k-neighbours":
        plt.plot(range(1, n_estimators), mse)
        plt.title('MSE for k-neighbours')
        plt.xlabel('k')

    if algo=="rf":
        criterion = kwargs.get('criterion', None)
        plt.plot(range(1, n_estimators), mse)
        plt.title('MSE for random forests with {} as criterion'.format(criterion))
        plt.xlabel('estimators')
    plt.ylabel('MSE')
    plt.show()

def loss_function(history):
    """Plot loss function"""

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