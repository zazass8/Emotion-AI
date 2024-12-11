# EEG emotion recognition from facial expression stimuli with the presence of noisy labels

Instructions in the content of each file uploaded to this repository:

## DEAP DATASET:

- **ICA_DEAP_WITH_RESULTS:** Contains the preprocessing pipeline with ICA and results from any regression algorithm, including curves drawn for hyperparameter tuning.
- **ICA_DEAP_VISUALS:** Contains any plots regarding preprocessing with ICA, such as heatmaps, independent components and EEG before and after ICA.
- **VAE_DEAP:** Contains the preprocessing pipeline with VAE and results from any regression algorithm, including curves drawn for hyperparameter tuning.

## MY DATA:

- **ICA_MYDATA_WITH_RESULTS:** Contains the preprocessing pipeline with ICA and results from any regression algorithm, including curves drawn for hyperparameter tuning.
- **ICA_MYDATA_VISUALS:** Contains any plots regarding preprocessing with ICA, such as heatmaps, independent components and EEG before and after ICA.
- **VAE_MYDATA:** Contains the preprocessing pipeline with VAE and results from any regression algorithm, including curves drawn for hyperparameter tuning.


## EXCEL SPREADSHEET:

- **MyData.xlsx:** Data analysis that was conducted for our dataset, to detect where have participant misclassified labels of valence, arousal and dominance. A few comparisons drawn on the last sheet.


## Notes:
The code for the implementation of the VAE was inspired by [Li et.al](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00087/full) although a lot of changes has been done to the original code.
