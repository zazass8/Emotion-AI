# EEG emotion recognition from facial expression stimuli with the presence of noisy labels

Instructions in the content of each file uploaded to this repository:

## DEAP DATASET:

1) ICA_DEAP_WITH_RESULTS: Contains the preprocessing pipeline with ICA and results from any regression algorithm, including curves drawn for hyperparameter tuning.
2) ICA_DEAP_VISUALS: Contains any plots regarding preprocessing with ICA, such as heatmaps, independent components and EEG before and after ICA.
3) VAE_DEAP: Contains the preprocessing pipeline with VAE and results from any regression algorithm, including curves drawn for hyperparameter tuning.

## MY DATA:

1) ICA_MYDATA_WITH_RESULTS: Contains the preprocessing pipeline with ICA and results from any regression algorithm, including curves drawn for hyperparameter tuning.
2) ICA_MYDATA_VISUALS: Contains any plots regarding preprocessing with ICA, such as heatmaps, independent components and EEG before and after ICA.
3) VAE_MYDATA: Contains the preprocessing pipeline with VAE and results from any regression algorithm, including curves drawn for hyperparameter tuning.


## EXCEL SPREADSHEET:
MyData.xlsx: Data analysis that was conducted for our dataset, to detect where have participant misclassified labels of valence, arousal and dominance. A few comparisons drawn on the last sheet.

## Notes:
The code for the implementation of the VAE was inspired by Li [1] et.al although a lot of changes has been done to the original code.


## References:

[1] Li, X., Zhao, Z., Song, D., Zhang, Y., Pan, J., Wu, L., Huo, J., Niu, C. and Wang, D., 2020. Latent factor decoding of multi-channel EEG for emotion recognition through autoencoder-like neural networks. Frontiers in neuroscience, 14, p.87.
