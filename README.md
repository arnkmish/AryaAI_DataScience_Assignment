# AryaAI_DataScience_Assignment
This repository contains the Python3 Jupyter Notebook based solution to the Interview Assignment for Arya.AI Deep Learning Engineer position.

The Jupyter Notebook makes use of the Google Colab environment and the following packages for the implementation-

- Numpy Version 1.21.6
- Pandas Version 1.3.5
- Scikit-Learn Version 1.0.2
- Matplotlib Version 3.2.2
- Seaborn Version 0.11.2

The Jupyter Notebook firstly loads the training_set.csv and test_set.csv datasets from Google Drive in the Google Colab environment using the Pandas package into dataframes.

After this firstly the data is studied via dataframe.describe(), to see what kinds of data are available. The presence of Null values are also checked.
No Null values were found. The data features were of vastly different ranges, with some features having a mean value near 0, whereas some other features having the mean value near 300. This suggests the need for feature scaling.

After this, the feature correlation is studied by plotting the correlation heatmap using the Seaborn library. Features were found to be not very highly correlated.
This suggests that correlation based feature reduction might not be a goodd idea.

After this, an 80-20 train-validation split of the data was performed and a Random Forest Classifier based baseline model was built.

The **baseline** model gave the following predictive performances on the validation data:

**Accuracy: 0.94, Precision: 0.94, Recall: 0.94**

Such high performance suggests that there is not a lot of room for improving. But next Feature Scaling is explored.

For Feature Scaling we firstly explore **StandardScaler** module from Scikit-Learn. Upon performing StandardScaler based scaling, the following performance was obtained on the validation data:

**Accuracy: 0.95, Precision: 0.95, Recall: 0.94**

Apart from StandardScaler, we have also explored **MinMaxScaler** module from Scikit-Learn for Feature Scaling. Upon performing MinMaxScaler based scaling, the following performance was obtained on the validation data:

**Accuracy: 0.95, Precision: 0.95, Recall: 0.95**

We see that with StandardScaler as well as MinMaxScaler slight improvements could be obtained. Next we explore Feature Selection.

For Feature Selection the **Recursive Feature Elimination via Cross-Validation(RFECV)** approach is considered, where the main idea is to recursively remove less important features and fit new models with reduced feature sets. The feature importances are calculated by an underlying predictive model(RFC in this Notebook).
The RFECV approach has been explored for selecting the top 10,20,30,40 and 50 features, and for each of these scenarios the results on the validation set is shown below:

**RFECV 10 Features: Accuracy- 0.93, Precision- 0.93, Recall- 0.93**

**RFECV 20 Features: Accuracy- 0.94, Precision- 0.94, Recall- 0.93**

**RFECV 30 Features: Accuracy- 0.94, Precision- 0.94, Recall- 0.94**

**RFECV 40 Features: Accuracy- 0.94, Precision- 0.94, Recall- 0.94**

**RFECV 50 Features: Accuracy- 0.94, Precision- 0.94, Recall- 0.94**


We observe that Feature Selection did not help with improving the model performance as compared to using all the features after MinMaxScaler is applied.

Next we perform **Model Selection via GridSearchCV based Hyperparameter Tuning**. 
For this we consider several hyperparameters and their possible values: 

{'criterion':('gini', 'entropy'),
'min_samples_split':[2,4,8,16,32,64,128],
'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],
'class_weight':('balanced','balanced_subsample')}

After the GridSearchCV based tuning is performed, the tuned model performance did not improve on the validation set:

**Accuracy: 0.94, Precision: 0.94, Recall: 0.93**

Based on all of the above modeling steps we see that the best performances for the RFC model could be obtained upon performing MinMaxScaler based feature scaling and then RFC based model training.

Therefore, for predicting on the test set also, firstly MinMaxScaler based Feature Scaling is performed and then RFC model training and classification are done.

The final predictions on the test set are updated in the test_data_with_predictions.csv file in the repository.
