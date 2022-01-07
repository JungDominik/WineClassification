
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing





path = 'C:\\Pythontest Anaconda\\Statquest\\projects\\wine\\'
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', ))

df_data_raw = pd.read_csv(ROOT_DIR + '\\00_data\\' + 'wine.data')

names = ['WineClass',
         'Alcohol'
 	, 'Malic acid'
 	, 'Ash'
	, 'Alcalinity of ash'
 	, 'Magnesium'
	, 'Total phenols'
 	, 'Flavanoids'
 	, 'Nonflavanoid phenols'
 	, 'Proanthocyanins'
	,'Color intensity'
 	,'Hue'
 	,'OD/OD of diluted wines'
 	,'Proline']

df_data_raw.columns = names

##Show dataframe
#df_data_raw
#Outline of the data
#n = 177 Observations in the dataset 
#p = 13 independent variables / attributes
#y (Attribute 'WineClass') can take values (1,2,3) -> Represents three different types of wine


###Check missing --> Seems no missing
###Create dummies --> Not necessary, no categorical variables, all float

###Modeling Prep: Split into X and y
X = df_data_raw.drop(columns = 'WineClass')
y = df_data_raw['WineClass']
names = names[1:]

###Split X into Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)



###Exploratory: Principle Component Analysis
X_transformed = preprocessing.scale(X)

pca = PCA(n_components = 10)
pca.fit(X_transformed)
data_pca_transformed = pca.transform(X_transformed)


#How many Principal Components do we need to show? Decide via Scree Plot
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1 )
labels = ['PC' + str(x) for x in range (1, len(per_var)+1)]

plt.bar(x = range (1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel ('Principal Component #')
plt.title('Scree Plot for PCA')
plt.show()
#Evaluation of Scree Plot: Which percentage of variation does each Principal Component explain?
#PC1 35% of explained variance, PC2 ~20, PC3 ~12%. 
#--> Focus on PC1 and PC2 (James et al.: "smallest number of principal components that are required in order to explain a sizable amount of the variation in the data)

#Check loading scores
loading_scores = pd.Series(pca.components_[0], index = names)
sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
top_10_attributes = sorted_loading_scores[0:10].index.values

print(loading_scores[top_10_attributes])
#Evaluation of Loadingscores: No one attribute with the strongest loadingscore

df_pca = pd.DataFrame(data_pca_transformed, columns = labels) #For plotting: Extract data out of PCA-Transformation (numpy-array) into Pandas dataframe --> Has indices and labels

plt.scatter(df_pca.PC1, df_pca.PC2)
plt.title('PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

for sample in df_pca.index:
     #plt.annotate(sample, (df_pca.PC1.loc[sample], df_pca.PC2.loc[sample]))
     #plt.annotate(sample, (df_data_raw.WineClass.loc[sample]))
     plt.annotate(
         text = df_data_raw.WineClass[sample], 
         xy = (df_pca.PC1.loc[sample], df_pca.PC2.loc[sample]) 
         )
     
plt.show()







###First model: Random Forest Classifier

clf_rf = RandomForestClassifier(max_depth=2, random_state = 42)
clf_rf.fit(X_train, y_train)

###Confusionmatrix for Test Set Predictions

predictions_rf = clf_rf.predict(X_test)

##Confusion Matrix for Random Forest: Absolute Values
cm_rf = confusion_matrix(y_test, predictions_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix = cm_rf)
disp_rf.plot()
disp_rf.ax_.set_title('Confusion Matrix for Random Forest: Absolute Values')



#Confusion Matrix for Random Forest: Relative Values
cm_rf = confusion_matrix(y_test, predictions_rf, normalize = 'all')
disp_rf = ConfusionMatrixDisplay(confusion_matrix = cm_rf)
disp_rf.plot()
disp_rf.ax_.set_title('Confusion Matrix for Random Forest: Relative Values')

#Evaluation: Prediction is quite good out of the box --> Likely the dataset is not challenging


### Comparison with AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier

clf_ada = AdaBoostClassifier(n_estimators = 5000, random_state = 42)
clf_ada.fit (X_train, y_train)

predictions_ada = clf_ada.predict(X_test)


##Confusion Matrix for AdaBoost: Absolute Values
cm_ada = confusion_matrix(y_test, predictions_ada)
disp_ada = ConfusionMatrixDisplay(confusion_matrix = cm_ada)
disp_ada.plot()
disp_ada.ax_.set_title('Confusion Matrix for AdaBoost: Absolute Values')

##Confusion Matrix for AdaBoost: Relative Values
cm_ada = confusion_matrix(y_test, predictions_ada, normalize = 'all')
disp_ada = ConfusionMatrixDisplay(confusion_matrix = cm_ada)
disp_ada.plot()
disp_ada.ax_.set_title('Confusion Matrix for AdaBoost: Relative Values')


###Compare with k-nearest-neighbours
clf_knn = KNeighborsClassifier(n_neighbors = 10)
clf_knn.fit(X_train, y_train)

predictions_knn = clf_knn.predict(X_test)


##Confusion Matrix for K-Nearest Neighbors: Absolute Values
cm_knn = confusion_matrix(y_test, predictions_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix = cm_knn)
disp_knn.plot()
disp_knn.ax_.set_title('Confusion Matrix for K-Nearest Neighbors: Absolute Values')

##Confusion Matrix for K-Nearest Neighbors: Relative Values
cm_knn = confusion_matrix(y_test, predictions_knn, normalize = 'all')
disp_knn = ConfusionMatrixDisplay(confusion_matrix = cm_knn)
disp_knn.plot()
disp_knn.ax_.set_title('Confusion Matrix for K-Nearest Neighbors: Relative Values')


###Compare with Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X_train, y_train)

predictions_lda = clf_lda.predict(X_test)

##Confusion Matrix for Linear Discriminant Analysis: Absolute Values
cm_lda = confusion_matrix(y_test, predictions_lda)
disp_lda = ConfusionMatrixDisplay(confusion_matrix = cm_lda)
disp_lda.plot()
disp_lda.ax_.set_title('Linear Discriminant Analysis: Absolute Values')

##Confusion Matrix for Linear Discriminant Analysis: Relative Values
cm_lda = confusion_matrix(y_test, predictions_lda, normalize = 'all')
disp_lda = ConfusionMatrixDisplay(confusion_matrix = cm_lda)
disp_lda.plot()
disp_lda.ax_.set_title('Linear Discriminant Analysis: Relative Values')





## Evaluation/Comparison: 
# - In general the comparison seems to be not too challenging. All methods had an acceptable performance.
# - Comparison
#    - Linear Discriminant Analysis has the best performance
#    - RandomForest performs slightly better on predicting observations with WineCategory == 1
#    - AdaBoost performs slightly better on predicting observations with WineCategory == 1 or == 3
#    - K-Nearest-Neighbors (k = 10) has a sub-par performance


