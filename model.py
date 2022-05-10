import numpy as np
import pandas as pd
import sklearn.datasets
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
# print(breast_cancer_dataset)
# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
# print the first 5 rows of the dataframe
# print(data_frame.head(5))
# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target
# print(data_frame)
# print last 5 rows of the dataframe
# print(data_frame.tail())
# getting some information about the data
# data_frame.info()
# checking for missing values
#print(data_frame.isnull().sum())
# statistical measures about the data
#print(data_frame.describe())
# checking the distribution of Target values
#print(data_frame['label'].value_counts())
data_frame.groupby('label').mean()
# Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
# Splitting the data into training data & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)
# Logistic Regression
model = LogisticRegression()
# training the Logistic Regression model using Training data

model.fit(X_train, Y_train)
pickle.dump(model,open('model.pkl', 'wb'))
# accuracy on training data
#X_train_prediction = model.predict(X_train)
#training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
#print(training_data_accuracy)
# accuracy on test data
#X_test_prediction = model.predict(X_test)
#test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
#print(test_data_accuracy)
#input_data = list(map(float, input('Enter the numbers separated by space').strip().split()))[:30]
# change the input data to a numpy array
#input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
#input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#prediction = model.predict(input_data_reshaped)
#print(prediction)

#if ( prediction (0) == 0 ):
    #print('The Breast cancer is Malignant')

#else:
    #print('The Breast Cancer is Benign')