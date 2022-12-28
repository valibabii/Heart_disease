import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the cvs Data to Pandas DataFrame
path = r"C:\Users\User\PycharmProjects\pythonProject\final project\heart.csv"
heart_data = pd.read_csv(path)
#print first 5 rows of the dataset
#print(heart_data.head())
#print the last 5 rows of the dataset
#print(heart_data.tail())
#numbers of rows and coloms in the dataset
#print(heart_data.shape)
#print(heart_data.info)
#checking for missing values
#print(heart_data.isnull)
#statistical measures about the data
#print(heart_data.describe())
#checking the distribution of target Variable
#print(heart_data['target'].value_counts()) #1--healthy heart 0--defective heart
#spleating the features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
#print(Y)
#Splitting the Data into Training Data & Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#print(X.shape, X.train.shape, X_test.shape)
#Logistic regression
model = LogisticRegression()
print(model.fit(X.train, Y_train))
