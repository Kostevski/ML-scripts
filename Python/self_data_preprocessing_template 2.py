# -*- coding: utf-8 -*-

#Data preprocessing

#1. Import the libraries_________________________________________________
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1.1 Importing dataset_________________________________________________
dataset = pd.read_csv('Data.csv')

#2. Independent matrix of feature_______________________________________________________________________
X = dataset.iloc[:, :-1].values

#2.1 Dependent variable vector
y = dataset.iloc[:, 3].values

#3. Take care of missing data_________________________________________________
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#4. Encoding categorical data_________________________________________________
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, LabelBinarizer

ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), [0]),],
    remainder='passthrough' # this will add the remaining columns also
)
X = ct.fit_transform(X)

cy = LabelBinarizer()
y= cy.fit_transform(y)

#5. Splitting the dataset into the Training set and Test set_________________________________________________
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#6. Feature scaling_________________________________________________
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)