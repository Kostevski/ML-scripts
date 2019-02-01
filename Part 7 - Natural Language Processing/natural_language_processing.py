# -*- coding: utf-8 -*-
""" Problem: NLL that can detect if review is positive or negative"""
# NLL

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import TSV (CSV bad for text)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter="\t", quoting=3)

# Step 1. Cleaning text

# Imports
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
length = len(dataset)
for i in range(0, length):
    # Select alphabetical letters, lowercase, stem and remove stopwords
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Step 2. Creating the Bag of Words model
    ## By creating sparse matrix, reviews as rows and words as cols

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Several text-cleaning features in class

# Create sparse matrix of features and dependent variable
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

# Step 3. Train the model. Naive bayes, decision tree and random forest commmon in NLP
    ## Find correlation between words and positive/negative reviews with Naive Bayes

"""Homework: Try different categorization models and compare"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def quality(cm):
    """ Calculates accuracy, precision, recall and f1 score from confusion
    matrix """

    accuracy = cm.trace()/cm.sum() # (TP + TN)/(TP + TN + FP + FN)
    precision = cm[0,0]/cm[:,0].sum()# TP / (TP + FP)
    recall = cm[0,0]/cm[0,:].sum()  #TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    scores = np.array([accuracy, precision, recall, f1_score]) * 100
    return scores

def kernel_svm(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)

    scores = quality(cm)
    return scores

def knn(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    scores = quality(cm)
    return scores

def svm(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    scores = quality(cm)
    return scores

def log_reg(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)

    scores = quality(cm)
    return scores

def naive_bayes(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    scores = quality(cm)
    return scores

def decision_tree(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    scores = quality(cm)
    return scores

def random_forest(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=0)

    classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    scores = quality(cm)
    return scores

d={"Naive Bayes": naive_bayes(X, y),
    "Log Reg": log_reg(X, y),
    "Random Forest": random_forest(X, y),
    "Decision Tree": decision_tree(X, y),
    "K-NN": knn(X, y),
    "SVM": svm(X, y),
    "Kernel SVM": kernel_svm(X, y)
    }

df = pd.DataFrame(data=d,
                  index=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                  dtype=np.uint8)

plt.figure()
df.plot(kind='bar')