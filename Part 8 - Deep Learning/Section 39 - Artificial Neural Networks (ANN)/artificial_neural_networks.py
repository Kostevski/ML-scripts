# Importing the libraries
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)


    plt.imshow(cm, cmap="YlGn", interpolation='nearest')
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=23)
    plt.yticks(tick_marks, classes)
    labels = np.array([["True positive:", "False Positive:"], ["False Negative:", "True negative:"]])

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, labels[i, j],
                 horizontalalignment="center",
                 verticalalignment="top",
                 color="black")
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color= "black")

    plt.ylabel('Actual outcome')
    plt.xlabel('Predicted outcome')
    plt.tight_layout()

"""
This is our dataset: 10 variables and house price. 10 000 entries in total

This is data from three banks. They gathered 10 variables on 10000 clients, and
on follow up 6 months later saw which ones had left the bank.

Now they want us to build a model that can predict based on these 10 variables
wether the client will leave the bank or not in 6 months.

# First we umport data and split into X (matrixof 10 variables) and
outcome y (Last column "Exited", 0 = Stayed at the bank and 1 = Left).
"""

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

"""
Here we process the data
"""
# Encoding categorical data
labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 2].astype('float64')

ct = ColumnTransformer(
    [('oh_enc_1', OneHotEncoder(sparse=False, dtype="float"), [1])],
    remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

"""
Data splitting and scaling.

In essence, we split our data of 10000 entries into 2000 + 8000 entries.

We then "train" the data on the first 2000 entries to build the model

Then we use that model on the rest of the 8000 houses to test the accuracy of
the model.

Scaling means that we scale all our variables from 0 to 1, to make sure all
entries have the same weight and don't get screwed by differents units of measurements
"""

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=0)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""
These lines of code is the whole Artificial Neural Network and trains it.
Can be written in one line but for pedagogic reasons kept it like this
"""

# Initializing the ANN
classifier = Sequential()
#Adding the input layer and first hidden layer
classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform', input_dim=11))
# Adding second later:
classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform'))
# Add output layer:
classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


"""
Thats it, model built. Let's see how well it predicts out come on the 8000
other customers
"""
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Create confusion matrix
class_names = ["Positive", "Negative"]
cm = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm, classes=class_names,
                      title='Confusion matrix, without normalization',
                      )

print(f"\n \nAccuracy: {100*cm.trace()/cm.sum()}%,    (TP + TN)/(TP + TN + FP + FN)\n")
print(f"Precision: {round(100*cm[0,0]/cm[:,0].sum(),2)}%    (TP / (TP + FP)\n")
print(f"Recall: {round(100*cm[0,0]/cm[0,:].sum())}%        (TP / (TP + FN)\n")
print(f"F1 Score: {round(200 * cm[0,0]/cm[:,0].sum() * cm[0,0]/cm[0,:].sum() / (cm[0,0]/cm[:,0].sum() + cm[0,0]/cm[0,:].sum()),2)}%\
     (2 * Prec * Recall / (Precis + Recall)")
