# HGFPC

# KNN
import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier


# load up the train and test data
X = pd.read_csv("X_train.csv", index_col=0)
y = pd.read_csv("y_train.csv", index_col=0)

# standardize the data
def standardize(x):
    return (x - np.mean(x)) / np.std(x)
# apply rowwise
X = X.apply(standardize, axis=1)

# Fix k
k = 150

# implement a KNN classifier
clf = KNeighborsClassifier(n_neighbors=k, weights = 'distance', metric = 'manhattan')

clf.fit(X, y)

# Read in the test data
X_test = pd.read_csv("X_test.csv", index_col=0)


# apply rowwise
X_test = X_test.apply(standardize, axis=1)

# predict the test data
results = clf.predict_proba(X_test)

mat = np.zeros(shape=(X_test.shape[0], 200))
for ix, i in enumerate(results):
    for jx, j in enumerate(results[ix]):
        mat[jx, ix] = j[1]
        

# Extract the sample and class names from the test submission
y_test_sample = pd.read_csv("y_test_sample.csv", index_col=0)

# build the dataframe with proper index and column names
df_results = pd.DataFrame(data=mat, index=y_test_sample.index, columns=y_test_sample.columns)

# save to a file for submission
df_results.to_csv("kNNw_m150.csv")