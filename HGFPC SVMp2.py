# Get SVM results


# Imports
import pandas as pd

import numpy as np

from sklearn import svm

# Get data
X = pd.read_csv("X_train.csv", index_col=0)
y = pd.read_csv("y_train.csv", index_col=0)

# standardize the data
def standardize(x):
    return (x - np.mean(x)) / np.std(x)
# apply rowwise
X = X.apply(standardize, axis=1)

# Use slicing or select specific GO terms
annotations = y.columns.values
        
# Extract the sample and class names from the test submission
y_test_sample = pd.read_csv("y_test_sample.csv", index_col=0)

# Read in the test data
X_test = pd.read_csv("X_test.csv", index_col=0)

# apply rowwise
X_test = X_test.apply(standardize, axis=1)

mat = np.zeros(shape=(X_test.shape[0], 200))

# Predict every annotation
for (ind,annotation) in enumerate(annotations):
    temp_clf = svm.SVC(probability = True, kernel = 'poly', degree = 2)
    temp_y = y.loc[:, annotation]
    
    temp_clf.fit(X,y.loc[:,annotation])
    
    pred = temp_clf.predict_proba(X_test)
    output = []
    
    for p in pred:
        output.append(p[1])
    
    mat[:,ind] = output

# build the dataframe with proper index and column names
df_results = pd.DataFrame(data=mat, index=y_test_sample.index, columns=y_test_sample.columns)

# save to a file for submission
df_results.to_csv("SVMp2.csv")