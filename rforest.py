# -*- coding: utf-8 -*-

"""
To get metrics and perform tests on test data set
Copy ' test_clf(test_x, test_y) ' to bottom of code
You will need to define a test_y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from comet_ml import Experiment
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer

# Connect to Comet ML for experiment logging
experiment = Experiment(
    api_key="______", project_name="_____")

# Read in Data
train_data = pd.read_csv("public_train.csv")
test_data = pd.read_csv("public_test.csv")

# Check distribution of data col 3 was initially removed due to high variance
plt.figure(figsize = (10,10))
train_data.hist()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("distributions.png")
plt.show()
train_data.describe()

# Show number of non-zero elements in a col.
# 2, 6 and 8 were removed due to mostly 0 values.
# This had a negative impact (up to -0.05 AUC), so they were added back
num_non_zeros = train_data.astype(bool).sum(axis=0)
num_non_zeros.plot.bar()
plt.show()

# Split training data into train and val
# NOTE: Can set 0.7 to 1.0 and remove test_clf(val_x, val_y) at bottom to
# make full use of training data
split = np.random.rand(len(train_data)) < 0.7
train = train_data[split]
val = train_data[~split]
val_counts = val["class_col"].value_counts()
train_counts = train["class_col"].value_counts()

# Split training data further, into x and y, inputs and target values
train_x = train[["variable_0", "variable_1", "variable_2", "variable_3",
                 "variable_4", "variable_5", "variable_6", "variable_7", "variable_8", "variable_9"]]
train_y = train[["class_col"]]

val_x = val[["variable_0", "variable_1", "variable_2", "variable_3",
             "variable_4", "variable_5", "variable_6", "variable_7", "variable_8", "variable_9"]]
val_y = val[["class_col"]]

test_x = test_data[["variable_0", "variable_1", "variable_2", "variable_3",
                    "variable_4", "variable_5", "variable_6", "variable_7", "variable_8", "variable_9"]]


"""#  Apply log transform (yeo-johnson because data include negative values)
"""
pt = preprocessing.PowerTransformer(method='yeo-johnson')
train_x = pt.fit_transform(train_x)

"""
Implement SMOTE, an over sampling technique to deal with imbalance
Works brilliantly if SMOTE also applied in test_clf(AUC:~0.84) without pt.fit_transform
If applied within testing, SMOTE would also have to applied to production data to reproduce results
Not sure if this is a viable option considering above.
"""
smote = SMOTE(ratio=1.0, random_state=12)
train_x, train_y = smote.fit_sample(train_x, train_y)

clf = RandomForestClassifier(n_estimators=200, min_samples_split=100,
                             min_samples_leaf=100, class_weight='balanced').fit(train_x, train_y)


# Function to test classifier on some data, Will return some metric to
# assess performance.
def test_clf(x, y):
    x = pt.fit_transform(x)
    x, y = smote.fit_sample(x, y)
    # Apply classifier to gret predictios and probabilities on val set
    predictions = clf.predict(x)
    probabilities = clf.predict_proba(x)

    # Output some metrics to give understanding
    print "Train Class Counts: \n:", train_counts
    print "Val Class Counts: \n", val_counts
    print(' \n Accuracy on val set: {:.3f}'.format(clf.score(x, y)))

    # Create confustion matrix and class report
    conf_matrix = confusion_matrix(y, predictions)
    class_report = classification_report(y, predictions)
    print (conf_matrix)
    print (class_report)

    # Output ROC
    roc_auc = roc_auc_score(y, predictions)
    fpr, tpr, thresholds = roc_curve(y, probabilities[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'g--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig('ROC_RF')
    # plt.show()
    print ("\nAUC: {0:.2f}".format(roc_auc))
    experiment.log_metric("AUC", roc_auc)  # Log AUC score to comet_ml


test_clf(val_x, val_y)

test_predictions = clf.predict(test_x)
test_probabilities = clf.predict_proba(test_x)
np.savetxt("rf_test_predictions.csv",
           test_predictions, delimiter=",", fmt='%.0f')

# Will output probabilities of output belonging to Positive class(1)
np.savetxt("rf_test_probabilities.csv",
           test_probabilities[:, 1], delimiter=",", fmt='%.5f')
