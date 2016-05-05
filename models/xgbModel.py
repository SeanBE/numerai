import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier

trainFile = "../data/numerai_training_data.csv"
trainDf = pd.read_csv(trainFile)

print trainDf.head()

# Looks like continous values as features + binary classification problem (1,0)
# Features are unknown. Not much room to engineer some new features.
# value counts, top pca components

print 'Class balances\n ', pd.value_counts(trainDf.target.values, sort=False)
# Classes look balanced enough.

features = trainDf.drop("target", axis=1)
labels = trainDf.target

scaler = StandardScaler()
features = scaler.fit_transform(features)

xTrain, xValid, yTrain, yValid = train_test_split(features, labels, test_size=0.2, random_state=37)

# use cross_val_score as well with kFold??
# Use pipeline and polynomialFeatures (2?) ??


clf = XGBClassifier(seed=37)
clf.fit(xTrain,yTrain,eval_metric='logloss')

# Checking classifier predictions on training data.
print "Accuracy : %.4g" % accuracy_score(yValid, clf.predict(xValid))
print "Log loss: %f" % log_loss(yValid, clf.predict_proba(xValid))

testFile = "../data/numerai_tournament_data.csv"
testDf = pd.read_csv(testFile)

print testDf.head()

testX = testDf.drop('t_id', axis=1)
testX = scaler.fit_transform(testX)

testY = clf.predict_proba(testX)
print testY.shape

testDf['probability'] = testY[:,1]
testDf.to_csv('../output/submission.csv', columns = ( 't_id', 'probability' ), index = False)


