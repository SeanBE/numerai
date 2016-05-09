import numpy as np
import pandas as pd
import datetime as dt

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier

ID_COL = 't_id'
LABEL_COL = 'probability'

TRAIN_FILENAME = '../data/numerai_training_data.csv'
TEST_FILENAME = '../data/numerai_tournament_data.csv'

trainDf = pd.read_csv(TRAIN_FILENAME)

# Continous values as features + binary classification problem (1,0)

# Classes look balanced enough.
# print 'Class balances\n ', pd.value_counts(trainDf.target.values, sort=False)

features = trainDf.drop("target", axis=1)
labels = trainDf.target

# Feature engineering

poly = PolynomialFeatures(degree=2, include_bias=False)
features = poly.fit_transform(features)

# Normalising!
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Time to train ze models!
xTrain, xValid, yTrain, yValid = train_test_split(
    features, labels, test_size=0.2, random_state=37)

stratifiedKFold = StratifiedKFold(yTrain, n_folds=5, shuffle=True)

# Default XGBClassifier gets logloss 0.694..at least I'm not last.
clf = XGBClassifier(seed=37, n_estimators=1000)

clf.fit(xTrain, yTrain, eval_metric='logloss')

# Checking classifier predictions on training data.
print "Log loss: %f" % log_loss(yValid, clf.predict_proba(xValid))

# Prediction
testDf = pd.read_csv(TEST_FILENAME)

testX = testDf.drop(ID_COL, axis=1)
testX = poly.transform(testX)
testX = scaler.fit_transform(testX)

testY = clf.predict_proba(testX)
testDf[LABEL_COL] = testY[:, 1]

currentDt = dt.datetime.now().isoformat()
outputFilename = '../output/submission' + currentDt + '.csv'
testDf.to_csv(outputFilename, columns=(ID_COL, LABEL_COL), index=False)
