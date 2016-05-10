import numpy as np
import pandas as pd
import datetime as dt

from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import log_loss
from xgboost.sklearn import XGBClassifier

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials

ID_COL = 't_id'
LABEL_COL = 'probability'

# TODO use argparse..
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

# Default XGBClassifier gets logloss 0.694..at least I'm not last.

# Hyperparameter optimisation


def objective(space):

    clf = XGBClassifier(n_estimators=int(space['n_estimators']),
                        objective='binary:logistic',
                        seed=37,
                        learning_rate=space['learning_rate'],
                        max_depth=space['max_depth'],
                        min_child_weight=space['min_child_weight'],
                        colsample_bytree=space['colsample_bytree'],
                        subsample=space['subsample'])

    clf.fit(xTrain, yTrain, eval_metric="logloss")
    pred = clf.predict_proba(xValid)[:, 1]
    loss = log_loss(yValid, pred)
    return{'loss': loss, 'status': STATUS_OK}


space = {
    'n_estimators': hp.quniform('n_estimators', 100, 300, 100),
    'learning_rate': hp.quniform('learning_rate', 0.02, 0.05, 0.01),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'min_child_weight': hp.quniform('min_child_weight', 3, 8, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
}

trials = Trials()
bestParams = fmin(fn=objective,
                  space=space,
                  algo=tpe.suggest,
                  max_evals=100,
                  trials=trials)


clf = XGBClassifier(**bestParams)
clf.seed = 37

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
