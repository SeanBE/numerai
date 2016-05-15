import numpy as np
import pandas as pd
import datetime as dt
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import log_loss

ID_COL = 't_id'
LABEL_COL = 'probability'

os.chdir(os.path.expanduser('~/code/python/numerai'))

TRAIN_FILENAME = 'data/numerai_training_data.csv'
TEST_FILENAME = 'data/numerai_tournament_data.csv'

trainDf = pd.read_csv(TRAIN_FILENAME)

features = trainDf.drop("target", axis=1)
labels = trainDf.target

poly = PolynomialFeatures(degree=2, include_bias=False)
features = poly.fit_transform(features)

scaler = StandardScaler()
features = scaler.fit_transform(features)

testDf = pd.read_csv(TEST_FILENAME)
testX = testDf.drop(ID_COL, axis=1)
testX = poly.transform(testX)
testX = scaler.fit_transform(testX)


class Ensemble(object):

    def __init__(self, folds, stacker, models):
        self.n_folds = folds
        self.stacker = stacker
        self.base_models = models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(KFold(len(y), n_folds=self.n_folds,
                           shuffle=True, random_state=37))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                clf.fit(X_train, y_train, eval_metric='logloss')
                y_pred = clf.predict_proba(X_holdout)[:]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:]

            S_test[:, i] = S_test_i.mean(1)

        self.stacker.fit(S_train, y, eval_metric='logloss')
        y_pred = self.stacker.predict_proba(S_test)[:]
        return y_pred


xgbStacker = XGBClassifier()

xgbClf = XGBClassifier()
extraClf = ExtraTreesClassifier()
gbClf = GradientBoostingClassifier()
rfClf = RandomForestClassifier()

ensemble = Ensemble(5, xgbStacker, [xgbClf, extraClf, gbClf, rfClf])

predictions = ensemble.fit_predict(features, labels, testX)

testDf[LABEL_COL] = predictions[:, 1]

currentDt = dt.datetime.now().isoformat()
outputFilename = '../output/submission' + currentDt + '.csv'
testDf.to_csv(outputFilename, columns=(ID_COL, LABEL_COL), index=False)
