# This is the source code for example 1 of the manuscript. Dataset not included.
#
# jtorniainen // UEF
# 2021
# MIT License

import tpot
import nippy
import pickle
import pandas as pd
import sklearn
import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import NuSVC, SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder


N_ITER = 1000  # Number of iterations from random search


if __name__ == '__main__':
    np.random.seed(704)
    results = {}

    # Dataset can be retrieved from  https://etsin.fairdata.fi/dataset/925999ad-8127-4428-b7cc-c03c12e62d3c/data
    filename = 'Data_NIRS_AvaSpec-NIR256-2.5-HSC.csv'
    df = pd.read_csv(filename, delimiter=';')
    x = df.to_numpy()[:, 2:].astype(np.float64)  # NIR spectra
    y = np.array([i.split('_')[1] for i in df.id.to_numpy()])  # Ligament type (ACL, PCL, MCL, LCL, PT)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    le = LabelEncoder()
    y_train_ = le.fit_transform(y_train)
    y_test_ = le.transform(y_test)

    # Baseline performance with randomly optimized nuSVC-classifier
    params = dict(kernel=['linear', 'poly', 'rbf', 'sigmoid'], degree=[1, 2, 3],
                  C=[1e-4, 1e-3, 1e-2, 1e-1, .5, 1, 5, 10, 15, 20, 25])
    svc = SVC()
    baseline = RandomizedSearchCV(svc, params, cv=5, n_iter=N_ITER)
    baseline.fit(x_train, y_train)

    results['Baseline CV'] = np.mean(cross_validate(baseline.best_estimator_, x_train, y_train)['test_score'])
    results['Baseline Test'] = baseline.score(x_test, y_test)

    # Expert-derived analysis pipeline with four operations
    estimators = [('snv', nippy.StandardNormalVariate()),
                  ('savgol', nippy.SavitzkyGolay()),
                  ('scale_feat', StandardScaler()),
                  ('classify', SVC())]

    pipeline = Pipeline(estimators)

    params = dict(savgol__filter_win=randint(7, 25), savgol__deriv_order=randint(1, 4),
                  scale_feat__with_mean=[True, False], scale_feat__with_std=[True, False],
                  classify__kernel=['linear', 'poly', 'rbf', 'sigmoid'], classify__degree=[1, 2, 3],
                  classify__C=[1e-4, 1e-3, 1e-2, 1e-1, .5, 1, 5, 10, 15, 20, 25])

    expert = RandomizedSearchCV(pipeline, params, cv=5, n_iter=N_ITER)
    expert.fit(x_train, y_train)

    results['Expert CV'] = np.mean(cross_validate(expert.best_estimator_, x_train, y_train)['test_score'])
    results['Expert Test'] = expert.score(x_test, y_test)

    # AutoML-optimized pipeline
    config = {

        'nippy.NoPreprocessing': {
        },

        'nippy.SavitzkyGolay': {
            'filter_win': [7, 11, 13, 15, 17, 19, 21, 23, 25],
            'deriv_order': [0, 1, 2]
        },

        'nippy.LocalStandardNormalVariate': {
            'num_windows': [2, 3, 4, 5, 6, 7],
        },

        'nippy.RobustNormalVariate': {
            'iqr1': [.10, .15, .25, .30, .35, .40, .45],
            'iqr2': [.55, .60, .65, .70, .75, .85, .90],
        },

        'nippy.MultipleScatterCorrection': {
        },

        'nippy.Detrend': {
        },

        'sklearn.preprocessing.RobustScaler': {
        },

        'sklearn.preprocessing.StandardScaler': {
        },

        'sklearn.decomposition.FastICA': {
            'n_components': np.arange(1, 187, 1),
            'tol': np.arange(0.0, 1.01, 0.05),
            'whiten': [True, False],
            'algorithm': ['parallel', 'deflation'],
            'fun': ['logcosh', 'exp', 'cube']
        },

        'sklearn.decomposition.PCA': {
            'n_components': np.arange(1, 187, 1)
        },

        'sklearn.svm.SVC': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [1, 2, 3],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
        },

        'sklearn.neighbors.KNeighborsClassifier': {
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
        },

        'sklearn.linear_model.LogisticRegression': {
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False]
        },

        'sklearn.linear_model.SGDClassifier': {
            'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['elasticnet'],
            'alpha': [0.0, 0.01, 0.001],
            'learning_rate': ['invscaling', 'constant'],
            'fit_intercept': [True, False],
            'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
            'eta0': [0.1, 1.0, 0.01],
            'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
        },

        'sklearn.feature_selection.VarianceThreshold': {
            'threshold': [0, .01, .1, .5]
        },

        'sklearn.feature_selection.SelectPercentile': {
            'percentile': [5, 7.5, 10, 12, 15]
        },

        'sklearn.feature_selection.RFE': {
            'estimator': [sklearn.svm.LinearSVC],
            'n_features_to_select': np.arange(.01, 1.0, .05)
        },
    }

    pipeline_structure = 'Transformer-Transformer-Selector-Classifier'

    automl = tpot.TPOTClassifier(generations=100, population_size=100, cv=5, template=pipeline_structure,
                                   config_dict=config, verbosity=2, n_jobs=-1, max_eval_time_mins=10)

    automl.fit(x_train, y_train_)

    results['AutoML CV'] = np.mean(cross_validate(automl.fitted_pipeline_, x_train, y_train_)['test_score'])
    results['AutoML Test'] = automl.score(x_test, y_test_)

    print(results)
