import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .metrics import logistic_nll

CALIBRATION_EPSILON = 1e-06


def preprocess(features, scale):
    categorical = [feature for feature in features if feature == 'ICUType']
    numeric = [feature for feature in features if feature != 'ICUType']
    numeric_steps = [('impute', SimpleImputer(strategy = 'median', keep_empty_features = True))]
    
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    
    return ColumnTransformer(
        [
            ('numeric', Pipeline(numeric_steps), numeric),
            (
                'categorical',
                Pipeline(
                    [
                        ('impute', SimpleImputer(strategy = 'most_frequent', keep_empty_features = True)),
                        ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
                    ]
                ),
                categorical
            )
        ],
        sparse_threshold = 0
    )


def fit_model(x, y, family, parameters):
    transform = preprocess(list(x), scale = family == 'logistic')
    matrix = transform.fit_transform(x)
    
    constructors = {
        'xgboost': XGBClassifier,
        'logistic': LogisticRegression,
        'histogram': HistGradientBoostingClassifier,
        'random_forest': RandomForestClassifier
    }
    
    estimator = constructors[family](**parameters)
    estimator.fit(matrix, np.asarray(y))
    
    return {'features': list(x), 'transform': transform, 'estimator': estimator}


def predict(model, x):
    matrix = model['transform'].transform(x[model['features']])
    estimator = model['estimator']
    
    if isinstance(estimator, LogisticRegression):
        return expit(np.einsum('ij,kj->ik', np.asarray(matrix), estimator.coef_).ravel() + estimator.intercept_[0])
    
    return estimator.predict_proba(matrix)[:, 1]


def fit_source_model(x, y, family, parameters, folds = 5, seed = 77):
    if not x.index.is_unique or not x.index.equals(y.index):
        raise ValueError('Source rows must be unique and outcomes aligned')
    
    splitter = StratifiedKFold(folds, shuffle = True, random_state = seed)
    oof = pd.Series(index = x.index, dtype = float, name = 'risk')
    fold_ids = pd.Series(index = x.index, dtype = int, name = 'fold')
    
    for fold, (train, test) in enumerate(splitter.split(x, y)):
        fitted = fit_model(x.iloc[train], y.iloc[train], family, parameters)
        oof.iloc[test] = predict(fitted, x.iloc[test])
        fold_ids.iloc[test] = fold
    
    corrections = {
        method: fit_recalibrator(y, oof, method)
        for method in ('intercept', 'platt')
    }
    
    return {
        'model': fit_model(x, y, family, parameters),
        'corrections': corrections,
        'oof': oof,
        'oof_fold': fold_ids.astype(int)
    }


def apply_calibration(coefficients, risk):
    return expit(coefficients[0] + coefficients[1] * probability_logit(risk))


def probability_logit(probabilities, epsilon = CALIBRATION_EPSILON):
    risk = np.asarray(probabilities, dtype = float)
    
    if not np.isfinite(risk).all():
        raise ValueError('Calibration requires finite probabilities')
    
    if np.any((risk < 0) | (risk > 1)):
        raise ValueError('Probabilities must lie in [0, 1]')
    
    clipped = np.clip(risk, epsilon, 1 - epsilon)
    
    return np.log(clipped) - np.log1p(-clipped)


def binary_outcome(y):
    outcome = np.asarray(y, dtype = float)
    
    if not np.isfinite(outcome).all():
        raise ValueError('Calibration requires finite outcomes')
    
    if outcome.ndim != 1 or set(np.unique(outcome)) != {0, 1}:
        raise ValueError('Calibration requires both outcome classes')
    
    return outcome


def fit_recalibrator(y, probabilities, method = 'platt'):
    outcome = binary_outcome(y)
    logit = probability_logit(probabilities)
    
    if method == 'intercept':
        def objective(value):
            return logistic_nll(outcome, value + logit)

        result = minimize_scalar(objective, bounds = (-20.0, 20.0), method = 'bounded')
        
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f'Intercept recalibration failed: {result.message}')
        
        return np.array([float(result.x), 1.0])
    
    if method != 'platt':
        raise ValueError(f'Unknown recalibration method: {method}')
    
    centre = float(logit.mean())
    spread = float(logit.std())
    if spread <= np.finfo(float).eps:
        raise ValueError('Calibration slope is unidentified for constant predictions')
    standardised = (logit - centre) / spread

    def objective(theta):
        linear = theta[0] + theta[1] * standardised
        loss = logistic_nll(outcome, linear)
        residual = expit(linear) - outcome
        gradient = np.array([residual.sum(), np.einsum('i,i->', residual, standardised)], dtype = float)
        
        return (loss, gradient)

    def score(theta):
        return objective(theta)[1] / len(outcome)

    def information(theta):
        probability = expit(theta[0] + theta[1] * standardised)
        weights = probability * (1 - probability)
        design = np.column_stack([np.ones(len(outcome)), standardised])
        return np.einsum('ni,n,nj->ij', design, weights, design) / len(outcome)

    result = least_squares(
        score,
        np.array([centre, spread]),
        jac = information,
        xtol = 1e-12,
        ftol = 1e-12,
        gtol = 1e-12,
        max_nfev = 10000
    )
    
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or np.max(np.abs(score(result.x))) > 1e-07
        or (np.max(np.abs(result.x)) > 100)
        or (np.linalg.eigvalsh(information(result.x)).min() < 1e-10)
    ):
        raise RuntimeError(f'Platt recalibration unidentified or failed: {result.message}')
    
    slope = float(result.x[1] / spread)
    intercept = float(result.x[0] - slope * centre)
    if not np.isfinite([intercept, slope]).all():
        raise FloatingPointError('Calibration returned non-finite coefficients')
    
    return np.array([intercept, slope])


def calibration_summary(y, probabilities):
    outcome = binary_outcome(y)
    risk = np.asarray(probabilities, dtype = float)
    calibrator = fit_recalibrator(outcome, risk, method = 'platt')
    
    return {
        'calibration_intercept': calibrator[0],
        'calibration_slope': calibrator[1],
        'mean_predicted_mortality': float(risk.mean()),
        'observed_mortality': float(outcome.mean())
    }


def calibration_curve_frame(y, probabilities, n_bins = 10):
    observed, predicted = calibration_curve(y, probabilities, n_bins = n_bins, strategy = 'quantile')
    
    return pd.DataFrame(
        {
            'bin': range(1, len(observed) + 1),
            'mean_predicted_mortality': predicted,
            'observed_mortality': observed
        }
    )
