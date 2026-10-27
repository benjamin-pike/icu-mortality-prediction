import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .metrics import logistic_nll


def clip_risk(probabilities):
    return np.clip(np.asarray(probabilities, dtype = float), 1e-06, 1 - 1e-06)


def optimise_coefficients(initial, design, outcome, offset, ridge):
    initial = np.asarray(initial, dtype = float)
    penalty = np.full(len(initial), ridge, dtype = float)
    penalty[0] = 0

    def objective(coefficients):
        eta = offset + np.sum(design * coefficients, axis = 1)
        return logistic_nll(outcome, eta) + 0.5 * np.sum(
            penalty * (coefficients - initial) ** 2
        )

    def gradient(coefficients):
        eta = offset + np.sum(design * coefficients, axis = 1)
        residual = expit(eta) - outcome
        data_gradient = np.sum(design * residual[:, None], axis = 0)
        return data_gradient + penalty * (coefficients - initial)

    result = minimize(objective, initial, jac = gradient, method = 'L-BFGS-B')
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(result.message)
    
    return result.x


def fit_score_updates(frame, ridge):
    y = frame.label.to_numpy(dtype = float)
    l2, l3 = (logit(clip_risk(frame.R2)), logit(clip_risk(frame.R3)))
    process = l3 - l2
    intercept = np.ones(len(y))

    def platt(score):
        return optimise_coefficients([0.0, 1.0], np.column_stack([intercept, score]), y, 0.0, ridge)

    return {
        'R3_intercept': optimise_coefficients([0.0], intercept[:, None], y, l3, ridge),
        'R3_platt': platt(l3),
        'process_update': optimise_coefficients([0.0, 1.0], np.column_stack([intercept, process]), y, l2, ridge),
        'unrestricted_stack': optimise_coefficients([0.0, 0.0, 1.0], np.column_stack([intercept, l2, l3]), y, 0.0, ridge)
    }


def predict_score_updates(frame, fitted):
    l2, l3 = (logit(clip_risk(frame.R2)), logit(clip_risk(frame.R3)))
    intercept, platt, process, stack = (
        fitted[name] for name in ('R3_intercept', 'R3_platt', 'process_update', 'unrestricted_stack')
    )

    return {
        'R2_raw': clip_risk(frame.R2),
        'R3_raw': clip_risk(frame.R3),
        'R3_intercept': expit(l3 + intercept[0]),
        'R3_platt': expit(platt[0] + platt[1] * l3),
        'process_update': expit(l2 + process[0] + process[1] * (l3 - l2)),
        'unrestricted_stack': expit(stack[0] + stack[1] * l2 + stack[2] * l3)
    }


def metric_rows(frame, risks, **settings):
    return [
        dict(
            settings,
            method = method,
            patients = len(frame),
            deaths = int(frame.label.sum()),
            log_loss = log_loss(frame.label, risk),
            Brier = brier_score_loss(frame.label, risk),
            AUROC = roc_auc_score(frame.label, risk)
        ) for method, risk in risks.items()
    ]