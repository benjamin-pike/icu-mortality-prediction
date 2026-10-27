import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def probability_metrics(y, risk):
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Discrimination metrics require both binary outcome classes')
    
    return {
        'AUROC': roc_auc_score(y, risk),
        'Brier': brier_score_loss(y, risk),
        'log_loss': log_loss(y, np.clip(risk, 1e-08, 1 - 1e-08))
    }


def logistic_nll(outcome, linear_predictor):
    outcome = np.asarray(outcome, dtype = float)
    
    return float(
        np.sum(np.logaddexp(0.0, linear_predictor) - outcome * linear_predictor)
    )


def binary_losses(labels, risks, *, epsilon):
    labels = np.asarray(labels, dtype = float)
    risks = np.clip(np.asarray(risks, dtype = float), epsilon, 1 - epsilon)
    
    return -(labels * np.log(risks) + (1 - labels) * np.log1p(-risks))
