import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def risk_matrix(frame, ids):
    return (
        frame.pivot(index = 'stay_id', columns = ['repeat', 'draw'], values = 'risk')
        .reindex(ids)
        .sort_index(axis = 1)
    )


def loss_matrix(labels, risks):
    values = np.clip(np.asarray(risks, dtype = float), 1e-8, 1 - 1e-8)
    if values.ndim == 1:
        values = values[:, None]

    y = np.asarray(labels)[:, None]
    losses = -(y * np.log(values) + (1 - y) * np.log1p(-values))

    if isinstance(risks, pd.DataFrame):
        return pd.DataFrame(losses, index = risks.index, columns = risks.columns)
    
    return losses


def mean_draw_losses(losses):
    return losses.T.groupby(level = 'repeat', sort = True).mean().T


def paired_bootstrap(metadata, difference, replicates = 2000, seed = 77, cluster = False, equal_hospital = False):
    values = difference.reindex(metadata.index).to_numpy(float)
    repeats = difference.columns
    if len(values) != len(metadata) or np.isinf(values).any():
        raise ValueError('Invalid paired patient losses')

    valid = np.isfinite(values)
    if not valid.any(axis = 0).all():
        raise ValueError('An allocation has no paired evaluation patients')

    sums = np.where(valid, values, 0)
    counts = valid.astype(float)

    if cluster:
        codes, hospitals = pd.factorize(metadata.hospital_id, sort = True)
        if (codes < 0).any() or len(hospitals) < 2:
            raise ValueError('At least two identified hospitals are required')

        sums = np.stack([
            np.bincount(codes, weights = column, minlength = len(hospitals)) for column in sums.T
        ], axis = 1)

        counts = np.stack([
            np.bincount(codes, weights = column, minlength = len(hospitals)) for column in counts.T
        ], axis = 1)

    if equal_hospital:
        rates = np.divide(sums, counts, out = np.zeros_like(sums), where = counts > 0)
        sums, counts = rates, (counts > 0).astype(float)

    points = sums.sum(axis = 0) / counts.sum(axis = 0)
    if np.array_equal(counts, np.broadcast_to(counts[:, :1], counts.shape)):
        sums, counts = sums.mean(axis = 1, keepdims = True), counts[:, :1]

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        weights = np.bincount(rng.integers(len(sums), size = len(sums)), minlength = len(sums))

        denominator = np.einsum('i,ij->j', weights, counts)
        if (denominator > 0).all():
            numerator = np.einsum('i,ij->j', weights, sums)
            draws.append(float(np.mean(numerator / denominator)))

    draws = np.asarray(draws)
    
    return {
        'points': pd.Series(points, index = repeats),
        'draws': draws,
        'patients': int(valid.any(axis = 1).sum())
    }


def bootstrap_difference(metadata, difference, replicates = 2000, seed = 77, cluster = False, equal_hospital = False):
    sampled = paired_bootstrap(metadata, difference, replicates, seed, cluster, equal_hospital)
    points, draws = sampled['points'], sampled['draws']

    point = float(points.mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    result = {
        'difference': point,
        'lower': float(low),
        'upper': float(high),
        'p_value': float((1 + np.count_nonzero(np.abs(draws - point) >= abs(point))) / (len(draws) + 1)),
        'patients': sampled['patients'],
        'replicates': replicates,
        'uncertainty': 'hospital' if cluster else 'patient',
        'weighting': 'equal_hospital' if equal_hospital else 'patient',
        'allocations': len(points),
        'allocation_median': float(np.median(points)),
        'allocation_q10': float(np.quantile(points, 0.1)),
        'allocation_q90': float(np.quantile(points, 0.9)),
        'allocation_minimum': float(points.min()),
        'allocation_maximum': float(points.max()),
        'allocation_fraction_below_zero': float(np.mean(points < 0))
    }

    if 0 in points.index:
        result['seed_0_difference'] = float(points.loc[0])
    
    return result


def mean_auc(y, risks):
    values = []
    for _, draw_columns in risks.T.groupby(level = 'repeat'):
        aucs = []
        for column in draw_columns.to_numpy():
            valid = np.isfinite(column)
            aucs.append(roc_auc_score(np.asarray(y)[valid], column[valid]))

        values.append(np.mean(aucs))
    
    return float(np.mean(values))


def paired_difference(metadata, candidate, reference, cluster, equal_hospital = False):
    left = mean_draw_losses(loss_matrix(metadata.in_hospital_mortality, candidate))
    right = mean_draw_losses(loss_matrix(metadata.in_hospital_mortality, reference))
    if not left.notna().equals(right.notna()):
        raise ValueError('Unmatched evaluation patients')

    result = bootstrap_difference(
        metadata, left - right, cluster = cluster, equal_hospital = equal_hospital
    )
    result['AUROC_difference'] = mean_auc(
        metadata.in_hospital_mortality, candidate
    ) - mean_auc(metadata.in_hospital_mortality, reference)

    return result


def auc_order(labels, scores):
    valid = np.isfinite(scores)
    ids = np.flatnonzero(valid)
    order = np.argsort(scores[valid], kind = 'stable')
    ids = ids[order]

    ordered = scores[ids]
    ties = np.cumsum(np.r_[True, np.diff(ordered) != 0]) - 1
    
    return ids, ties, np.asarray(labels)[ids]


def weighted_auc(order, weights):
    ids, ties, labels = order
    positive = np.bincount(ties, weights = weights[ids] * labels)
    negative = np.bincount(ties, weights = weights[ids] * (1 - labels))
    denominator = positive.sum() * negative.sum()

    if denominator == 0:
        return np.nan
    
    return float(np.sum(positive * (np.cumsum(negative) - 0.5 * negative)) / denominator)


def paired_auc_interval(metadata, candidate, reference, cluster, replicates, seed):
    labels = metadata.in_hospital_mortality.to_numpy()
    if not candidate.columns.equals(reference.columns) or not candidate.isna().equals(reference.isna()):
        raise ValueError('AUROC intervals require matched allocations and draws')

    orders = [
        (
            auc_order(labels, candidate[column].to_numpy()),
            auc_order(labels, reference[column].to_numpy())
        )
        for column in candidate
    ]

    if cluster:
        codes, hospitals = pd.factorize(metadata.hospital_id, sort = True)
        if (codes < 0).any() or len(hospitals) < 2:
            raise ValueError('At least two identified hospitals are required')

    size = len(hospitals) if cluster else len(labels)
    rng = np.random.default_rng(seed)
    draws = []

    for _ in range(replicates):
        weights = np.bincount(rng.integers(size, size = size), minlength = size)
        patient_weights = weights[codes] if cluster else weights

        differences = [
            weighted_auc(left, patient_weights) - weighted_auc(right, patient_weights) for left, right in orders
        ]

        per_repeat = (pd.Series(differences, index = candidate.columns).groupby(level = 'repeat').mean())

        if np.isfinite(per_repeat).all():
            draws.append(per_repeat.mean())

    lower, upper = np.quantile(draws, [0.025, 0.975])
    
    return {'AUROC_lower': float(lower), 'AUROC_upper': float(upper)}


def prediction_rows(metadata, risks, **settings):
    values = np.asarray(risks)
    if values.ndim == 1:
        values = values[:, None]

    base = metadata.rename_axis('stay_id').reset_index()
    
    return pd.concat([
        base.assign(risk = values[:, draw], draw = draw, **settings)
        for draw in range(values.shape[1])
    ], ignore_index = True)


def joint_draws(loss_differences, hospitals, replicates = 5000, seed = 77):
    values = np.asarray(loss_differences, dtype = float)
    hospitals = pd.Series(hospitals)

    if (
        values.ndim != 3
        or values.shape[0] < 2
        or values.shape[1] != len(hospitals)
        or values.shape[2] == 0
        or not np.isfinite(values).all()
        or hospitals.isna().any()
        or hospitals.nunique() < 2
        or replicates < 2
    ):
        raise ValueError('Invalid source/patient/contrast array or hospital identifiers')

    codes, names = pd.factorize(hospitals, sort = True)
    sizes = np.bincount(codes)

    sums = np.stack([
        np.stack([np.bincount(codes, weights = v[:, c]) for c in range(values.shape[2])], axis = 1) for v in values
    ])

    rng = np.random.default_rng(seed)
    source = rng.integers(len(values), size = replicates)
    patient, equal = [], []

    for fit in source:
        h = rng.integers(len(names), size = len(names))
        patient.append(sums[fit, h].sum(axis = 0) / sizes[h].sum())
        equal.append((sums[fit, h] / sizes[h, None]).mean(axis = 0))

    return {
        'source_index': source,
        'patient': np.asarray(patient),
        'equal_hospital': np.asarray(equal)
    }
