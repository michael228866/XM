"""Pure, label-only transformations. Never supply these columns to a model."""
import numpy as np

FORMULAS = {
    'L0_CURRENT': 'C1_NET_R > 0',
    'L1_R_GE_025': 'C1_NET_R >= 0.25',
    'L1_R_GE_050': 'C1_NET_R >= 0.50',
    'L2_NONLOSS_GT_M025': 'C1_NET_R > -0.25',
    'L2_NONLOSS_GT_M050': 'C1_NET_R > -0.50',
}


def make_label(label_id, realized_r):
    r = np.asarray(realized_r, dtype=np.float64)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError('Missing/nonfinite realized R; never impute labels')
    if label_id == 'L0_CURRENT':
        result = r > 0
    elif label_id == 'L1_R_GE_025':
        result = r >= .25
    elif label_id == 'L1_R_GE_050':
        result = r >= .5
    elif label_id == 'L2_NONLOSS_GT_M025':
        result = r > -.25
    elif label_id == 'L2_NONLOSS_GT_M050':
        result = r > -.5
    else:
        raise ValueError('Unsupported frozen label: '+label_id)
    return result.astype(np.int8)


def class_summary(target, minimum=100):
    y = np.asarray(target)
    if y.ndim != 1 or not np.isin(y, [0,1]).all():
        raise ValueError('Binary labels required')
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    rate = positive/len(y) if len(y) else 0.
    if min(positive, negative) < minimum or not .01 <= rate <= .99:
        raise ValueError('LABEL_DEGENERACY: class minimum or prevalence')
    return dict(positive_count=positive, negative_count=negative, positive_rate=rate,
                class_weight={'0':1., '1':1.}, weighting='NONE')


def partitions(data, validation_start, development_start, calibration_start=None):
    t = np.asarray(data['train_ns'])
    feature = np.asarray(data['feature_cutoff_time'])
    outcome = np.asarray(data['maturity'])
    legacy = np.asarray(data['legacy_maturity'])
    if not (t.ndim == 1 and all(x.shape == t.shape for x in (feature,outcome,legacy))
            and np.all(feature <= t) and np.all(outcome >= t)
            and np.all(t < validation_start) and np.all(outcome < validation_start)
            and np.all(legacy < validation_start) and np.all(np.diff(t) > 0)):
        raise ValueError('Temporal leakage or misaligned outcome timestamps')
    fit_end = development_start if calibration_start is None else calibration_start
    if not fit_end <= development_start < validation_start:
        raise ValueError('Invalid development boundaries')
    subset = np.asarray(data['b0_train']) < .75
    fit = np.flatnonzero(subset & (t < fit_end) & (outcome < fit_end) & (legacy < fit_end))
    dev = np.flatnonzero(subset & (t >= development_start))
    cal = np.array([],dtype=np.int64) if calibration_start is None else np.flatnonzero(
        subset & (t >= calibration_start) & (t < development_start)
        & (outcome < development_start) & (legacy < development_start))
    if not len(fit) or not len(dev) or (calibration_start is not None and not len(cal)):
        raise ValueError('Insufficient purged partition')
    return fit, cal, dev


def choose_threshold(probability, target, grid):
    """Development-only balanced accuracy; ties recall then higher threshold."""
    p, y = np.asarray(probability), np.asarray(target)
    class_summary(y)
    if p.shape != y.shape or not np.isfinite(p).all() or not np.all((p >= 0) & (p <= 1)):
        raise ValueError('Invalid development probabilities')
    rows = []
    for threshold in grid:
        prediction = p >= threshold
        recall = float(prediction[y == 1].mean())
        specificity = float((~prediction[y == 0]).mean())
        rows.append(dict(threshold=threshold, balanced_accuracy=(recall+specificity)/2,
                         recall=recall, specificity=specificity))
    if not rows:
        raise ValueError('Empty frozen threshold grid')
    best = max(rows,key=lambda r:(r['balanced_accuracy'],r['recall'],r['threshold']))
    return best['threshold'], rows
