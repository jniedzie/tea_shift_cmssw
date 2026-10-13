# Code style: AI-code. AI-generated BDT diagnostic utility.
"""Paired, event-grouped statistics for fixed BDT acceptance comparisons."""
import numpy as np


def paired_support(pairs, events, event_weight_effective_n, probability_available, minimum=20):
    """Minimum sample-support gate, distinct from a defined probability ratio.

    Passing this floor does not establish precision or validate an interval.
    """
    if minimum < 1:
        raise ValueError('Minimum support must be positive')
    return bool(probability_available and pairs >= minimum and events >= minimum
                and event_weight_effective_n is not None
                and np.isfinite(event_weight_effective_n)
                and event_weight_effective_n >= minimum)


def paired_acceptance_difference(first, second, groups, weights=None, repeats=200, seed=12071, minimum=20):
    """First-minus-second acceptance, resampling shared events identically.

    Signed weights retain a signed ratio but cannot define probability intervals.
    A deterministic empirical comparison is still returned for unit weights.
    """
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    if first.shape != second.shape or first.ndim != 1 or len(groups) != len(first):
        raise ValueError('Paired decisions and event identities are not aligned')
    weights = np.ones(len(first)) if weights is None else np.asarray(weights, float)
    if weights.shape != first.shape or not np.isfinite(weights).all():
        raise ValueError('Paired comparison weights must be finite and aligned')
    keys, inverse = {}, []
    for key in groups:
        key = tuple(key)
        if key not in keys:keys[key] = len(keys)
        inverse.append(keys[key])
    delta = first.astype(float)-second.astype(float)
    denominator = float(weights.sum())
    numerator = float(np.sum(weights*delta))
    result = dict(pairs=len(first), events=len(keys), first_only_pairs=int((first & ~second).sum()),
        second_only_pairs=int((second & ~first).sum()), both_pass_pairs=int((first & second).sum()),
        difference=None, ci95_event_bootstrap=[None,None], bootstrap_replicates=0,
        probability_available=False, supported=False, support_minimum=minimum,
        event_weight_effective_n=None, negative_weight_pairs=int((weights<0).sum()),
        signed_difference_ratio=numerator/denominator if denominator else None,
        difference_definition='first acceptance minus second acceptance; shared events resampled together')
    paired_support(0, 0, None, False, minimum)  # Validate even empty inputs.
    if not len(first) or denominator<=0 or (weights<0).any():return result
    result.update(difference=numerator/denominator,probability_available=True)
    event_total=np.bincount(inverse,weights=weights,minlength=len(keys))
    result['event_weight_effective_n']=float(event_total.sum()**2/(event_total@event_total))
    result['supported']=paired_support(result['pairs'],result['events'],result['event_weight_effective_n'],True,minimum)
    if len(keys)<2 or repeats<=0:return result
    event_delta=np.bincount(inverse,weights=weights*delta,minlength=len(keys))
    rng=np.random.default_rng(seed)
    differences=[]
    for _ in range(repeats):
        counts=np.bincount(rng.integers(len(keys),size=len(keys)),minlength=len(keys))
        total=counts@event_total
        if total>0:differences.append(float((counts@event_delta)/total))
    if differences:
        result.update(ci95_event_bootstrap=np.quantile(differences,[.025,.975]).tolist(),bootstrap_replicates=len(differences))
    return result


def event_histogram_variance(bin_indices, groups, weights, bins):
    """Poisson event-bootstrap variance; correlated pairs in one bin add first."""
    bin_indices,weights=np.asarray(bin_indices),np.asarray(weights,float)
    if bin_indices.shape!=weights.shape or len(groups)!=len(weights):raise ValueError('Histogram arrays differ')
    if not np.isfinite(weights).all() or np.any(bin_indices<0) or np.any(bin_indices>=bins):raise ValueError('Invalid histogram values')
    totals={}
    for b,g,w in zip(bin_indices,groups,weights):
        key=(tuple(g),int(b));totals[key]=totals.get(key,0.)+float(w)
    result=np.zeros(bins)
    for (_,b),w in totals.items():result[b]+=w*w
    return result
