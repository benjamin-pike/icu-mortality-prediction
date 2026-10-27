from bisect import bisect_left

import numpy as np
import pandas as pd

from .data import observation_batches
from .eicu import append_gcs
from .features import RULES, build_features

UNIT_KEYS = ['stay_id', 'canonical_variable', 'unit_time']


def uniform_keys(hashes, seed):
    values = hashes.copy()
    
    with np.errstate(over = 'ignore'):
        values ^= np.uint64(seed)
        values = (values ^ values >> np.uint64(30)) * np.uint64(13787848793156543929)
        values = (values ^ values >> np.uint64(27)) * np.uint64(10723151780598845931)
        values ^= values >> np.uint64(31)
    
    return (values >> np.uint64(11)).astype(float) / 2**53


def spread_indices(times, count):
    targets = np.linspace(times[0], times[-1], count) if count > 1 else [times[-1]]
    available = list(range(len(times)))
    
    chosen = []
    for target in targets:
        position = bisect_left(available, int(np.searchsorted(times, target)))
        
        if position == 0:
            use = 0
        elif (
            position == len(available)
            or target - times[available[position - 1]] <= times[available[position]] - target
        ):
            time = times[available[position - 1]]
            use = bisect_left(available, int(np.searchsorted(times, time)))
        else:
            use = position
        
        chosen.append(available.pop(use))
    
    return np.asarray(chosen)


def restriction_mask(prepared, mode, seed, spec, *, retention = 1.0):
    units = prepared['units']
    protected = units.canonical_variable.isin(spec['protected_variables']).to_numpy()
    minute = units.minute_from_icu_admission.to_numpy()
    
    priority = (uniform_keys(prepared['hashes'], seed) if mode in {'thinning', 'random_count', 'gap_random'} else None)
    
    if mode == 'full':
        keep = np.ones(len(units), dtype = bool)
    
    elif mode == 'thinning':
        keep = priority < retention

    elif mode in {'random_count', 'spread', 'recent'}:
        keep = np.zeros(len(units), dtype = bool)
        
        for index, required in zip(prepared['groups'], prepared['required_counts'], strict = True):
            count = max(1, int(np.floor(retention * len(index))), required)
            
            if mode == 'random_count':
                chosen = index[np.argsort(priority[index])[:count]]
            elif mode == 'recent':
                chosen = index[-count:]
            else:
                chosen = index[spread_indices(minute[index], count)]
            
            keep[chosen] = True
    
    elif mode in {'terminal_outage', 'gap_random'}:
        keep = minute < 2880 - spec['terminal_outage_hours'] * 60

        if mode == 'gap_random':
            for chronological in prepared['groups']:
                index = np.sort(chronological)
                count = int(keep[index].sum())
                
                keep[index] = False
                keep[index[np.argsort(priority[index])[:count]]] = True
    
    else:
        raise ValueError(f'Unknown observation restriction: {mode}')
    
    return np.asarray(keep | protected, dtype = bool)


def restriction_settings(restriction, seed = 77):
    settings = {'mode': restriction, 'seed': seed}
    
    for prefix in ['retain', 'random_count', 'spread', 'recent']:
        if restriction.startswith(prefix + '_'):
            settings.update(
                mode = 'thinning' if prefix == 'retain' else prefix,
                retention = float(restriction[len(prefix) + 1 :]),
            )
            
            return settings
    
    if restriction == 'gap_random_12':
        settings['mode'] = 'gap_random'
    
    return settings


def schedule_requirements(inputs, units, groups):
    observations = inputs['observations'].loc[inputs['observations'].canonical_variable.ne('GCS')]
    
    values = pd.to_numeric(observations.value, errors = 'coerce').astype(float)
    percentage = observations.canonical_variable.eq('FiO2') & values.between(21, 100)
    values = values.where(~percentage, values / 100)
    
    low = observations.canonical_variable.map({name: limits[0] for name, limits in RULES.items()})
    high = observations.canonical_variable.map({name: limits[1] for name, limits in RULES.items()})
    valid = values.notna() & (low.isna() | values.between(low, high))
    last_valid = (
        observations.loc[valid]
        .groupby(['stay_id', 'canonical_variable'])
        .minute_from_icu_admission.max()
    )
    
    counts = []
    for index in groups:
        ordered = units.iloc[index]
        variable = ordered.canonical_variable.iloc[0]
    
        if variable == 'GCS':
            times = pd.to_datetime(ordered.unit_time)
            first_latest = times.loc[
                ordered.minute_from_icu_admission.eq(ordered.minute_from_icu_admission.iloc[-1])
            ].min()
            counts.append(int(times.ge(first_latest - pd.Timedelta(hours = 6)).sum()))
        else:
            last_time = last_valid.get((ordered.stay_id.iloc[0], variable))
            counts.append(
                int(ordered.minute_from_icu_admission.eq(last_time).sum())
                if last_time is not None else 1
            )
    
    return counts


def subset_inputs(inputs, ids):
    return dict(
        inputs,
        stays = inputs['stays'].loc[inputs['stays'].stay_id.isin(ids)],
        observations = inputs['observations'].loc[inputs['observations'].stay_id.isin(ids)],
        components = inputs['components'].loc[inputs['components'].RecordID.isin(ids)],
    )


def observation_units(inputs):
    events = inputs['observations']
    
    scalar = events.loc[
        events.canonical_variable.ne('GCS'),
        ['stay_id', 'canonical_variable', 'minute_from_icu_admission']
    ].copy()
    scalar['unit_time'] = scalar.minute_from_icu_admission.astype(str)
    
    components = (
        inputs['components'][['RecordID', 'CHARTTIME', 'minute_from_icu_admission']]
        .rename(columns = {'RecordID': 'stay_id'})
        .copy()
    )
    components['canonical_variable'] = 'GCS'
    components['unit_time'] = components.CHARTTIME.astype(str)
    
    return (
        pd.concat([scalar, components.drop(columns = 'CHARTTIME')], ignore_index = True)
        .drop_duplicates(UNIT_KEYS)
        .sort_values(UNIT_KEYS)
        .reset_index(drop = True)
    )


def prepare_restrictions(inputs, modes):
    units = observation_units(inputs)

    groups = []
    if set(modes) & {'random_count', 'recent', 'spread', 'gap_random'}:
        ordered = units.sort_values(['stay_id', 'canonical_variable', 'minute_from_icu_admission', 'unit_time'])
        order = ordered.index.to_numpy()
        groups = [
            order[index]
            for index in ordered.groupby(['stay_id', 'canonical_variable'], sort = False).indices.values()
        ]

    scalar = (
        inputs['observations']
        .loc[inputs['observations'].canonical_variable.ne('GCS')]
        .copy()
    )
    scalar['unit_time'] = scalar.minute_from_icu_admission.astype(str)

    components = inputs['components'].assign(
        stay_id = inputs['components'].RecordID,
        canonical_variable = 'GCS',
        unit_time = inputs['components'].CHARTTIME.astype(str)
    )

    return {
        'inputs': inputs,
        'units': units,
        'groups': groups,
        'scalar': scalar,
        'components': components,
        'hashes': pd.util.hash_pandas_object(units[UNIT_KEYS].astype(str), index = False).to_numpy(dtype = np.uint64)
        if set(modes) & {'thinning', 'random_count', 'gap_random'} else None,
        'required_counts': schedule_requirements(inputs, units, groups)
        if set(modes) & {'random_count', 'recent', 'spread'} else None
    }


def restricted_inputs(prepared, settings, rules):
    keep = restriction_mask(prepared, spec = rules, **settings)
    selected = prepared['units'].loc[keep, UNIT_KEYS]
    
    scalar = (
        prepared['scalar']
        .merge(selected, on = UNIT_KEYS, how = 'inner')
        .drop(columns = 'unit_time')
    )
    components = (
        prepared['components']
        .merge(selected, on = UNIT_KEYS, how = 'inner')
        .drop(columns = ['stay_id', 'unit_time'])
    )
    
    return dict(
        prepared['inputs'],
        observations = append_gcs(scalar, components),
        components = components,
    )


def recording_view(inputs):
    events, components = inputs['observations'], inputs['components']
    keep = events.available_minute.le(2880) | events.canonical_variable.isin(['Height', 'MechVent'])
    events = events.loc[keep]
    components = components.loc[components.available_minute.le(2880)]
    
    return dict(inputs, observations = append_gcs(events, components), components = components)


def feature_view(name, original, restriction, seed, rules):
    frames = []
    for inputs in observation_batches(name):
        inputs = subset_inputs(inputs, original.index)
        
        if inputs['stays'].empty:
            continue
        
        if restriction == 'known_available':
            altered = recording_view(inputs)
        else:
            settings = restriction_settings(restriction, seed)
            prepared = prepare_restrictions(inputs, [settings['mode']])
            altered = restricted_inputs(prepared, settings, rules)
        
        frames.append(build_features(altered))

    frame = pd.concat(frames).loc[original.index]
    protected = [c for c in frame if c.split('_')[0] in rules['protected_variables']]
    frame.loc[:, protected] = original[protected]

    return frame
