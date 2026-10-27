import numpy as np
import pandas as pd

COMPONENTS = ['eye', 'verbal', 'motor']
COMPONENT_ITEMS = {
    184: 'eye',
    723: 'verbal',
    454: 'motor',
    220739: 'eye',
    223900: 'verbal',
    223901: 'motor'
}
FIO2_ITEMIDS = frozenset({223835, 3420, 190})
CAREVUE_GCS_COMPONENTS = (184, 723, 454)
METAVISION_GCS_COMPONENTS = (220739, 223900, 223901)
GCS_TOTAL_ITEMIDS = frozenset({198})
GCS_COMPONENT_ITEMIDS = frozenset(CAREVUE_GCS_COMPONENTS + METAVISION_GCS_COMPONENTS)
MECHVENT_ITEMIDS = frozenset(
    {
        224419,
        224684,
        224685,
        224686,
        224687,
        220339,
        224695,
        224696,
        224697,
        226873,
        224700,
        224701,
        224702,
        224705,
        224706,
        224707,
        224709,
        720,
        224738,
        223848,
        223849,
        224746,
        224747,
        224750,
        227187
    }
)


def availability_minutes(frame):
    stored = pd.to_datetime(frame.get('STORETIME', pd.Series(pd.NaT, index = frame.index)), errors = 'coerce')
    measured = pd.to_datetime(frame.CHARTTIME, errors = 'coerce')
    available = stored.where(stored.ge(measured), measured).where(stored.notna())
    
    return (available - frame.INTIME).dt.total_seconds() / 60


def conservative_latest(values):
    return np.nan if values.isna().any() else float(values.max())


def derive_gcs(events):
    components = events.loc[events.ITEMID.isin(COMPONENT_ITEMS)].copy()
    
    if components.empty:
        return events.iloc[:0].drop(columns = 'raw_value', errors = 'ignore')
    
    components['component'] = components.ITEMID.map(COMPONENT_ITEMS)
    values = pd.to_numeric(components.value, errors = 'coerce')
    raw = components.raw_value.astype('string').str.strip()
    tube = components.ITEMID.eq(723) & raw.eq('1.0 ET/Trach') | components.ITEMID.eq(223900) & raw.eq('No Response-ETT')
    upper = components.component.map({'eye': 4, 'verbal': 5, 'motor': 6})
    
    components['component_value'] = values.where(values.ge(1) & values.le(upper))
    components.loc[tube.fillna(False), 'component_value'] = 0.0
    
    keys = [
        'RecordID',
        'SUBJECT_ID',
        'HADM_ID',
        'ICUSTAY_ID',
        'CHARTTIME',
        'minute_from_icu_admission'
    ]
    
    timeline = (
        components.groupby(keys + ['component'], dropna = False)
        .component_value.max()
        .unstack('component')
        .reindex(columns = COMPONENTS)
        .reset_index()
        .sort_values(['RecordID', 'CHARTTIME'])
        .reset_index(drop = True)
    )
    
    previous = timeline.groupby('RecordID')[COMPONENTS].shift()
    gap = timeline.groupby('RecordID').CHARTTIME.diff().dt.total_seconds() / 60
    previous.loc[~gap.lt(360)] = np.nan

    normal = {'eye': 4, 'verbal': 5, 'motor': 6}
    current = timeline[COMPONENTS]
    total = current.fillna(previous).fillna(normal).sum(axis = 1)

    previous_tube = previous.verbal.eq(0)
    total.loc[previous_tube] = current.fillna(normal).sum(axis = 1)
    tube = current.verbal.eq(0) | current.verbal.isna() & previous_tube
    timeline['value'] = total.mask(tube, 15.0)
    
    provenance = components.groupby(keys, dropna = False).agg(
        source_row_id = ('source_row_id', 'min'),
        available_minute = ('available_minute', conservative_latest),
    ).reset_index()
    result = timeline.merge(provenance, on = keys, validate = 'one_to_one')

    unknown = result.available_minute.isna().groupby(result.RecordID).cummax()
    result['available_minute'] = (result.groupby('RecordID').available_minute.cummax().mask(unknown))

    minute_keys = [key for key in keys if key != 'CHARTTIME']
    result = result.groupby(minute_keys, as_index = False, dropna = False).agg(
        CHARTTIME = ('CHARTTIME', 'min'),
        source_row_id = ('source_row_id', 'min'),
        available_minute = ('available_minute', conservative_latest),
        value = ('value', 'median')
    )

    return result.assign(
        canonical_variable = 'GCS',
        source_table = 'CHARTEVENTS',
        ITEMID = pd.NA,
        VALUEUOM = 'score',
    )


def normalise(variable, label, value, raw_value, itemid):
    result = pd.to_numeric(value, errors = 'coerce')

    fio2 = variable.eq('FiO2') & itemid.isin(FIO2_ITEMIDS)
    fractional = result.between(0.21, 1.0)
    percentage = result.between(21.0, 100.0)
    result.loc[fio2 & percentage] /= 100
    result.loc[fio2 & ~(fractional | percentage)] = np.nan

    magnesium = variable.eq('Mg')
    result.loc[magnesium] *= 0.4114

    fahrenheit = variable.eq('Temp') & label.str.contains('temperature f', case = False, na = False)
    result.loc[fahrenheit] = (result.loc[fahrenheit] - 32) * 5 / 9

    pounds = variable.eq('Weight') & label.str.contains('lb', case = False, na = False)
    result.loc[pounds] *= 0.45359237

    inches = variable.eq('Height') & (
        label.str.contains('inch', case = False, na = False) | itemid.isin([920, 1394, 3486, 4187, 226707])
    )
    result.loc[inches] *= 2.54

    ventilated = variable.eq('MechVent') & raw_value.notna()
    result.loc[ventilated] = 1.0
    
    return result


def derive_mechvent(events):
    signals = events.loc[events.ITEMID.isin(MECHVENT_ITEMIDS)].copy()
    raw = signals.raw_value.astype('string').str.strip()
    accepted = raw.ne('') & ~(signals.ITEMID.eq(223848) & raw.eq('Other'))
    signals = signals.loc[accepted]

    if signals.empty:
        return events.iloc[0:0].drop(columns = 'raw_value', errors = 'ignore')

    keys = ['RecordID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'minute_from_icu_admission']

    result = signals.groupby(keys, as_index = False, dropna = False).agg(
        source_row_id = ('source_row_id', 'min'),
        available_minute = ('available_minute', conservative_latest)
    )

    result['canonical_variable'] = 'MechVent'
    result['source_table'] = 'CHARTEVENTS'
    result['ITEMID'] = pd.NA
    result['value'] = 1.0
    result['VALUEUOM'] = 'indicator'

    return result[[
        'RecordID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',
        'canonical_variable', 'source_table', 'ITEMID', 'source_row_id',
        'CHARTTIME', 'available_minute', 'minute_from_icu_admission',
        'value', 'VALUEUOM'
    ]]


def derive_chart_events(events):
    events = events.copy()
    
    if 'available_minute' not in events:
        events['available_minute'] = np.nan
    
    direct = events.loc[
        ~events.ITEMID.isin(GCS_COMPONENT_ITEMIDS | GCS_TOTAL_ITEMIDS | MECHVENT_ITEMIDS)
    ].drop(columns = 'raw_value', errors = 'ignore')
    
    return pd.concat([direct, derive_gcs(events), derive_mechvent(events)], ignore_index = True)


def valid_fio2(values):
    numeric = pd.to_numeric(values, errors = 'coerce')
    
    return numeric.between(0.21, 1.0) | numeric.between(21.0, 100.0)
