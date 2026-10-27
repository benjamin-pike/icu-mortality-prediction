import pandas as pd

from .eicu import load_bucket
from .io import WORK
from .statistics import risk_matrix

MIMICIII_ICU_TYPE_CODES = {'CCU': 1, 'CSRU': 2, 'MICU': 3, 'SICU': 4, 'TSICU': 4}
MIMICIV_ICU_TYPE_CODES = {
    'Coronary Care Unit (CCU)': 1,
    'Cardiac Vascular Intensive Care Unit (CVICU)': 2,
    'Medical Intensive Care Unit (MICU)': 3,
    'Medical/Surgical Intensive Care Unit (MICU/SICU)': 3,
    'Surgical Intensive Care Unit (SICU)': 4,
    'Trauma SICU (TSICU)': 4,
    'Neuro Surgical Intensive Care Unit (Neuro SICU)': 4
}
ALL_ICU_TYPE_CODES = MIMICIII_ICU_TYPE_CODES | MIMICIV_ICU_TYPE_CODES

def cohort(name):
    return pd.read_parquet(WORK / 'data' / name / 'cohort.parquet').set_index('stay_id')


def features(name):
    return pd.read_parquet(WORK / 'features' / f'{name}.parquet')


def record_cohort_step(rows, domain, step, before, after, id_columns):
    step_number = 1 + max((row['step_number'] for row in rows if row['domain'] == domain), default = 0)
    
    for unit, column in id_columns.items():
        n_before = int(before[column].nunique())
        n_after = int(after[column].nunique())
        
        rows.append(
            {
                'domain': domain,
                'step_number': step_number,
                'step': step,
                'unit': unit,
                'before': n_before,
                'excluded': n_before - n_after,
                'remaining': n_after
            }
        )
    
    return after


def canonical_stays(frame, name):
    columns = {
        'RecordID': 'stay_id',
        'SUBJECT_ID': 'subject_id',
        'HADM_ID': 'hadm_id',
        'INTIME': 'intime',
        'OUTTIME': 'outtime',
        'LOS': 'los_days',
        'AGE_AT_ICU_ADMISSION': 'age',
        'GENDER': 'sex',
        'FIRST_CAREUNIT': 'icu_type',
        'IN_HOSPITAL_MORTALITY': 'in_hospital_mortality'
    }

    result = frame.rename(columns = columns).copy()
    result['icu_type_code'] = result.icu_type.map(ALL_ICU_TYPE_CODES).astype('Int64')
    result['domain'] = name
    
    return result[
        [
            'domain',
            'stay_id',
            'subject_id',
            'hadm_id',
            'intime',
            'outtime',
            'los_days',
            'age',
            'sex',
            'icu_type',
            'icu_type_code',
            'in_hospital_mortality'
        ]
    ]


def prediction_matrix(name, ids, filters):
    frame = pd.read_parquet(
        WORK / 'predictions' / name,
        columns = ['stay_id', 'repeat', 'draw', 'risk'],
        filters = filters,
    )
    
    return risk_matrix(frame, ids)


def observation_batches(name):
    directory = WORK / 'data' / name
    stays = cohort(name).reset_index()
    
    if name == 'eicu':
        for bucket in sorted(stays.bucket.unique()):
            yield load_bucket(directory / 'events', stays, int(bucket))
    else:
        columns = {
            'RecordID': 'stay_id',
            'ITEMID': 'source_itemid',
            'VALUEUOM': 'source_unit',
            'SUBJECT_ID': 'subject_id',
            'HADM_ID': 'hadm_id'
        }
        
        events = pd.read_parquet(directory / 'observations.parquet').rename(columns = columns)
        
        yield {
            'name': name,
            'stays': stays,
            'observations': events,
            'components': pd.read_parquet(directory / 'gcs_components.parquet')
        }


def prediction_metadata(metadata):
    result = metadata[['in_hospital_mortality', 'partition']].copy()
    result['hospital_id'] = (
        metadata.hospital_id if 'hospital_id' in metadata else metadata.domain
    ).astype('string')
    
    return result
