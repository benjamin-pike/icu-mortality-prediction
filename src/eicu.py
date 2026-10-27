import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .features import clean_value
from .readers import read_chunks
from .signals import derive_gcs

COMPONENT_IDS = {'GCSEye': 184, 'GCSVerbal': 723, 'GCSMotor': 454}
COMPONENT_COLUMNS = [
    'RecordID',
    'SUBJECT_ID',
    'HADM_ID',
    'ICUSTAY_ID',
    'CHARTTIME',
    'minute_from_icu_admission',
    'ITEMID',
    'value',
    'raw_value',
    'source_row_id',
    'available_minute'
]
OBSERVATION_COLUMNS = {
    'domain',
    'subject_id',
    'hadm_id',
    'stay_id',
    'minute_from_icu_admission',
    'canonical_variable',
    'value',
    'source_table',
    'source_itemid',
    'source_unit',
    'source_row_id'
}
EVENT_COLUMNS = sorted(OBSERVATION_COLUMNS) + ['available_minute']
WIDE_VITALS = {
    'vitalperiodic': {
        'heartrate': ('HR', 'bpm'),
        'respiration': ('RespRate', '/min'),
        'temperature': ('Temp', 'C'),
        'sao2': ('SaO2', '%'),
        'systemicsystolic': ('SysABP', 'mmHg'),
        'systemicdiastolic': ('DiasABP', 'mmHg'),
        'systemicmean': ('MAP', 'mmHg')
    },
    'vitalaperiodic': {
        'noninvasivesystolic': ('NISysABP', 'mmHg'),
        'noninvasivediastolic': ('NIDiasABP', 'mmHg'),
        'noninvasivemean': ('NIMAP', 'mmHg')
    }
}


def priority(value, purpose):
    return int.from_bytes(
        hashlib.sha256(f'{purpose}:{value}'.encode()).digest()[:8], 'big'
    ) / 2**64


def numeric(series):
    return pd.to_numeric(series.replace('', pd.NA), errors = 'coerce')



def event_keys(frame, schema):
    values = frame[schema['keys']].fillna('').astype(str)
    
    return pd.Series(
        [
            json.dumps(list(row), ensure_ascii = True, separators = (',', ':'))
            for row in values.itertuples(index = False, name = None)
        ],
        index = frame.index,
        dtype = 'string'
    )


def joined_events(frame, schema, stays):
    identifier = schema.get('stay', schema.get('admission'))
    canonical = 'stay_id' if schema.get('stay') else 'hadm_id'
    left = frame.copy()
    left['_join'] = numeric(left[identifier]).astype('Int64')
    right = stays.copy()
    right['_join'] = numeric(right[canonical]).astype('Int64')
    right = right.rename(columns = {name: '_cohort_' + name for name in stays.columns})
    result = left.merge(right, on = '_join', how = 'inner', validate = 'many_to_one')
    if schema['time_kind'] == 'offset':
        result['_minute'] = numeric(result[schema['time']])
        result['_available'] = (
            numeric(result[schema['available']]) if schema.get('available') else np.nan
        )
    else:
        chart = pd.to_datetime(
            result[schema['time']].replace('', pd.NA), errors = 'raise'
        )
        result['_minute'] = (chart - result._cohort_intime).dt.total_seconds() / 60
        result['_available'] = (
            (
                pd.to_datetime(
                    result[schema['available']].replace('', pd.NA), errors = 'raise'
                )
                - result._cohort_intime
            ).dt.total_seconds()
            / 60
            if schema.get('available')
            else np.nan
        )
    return result.loc[result._minute.between(0, 2880)].copy()


def convert_chunk(frame, table, schema, stays, mapping, row_start):
    frame = frame.copy()
    frame['_row'] = [
        f'{table}:{row_start + position}' for position in range(len(frame))
    ]
    values = joined_events(frame, schema, stays)
    
    if schema.get('wide'):
        variables = WIDE_VITALS[table]
        
        values = values.melt(
            id_vars = [name for name in values if name not in variables],
            value_vars = list(variables),
            var_name = '_key',
            value_name = '_raw'
        )
        
        values['_variable'] = values._key.map({key: item[0] for key, item in variables.items()})
        values['_unit'] = values._key.map({key: item[1] for key, item in variables.items()})
        values['_value'] = numeric(values._raw)
    
    else:
        values['_key'] = event_keys(values, schema)
        values['_unit'] = values[schema['unit']] if schema.get('unit') else ''
        selected_mapping = mapping.loc[mapping.table.eq(table)].rename(
            columns = {'key': '_key', 'unit': '_unit'}
        )
        
        values = values.merge(
            selected_mapping,
            on = ['_key', '_unit'],
            how = 'left',
            validate = 'many_to_one',
            indicator = True
        )
        values = values.loc[
            values._merge.eq('both') & values.canonical_variable.ne('')
        ].copy()
        
        values['_variable'] = values.canonical_variable
        values['_raw'] = values[schema['value']].astype(str)
        values['_value'] = numeric(values._raw) * values.factor.astype(
            float
        ) + values.offset.astype(float)
    
    values['_value'] = [
        clean_value(variable, value)
        for variable, value in zip(values._variable, values._value, strict = True)
    ]
    
    component = values._variable.isin(COMPONENT_IDS)
    bounds = values._variable.map({'GCSEye': 4, 'GCSVerbal': 5, 'GCSMotor': 6})
    invalid_component = component & (
        ~values._value.between(1, bounds) | values._value.mod(1).ne(0)
    )
    values.loc[invalid_component, '_value'] = np.nan
    values = values.loc[values._value.notna()].copy()
    component = values._variable.isin(COMPONENT_IDS)
    
    rows = pd.DataFrame(
        {
            'domain': values._cohort_domain,
            'subject_id': values._cohort_subject_id,
            'hadm_id': values._cohort_hadm_id,
            'stay_id': values._cohort_stay_id,
            'minute_from_icu_admission': values._minute,
            'canonical_variable': values._variable,
            'value': values._value,
            'source_table': table,
            'source_itemid': values._key,
            'source_unit': values._unit,
            'source_row_id': values._row + ':' + values._variable,
            'available_minute': values._available,
            'bucket': values._cohort_bucket
        }
    )
    
    components = (
        rows.loc[component]
        .rename(
            columns = {
                'stay_id': 'RecordID',
                'subject_id': 'SUBJECT_ID',
                'hadm_id': 'HADM_ID'
            }
        )
        .copy()
    )
    
    components['ICUSTAY_ID'] = components.RecordID
    components['ITEMID'] = components.canonical_variable.map(COMPONENT_IDS)
    components['CHARTTIME'] = values.loc[component, '_cohort_intime'] + pd.to_timedelta(
        components.minute_from_icu_admission, unit = 'm'
    )
    components['raw_value'] = components.value.astype(str)
    
    return (
        rows.loc[~component, EVENT_COLUMNS + ['bucket']],
        components[COMPONENT_COLUMNS + ['bucket']]
    )


def extract_events(source, schema, stays, mapping, directory):
    directory.mkdir(parents = True, exist_ok = True)
    
    writers = {}
    try:
        for table, table_schema in schema.items():
            row_start = 0
            
            for chunk in read_chunks(source, table):
                observations, components = convert_chunk(
                    chunk, table, table_schema, stays, mapping, row_start
                )
                
                for kind, frame in [('observations', observations), ('components', components)]:
                    for bucket, part in frame.groupby('bucket'):
                        key = (kind, int(bucket))
                        part = part.drop(columns = 'bucket')
                        
                        for column in (
                            'minute_from_icu_admission',
                            'available_minute',
                            'value'
                        ):
                            part[column] = pd.to_numeric(part[column], errors = 'coerce').astype(float)
                        
                        part['source_itemid'] = part.get(
                            'source_itemid', pd.Series(index = part.index, dtype = str)
                        ).astype(str)
                        part['source_unit'] = part.get(
                            'source_unit', pd.Series(index = part.index, dtype = str)
                        ).astype(str)
                        
                        arrow = pa.Table.from_pandas(part, preserve_index = False)
                        
                        if key not in writers:
                            writers[key] = pq.ParquetWriter(
                                directory / f'{kind}_{int(bucket):03d}.parquet',
                                arrow.schema,
                            )
                        
                        writers[key].write_table(arrow.cast(writers[key].schema))
                
                row_start += len(chunk)
            
            print(table)
    
    finally:
        for writer in writers.values():
            writer.close()


def load_bucket(directory, stays, bucket):
    cohort = stays.loc[stays.bucket.eq(bucket)].copy()
    events = pd.read_parquet(directory / f'observations_{bucket:03d}.parquet')
    
    component_path = directory / f'components_{bucket:03d}.parquet'
    components = (
        pd.read_parquet(component_path)
        if component_path.exists()
        else pd.DataFrame(columns = COMPONENT_COLUMNS)
    )
    
    height = cohort.loc[cohort.height.between(100, 250)]
    height_events = pd.DataFrame(
        {
            'stay_id': height.stay_id,
            'canonical_variable': 'Height',
            'minute_from_icu_admission': 0,
            'value': height.height,
            'source_table': 'patient',
            'available_minute': np.nan
        }
    )
    
    events = pd.concat([events, height_events], ignore_index = True)
    
    return {
        'name': 'eicu',
        'stays': cohort,
        'observations': append_gcs(events, components),
        'components': components
    }


def append_gcs(events, components):
    gcs = derive_gcs(components).rename(
        columns = {
            'RecordID': 'stay_id',
            'SUBJECT_ID': 'subject_id',
            'HADM_ID': 'hadm_id',
            'ITEMID': 'source_itemid',
            'VALUEUOM': 'source_unit'
        }
    )
    
    return pd.concat(
        [
            events.loc[events.canonical_variable.ne('GCS')],
            gcs.reindex(columns = events.columns)
        ],
        ignore_index = True
    )
