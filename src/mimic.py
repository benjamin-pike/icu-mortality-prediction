import sqlite3
from itertools import groupby

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .readers import table_source
from .signals import (
    GCS_COMPONENT_ITEMIDS,
    GCS_TOTAL_ITEMIDS,
    MECHVENT_ITEMIDS,
    availability_minutes,
    derive_chart_events,
    normalise,
    valid_fio2
)

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
EVENT_COLUMNS = [
    'RecordID',
    'SUBJECT_ID',
    'HADM_ID',
    'ICUSTAY_ID',
    'canonical_variable',
    'source_table',
    'ITEMID',
    'source_row_id',
    'CHARTTIME',
    'available_minute',
    'minute_from_icu_admission',
    'value',
    'VALUEUOM',
    'raw_value'
]
SPECIAL_ITEMS = GCS_COMPONENT_ITEMIDS | GCS_TOTAL_ITEMIDS | MECHVENT_ITEMIDS


def extract_mimic(source, cohort, mapping, directory):
    directory.mkdir(parents = True, exist_ok = True)
    
    writers = {}
    store_path = directory / 'chart_signals.sqlite'
    store_path.unlink(missing_ok = True)
    connection = sqlite3.connect(store_path)
    chart_stays = set()
    try:
        for table in ('CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'):
            selected = mapping.loc[mapping.inclusion_status.eq('included') & mapping.source_table.eq(table)]
            lookup = selected.set_index('ITEMID')[['canonical_variable', 'item_label']]
            
            with table_source(source, table) as stream:
                for chunk in pd.read_csv(stream, chunksize = 100_000, low_memory = False):
                    chunk['source_row_id'] = chunk.get('ROW_ID', chunk.index.to_series())
                    chunk = chunk.rename(columns = str.upper).rename(
                        columns = {
                            'STAY_ID': 'ICUSTAY_ID',
                            'SOURCE_ROW_ID': 'source_row_id'
                        }
                    )
                    
                    if table == 'CHARTEVENTS':
                        chart_stays.update(pd.to_numeric(chunk.ICUSTAY_ID, errors = 'coerce').dropna().astype(int))
                    
                    chunk = chunk.loc[chunk.ITEMID.isin(lookup.index)].copy()
                    if 'ERROR' in chunk:
                        chunk = chunk.loc[~pd.to_numeric(chunk.ERROR, errors = 'coerce').eq(1)]
                    chunk['CHARTTIME'] = pd.to_datetime(chunk.CHARTTIME, errors = 'coerce')
                    
                    keys = ['SUBJECT_ID', 'HADM_ID']
                    if 'ICUSTAY_ID' in chunk and chunk.ICUSTAY_ID.notna().any():
                        keys.append('ICUSTAY_ID')
                    
                    chunk = chunk.merge(
                        cohort[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME']],
                        on = keys,
                        how = 'inner',
                    )
                    chunk['minute_from_icu_admission'] = (
                        ((chunk.CHARTTIME - chunk.INTIME).dt.total_seconds() / 60)
                        .round()
                        .astype('Int64')
                    )
                    chunk = chunk.loc[chunk.minute_from_icu_admission.between(0, 2880)].copy()
                    
                    if chunk.empty:
                        continue
                    
                    chunk['available_minute'] = availability_minutes(chunk)
                    chunk = chunk.join(lookup, on = 'ITEMID')
                    chunk['VALUE'] = chunk.get('VALUE', pd.Series(pd.NA, index = chunk.index))
                    chunk['VALUENUM'] = chunk.get('VALUENUM', pd.to_numeric(chunk.VALUE, errors = 'coerce'))
                    chunk['VALUEUOM'] = chunk.get('VALUEUOM', pd.Series(pd.NA, index = chunk.index))
                    chunk = chunk.loc[chunk.canonical_variable.ne('FiO2') | valid_fio2(chunk.VALUENUM)].copy()
                    chunk['value'] = normalise(
                        chunk.canonical_variable,
                        chunk.item_label,
                        chunk.VALUENUM,
                        chunk.VALUE,
                        chunk.ITEMID
                    )
                    chunk['raw_value'] = chunk.VALUE
                    chunk['RecordID'] = chunk.ICUSTAY_ID.astype(int)
                    chunk['source_table'] = table
                    
                    events = chunk[EVENT_COLUMNS]
                    
                    if table == 'CHARTEVENTS':
                        signals = events.loc[events.ITEMID.isin(SPECIAL_ITEMS)].copy()
                        
                        if len(signals):
                            signals['CHARTTIME'] = signals.CHARTTIME.astype(str)
                            signals.to_sql('signals', connection, if_exists = 'append', index = False)
                        events = events.loc[~events.ITEMID.isin(SPECIAL_ITEMS)]
                    
                    append_events(writers, directory / 'observations.parquet', events)

        has_signals = connection.execute("SELECT name FROM sqlite_master WHERE name = 'signals'").fetchone()
        
        if has_signals:
            connection.execute('CREATE INDEX patient_order ON signals(RecordID, CHARTTIME, source_row_id)')
            rows = connection.execute('SELECT * FROM signals ORDER BY RecordID, CHARTTIME, source_row_id')

            for _, patient_rows in groupby(rows, key = lambda row: row[0]):
                signals = pd.DataFrame(list(patient_rows), columns = EVENT_COLUMNS)
                signals['CHARTTIME'] = pd.to_datetime(signals.CHARTTIME)
                components = signals.loc[signals.ITEMID.isin(GCS_COMPONENT_ITEMIDS), COMPONENT_COLUMNS]
                
                append_events(writers, directory / 'gcs_components.parquet', components)
                append_events(writers, directory / 'observations.parquet', derive_chart_events(signals))
        
        else:
            pd.DataFrame(columns = COMPONENT_COLUMNS).to_parquet(directory / 'gcs_components.parquet')
        
        return chart_stays
    
    finally:
        for writer in writers.values():
            writer.close()
        connection.close()
        store_path.unlink(missing_ok = True)


def append_events(writers, path, frame):
    if frame.empty:
        return
    
    frame = frame.copy()
    
    if path.name == 'observations.parquet':
        frame = frame.drop(columns = 'raw_value', errors = 'ignore')
    
    for column in (
        'ITEMID',
        'source_row_id',
        'minute_from_icu_admission',
        'RecordID',
        'SUBJECT_ID',
        'HADM_ID',
        'ICUSTAY_ID'
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors = 'coerce').astype('Int64')

    for column in ('canonical_variable', 'source_table', 'VALUEUOM', 'raw_value'):
        if column in frame:
            frame[column] = frame[column].astype('string')
    frame['value'] = pd.to_numeric(frame.value, errors = 'coerce').astype(float)
    frame['available_minute'] = frame.available_minute.astype(float)
    
    arrow = pa.Table.from_pandas(frame, preserve_index = False)
    if path not in writers:
        writers[path] = pq.ParquetWriter(path, arrow.schema)
    
    writers[path].write_table(arrow.cast(writers[path].schema))


def mimiciv_heights(source, cohort):
    with table_source(source, 'OMR') as stream:
        for chunk in pd.read_csv(
            stream,
            chunksize = 100_000,
            usecols = ['subject_id', 'chartdate', 'result_name', 'result_value'],
        ):
            chunk['source_row_id'] = chunk.index
            
            heights = chunk.loc[chunk.result_name.eq('Height (Inches)')].merge(
                cohort[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME']],
                left_on = 'subject_id',
                right_on = 'SUBJECT_ID',
                how = 'inner'
            )
            heights['CHARTTIME'] = pd.to_datetime(heights.chartdate)
            heights = heights.loc[
                heights.CHARTTIME.between(
                    heights.INTIME.dt.normalize(),
                    (heights.INTIME + pd.Timedelta(hours = 48)).dt.normalize(),
                )
            ]
            
            yield pd.DataFrame(
                {
                    'RecordID': heights.ICUSTAY_ID,
                    'SUBJECT_ID': heights.SUBJECT_ID,
                    'HADM_ID': heights.HADM_ID,
                    'ICUSTAY_ID': heights.ICUSTAY_ID,
                    'canonical_variable': 'Height',
                    'source_table': 'OMR',
                    'ITEMID': pd.NA,
                    'source_row_id': heights.source_row_id,
                    'CHARTTIME': heights.CHARTTIME,
                    'available_minute': np.nan,
                    'minute_from_icu_admission': (
                        (heights.CHARTTIME - heights.INTIME).dt.total_seconds() / 60
                    ).clip(lower = 0).round(),
                    'value': pd.to_numeric(heights.result_value, errors = 'coerce') * 2.54,
                    'VALUEUOM': 'inches'
                }
            )

def calculate_age(dob, intime):
    age = intime.dt.year - dob.dt.year
    birthday_pending = (intime.dt.month < dob.dt.month) | (
        (intime.dt.month == dob.dt.month) & (intime.dt.day < dob.dt.day)
    )
    age = age - birthday_pending.astype(int)
    
    return age.where(age <= 89, 90.0).astype(float)