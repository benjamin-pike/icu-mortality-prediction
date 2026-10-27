import numpy as np
import pandas as pd

from .io import RESOURCES

WINDOW_MINUTES = 2880
ADULT_MIN_AGE = 18

STATIC_RULES = {
    'Age': ('years', ADULT_MIN_AGE, 130),
    'Gender': ('binary', 0, 1),
    'Height': ('cm', 50, 260),
    'ICUType': ('category', 1, 4)
}
RULES = {
    'Na': (80, 220),
    'SaO2': (0, 100),
    'Lactate': (0, 40),
    'NIMAP': (0, 300),
    'TroponinI': (0, 500),
    'Temp': (20, 45),
    'DiasABP': (0, 250),
    'Mg': (0, 20),
    'HR': (0, 300),
    'FiO2': (0.21, 1),
    'Urine': (0, 10000),
    'MAP': (0, 300),
    'MechVent': (0, 1),
    'Glucose': (0, 2000),
    'Creatinine': (0, 30),
    'Platelets': (0, 2000),
    'ALT': (0, 20000),
    'NIDiasABP': (0, 250),
    'K': (0, 20),
    'NISysABP': (0, 350),
    'pH': (6.5, 8),
    'PaCO2': (0, 250),
    'WBC': (0, 500),
    'Cholesterol': (0, 2000),
    'Albumin': (0, 10),
    'RespRate': (0, 100),
    'ALP': (0, 10000),
    'TroponinT': (0, 100),
    'AST': (0, 20000),
    'BUN': (0, 500),
    'HCO3': (0, 100),
    'Weight': (1, 500),
    'GCS': (3, 15),
    'Bilirubin': (0, 100),
    'SysABP': (0, 350),
    'PaO2': (0, 800),
    'HCT': (0, 100)
}
DYNAMIC_VARS = list(RULES)


def clean_value(variable, raw):
    value = pd.to_numeric(raw, errors = 'coerce')
    
    if pd.isna(value):
        return np.nan
    
    if variable == 'FiO2' and 21 <= value <= 100:
        value /= 100
    
    if variable in RULES:
        lower, upper = RULES[variable]
        if not lower <= value <= upper:
            return np.nan
    
    return float(value)


def empty_dynamic_features(variable):
    values = {
        f'{variable}_observed': 0,
        f'{variable}_count': 0,
        f'{variable}_valid_count': 0,
        f'{variable}_count_d1': 0,
        f'{variable}_count_d2': 0,
        f'{variable}_count_change': 0,
        f'{variable}_first_measurement_hour': np.nan,
        f'{variable}_last_measurement_hour': np.nan,
        f'{variable}_hours_since_last_measurement': np.nan,
        f'{variable}_median_intermeasurement_interval': np.nan
    }
    
    if variable == 'MechVent':
        values |= {
            'MechVent_ever': 0,
            'MechVent_first': np.nan,
            'MechVent_last': np.nan,
            'MechVent_fraction_positive': np.nan
        }
    else:
        values |= {
            f'{variable}_{statistic}': np.nan
            for statistic in ('mean_d1', 'mean_d2', 'd2_minus_d1')
        }
    
    if variable == 'Urine':
        values |= {
            'Urine_total': np.nan,
            'Urine_total_d1': np.nan,
            'Urine_total_d2': np.nan
        }
    
    return values


EMPTY_DYNAMIC_FEATURES = {
    variable: empty_dynamic_features(variable) for variable in DYNAMIC_VARS
}


def summaries(values, times):
    if not len(values):
        return {}

    result = {
        'mean': values.mean(),
        'median': np.median(values),
        'min': values.min(),
        'max': values.max(),
        'first': values[0],
        'last': values[-1]
    }

    if len(values) >= 2:
        result |= {
            'trend': values[-1] - values[0],
            'sd': values.std(ddof = 1),
            'range': values.max() - values.min(),
            'iqr': np.quantile(values, 0.75) - np.quantile(values, 0.25),
            'mad': np.median(np.abs(values - np.median(values)))
        }

    if len(np.unique(times)) >= 2:
        result['slope'] = np.polyfit(times / 60, values, 1)[0]

    return result


def nearest_values(query_times, observations, minimum):
    by_time = {}
    
    for time, value in observations:
        if pd.notna(value) and value >= minimum:
            by_time[time] = min(value, by_time.get(time, value))
    
    if not by_time:
        return np.full(len(query_times), np.nan)
    
    times = np.array(sorted(by_time))
    values = np.array([by_time[time] for time in times])
    query = np.asarray(query_times)
    right = np.searchsorted(times, query).clip(0, len(times) - 1)
    left = (np.searchsorted(times, query) - 1).clip(0, len(times) - 1)
    left_distance, right_distance = (
        abs(query - times[left]),
        abs(query - times[right])
    )
    choose_left = (left_distance < right_distance) | (
        left_distance == right_distance
    ) & (values[left] <= values[right])
    nearest = np.where(choose_left, left, right)
    
    return np.where(abs(query - times[nearest]) <= 60, values[nearest], np.nan)


def paired_clinical_features(series):
    cleaned = {
        variable: [
            (time, clean_value(variable, value))
            for time, value in series.get(variable, [])
        ]
        for variable in ('PaO2', 'FiO2', 'HR', 'SysABP', 'NISysABP')
    }
    
    oxygen = [(time, value) for time, value in cleaned['PaO2'] if pd.notna(value)]
    fio2 = nearest_values([time for time, _ in oxygen], cleaned['FiO2'], minimum = 0.01)
    pf = np.array([value for _, value in oxygen]) / fio2
    pf = pf[np.isfinite(pf)]
    heart_rate = [(time, value) for time, value in cleaned['HR'] if pd.notna(value)]
    times = [time for time, _ in heart_rate]
    invasive = nearest_values(times, cleaned['SysABP'], minimum = 40)
    noninvasive = nearest_values(times, cleaned['NISysABP'], minimum = 40)
    pressure = np.where(np.isfinite(invasive), invasive, noninvasive)
    shock = np.array([value for _, value in heart_rate]) / pressure
    shock = shock[np.isfinite(shock)]
    
    return {
        'PF_min': np.min(pf) if len(pf) else np.nan,
        'PF_mean': np.mean(pf) if len(pf) else np.nan,
        'PF_last': pf[-1] if len(pf) else np.nan,
        'ShockIndex_mean': np.mean(shock) if len(shock) else np.nan,
        'ShockIndex_max': np.max(shock) if len(shock) else np.nan,
        'ShockIndex_last': shock[-1] if len(shock) else np.nan
    }


def build_predictors_from_records(records):
    split_minutes = WINDOW_MINUTES // 2
    window_hours = WINDOW_MINUTES / 60.0
    rows = []
    
    for record_id, static, series in records:
        row = {'RecordID': record_id}
        
        for variable, (_, lower, upper) in STATIC_RULES.items():
            value = pd.to_numeric(static.get(variable), errors = 'coerce')
            value = (
                np.nan
                if value == -1 or pd.isna(value) or (not lower <= value <= upper)
                else float(value)
            )
            
            row[variable], row[f'{variable}_missing'] = (value, int(pd.isna(value)))
        
        total, breadth = ([0, 0], [set(), set()])
        
        for variable in DYNAMIC_VARS:
            if not series.get(variable):
                row |= EMPTY_DYNAMIC_FEATURES[variable]
                continue
            
            raw = sorted(
                [
                    (minute, value)
                    for minute, value in series.get(variable, [])
                    if 0 <= minute <= WINDOW_MINUTES
                ]
            )
            
            cleaned = []
            
            for minute, value in raw:
                cleaned_value = clean_value(variable, value)
                if pd.notna(cleaned_value):
                    cleaned.append((minute, cleaned_value))
            
            raw_times = np.array([minute for minute, _ in raw])
            times = np.array([minute for minute, _ in cleaned])
            values = np.array([value for _, value in cleaned], dtype = float)
            
            d1, d2 = (values[times < split_minutes], values[times >= split_minutes])
            raw_d1, raw_d2 = (
                int((raw_times < split_minutes).sum()),
                int((raw_times >= split_minutes).sum())
            )
            
            row |= {
                f'{variable}_observed': int(bool(raw)),
                f'{variable}_count': len(raw),
                f'{variable}_valid_count': len(values),
                f'{variable}_count_d1': raw_d1,
                f'{variable}_count_d2': raw_d2,
                f'{variable}_count_change': raw_d2 - raw_d1,
                f'{variable}_first_measurement_hour': raw_times[0] / 60 if len(raw) else np.nan,
                f'{variable}_last_measurement_hour': raw_times[-1] / 60 if len(raw) else np.nan,
                f'{variable}_hours_since_last_measurement': window_hours - raw_times[-1] / 60 if len(raw) else np.nan
            }

            unique_times = np.unique(raw_times)
            
            row[f'{variable}_median_intermeasurement_interval'] = (
                np.median(np.diff(unique_times)) / 60 if len(unique_times) >= 2 else np.nan
            )
            
            if variable == 'MechVent':
                row |= {
                    'MechVent_ever': int(len(values) and values.max() > 0),
                    'MechVent_first': values[0] if len(values) else np.nan,
                    'MechVent_last': values[-1] if len(values) else np.nan,
                    'MechVent_fraction_positive': values.mean() if len(values) else np.nan
                }
            
            else:
                row |= {
                    f'{variable}_{statistic}': value
                    for statistic, value in summaries(values, times).items()
                }
                row |= {
                    f'{variable}_mean_d1': d1.mean() if len(d1) else np.nan,
                    f'{variable}_mean_d2': d2.mean() if len(d2) else np.nan,
                    f'{variable}_d2_minus_d1': d2.mean() - d1.mean() if len(d1) and len(d2) else np.nan
                }

            if variable == 'Urine':
                row |= {
                    'Urine_total': values.sum() if len(values) else np.nan,
                    'Urine_total_d1': d1.sum() if len(d1) else np.nan,
                    'Urine_total_d2': d2.sum() if len(d2) else np.nan
                }
            
            total[0] += raw_d1
            total[1] += raw_d2
            
            if raw_d1:
                breadth[0].add(variable)
            if raw_d2:
                breadth[1].add(variable)
        
        row |= {
            'total_measurements_d1': total[0],
            'total_measurements_d2': total[1],
            'total_measurements': sum(total),
            'total_measurement_change': total[1] - total[0],
            'distinct_variables_measured_d1': len(breadth[0]),
            'distinct_variables_measured_d2': len(breadth[1]),
            'distinct_variables_measured': len(breadth[0] | breadth[1]),
            'distinct_variables_change': len(breadth[1]) - len(breadth[0])
        }
        
        row['BUN_Cr_ratio'] = (
            row.get('BUN_mean', np.nan) / row.get('Creatinine_mean', np.nan)
            if pd.notna(row.get('Creatinine_mean')) and row['Creatinine_mean'] > 0 else np.nan
        )
        
        window_series = {
            variable: [(minute, value) for minute, value in values if 0 <= minute <= WINDOW_MINUTES]
            for variable, values in series.items()
        }
        
        row |= paired_clinical_features(window_series)
        rows.append(row)
    
    return pd.DataFrame(rows).set_index('RecordID')


def build_features(domain):
    observations = domain['observations']
    
    events = observations.loc[observations.canonical_variable.ne('Height')].sort_values(
        ['stay_id', 'canonical_variable', 'minute_from_icu_admission']
    )
    event_groups = events.groupby('stay_id', sort = False)
    
    heights = (
        observations.loc[observations.canonical_variable.eq('Height')]
        .sort_values(['stay_id', 'minute_from_icu_admission'])
        .drop_duplicates('stay_id')
        .set_index('stay_id')
        .value
    )

    def records():
        for stay in domain['stays'].sort_values('stay_id').itertuples(index = False):
            series = {}
            
            if stay.stay_id in event_groups.indices:
                for variable, group in event_groups.get_group(stay.stay_id).groupby(
                    'canonical_variable', sort = False
                ):
                    series[variable] = list(
                        zip(
                            group.minute_from_icu_admission.astype(int).tolist(),
                            group.value.astype(float).tolist(),
                            strict = True
                        )
                    )
            
            yield (
                int(stay.stay_id),
                {
                    'Age': stay.age,
                    'Gender': {'F': 0, 'M': 1}.get(stay.sex),
                    'Height': heights.get(stay.stay_id, float('nan')),
                    'ICUType': stay.icu_type_code
                },
                series
            )

    return (
        build_predictors_from_records(records())
        .reindex(
            columns = pd.read_csv(RESOURCES / 'feature_dictionary.csv').feature.tolist()
        )
        .rename_axis('stay_id')
    )
