'''Data snapshot: 2023/03/10 4:42 PM PST
'''
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


IDS_MISSING_MEAL_RECORDS = []

os.chdir(os.path.dirname(__file__))
if not os.path.exists("output"):
    os.makedirs("output")


def midpoint(x, y):
    return x + abs(x - y) / 2.0


# ========== Load CGMS Data ==========
DATA_XLSX_CGMS = pd.read_excel("data/cgms timeseries data.xlsx")
# Drop any rows with an empty value
DATA_XLSX_CGMS = DATA_XLSX_CGMS.dropna(how="any")
# Create datetime column by combining the date and time columns
DATA_XLSX_CGMS['dt'] = DATA_XLSX_CGMS.apply(
    lambda row: datetime.combine(row['date'], row['time']),
    axis=1
)
DATA_XLSX_CGMS = DATA_XLSX_CGMS.loc[
    DATA_XLSX_CGMS["record"] == 0,
    ['id', 'dt', 'glucose']
]

# ========== Load Lifestyle Data ==========
def lifestyle_dt(row):
    if row['st_h'] >= 24:
        row['st_h'] = 0
    elif row['end_h'] >= 24:
        row['end_h'] = 0 

    st = datetime(
        year = pd.to_datetime(row['date']).year,
        month = pd.to_datetime(row['date']).month,
        day = pd.to_datetime(row['date']).day,
        hour = int(row['st_h']),
        minute = int(row['st_m'])
    )

    try:
        et = datetime(
            year = pd.to_datetime(row['date']).year,
            month = pd.to_datetime(row['date']).month,
            day = pd.to_datetime(row['date']).day,
            hour = int(row['end_h']),
            minute = int(row['end_m'])
        )
    except ValueError:
        return st
    else:
        return midpoint(et, st)


DATA_XLSX_LIFE = pd.read_excel("data/lifestyle_timeseries_data.xlsx")
# Remove rows with missing date or time information / lifestyle information
DATA_XLSX_LIFE = DATA_XLSX_LIFE.dropna(subset=['date', 'st_h', 'st_m', 'lifestyle'], how="any")
DATA_XLSX_LIFE['dt'] = DATA_XLSX_LIFE.apply(lifestyle_dt, axis=1)
DATA_XLSX_LIFE = DATA_XLSX_LIFE[['id', 'dt', 'lifestyle', 'pa_type', 'stress']]

# ========== Load Meal Data ==========
def meal_dt(row):
    if row['st_h'] >= 24:
        row['st_h'] = 0
    elif row['end_h'] >= 24:
        row['end_h'] = 0

    return midpoint(
        datetime(
            year = pd.to_datetime(row['date']).year,
            month = pd.to_datetime(row['date']).month,
            day = pd.to_datetime(row['date']).day,
            hour = row['st_h'],
            minute = row['st_m']
        ),
        datetime(
            year = pd.to_datetime(row['date']).year,
            month = pd.to_datetime(row['date']).month,
            day = pd.to_datetime(row['date']).day,
            hour = row['end_h'],
            minute = row['end_m']
        )
    )


DATA_CSV_MEAL = pd.read_csv("data/permeal.csv", header=0)
# Drop empty rows
DATA_CSV_MEAL = DATA_CSV_MEAL.dropna(how='all')
DATA_CSV_MEAL['dt'] = DATA_CSV_MEAL.apply(meal_dt, axis=1)
DATA_CSV_MEAL['meal_data'] = list(zip(
    DATA_CSV_MEAL['d_rice_g'],
    DATA_CSV_MEAL['d_bread_g'],
    DATA_CSV_MEAL['d_pickled_g'],
    DATA_CSV_MEAL['d_fried_g'],
    DATA_CSV_MEAL['d_drink_g'],
    DATA_CSV_MEAL['d_nodl_g'],
))
DATA_CSV_MEAL = DATA_CSV_MEAL[["id", "dt", "auc", "meal_data"]]


# ========== Combine All Time Series Datasets ==========
def merge_timestamps(rows):
    # No conflict - no repeated value for the timestamp within the column
    candidates = tuple([x for x in set(rows) if not pd.isnull(x)])

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        return None
    else:
        return candidates

df = pd.concat([
    DATA_XLSX_CGMS,
    DATA_XLSX_LIFE,
    DATA_CSV_MEAL,
], axis=0, sort=True)
df = df.sort_values(by=['id', 'dt'])
# meal.auc column is only needed for PPGR comparison
df = df.drop("auc", axis=1)
# May be duplicate timestamps from different datasets
# Groupby operation to merge duplicates
df = df.groupby(['id', 'dt']).agg(merge_timestamps)
df = df.reset_index()
df.to_csv(f'output/big_dataset_{datetime.today().date()}.csv')


# ========== Measure PPGR ==========
def ppgr(r, data):
    baseline = data.loc[(data['dt'] >= r[0]) & (data['dt'] < r[1]), 'glucose'].mean()
    
    fdf = data.loc[(data['dt'] >= r[1]) & (data['dt'] <= r[2]), ['dt', 'glucose']].dropna()
    # only one glucose reading allowed for each timestamp
    fdf = fdf.drop_duplicates(subset='dt')

    try:
        # Trapezoidal area. np.trapz(y, x)
        area_under_curve = np.trapz(np.maximum(fdf['glucose'].values - baseline, 0), fdf['dt'].values)
        return area_under_curve / np.timedelta64(1, 'm')
    except TypeError as e:
        return None


for _id in set(df["id"]):
    if not os.path.exists(os.path.join("output", _id)):
        os.makedirs(os.path.join("output", _id))

    subset = df.loc[df["id"] == _id, :]
    glucose_records_cnt = len(subset.loc[~pd.isnull(subset['glucose']), :])
    lifestyle_records_cnt = len(subset.loc[~pd.isnull(subset['lifestyle']), :])
    meal_records_cnt = len(subset.loc[~pd.isnull(subset['meal_data']), :])
    assert glucose_records_cnt == len(DATA_XLSX_CGMS.loc[DATA_XLSX_CGMS["id"] == _id, :])
    #assert lifestyle_records_cnt == len(DATA_XLSX_LIFE.loc[DATA_XLSX_LIFE["id"] == _id, :])
    #assert meal_records_cnt == len(DATA_CSV_MEAL.loc[DATA_CSV_MEAL["id"] == _id, :])

    for ts in list(subset.loc[~pd.isnull(subset["meal_data"]), "dt"]):
        auc_range = (ts - timedelta(minutes=30), ts, ts + timedelta(hours=2))
        auc = ppgr(auc_range, subset)
        subset.loc[subset['dt'] == ts, "ppgr"] = auc
    PPGR_FILE_NAME = f"output/{_id}/ppgr_{_id}_{datetime.today().date()}.csv"
    subset.to_csv(PPGR_FILE_NAME)
    
    print(f'ID: {_id} Glucose Records', glucose_records_cnt)
    print(f'ID: {_id} Lifestyle Records', lifestyle_records_cnt)
    print(f'ID: {_id} Meal Records', meal_records_cnt)
    print(f'ID: {_id} All Records', len(subset.loc[
        ~pd.isnull(subset['glucose']) &
        ~pd.isnull(subset['lifestyle']) &
        ~pd.isnull(subset['meal_data'])
    ]))
    try:
        print(f'ID: {_id} PPGR Sample', subset[["dt", "ppgr"]].dropna(subset=["ppgr"], axis=0))
    except Exception as e:
        print(e, "Missing PPGR for ", _id)
        subset["ppgr"] = 0
        IDS_MISSING_MEAL_RECORDS.append(_id)
    print(f'Output file: {PPGR_FILE_NAME}')
    print('---')


# ========== Compare Original AUC ==========
    og_auc = DATA_CSV_MEAL.loc[DATA_CSV_MEAL["id"] == _id, ['dt', 'auc']]
    subset['og_auc'] = 0
    for tt in DATA_CSV_MEAL['dt']:
        if len(og_auc.loc[og_auc['dt'] == tt, 'auc'].values) > 0:
            subset.loc[subset['dt'] == tt, 'og_auc'] = og_auc.loc[og_auc['dt'] == tt, 'auc'].values[0]

    CHECK_FILE_NAME = f"output/{_id}/check_{_id}_{datetime.today().date()}.csv"
    subset[['dt', 'ppgr', 'og_auc']].to_csv(CHECK_FILE_NAME)

    print(f'Output file: {CHECK_FILE_NAME}')
    print('---')


print("IDs without meal records: ", IDS_MISSING_MEAL_RECORDS)
