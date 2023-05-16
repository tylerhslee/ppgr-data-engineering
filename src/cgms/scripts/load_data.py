import numpy as np
import pandas as pd
from datetime import datetime

CGMS_COLS = [
    'id',
    'dt',
    'glucose'
]

LIFE_COLS = [
    'id',
    'dt',
    'lifestyle',
    'smk',
    'stress'
]

DEMO_COLS = [
    'id',
    'dt',
    'meal',
    'sex',
    'age',
]

MEAL_COLS = [
    'f_cereals_fiber',
    'f_potatos_fiber',
    'f_sugars_fiber',
    'f_beans_fiber',
    'f_nuts_fiber',
    'f_veges_fiber',
    'f_mushs_fiber',
    'f_fruits_fiber',
    'f_meats_fiber',
    'f_eggs_fiber',
    'f_fishs_fiber',
    'f_seawds_fiber',
    'f_milks_fiber',
    'f_oils_fiber',
    'f_drinks_fiber',
    'f_seasns_fiber',
    'f_etcs_fiber',

    'f_cereals_g',
    'f_potatos_g',
    'f_sugars_g',
    'f_beans_g',
    'f_nuts_g',
    'f_veges_g',
    'f_mushs_g',
    'f_fruits_g',
    'f_meats_g',
    'f_eggs_g',
    'f_fishs_g',
    'f_seawds_g',
    'f_milks_g',
    'f_oils_g',
    'f_drinks_g',
    'f_seasns_g',
    'f_etcs_g',

    'n_g'
]

GP_COLS = [
    'f_cereals_fiber',
    'f_potatos_fiber',
    'f_sugars_fiber',
    'f_beans_fiber',
    'f_nuts_fiber',
    'f_veges_fiber',
    'f_mushs_fiber',
    'f_fruits_fiber',
    'f_meats_fiber',
    'f_eggs_fiber',
    'f_fishs_fiber',
    'f_seawds_fiber',
    'f_milks_fiber',
    'f_oils_fiber',
    'f_drinks_fiber',
    'f_seasns_fiber',
    'f_etcs_fiber',

    'f_cereals_gp',
    'f_potatos_gp',
    'f_sugars_gp',
    'f_beans_gp',
    'f_nuts_gp',
    'f_veges_gp',
    'f_mushs_gp',
    'f_fruits_gp',
    'f_meats_gp',
    'f_eggs_gp',
    'f_fishs_gp',
    'f_seawds_gp',
    'f_milks_gp',
    'f_oils_gp',
    'f_drinks_gp',
    'f_seasns_gp',
    'f_etcs_gp',
]

PEOP_COLS = [
    'id',
    'wc',
    'bmi_s',
    'hba1c',
    'ogtt'
]


def midpoint(x, y):
    return x + abs(x - y) / 2.0


cgms_table = pd.read_excel("data/cgms timeseries data.xlsx")
life_table = pd.read_excel("data/lifestyle_timeseries_data.xlsx")
meal_table = pd.read_csv("data/permeal.csv")
peop_table = pd.read_csv("data/perid.csv")

# ================================ CGMS Data Cleaning Protocol ====================================
# 1. Remove any rows with missing data.
cgms_table = cgms_table.dropna(how = "any")

# 2. Create datetime based on `date` and `time`.
cgms_table["dt"] = cgms_table.apply(
    lambda row: datetime.combine(row['date'], row['time']),
    axis=1
)

# 3. Filter for valid records.
cgms_table = cgms_table.loc[
    cgms_table["record"] == 0,
    CGMS_COLS
]

# 4. Output contains CGMS reading for each person every 15 minutes.
cgms_table = cgms_table.sort_values(by=["id", "dt"])
cgms_table = cgms_table[CGMS_COLS]


# ============================== Lifestyle Data Cleaning Protocol =================================
# 1. Remove any rows with missing data in specific columns.
life_table = life_table.dropna(
    subset = ['date', 'st_h', 'st_m', 'lifestyle'],
    how = "any"
)

# 2. Ensure all time data are within 24 hours.
life_table['st_h'] = np.where(life_table['st_h'] >= 24, 0, life_table['st_h'])
life_table['end_h'] = np.where(life_table['end_h'] >= 24, 0, life_table['end_h'])

# 3. For activities with a duration, record the end datetime.
life_st = pd.to_datetime({
    'year': pd.to_datetime(life_table['date']).dt.year,
    'month': pd.to_datetime(life_table['date']).dt.month,
    'day': pd.to_datetime(life_table['date']).dt.day,
    'hour': life_table['st_h'].astype(np.int64),
    'minute': life_table['st_m'].astype(np.int64)
})

try:
    life_et = pd.to_datetime({
        'year': pd.to_datetime(life_table['date']).dt.year,
        'month': pd.to_datetime(life_table['date']).dt.month,
        'day': pd.to_datetime(life_table['date']).dt.day,
        'hour': life_table['end_h'].astype(np.int64),
        'minute': life_table['end_m'].astype(np.int64)
    })
except ValueError:
    life_table['dt'] = life_st
else:
    #life_table['dt'] = midpoint(life_et, life_st)
    life_table['dt'] = life_et

# 4. Output contains lifestyle records for each person.
life_table = life_table.sort_values(by = ['id', 'dt'])
life_table = life_table[LIFE_COLS]


# ============================== Meal Data Cleaning Protocol ======================================
# 1. Remove any rows with missing data in specific columns.
meal_table = meal_table.dropna(
    subset = ['date', 'st_h', 'st_m', 'meal'],
    how = "any"
)

# 2. Ensure all time data are within 24 hours.
meal_table['st_h'] = np.where(meal_table['st_h'] >= 24, 0, meal_table['st_h'])
meal_table['end_h'] = np.where(meal_table['end_h'] >= 24, 0, meal_table['end_h'])

# 3. For activities with a duration, record the midpoint datetime.
meal_st = pd.to_datetime({
    'year': pd.to_datetime(meal_table['date']).dt.year,
    'month': pd.to_datetime(meal_table['date']).dt.month,
    'day': pd.to_datetime(meal_table['date']).dt.day,
    'hour': meal_table['st_h'].astype(np.int64),
    'minute': meal_table['st_m'].astype(np.int64)
})

try:
    meal_et = pd.to_datetime({
        'year': pd.to_datetime(meal_table['date']).dt.year,
        'month': pd.to_datetime(meal_table['date']).dt.month,
        'day': pd.to_datetime(meal_table['date']).dt.day,
        'hour': meal_table['end_h'].astype(np.int64),
        'minute': meal_table['end_m'].astype(np.int64)
    })
except ValueError:
    meal_table['dt'] = meal_st
else:
    meal_table['dt'] = midpoint(meal_et, meal_st)

# 4. Output contains meal records for each person.
meal_table = meal_table.sort_values(by = ['id', 'dt'])
meal_table = meal_table[DEMO_COLS + MEAL_COLS]

# Add GP columns
meal_table['f_cereals_gp'] = meal_table['f_cereals_g'] / meal_table['n_g']
meal_table['f_potatos_gp'] = meal_table['f_potatos_g'] / meal_table['n_g']
meal_table['f_sugars_gp'] = meal_table['f_potatos_g'] / meal_table['n_g']
meal_table['f_beans_gp'] = meal_table['f_beans_g'] / meal_table['n_g']
meal_table['f_nuts_gp'] = meal_table['f_nuts_g'] / meal_table['n_g']
meal_table['f_veges_gp'] = meal_table['f_veges_g'] / meal_table['n_g']
meal_table['f_mushs_gp'] = meal_table['f_mushs_g'] / meal_table['n_g']
meal_table['f_fruits_gp'] = meal_table['f_fruits_g'] / meal_table['n_g']
meal_table['f_meats_gp'] = meal_table['f_meats_g'] / meal_table['n_g']
meal_table['f_eggs_gp'] = meal_table['f_eggs_g'] / meal_table['n_g']
meal_table['f_fishs_gp'] = meal_table['f_fishs_g'] / meal_table['n_g']
meal_table['f_seawds_gp'] = meal_table['f_seawds_g'] / meal_table['n_g']
meal_table['f_milks_gp'] = meal_table['f_milks_g'] / meal_table['n_g']
meal_table['f_oils_gp'] = meal_table['f_oils_g'] / meal_table['n_g']
meal_table['f_drinks_gp'] = meal_table['f_drinks_g'] / meal_table['n_g']
meal_table['f_seasns_gp'] = meal_table['f_seasns_g'] / meal_table['n_g']
meal_table['f_etcs_gp'] = meal_table['f_etcs_g'] / meal_table['n_g']

meal_table.drop([
    'f_cereals_g',
    'f_potatos_g',
    'f_sugars_g',
    'f_beans_g',
    'f_nuts_g',
    'f_veges_g',
    'f_mushs_g',
    'f_fruits_g',
    'f_meats_g',
    'f_eggs_g',
    'f_fishs_g',
    'f_seawds_g',
    'f_milks_g',
    'f_oils_g',
    'f_drinks_g',
    'f_seasns_g',
    'f_etcs_g',
    'n_g' 
], axis=1)

# ============================== Person Data Cleaning Protocol ====================================
# 1. Remove empty rows.
peop_table = peop_table.dropna(how = "all")
peop_table = peop_table.sort_values(by = ['id'])
peop_table = peop_table[PEOP_COLS]


# ======================================= Data Merge ==============================================
# Merge by taking in time history data.
def merge_data(row):
    '''Merge & append datasets based on each CGMS record timings and latest meal & activity
    records.
    '''
    cgms_ts = row['dt']
    meal_range = meal_table.loc[
        (meal_table['id'] == row['id']) &
        (meal_table['dt'] <= cgms_ts),
        :
    ]
    act_range = life_table.loc[
        (life_table['id'] == row['id']) &
        (life_table['dt'] <= cgms_ts),
        :
    ]

    # Append meal data
    if len(meal_range) > 0:
        latest_meal = meal_range.loc[meal_range['dt'].idxmax(), ['dt'] + GP_COLS + DEMO_COLS]
        row['fasting'] = pd.to_timedelta(cgms_ts - latest_meal[0])
        row = pd.concat([row, latest_meal.drop(['dt'])], axis=1)
        row = row.iloc[:, 0].fillna(row.iloc[:, 1])
 
    # Append lifestyle data
    if len(act_range) > 0:
        latest_action = act_range.loc[act_range['dt'].idxmax(), :]
        row['resting'] = cgms_ts - latest_action['dt']
        row = pd.concat([row, latest_action.drop(columns=['id', 'dt'])], axis=1)
        row = row.iloc[:, 0].fillna(row.iloc[:, 1])
   
    # Append demographic data
    person = peop_table.loc[
        (peop_table['id'] == row['id']),
        :
    ]
    row = pd.concat([pd.DataFrame(row).T, person.drop(columns=['id'])])

    # Merge by filling NA from values in other rows
    if len(row) > 1:
        row = pd.Series(row.iloc[0, :].fillna(row.iloc[1, :]))
    else:
        row = pd.Series(row.iloc[0, :])

    print(f'Merged {row["id"]} {row["dt"]}')
    return row

    # Append activity data

dataset = cgms_table.apply(merge_data, axis=1)
#dataset = dataset.reset_index()
print(dataset.head())
dataset.to_csv('ml_input.csv')


# ======================================= Data Validation =========================================
# 1. Per-ID validation.
ids = set(dataset['id'])

for i in ids:
    df = dataset.loc[dataset['id'] == i, :]
    assert sum(pd.isnull(dataset['glucose']).astype(np.int64)) == 0

    print('=============================')
    print('Person ID: ', i)
    print('-----------------------------')
    print('Glucose records (cnt): ', len(df))
    print('Lifestyle records (cnt): ', len(df.loc[~pd.isnull(df['lifestyle']), :]))
    print('Meal records (cnt): ', len(df.loc[~pd.isnull(df['meal']), :]))
    print('\n')
