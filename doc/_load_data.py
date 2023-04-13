import pandas as pd

from datetime import datetime


MEAL_COLS = [
    'age',
    'sex',
    'meal',
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
]


def midpoint(x, y):
    return x + abs(x - y) / 2.0


def cgms(path):
    data = pd.read_excel(path).dropna(how = "any")

    # Add datetime column for merge index
    data["dt"] = data.apply(
        lambda row: datetime.combine(row['date'], row['time']),
        axis=1
    )

    # Invalid glucose records are marked with "1".
    # Filter for only valid records.
    data = data.loc[
        data["record"] == 0,
        :
    ]

    return data


def life(path):
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


    data = pd.read_excel(path).dropna(subset=['date', 'st_h', 'st_m', 'lifestyle'], how="any")
    # Remove rows with missing date or time information / lifestyle information
    data['dt'] = data.apply(lifestyle_dt, axis=1)
    return data


def meal(path):
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


    data = pd.read_csv(path, header=0).dropna(how = "all")
    # Drop empty rows
    data['dt'] = data.apply(meal_dt, axis=1)
    return data[MEAL_COLS]


def clean_data(
        cgms_data = "data/cgms timeseries data.xlsx",
        life_data = "data/lifestyle_timeseries_data.xlsx",
        meal_data = "data/permeal.csv"
):
    clean_cgms = cgms(cgms_data)
    print('CGMS Done.')
    clean_life = life(life_data)
    print('Lifestyle Done.')
    clean_meal = meal(meal_data)
    print('Meal Done.')

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
        clean_cgms,
        clean_life,
        clean_meal,
    ], axis=0, sort=True)
    print('Concat Done.')

    df = df.sort_values(by=['id', 'dt'])
    print('Sort Done.')
    # meal.auc column is only needed for PPGR comparison
    df = df.drop("auc", axis=1)
    # May be duplicate timestamps from different datasets
    # Groupby operation to merge duplicates
    ids = set(df['id'])
    dfs = []
    for i in ids:
        df = df.loc[df['id'] == i, :].groupby('dt').agg(merge_timestamps)
        df = df.reset_index()
        dfs.append(df)
        print(f'Merged {i}')
    ret = pd.concat(dfs, axis=0, sort=False)
    print('Merge Done.')

    return ret
