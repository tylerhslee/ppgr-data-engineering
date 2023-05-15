#!/bin/bash
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import functools
import numpy as np
import pandas as pd
from pymongo import MongoClient

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

FOOD_GROUPS = [
    'f_cereals',
    'f_potatos',
    'f_sugars',
    'f_beans',
    'f_nuts',
    'f_veges',
    'f_mushs',
    'f_fruits',
    'f_meats',
    'f_eggs',
    'f_fishs',
    'f_seawds',
    'f_milks',
    'f_oils',
    'f_drinks',
    'f_seasns',
    'f_etcs',
]

MEAL_SUFFIXES = [
    'gp'
]

MEAL_COLS = [
    'id',
    'dt',
    'n_g',
    'n_cho',
    'n_en',
    'n_fiber',
    'meal',
    'age',
    'sex'
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


cgms_table = pd.read_excel("data/raw/cgms timeseries data.xlsx")
life_table = pd.read_excel("data/raw/lifestyle_timeseries_data.xlsx")
meal_table = pd.read_csv("data/raw/permeal.csv")
peop_table = pd.read_csv("data/raw/perid.csv")

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
    # life_table['dt'] = life_et
    life_table['dt'] = life_st

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
    # meal_table['dt'] = midpoint(meal_et, meal_st)
    meal_table['dt'] = meal_st

# 4. Output contains meal records for each person.
meal_table = meal_table.sort_values(by = ['id', 'dt'])

for f in FOOD_GROUPS:
    gp_col = f + "_gp"
    g_col = f + "_g"
    meal_table[gp_col] = meal_table[g_col] / meal_table["n_g"]
    meal_table.drop(g_col, axis=1)
meal_table.drop("n_g", axis=1)

target_cols = ["_".join([x, s]) for s in MEAL_SUFFIXES for x in FOOD_GROUPS]
meal_table = meal_table[target_cols + MEAL_COLS]


# ============================== Person Data Cleaning Protocol ====================================
# 1. Remove empty rows.
peop_table = peop_table.dropna(how = "all")
peop_table = peop_table.sort_values(by = ['id'])
peop_table = peop_table[PEOP_COLS]


# ======================================= Data Merge ==============================================
# Merge by taking in time history data.

# Merge on DT - keep all records.

# Fill NA values - "", or 0.

# Treat Booleans as strings and add "unknown".
dfs = [cgms_table, life_table, meal_table]
dataset = pd.concat(dfs, axis=0, ignore_index=True)
dataset = dataset.sort_values(by=["id", "dt"])


#dataset = dataset.loc[dataset["id"] == "1-23-2", :]


def calculate_ppgr(row):
    ppgr_range = (row["dt"] - timedelta(minutes=30), row["dt"] + timedelta(hours=2))
    baseline = dataset.loc[
        (dataset["dt"] >= ppgr_range[0]) &
        (dataset["dt"] < row["dt"]) &
        ~pd.isnull(dataset["glucose"]) &
        (dataset["id"] == row["id"]),
        "glucose"
    ]
    if len(baseline) == 0:
        return np.nan
    else:
        baseline = baseline.mean()

    ppgr = dataset.loc[
        (dataset["dt"] >= row["dt"]) &
        (dataset["dt"] < ppgr_range[1]) &
        ~pd.isnull(dataset["glucose"]) &
        (dataset["id"] == row["id"]),
        ["dt", "glucose"]
    ]
    if len(ppgr) == 0:
        return np.nan

    ppgr = ppgr.fillna(0)
    ppgr = ppgr.groupby(by="dt", axis=0).sum()
    ppgr = ppgr.reset_index()

    ppgr["delta"] = 0
    ppgr["value"] = 0
    ppgr["adj"] = ppgr["glucose"] - baseline

    rows = len(ppgr)
    if rows > 1:
        last_row = ppgr.iloc[0]
        for i in range(1, ppgr.shape[0]):
            current_row = ppgr.iloc[i]

            if (last_row["adj"] <= 0).all() & (current_row["adj"] <= 0).all():
                last_row["delta"] = 0
            elif (last_row["adj"] <= 0).all() & (current_row["adj"] > 0).all():
                last_row["delta"] = (current_row["dt"] - last_row["dt"]).total_seconds() * current_row["adj"] / (current_row["adj"] + abs(last_row["adj"]))
            elif (ppgr["adj"] > 0).all() & (ppgr["adj"] <= 0).all():
                last_row["delta"] = (current_row["dt"] - last_row["dt"]).total_seconds() * last_row["adj"] / (abs(current_row["adj"]) + last_row["adj"])
            elif (last_row["adj"] > 0).all() & (current_row["adj"] > 0).all():
                last_row["delta"] = (current_row["dt"] - last_row["dt"]).total_seconds()

            last_row["value"] = (max(0, last_row["adj"]) + max(0, current_row["adj"])) * last_row["delta"] / 2
            ppgr.iloc[i-1] = last_row
            last_row = current_row

    print(ppgr)

    '''
    ppgr["adj"] = ppgr["glucose"] - baseline
    ppgr.loc[ppgr["adj"] < 0, "adj"] = 0
    ppgr["delta"] = (ppgr["dt"] - ppgr.shift(1)["dt"]).dt.seconds // 60
    ppgr["value"] = (ppgr.shift(1)["adj"] + ppgr["adj"]) * ppgr["delta"] / 2
    '''
    ppgr.to_csv("data/std/ppgr_history.csv", mode="a")

    row["ppgr"] = ppgr["value"].sum() // 60
    row["peak"] = (row["dt"] - ppgr.loc[ppgr["glucose"].idxmax(), "dt"]).seconds // 60
    return row

out_dfs = []
for x in set(list(dataset["id"])):
    subset = dataset.loc[
        ~pd.isnull(dataset["meal"]) &
        (dataset["id"] == x),
        :
    ]

    subset = subset.apply(calculate_ppgr, axis=1)
    out_dfs.append(subset)

dataset = pd.concat(out_dfs, axis=0)
dataset = dataset.dropna(how="all")
dataset = dataset.loc[
    ~pd.isnull(dataset["ppgr"]),
    :
]
dataset["fasting"] = (dataset["dt"] - dataset["dt"].shift(1)).dt.seconds // 60
dataset = dataset.dropna(subset=["fasting"], how="any")

seconds_in_day = 24*60*60
dt = pd.to_timedelta(dataset["dt"] - datetime(1970, 1, 1)).dt.total_seconds()
dataset["sin_dt"] = np.sin(2*np.pi*dt/seconds_in_day)
dataset["cos_dt"] = np.cos(2*np.pi*dt/seconds_in_day)
# dataset = dataset.drop("dt", axis=1)

dataset = pd.merge(dataset, peop_table, on="id")
print(dataset.head())


dataset.to_csv('data/std/ml_input.csv')

# Write output data into Mongo as flat JSON per row.
docs = dataset.to_dict(orient="records")

client = MongoClient(
    "mongodb+srv://cluster0.fmexqnq.mongodb.net/?retryWrites=true&w=majority",
    username = "ppgrdbuser",
    password = "mMNC24Zh3mTJ6Xrc"
)

db = client.ppgr

db.inputData.insert_many(docs)
