import numpy as np
import pandas as pd
import tensorflow as tf
import keras.api._v2.keras as keras

from sklearn.model_selection import train_test_split

from keras import Model
from keras.layers import Dense, Dropout, Input


BATCH_SIZE = 256

# Training Features
LABEL_COL = 'glucose'

CAT_X_STR_COLS = [
    'lifestyle',
    'stress',
    'smk',
    'sex'
]

CAT_X_INT_COLS = [
    'dt',
    'resting',
    'fasting',
    #'meal_n',
    'age'
]

CONT_X_COLS = [
    'bmi_s',
    'wc',
    'hba1c',
    'ogtt',
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
]


def load_data():
    dataset = pd.read_csv("ml_input.csv", header=0, dtype=str)

    # Only use data points reacting to activities and/or meals.
    # Removed records did not have any meal or activity records prior to the CGMS reading.
    dataset = dataset.loc[
        ~pd.isnull(dataset['meal']) |
        ~pd.isnull(dataset['lifestyle']),
        :
    ]

    # Drop ID column.
    dataset = dataset.drop(['id', 'Unnamed: 0'], axis=1)

    # Fill NA values.
    # Numeric     --> 0
    # Categorical --> "not_provided"
    dataset[CONT_X_COLS] = dataset[CONT_X_COLS].fillna(value = 0)
    dataset[CAT_X_INT_COLS] = dataset[CAT_X_INT_COLS].fillna(value = 0)
    dataset[CAT_X_STR_COLS] = dataset[CAT_X_STR_COLS].fillna(value = "not_provided")

    dataset['age'] = dataset['age'].astype(np.float64).astype(np.int64)
    dataset['dt'] = pd.to_datetime(dataset['dt']).astype(np.int64) // 10**9
    dataset['resting'] = pd.to_timedelta(dataset['resting']).astype(np.int64) // 10**9
    dataset['fasting'] = pd.to_timedelta(dataset['fasting']).astype(np.int64) // 10**9

    print('Dataset imported.')
    return dataset


def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32, dtype=str):
    df = dataframe.copy()
    df = {key: value[:,tf.newaxis].astype(dtype) for key, value in dataframe.items()}
    #[print(key, value[:,tf.newaxis].dtype) for key, value in dataframe.items()]
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def preprocessing_data(dataset, labels, dtype=str):
    dataset[LABEL_COL] = labels

    train, val, test = np.split(
        dataset.sample(frac=1), 
        [int(0.8*len(dataset)), int(0.9*len(dataset))]
    )

    ds = {
        'train': train,
        'val': val,
        'test': test,
    }

    for key, value in ds.items():
        labels = value.pop(LABEL_COL).astype(float)
        ds[key] = df_to_dataset(value, labels, batch_size=BATCH_SIZE, dtype=dtype)

    return ds


def normalizer_layer(cols, ds, inputs=None, features=None):
    print('normalize numerical features')
    for header in cols:
        numeric_col = Input(shape=(1,), name=header)

        normalizer = keras.layers.Normalization(axis=None)
        feature_ds = ds['train'].map(lambda x, y: x[header])
        normalizer.adapt(feature_ds)

        encoded_numeric_col = normalizer(numeric_col)
        inputs.append(numeric_col)
        features.append(encoded_numeric_col)
    

def categorical_layer(cols, ds, inputs=None, features=None):
    print('encode categorical features')
    for header in cols:
        categorical_col = Input(shape=(1,), name=header, dtype='string')

        index = keras.layers.StringLookup(output_mode="one_hot")
        feature_ds = ds['train'].map(lambda x, y: x[header])
        index.adapt(feature_ds)
        encoder = keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())
        encoding_layer = lambda feature: encoder(index(feature))

        encoded_categorical_col = encoding_layer(categorical_col)
        inputs.append(categorical_col)
        features.append(encoded_categorical_col)
    

def ordinal_layer(cols, ds, inputs=None, features=None):
    print('encode ordinal features')
    for header in cols:
        ordinal_col = Input(shape=(1,), name=header, dtype='int64')

        index = keras.layers.IntegerLookup()
        feature_ds = ds['train'].map(lambda x, y: x[header])
        index.adapt(feature_ds)
        encoder = keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size())
        encoding_layer = lambda feature: encoder(index(feature))

        encoded_ordinal_col = encoding_layer(ordinal_col)
        inputs.append(ordinal_col)
        features.append(encoded_ordinal_col)


def main():
    raw_data = load_data()

    labels = raw_data[LABEL_COL].astype(float)
    float_data = preprocessing_data(raw_data[CONT_X_COLS], labels, dtype=float)
    str_data = preprocessing_data(raw_data[CAT_X_STR_COLS], labels, dtype=str)
    int_data = preprocessing_data(raw_data[CAT_X_INT_COLS], labels, dtype=int)

    [(train_features, label_batch)] = float_data['train'].take(1)
    print('Every feature:', list(train_features.keys()))
    print('A batch of BMI:', train_features['bmi_s'])
    print('A batch of Glucose:', label_batch)

    inputs = []
    features = []

    normalizer_layer(CONT_X_COLS, float_data, inputs=inputs, features=features)
    categorical_layer(CAT_X_STR_COLS, str_data, inputs=inputs, features=features)
    ordinal_layer(CAT_X_INT_COLS, int_data, inputs=inputs, features=features)

    combined_features = keras.layers.concatenate(features)

    x0 = Dense(32, activation="relu", kernel_initializer='he_normal')(combined_features)
    x0 = Dropout(0.5)(x0)

    x1 = Dense(10, activation='relu', kernel_initializer='he_normal')(x0)
    x1 = Dropout(0.5)(x1)

    x2 = Dense(8, activation='relu', kernel_initializer='he_normal')(x1)
    x2 = Dropout(0.5)(x2)

    output = Dense(1)(x2)

    model = Model(inputs=inputs, outputs=output)

    print('compile the model')
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    
    # tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

    print('fit the model')
    #combined_train = tf.data.Dataset.concatenate([float_data['train'], str_data['train'], int_data['train']])
    #combined_val = tf.data.Dataset.concatenate([float_data['val'], str_data['val'], int_data['val']])
    #combined_test = tf.data.Dataset.concatenate([float_data['test'], str_data['test'], int_data['test']])
    combined_train = tf.data.Dataset.zip((
        (float_data['train'].map(lambda x, y: x),
        str_data['train'].map(lambda x, y: x),
        int_data['train'].map(lambda x, y: x)),
        float_data['train'].map(lambda x, y: y)
    ))

    combined_val = tf.data.Dataset.zip((
        (float_data['val'].map(lambda x, y: x),
        str_data['val'].map(lambda x, y: x),
        int_data['val'].map(lambda x, y: x)),
        float_data['val'].map(lambda x, y: y)
    ))

    combined_test = tf.data.Dataset.zip((
        (float_data['test'].map(lambda x, y: x),
        str_data['test'].map(lambda x, y: x),
        int_data['test'].map(lambda x, y: x)),
        float_data['test'].map(lambda x, y: y)
    ))

    model.fit(
        combined_train,
        validation_data=combined_val,
        epochs=150
    )

    print('evaluate')
    loss, accuracy = model.evaluate(combined_test)
    print("Loss", loss)
    print("Accuracy", accuracy)


if __name__ == '__main__':
    main()
