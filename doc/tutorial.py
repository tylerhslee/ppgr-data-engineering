from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Input

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# determine the number of input features
n_features = X_train.shape[1]

# define model
x_in = Input(shape=(n_features,))
x1 = Dense(10, activation='relu', kernel_initializer='he_normal')(x_in)
x2 = Dense(8, activation='relu', kernel_initializer='he_normal')(x1)
x_out = Dense(1)(x2)
model = Model(inputs=x_in, outputs=x_out)

# compile the model
model.compile(optimizer='adam', loss='mse')

# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))

# make a prediction
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)
