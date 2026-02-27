# calling to libraries
import numpy as np
from model import build_lstm_model
from data_utils import fetch_data, prepare_data
from plot_utils import plot_predictions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # to compute metrics
from tensorflow.keras.callbacks import EarlyStopping

# fetch historical bitcion price data
df = fetch_data("BTC-USD")
x, y, scaler = prepare_data(df, look_back=60)

# split the data
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# reshape data for lstm
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# build the model
model = build_lstm_model((x_train.shape[1],1))

# trian with earlystopping
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(
    x_train, y_train, batch_size=32, epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# prediction and scaling back
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1,1))

# compute metrics
mse = mean_squared_error(y_test_unscaled, predictions)
mae = mean_absolute_error(y_test_unscaled, predictions)
r2 = r2_score(y_test_unscaled, predictions)
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# plot result
plot_predictions(y_test_unscaled, predictions)