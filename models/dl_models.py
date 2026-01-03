
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

def lstm_model(shape):
    model = Sequential([LSTM(64, return_sequences=True, input_shape=shape),
                        LSTM(32), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model

def gru_model(shape):
    model = Sequential([GRU(64, return_sequences=True, input_shape=shape),
                        GRU(32), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model
