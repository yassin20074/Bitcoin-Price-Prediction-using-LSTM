#calling the libraries

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

# function to create the model
def build_lstm_model(input_shape):
    
    # LSTM model for Time Series prediction with Bidirectional LSTM and Dropout.
    
    model = Sequential([
        Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        #only one layer because we axpect only one price value
        Dense(1) 
    ])
    #model training 
    # adam=optimization algorithm
    #mse= suitable for problems of prediction continuous numbers

    model.compile(optimizer='adam', loss='mean_squared_error')
    #we return the ready made model yo our data for training
    return model
