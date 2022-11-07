import numpy as np
import tensorflow as tf
from tensorflow import keras


path = "test.npz"

# KOMMENTAREN----------------------------------------------------------------------------
with np.load(path) as data:
    input_data = data["all_events_input"]
    output_data = data["all_events_output"]
   

    output_data = output_data.flatten()

 
    print(input_data.shape)
    print(output_data.shape)

    X_train = input_data[:400]
    Y_train = output_data[:400]
    X_test  = input_data[400:]
    Y_test  = output_data[400:]

    model = keras.Sequential([keras.layers.Flatten(),
                          keras.layers.Dense(6000, activation = "relu"),
                          keras.layers.Dense(6000, activation = "relu"),
                          keras.layers.Dense(6000, activation = "relu"),
                          keras.layers.Dense(6000, activation = "relu"),
                          keras.layers.Dense(2)])
    model.compile(loss='mean_absolute_error',
              optimizer = keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=30)


                              
