import numpy as np
import tensorflow as tf
from tensorflow import keras


path = "test.npz"

# load data
with np.load(path) as data:
    input_data = data["all_events_input"]
    output_data = data["all_events_output"]

    print(input_data.shape)
    print(output_data.shape)

    # slice data
    X_train = input_data[:360]
    Y_train = output_data[:360]
    X_val   = input_data[360:400]
    Y_val   = output_data[360:400]
    X_test  = input_data[400:]
    Y_test  = output_data[400:]

    # define model
    model = keras.Sequential([keras.layers.InputLayer(input_shape = (12,2,32,2)),
                              keras.layers.Conv3D(filters = 10, kernel_size = [3,2,3]),
                              keras.layers.Flatten(),
                              keras.layers.Dense(5192, activation = "elu"),
                              keras.layers.Reshape((22,118,2,))])
    # compile model
    model.compile(loss='mean_absolute_error',
                  optimizer = keras.optimizers.Adam(0.001),
                  metrics=['accuracy']
                  )

    # train model
    model.fit(X_train, 
              Y_train, 
              epochs = 30,    
              validation_data = (X_val, Y_val), 
              callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3)],
              batch_size = 1
              )

    #evaluate model
    score = model.evaluate(X_test, Y_test, verbose = 0) 

    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
                              
