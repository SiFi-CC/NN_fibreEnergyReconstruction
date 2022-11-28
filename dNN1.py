import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

path = "FinalDetectorVersion_RasterCoupling_OPM_38e8protons.npz"

# load data
with np.load(path) as data:
    input_data = data["all_events_input"]
    output_data = data["all_events_output"]

    print(input_data.shape)
    print(output_data.shape)

    # slice data
    trainset_index  = int(input_data.shape[0]*0.7)
    valset_index    = int(input_data.shape[0]*0.8)
    print(trainset_index)
    print(valset_index)
    X_train = input_data[:trainset_index]
    Y_train = output_data[:trainset_index]
    X_val   = input_data[trainset_index:valset_index]
    Y_val   = output_data[trainset_index:valset_index]
    X_test  = input_data[valset_index:]
    Y_test  = output_data[valset_index:]

    # define model
    model = keras.Sequential([keras.layers.InputLayer(input_shape = (12,2,32,2)),
                              keras.layers.Conv3D(filters = 4, kernel_size = 2, padding='same'),
                              keras.layers.Flatten(),
                              keras.layers.Dense(10000, activation = "tanh"),
                              keras.layers.Dense(5192, activation = "tanh"),
                              keras.layers.Reshape((22,118,2,))])
    # compile model
    model.compile(loss="mse",
                  optimizer = "nadam",
                  metrics=['mae']
                  )


    model.summary()

    # train model
    history = model.fit(X_train, 
              Y_train, 
              epochs = 10000,    
              validation_data = (X_val, Y_val), 
              callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3)],
              batch_size = 1000
              )

    #evaluate model
    score = model.evaluate(X_test, Y_test, verbose = 0) 

    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])

    model.save("dNN1.h5")

    # summarize history for loss
    plt.figure(0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_hist_dNN1.png')
    

    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc_hist_dNN1.png')
