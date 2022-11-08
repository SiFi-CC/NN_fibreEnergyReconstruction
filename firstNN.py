import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

path = "1.npz"

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
    history = model.fit(X_train, 
              Y_train, 
              epochs = 1000,    
              validation_data = (X_val, Y_val), 
              callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3)],
              batch_size = 100
              )

    #evaluate model
    score = model.evaluate(X_test, Y_test, verbose = 0) 

    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_hist1.png')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc_hist1.png')
    plt.show()

    # save model
    model.save('firstNN_model1')
                              
