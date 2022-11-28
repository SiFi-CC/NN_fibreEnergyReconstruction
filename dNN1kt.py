import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import keras_tuner

path = "FinalDetectorVersion_RasterCoupling_OPM_38e8protons.npz"

def build_model(hp):
    af = hp.Choice("acitvation",["relu","elu","tanh"])
    model = keras.Sequential([keras.layers.InputLayer(input_shape = (12,2,32,2)),
                              keras.layers.Conv3D(filters = hp.Int("units", min_value=1, max_value=16,step=1), kernel_size = 2, padding='same'),
                              keras.layers.Flatten(),
                              keras.layers.Dense(hp.Int("units", min_value =500, max_value = 10000, step =500), activation = af),
                              keras.layers.Dense(5192, activation = af),
                              keras.layers.Reshape((22,118,2,))])
    model.compile(loss="mse",
                  optimizer="nadam")
    return model

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

    


    tuner = keras_tuner.RandomSearch(hypermodel=build_model,
                                     objective="val_loss",
                                     max_trials=3,
                                     executions_per_trial=2,
                                     overwrite=True,
                                     directory="RandomSearch",
                                     project_name="dNN1"
                                     )


    # train model
    tuner.search(X_train, 
              Y_train, 
              epochs = 10000,    
              validation_data = (X_val, Y_val), 
              callbacks = [tf.keras.callbacks.EarlyStopping('val_loss', patience=3)],
              batch_size = 1000
              )

    models = tuner.get_best_models(num_models=5)
    tuner.results_summary()
    for m in models:
        m.build(input_shape=(None,12,2,32))
        m.summary()
