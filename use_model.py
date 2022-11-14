import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

path = "1.npz"
index = 45

with np.load(path) as data:
    input_data  = data["all_events_input"]
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

    model       = keras.models.load_model('firstNN_model2.h5', custom_objects={ 'loss': custom_fn(penalty_weight) })
    model = load_model(modelFile)
    f_X_test    = model.predict(X_test)
    print("Xvalue=%s, Difference=%s" % (X_test[index], abs(f_X_test[index] - Y_test[index])))
    E_err = []
    p_err = []
    p_true = []
    p_reco = []
    for i in range(len(f_X_test[index] - Y_test[index])):
        Ex = []
        px = []
        ptx = []
        prx = []
        for j in range(len(f_X_test[index][i])):
            Ex.append(abs(f_X_test[index][i][j][0] - Y_test[index][i][j][0]))
            px.append(abs(f_X_test[index][i][j][1] - Y_test[index][i][j][1]))
            ptx.append(Y_test[index][i][j][1])
            prx.append(f_X_test[index][i][j][1])
        E_err.append(np.array(Ex))
        p_err.append(np.array(px))
        p_true.append(ptx)
        p_reco.append(prx)
    E_err = np.array(E_err)
    p_err = np.array(p_err)
    p_true = np.array(p_true)
    p_reco = np.array(p_reco)
            
    print(E_err)
    print(50*'-')
    print(p_err)

    plt.matshow(E_err)
    plt.show()
    plt.matshow(p_err)
    plt.show()
    plt.matshow(p_true)
    plt.show()
    plt.matshow(p_reco)
    plt.show()

