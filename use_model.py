import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

path = "1.npz"

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

    model = keras.models.load_model('firstNN_model1')
    f_X_test = model.predict(X_test)
    print("Xvalue=%s, Difference=%s" % (X_test[300], abs(f_X_test[300] - Y_test[300])))
    E_err = []
    p_err = []
    for i in range(len(f_X_test[300] - Y_test[300])):
        Ex = []
        px = []
        for j in range(len(f_X_test[300][i])):
            Ex.append(abs(f_X_test[300][i][j][0])) # - Y_test[300][i][j][0]))
            px.append(abs(f_X_test[300][i][j][1])) # - Y_test[300][i][j][1]))
        E_err.append(np.array(Ex))
        p_err.append(np.array(px))
    E_err = np.array(E_err)
    p_err = np.array(p_err)
            
    print(E_err)
    print(50*'-')
    print(p_err)

    plt.matshow(E_err)
    plt.show()
    plt.matshow(p_err)
    plt.show()

