import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

path = "1.npz"
index = 107

def custom_loss(penalty_weight=1):
    def custom_fn(y_true, y_pred):
        squared_loss    = tf.square(tf.abs(y_true - y_pred))
        true_std        = tfp.stats.stddev(y_true)
        pred_std        = tfp.stats.stddev(y_pred)
        var_loss        = tf.square(tf.abs(true_std - pred_std))
        return squared_loss + penalty_weight*var_loss
    return custom_fn

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

    model       = keras.models.load_model('firstNN_model2.h5', custom_objects={ 'custom_fn': custom_loss() })

    f_X_test    = model.predict(X_test)
    print("Xvalue=%s, Difference=%s" % (X_test[index], abs(f_X_test[index] - Y_test[index])))
    E_err = []
    p_err = []
    p_true = []
    p_reco = []
    E_true = []
    E_reco = []
    for i in range(len(f_X_test[index] - Y_test[index])):
        Ex = []
        px = []
        ptx = []
        prx = []
        Etx = []
        Erx = []
        for j in range(len(f_X_test[index][i])):
            Ex.append(abs(f_X_test[index][i][j][0] - Y_test[index][i][j][0]))
            px.append(abs(f_X_test[index][i][j][1] - Y_test[index][i][j][1]))
            ptx.append(Y_test[index][i][j][1])
            prx.append(f_X_test[index][i][j][1])
            Etx.append(Y_test[index][i][j][0])
            Erx.append(f_X_test[index][i][j][0])
        E_err.append(np.array(Ex))
        p_err.append(np.array(px))
        p_true.append(ptx)
        p_reco.append(prx)
        E_true.append(Etx)
        E_reco.append(Erx)
    E_err = np.array(E_err)
    p_err = np.array(p_err)
    p_true = np.array(p_true)
    p_reco = np.array(p_reco)+1
    E_true = np.array(E_true)
    E_reco = np.array(E_reco)
            
    print(E_err)
    print(50*'-')
    print(p_err)

    figure, axis = plt.subplots(3, 2)
  
    # E err
    axis[0, 0].matshow(E_err)
    axis[0, 0].set_title("E_err")
      
    # For Cosine Function
    axis[0, 1].matshow(p_err)
    axis[0, 1].set_title("p_err")
      
    # For Tangent Function
    axis[1, 0].matshow(p_true)
    axis[1, 0].set_title("p_true")
      
    # For Tanh Function
    axis[1, 1].matshow(p_reco)
    axis[1, 1].set_title("p_reco")

    # For Tangent Function
    axis[2, 0].matshow(E_true)
    axis[2, 0].set_title("E_true")
      
    # For Tanh Function
    axis[2, 1].matshow(E_reco)
    axis[2, 1].set_title("E_reco")

    plt.show()

