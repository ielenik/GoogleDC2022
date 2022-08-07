import numpy as np
import tensorflow as tf

def predict_distance(psevdo, delta, real):
    data = np.zeros((psevdo.shape[0],psevdo.shape[1],4), dtype = np.float32)
    data[:,:,0] = psevdo
    data[:,:,1] = 1
    data[np.isnan(psevdo),1] = 0
    data[np.isnan(psevdo),0] = 0

    data[:,:,2] = delta
    data[:,:,3] = 1
    data[np.isnan(delta),2] = 0
    data[np.isnan(delta),3] = 0
    real[np.isnan(real)] = 0
    real[np.isnan(psevdo)] = 0
    real -= data[:,:,0]

    inp = tf.keras.layers.Input((None,4))
    x = tf.keras.layers.Conv1D(256,3,padding='same',dilation_rate=1, activation = 'selu')(inp)
    x = tf.keras.layers.Conv1D(256,3,padding='same',dilation_rate=2, activation = 'selu')(x)
    x = tf.keras.layers.Conv1D(256,3,padding='same',dilation_rate=4, activation = 'selu')(x)
    x = tf.keras.layers.Conv1D(256,3,padding='same',dilation_rate=8, activation = 'selu')(x)
    x = tf.keras.layers.Conv1D(256,3,padding='same',dilation_rate=16, activation = 'selu')(x)
    x = tf.keras.layers.Conv1D(256,3,padding='same',dilation_rate=32, activation = 'selu')(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(1,3,padding='same',dilation_rate=16, activation = None)(x)*1e-1
    out = tf.keras.layers.Flatten()(x) 

    model = tf.keras.models.Model(inp,out)
    model.compile(tf.keras.optimizers.Adam(1e-3),'mae')
    model.fit(data,real, epochs = 1024, steps_per_epoch = 1024)
