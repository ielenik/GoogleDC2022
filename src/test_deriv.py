import tensorflow as tf
import numpy as np

size = 4096
size_fur = 128
class LinearFun(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LinearFun, self).__init__(**kwargs)
    def build(self, input_shape):
        super(LinearFun, self).build(input_shape)
        self.cos_values = self.add_weight(name='cos_values', shape=(1,size_fur), trainable=True, initializer=tf.keras.initializers.Constant(value=np.zeros((1,size_fur))))
        self.sin_values = self.add_weight(name='cos_values', shape=(1,size_fur), trainable=True, initializer=tf.keras.initializers.Constant(value=np.zeros((1,size_fur))))

    def get_value(self,index):
        indicies = tf.cast(tf.reshape(index,[-1]), tf.float32)
        xval = indicies/size*3.14
        xval = tf.reshape(xval,[-1,1])
        perd = tf.cast(tf.range(size_fur), tf.float32)
        perd = tf.reshape(perd, [1,-1])
        args = xval*perd
        return tf.reduce_sum(self.cos_values*tf.math.cos(args), axis = -1)+tf.reduce_sum(self.sin_values*tf.math.sin(args), axis = -1)

    def call(self, index):
        
        return self.get_value(index), self.get_value(index+1)

linear = LinearFun()

epoch_input = tf.keras.layers.Input((1,), dtype=tf.int32, name = 'epoch')
v1, v2 = linear(epoch_input)
model = tf.keras.Model(epoch_input, v1-v2)
model.compile(tf.keras.optimizers.Adam(learning_rate = 0.01), loss = 'MAE')

model.fit(np.arange(size), np.ones(size), epochs = 1024*2, batch_size=16,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience = 50,verbose=1,min_lr=1e-5)])

res = (model.predict(np.arange(size)))
res = res.reshape((-1,))
print(res )
