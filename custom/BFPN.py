from functools import reduce

import tensorflow as tf
from layers import wBiFPNAdd
keras=tf.keras
L=keras.layers
activations=keras.activations

class BFPN(L.Layer):
    def __init__(self,num_channels,index, epsilon=1e-4,momentum= 0.997, **kwargs):
        super(BFPN, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum=momentum
        self.num_channels=num_channels
        self.index=index

    def Preprocess(self,inputs):
        for i in range(len(inputs)):
            inputs[i]=L.Conv2D(self.num_channels,(1,1))(inputs[i])
        return inputs

    def ConvBlock(self,kernel_size=3, strides=2):
        f1 = L.SeparableConv2D(self.num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                    use_bias=True)
        f2 = L.BatchNormalization(momentum=self.momentum, epsilon=self.epsilon)
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

    def Process(self,inputs,type='up'):
        if type=='up':
            x=L.UpSampling2D()(inputs[0])
            add = wBiFPNAdd(name='add')([x,inputs[1]])

        if type=='max':
            x=L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='add')([x,inputs[1],inputs[2]])

        if type=='out':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='add')([x, inputs[1]])

        out = L.Activation(lambda y: tf.nn.swish(y))(add)
        out = self.ConvBlock(kernel_size=3, strides=1)(out)
        return out

    #def build(self, input_shape):

    def call(self, inputs, **kwargs):

        if self.index==0:
            inputs=self.Preprocess(inputs)

        i3,i4,i5,i6,i7=inputs

        m6=self.Process([i7,i6])
        m5=self.Process([m6,i5])
        m4=self.Process([m5,i4])

        o3=self.Process([m4,i3])

        o4=self.Process([o3,m4,i4],type='max')
        o5 = self.Process([o4, m5, i5],type='max')
        o6 = self.Process([o5, m6, i6],type='max')
        o7 = self.Process([o6, i7],type='out')
        return o3,o4,o5,o6,o7

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(BFPN, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config

