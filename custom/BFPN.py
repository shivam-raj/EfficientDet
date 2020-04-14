from functools import reduce

import tensorflow as tf
from layers import wBiFPNAdd
keras=tf.keras
L=keras.layers
activations=keras.activations

# class BFPN(keras.Model):
#     def __init__(self,num_channels,index, epsilon=1e-4,momentum= 0.997, **kwargs):
#         super(BFPN, self).__init__(**kwargs)
#         self.epsilon = epsilon
#         self.momentum=momentum
#         self.num_channels=num_channels
#         self.index=index
#
#     def Preprocess(self,inputs):
#         for i in range(len(inputs)):
#             inputs[i]=L.Conv2D(self.num_channels,(1,1))(inputs[i])
#         return inputs
#
#     def ConvBlock(self,training,kernel_size=3, strides=2):
#         f1 = L.SeparableConv2D(self.num_channels, kernel_size=kernel_size, strides=strides, padding='same',
#                                     use_bias=True)
#         f2 = L.BatchNormalization(training=training,momentum=self.momentum, epsilon=self.epsilon)
#         return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))
#
#     def Process(self,inputs,training,type='up'):
#         if type=='up':
#             x=L.UpSampling2D()(inputs[0])
#             add = wBiFPNAdd(name='add')([x,inputs[1]])
#
#         if type=='max':
#             x=L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
#             add = wBiFPNAdd(name='add')([x,inputs[1],inputs[2]])
#
#         if type=='out':
#             x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
#             add = wBiFPNAdd(name='add')([x, inputs[1]])
#
#         out = L.Activation(lambda y: tf.nn.swish(y))(add)
#         out = self.ConvBlock(training,kernel_size=3, strides=1)(out)
#         return out
#
#     def call(self, inputs, training=False):
#
#         if self.index==0:
#             inputs=self.Preprocess(inputs)
#
#         i3,i4,i5,i6,i7=inputs
#
#         m6=self.Process([i7,i6],training)
#         m5=self.Process([m6,i5],training)
#         m4=self.Process([m5,i4],training)
#
#         o3=self.Process([m4,i3],training)
#
#         o4=self.Process([o3,m4,i4],training,type='max')
#         o5 = self.Process([o4, m5, i5],training,type='max')
#         o6 = self.Process([o5, m6, i6],training,type='max')
#         o7 = self.Process([o6, i7],training,type='out')
#         return o3,o4,o5,o6,o7


def BiFPN(inputs,num_channels, index, epsilon=1e-4, momentum=0.997):
    def Preprocess(inputs):
        for i in range(len(inputs)):
            inputs[i] = L.Conv2D(num_channels, (1, 1),name='pre-{}'.format(i))(inputs[i])
        return inputs

    def ConvBlock(value,kernel_size=3, strides=2):
        f1 = L.SeparableConv2D(num_channels,name='conv-{0}-{1}'.format(index,value), kernel_size=kernel_size, strides=strides, padding='same',
                               use_bias=True)
        f2 = L.BatchNormalization(name='bn-{0}-{1}'.format(index,value),momentum=momentum, epsilon=epsilon)
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

    def Process(inputs,value, type='up'):
        if type == 'up':
            x = L.UpSampling2D()(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])

        if type == 'max':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1], inputs[2]])

        if type == 'out':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])

        out = L.Activation(lambda y: tf.nn.swish(y))(add)
        out = ConvBlock(value=value,kernel_size=3, strides=1)(out)
        return out

    if index == 0:
        inputs = Preprocess(inputs)

    i3, i4, i5, i6, i7 = inputs

    m6 = Process([i7, i6],value='m6')
    m5 = Process([m6, i5],value='m5')
    m4 = Process([m5, i4],value='m4')

    o3 = Process([m4, i3],value='o3')

    o4 = Process([o3, m4, i4],value='o4', type='max')
    o5 = Process([o4, m5, i5],value='o5', type='max')
    o6 = Process([o5, m6, i6],value='o6', type='max')
    o7 = Process([o6, i7],value='o7', type='out')
    return o3, o4, o5, o6, o7