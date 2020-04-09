from functools import reduce

import tensorflow as tf
from layers import wBiFPNAdd
keras=tf.keras
L=keras.layers
activations=keras.activations


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
            #x = L.Conv2DTranspose(num_channels, kernel_size=1, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])

        if type == 'max':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            #x = L.Conv2D(num_channels, kernel_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1], inputs[2]])

        if type == 'out':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            #x = L.Conv2D(num_channels, kernel_size=3, strides=2, padding='same')(inputs[0])
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


def QuadFPN(inputs,num_channels, index, epsilon=1e-4, momentum=0.997):
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

        if type == 'stretch':
            x = L.UpSampling2D()(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x,inputs[1], inputs[2]])

        if type == 'out':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])

        if type == 'put':
            x = L.UpSampling2D()(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])

        if type == 'eval':
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([inputs[0], inputs[1]])

        out = L.Activation(lambda y: tf.nn.swish(y))(add)
        out = ConvBlock(value=value,kernel_size=3, strides=1)(out)
        return out

    if index == 0:
        inputs = Preprocess(inputs)

    i3, i4, i5, i6, i7 = inputs

    n4 = Process([i3, i4],value='n3',type='out')
    n5 = Process([n4, i5],value='n4',type='out')
    n6 = Process([n5, i6],value='n5',type='out')
    p7 = Process([n6, i7],value='p7',type='out')
    p6 = Process([p7, n6, i6],value='p6',type='stretch')
    p5 = Process([p6, n5, i5],value='p5',type='stretch')
    p4 = Process([p5, n4, i4],value='p4',type='stretch')
    p3 = Process([p4, i3],value='p3',type='put')

    m6 = Process([i7, i6],value='m6')
    m5 = Process([m6, i5],value='m5')
    m4 = Process([m5, i4],value='m4')
    o3 = Process([m4, i3],value='o3')
    o4 = Process([o3, m4, i4],value='o4', type='max')
    o5 = Process([o4, m5, i5],value='o5', type='max')
    o6 = Process([o5, m6, i6],value='o6', type='max')
    o7 = Process([o6, i7],value='o7', type='out')

    w3 = Process([o3, p3], value='w3', type='eval')
    w4 = Process([o4, p4], value='w4', type='eval')
    w5 = Process([o5, p5], value='w5', type='eval')
    w6 = Process([o6, p6], value='w6', type='eval')
    w7 = Process([o7, p7], value='w7', type='eval')

    return w3,w4,w5,w6,w7


'''
    Replace UpSampling by : 
    x = L.Conv2DTranspose(num_channels,kernel_size=1,strides=2,padding='same')(inputs[0])
          
    Cost : 49,920       
    
    Replace Maxpooling by :
    x = L.Conv2D(num_channels,kernel_size=3,strides=2,padding='same')(inputs[0])
              
    Cost : 443,136
    
'''