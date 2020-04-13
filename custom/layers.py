from functools import reduce

import tensorflow as tf
from layers import wBiFPNAdd
keras=tf.keras
L=keras.layers
layers=keras.layers
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


def QuadFPN(inputs, num_channels, index, epsilon=1e-4, momentum=0.997):
    def Preprocess(inputs):
        for i in range(len(inputs)):
            inputs[i] = L.Conv2D(num_channels, (1, 1), name='pre-{}'.format(i))(inputs[i])
        return inputs

    def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
        f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                    use_bias=True, name=f'{name}/conv')
        f2 = layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=f'{name}/bn')
        # f2 = BatchNormalization(freeze=freeze_bn, name=f'{name}/bn')
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

    def ConvBlock(value, kernel_size=3, strides=2):
        f1 = L.SeparableConv2D(num_channels, name='conv-{0}-{1}'.format(index, value), kernel_size=kernel_size,
                               strides=strides, padding='same',
                               use_bias=True)
        f2 = L.BatchNormalization(name='bn-{0}-{1}'.format(index, value), momentum=momentum, epsilon=epsilon)
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))

    def Process(inputs, value, type='up'):
        if type == 'up':
            x = L.UpSampling2D()(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index, value))([x, inputs[1]])

        if type == 'max':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index, value))([x, inputs[1], inputs[2]])

        if type == 'stretch':
            x = L.UpSampling2D()(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index, value))([x, inputs[1], inputs[2]])

        if type == 'out':
            x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index, value))([x, inputs[1]])

        if type == 'put':
            x = L.UpSampling2D()(inputs[0])
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index, value))([x, inputs[1]])

        if type == 'eval':
            add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index, value))([inputs[0], inputs[1]])

        out = L.Activation(lambda y: tf.nn.swish(y))(add)
        out = ConvBlock(value=value, kernel_size=3, strides=1)(out)
        return out

    if index == 0:
        j = 0
        _, _, C3, C4, C5 = inputs
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name='resample_p6/bn')(P6_in)
        j += 1
        P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)
        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p7/maxpool')(P6_in)
        P7_U = layers.UpSampling2D()(P7_in)
        P6_td = wBiFPNAdd(name='add-0-1')([P6_in, P7_U])
        P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P6_td)
        j += 1
        P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P5_in)
        P5_in_1 = layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(P5_in_1)
        P6_U = layers.UpSampling2D()(P6_td)
        P5_td = wBiFPNAdd(name='add-0-2')([P5_in_1, P6_U])
        P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P5_td)
        P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P4_in)
        P4_in_1 = layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(P4_in_1)
        j += 1
        P5_U = layers.UpSampling2D()(P5_td)
        P4_td = wBiFPNAdd(name='add-0-3')([P4_in_1, P5_U])
        P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P4_td)
        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P3_in)
        P3_in = layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(P3_in)
        j += 1
        P4_U = layers.UpSampling2D()(P4_td)
        P3_out = wBiFPNAdd(name='add-0-4')([P3_in, P4_U])
        P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P3_out)
        P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P4_in)
        P4_in_2 = layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(P4_in_2)
        j += 1
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = wBiFPNAdd(name='add-0-5')([P4_in_2, P4_td, P3_D])
        P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P4_out)
        j += 1
        P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P5_in)
        P5_in_2 = layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(P5_in_2)
        # P5_in_2 = BatchNormalization(freeze=freeze_bn, name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = wBiFPNAdd(name='add-0-6')([P5_in_2, P5_td, P4_D])
        P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P5_out)
        j += 1
        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = wBiFPNAdd(name='add-0-7')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P6_out)
        j += 1
        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = wBiFPNAdd(name='add-0-8')([P7_in, P6_D])
        P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, name="block-{}".format(j))(
            P7_out)

        return P3_out, P4_td, P5_td, P6_td, P7_out

    else:

        i3, i4, i5, i6, i7 = inputs

        n4 = Process([i3, i4], value='n3', type='out')
        n5 = Process([n4, i5], value='n4', type='out')
        n6 = Process([n5, i6], value='n5', type='out')
        p7 = Process([n6, i7], value='p7', type='out')
        p6 = Process([p7, n6, i6], value='p6', type='stretch')
        p5 = Process([p6, n5, i5], value='p5', type='stretch')
        p4 = Process([p5, n4, i4], value='p4', type='stretch')
        p3 = Process([p4, i3], value='p3', type='put')

        m6 = Process([i7, i6], value='m6')
        m5 = Process([m6, i5], value='m5')
        m4 = Process([m5, i4], value='m4')
        o3 = Process([m4, i3], value='o3')
        o4 = Process([o3, m4, i4], value='o4', type='max')
        o5 = Process([o4, m5, i5], value='o5', type='max')
        o6 = Process([o5, m6, i6], value='o6', type='max')
        o7 = Process([o6, i7], value='o7', type='out')

        w3 = Process([o3, p3], value='w3', type='eval')
        w4 = Process([o4, p4], value='w4', type='eval')
        w5 = Process([o5, p5], value='w5', type='eval')
        w6 = Process([o6, p6], value='w6', type='eval')
        w7 = Process([o7, p7], value='w7', type='eval')

        return w3, w4, w5, w6, w7


# def QuadFPN(inputs,num_channels, index, epsilon=1e-4, momentum=0.997):
#     def Preprocess(inputs):
#         for i in range(len(inputs)):
#             inputs[i] = L.Conv2D(num_channels, (1, 1),name='pre-{}'.format(i))(inputs[i])
#         return inputs
#
#     def ConvBlock(value,kernel_size=3, strides=2):
#         f1 = L.SeparableConv2D(num_channels,name='conv-{0}-{1}'.format(index,value), kernel_size=kernel_size, strides=strides, padding='same',
#                                use_bias=True)
#         f2 = L.BatchNormalization(name='bn-{0}-{1}'.format(index,value),momentum=momentum, epsilon=epsilon)
#         return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))
#
#     def Process(inputs,value, type='up'):
#         if type == 'up':
#             x = L.UpSampling2D()(inputs[0])
#             add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])
#
#         if type == 'max':
#             x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
#             add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1], inputs[2]])
#
#         if type == 'stretch':
#             x = L.UpSampling2D()(inputs[0])
#             add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x,inputs[1], inputs[2]])
#
#         if type == 'out':
#             x = L.MaxPooling2D(pool_size=3, strides=2, padding='same')(inputs[0])
#             add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])
#
#         if type == 'put':
#             x = L.UpSampling2D()(inputs[0])
#             add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([x, inputs[1]])
#
#         if type == 'eval':
#             add = wBiFPNAdd(name='bifpn-add-{0}-{1}'.format(index,value))([inputs[0], inputs[1]])
#
#         out = L.Activation(lambda y: tf.nn.swish(y))(add)
#         out = ConvBlock(value=value,kernel_size=3, strides=1)(out)
#         return out
#
#     if index == 0:
#         inputs = Preprocess(inputs)
#
#     i3, i4, i5, i6, i7 = inputs
#
#     n4 = Process([i3, i4],value='n3',type='out')
#     n5 = Process([n4, i5],value='n4',type='out')
#     n6 = Process([n5, i6],value='n5',type='out')
#     p7 = Process([n6, i7],value='p7',type='out')
#     p6 = Process([p7, n6, i6],value='p6',type='stretch')
#     p5 = Process([p6, n5, i5],value='p5',type='stretch')
#     p4 = Process([p5, n4, i4],value='p4',type='stretch')
#     p3 = Process([p4, i3],value='p3',type='put')
#
#     m6 = Process([i7, i6],value='m6')
#     m5 = Process([m6, i5],value='m5')
#     m4 = Process([m5, i4],value='m4')
#     o3 = Process([m4, i3],value='o3')
#     o4 = Process([o3, m4, i4],value='o4', type='max')
#     o5 = Process([o4, m5, i5],value='o5', type='max')
#     o6 = Process([o5, m6, i6],value='o6', type='max')
#     o7 = Process([o6, i7],value='o7', type='out')
#
#     w3 = Process([o3, p3], value='w3', type='eval')
#     w4 = Process([o4, p4], value='w4', type='eval')
#     w5 = Process([o5, p5], value='w5', type='eval')
#     w6 = Process([o6, p6], value='w6', type='eval')
#     w7 = Process([o7, p7], value='w7', type='eval')
#
#     return w3,w4,w5,w6,w7


'''
    Replace UpSampling by : 
    x = L.Conv2DTranspose(num_channels,kernel_size=1,strides=2,padding='same')(inputs[0])
          
    Cost : 49,920       
    
    Replace Maxpooling by :
    x = L.Conv2D(num_channels,kernel_size=3,strides=2,padding='same')(inputs[0])
              
    Cost : 443,136
    
'''