import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6
#from custom.layers import BiFPN
from layers import ClipBoxes, RegressBoxes, FilterDetections,wBiFPNAdd
from utils.anchors import anchors_for_shape
import numpy as np
from custom.detector import BoxNet
from custom.classifier import ClassNet
from functools import reduce


w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]

MOMENTUM = 0.997
EPSILON = 1e-4

keras=tf.keras
L=keras.layers
activations=keras.activations


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


def efficientdet(phi, num_classes=20, num_anchors=9, weighted_bifpn=True, freeze_bn=False,
                 score_threshold=0.01, detect_quadrangle=False, anchor_parameters=None, separable_conv=True):
    assert phi in range(7)
    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)
    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = d_bifpns[phi]
    w_head = w_bifpn
    d_head = d_heads[phi]
    backbone_cls = backbones[phi]
    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
    if weighted_bifpn:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features=QuadFPN(fpn_features,w_bifpn,index=i)
    else:
        raise ValueError('Only Weighted Network Supported')
    box_net = BoxNet(w_head, d_head,MOMENTUM=MOMENTUM,EPSILON=EPSILON,
                     num_anchors=num_anchors, separable_conv=separable_conv,
                     freeze_bn=freeze_bn,name='box_net')
    class_net = ClassNet(w_head, d_head,MOMENTUM=MOMENTUM,EPSILON=EPSILON,
                         num_classes=num_classes, num_anchors=num_anchors,
                         separable_conv=separable_conv, freeze_bn=freeze_bn, name='class_net')
    classification = [class_net([feature, i]) for i, feature in enumerate(fpn_features)]
    classification = layers.Concatenate(axis=1, name='classification')(classification)
    regression = [box_net([feature, i]) for i, feature in enumerate(fpn_features)]
    regression = layers.Concatenate(axis=1, name='regression')(regression)

    model = models.Model(inputs=[image_input], outputs=[classification, regression], name='efficientdet')

    # apply predicted regression to anchors
    anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    anchors_input = np.expand_dims(anchors, axis=0)
    boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    if detect_quadrangle:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold,
            detect_quadrangle=True
        )([boxes, classification, regression[..., 4:8], regression[..., 8]])
    else:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold
        )([boxes, classification])

    prediction_model = models.Model(inputs=[image_input], outputs=detections, name='efficientdet_p')
    return model, prediction_model
