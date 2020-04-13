import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6
from custom.layers import BiFPN,QuadFPN
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
