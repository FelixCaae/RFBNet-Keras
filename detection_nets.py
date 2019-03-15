from module import *
from keras.layers import Conv2D,SeparableConv2D,Dense,Input,Flatten,Reshape,MaxPooling2D,AveragePooling2D,Activation,Softmax,ZeroPadding2D
from keras.layers import Concatenate,Add,Multiply,Lambda
from keras.layers import BatchNormalization,Dropout 
from keras.models import Model,Sequential,load_model
import keras.applications.mobilenetv2 
from keras import backend as K
import numpy as np
import os
import warnings
def build_simple_model(base_model,source_layer,extractor,version_name):
    x = base_model.get_layer(source_layer).output
    for layer in extractor:
        x = layer(x)
    x0 = Dense(4,activation ='sigmoid')(x)
    x1 = Dense(num_classes,activation = 'softmax')(x)
    x = Concatenate()([x1,x0])
    model = Model(inputs = base_model.input, outputs = x)
    name = 'simple_detection_' + '_' + version_name
    
def build_simple_detection_net(base_model,source_layer,num_classes,version_name,base_name='mobilenetv2'):
    x = base_model.get_layer(source_layer).output
    x = AveragePooling2D(strides=2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(100,activation = 'relu')(x)
    x0 = Dense(4,activation ='sigmoid')(x)
    x1 = Dense(num_classes,activation = 'softmax')(x)
    x = Concatenate()([x1,x0])
    model = Model(inputs = base_model.input, outputs = x)
    name = 'simple_detection_' + base_name + '_' + version_name 
    return name,model

def build_double_tail_detection_net(base_model,source_layer,num_classes,version_name,base_name='mobilenetv2'):
#    x = BasicFC(base_model.get_layer('out_relu').output, 4 + num_classes, 'relu')
#    x = Dense(4 + num_classes,activation = 'relu')(base_model.output)
    x = base_model.get_layer(source_layer).output
    x0 = AveragePooling2D(strides=2)(x)
    x0 = Flatten()(x0)
    x0 = Dropout(0.5)(x0)
    x0 = Dense(100,activation = 'relu')(x0)
    x0 = Dense(4,activation ='sigmoid')(x0)
    x1 = AveragePooling2D(strides=7)(x)
    x1 = Flatten()(x1)
    x1 = Dense(1000,activation = 'relu')(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(num_classes,activation = 'softmax')(x1)
    x = Concatenate()([x1,x0])
    model = Model(inputs = base_model.input, outputs = x)
    name = 'double_tail_detection_' + base_name + '_' + version_name 
    return name,model

def build_feature_pyramid_detection_net(base_model,source_layers,num_classes):
    source_layers = []
    outputs = []
    for source_layer in source_layers:
        x = Flatten()(base_model.get_layer(source_layer).output)
        x = Dense(4096,activation = 'relu')(x)
        x = Dense(1024,activation = 'relu')(x)
        x = Dense(4 + num_classes,activation ='sigmoid')(x)
        outputs.append(x)
    output = Concatenate()(outputs)
    model = Model(inputs = base_model.input, outputs = output)
    
    return 'simple_detection_net_2',model

def add_extra_layers(x, extra_cfg, use_l2=False):
    source_layers = []
    flag = False
#     for k, v in enumerate(extra_cfg):
#         if v == 'S':
#             x = BasicRFB(x, extra_cfg[k+1], stride=2, scale = 1.0)
#             source_layers += [x]
#         else:
#             x = BasicRFB(x, v, scale = 1.0)
    x = BasicRFB(x, 512, stride=2, scale = 1.0,use_l2 = use_l2) 
    source_layers += [x]
    x = BasicConv(x,128,kernel_size = 1,stride = 1,use_l2 = use_l2) 
    x = BasicConv(x,256,kernel_size = 3,stride = 2,use_l2 = use_l2)
    source_layers += [x]
    x = BasicConv(x,128,kernel_size = 1,stride = 1,use_l2 = use_l2)
    x = BasicConv(x,256,kernel_size = 3,stride = 2,use_l2 = use_l2)
    source_layers += [x]
    x = BasicConv(x,64,kernel_size = 1,stride = 1,use_l2 = use_l2)
    x = BasicConv(x,128,kernel_size = 3, stride=2,use_l2 = use_l2)
    source_layers += [x]

    return source_layers

def multibox(x, prior_num, num_classes,softmax = False):
    #Input a feature map layer , add locating layer and classificaton layer upon
    #Output shape should be (batch_size,#boxes,num_classes + 4)
    n_H,n_W = x.get_shape().as_list()[1:3]
    num_classes = num_classes + 1
    loc_pred = Conv2D(prior_num * 4,kernel_size = 3 ,padding= 'same')(x)
    loc_pred = Reshape((-1,4))(loc_pred)
    conf = Conv2D(prior_num * num_classes, kernel_size = 3,padding = 'same')(x)
    conf = Reshape((-1, num_classes))(conf)
    if softmax:
        conf = Softmax()(conf)
    prediction = Concatenate()([conf,loc_pred])
    return prediction

def build_RFBNet( input_shape,
            phase,
            source_layers,
            base_model,
            extra_config,
            prior_config,
            num_classes,
            mean_color,
            swap_channels,
            use_l2 = 0.00005,
            return_predictor = False):
        """Applies network layers and ops on input image(s) x.
                     
           Source_layers and
        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """        
        sources = list()
        prediction_layers = list()
        
        for layer_name in source_layers:
            source_layer = base_model.get_layer(layer_name)
            out = BasicRFB_a(source_layer.output,512,stride = 1,scale=1.0,use_l2 = use_l2)            
            sources.append(out)
        
        last_layer_output = base_model.output
        sources.append(last_layer_output)
        sources += add_extra_layers(last_layer_output,extra_config,use_l2 = use_l2)
        predictor_size = []
        for k,x in enumerate(sources):
            predictor_size.append(x.get_shape().as_list()[1:3])
            prediction_layer = multibox(x,prior_config[k],num_classes,softmax=True)
            prediction_layers.append(prediction_layer)
         
        output = Concatenate(axis = 1)(prediction_layers)
        model = Model(inputs=base_model.input, outputs=output)
        #returns = [model]
#         x = Input(input_shape)
#         model = preprocess(x,model,mean_color,swap_channels)
        if return_predictor:
            print(np.array(predictor_size))
        return model
    
def preprocess(x,model,mean_color,swap_channels):
        def input_mean_normalization(tensor):
            return tensor - np.array(mean_color)

        def input_channel_swap(tensor):
            if len(swap_channels) == 3:
                return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
            elif len(swap_channels) == 4:
                return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)


        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
        input_shape = x.get_shape().as_list()
      
        x1 = x
        if not (mean_color is None):
            x1 = Lambda(input_mean_normalization, output_shape= input_shape, name='input_mean_normalization')(x1)
        if not (swap_channels is None):
            x1 = Lambda(input_channel_swap, output_shape=input_shape, name='input_channel_swap')(x1)
        x2 = model(x1)
        new_model = Model(x,x2)
        return new_model
def build_RFB_Mobilev2_300(phase,
                  mean_color,
                  swap_channels,
                  aspect_ratios,
                  num_classes,
                  return_predictor = False):
    mobilev2_300_path = 'mobilenet_300x300.h5'
    if not os.path.isfile(mobilev2_300_path):
        print('Transfering weights from old msodel(224x224x3) to new model(300x300x3)')
        base_model = keras.applications.mobilenetv2.MobileNetV2(input_shape =(224,224,3),include_top=False,weights='imagenet')
        #print(x.get_shape())
        #print(K.is_keras_tensor(x))
        new_model = keras.applications.mobilenetv2.MobileNetV2(input_shape = (300,300,3),include_top=False,weights=None)
        new_model.set_weights(base_model.get_weights())
        #         for new_layer, layer in zip(new_model.layers[1:], base_model.layers[1:]):
#             new_layer.set_weights(layer.get_weights())
        new_model.save(mobilev2_300_path)
#         base_model = new_model
    else:
        base_model = load_model(mobilev2_300_path)
#     base_model = keras.applications.mobilenetv2.MobileNetV2(input_shape =(300,300,3),include_top=False,weights=None)
    prior_config =  [2 + len(ar) * 2 for ar in aspect_ratios]  # number of boxes per feature map location
    model = build_RFBNet(input_shape = (300,300,3),
                  phase = phase,
                  source_layers = ['block_12_add'],
                  base_model = base_model,
                  extra_config = ['S',512],
                  prior_config = prior_config,
                  num_classes = num_classes,
                  mean_color = mean_color,
                  swap_channels = swap_channels,
                  return_predictor = return_predictor)
    return model
