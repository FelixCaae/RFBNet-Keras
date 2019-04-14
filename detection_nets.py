from module import *
from keras.layers import *
from keras.models import Model,Sequential,load_model
import keras.applications.mobilenetv2 
from keras import backend as K
import numpy as np
import os
import warnings

def load_mobilenetv2(size =300):
    mobilev2_300_path = 'mobilenet_300x300.h5'
    if not os.path.isfile(mobilev2_300_path):
        print('Transfering weights from old model(224x224x3) to new model(300x300x3)')
        base_model = keras.applications.mobilenetv2.MobileNetV2(input_shape =(224,224,3),include_top=False,weights='imagenet')
        #print(x.get_shape())
        #print(K.is_keras_tensor(x))
        new_model = keras.applications.mobilenetv2.MobileNetV2(input_shape = (300,300,3),include_top=False,weights=None)
        new_model.set_weights(base_model.get_weights())
        #         for new_layer, layer in zip(new_model.layers[1:], base_model.layers[1:]):
#             new_layer.set_weights(layer.get_weights())
        new_model.save(mobilev2_300_path)
        base_model = new_model
    else:
        base_model = load_model(mobilev2_300_path)
    return base_model


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


# def add_up_sampling_layers(source_layers):
#     merged_layer_0 = source_layers[-1]
#     #19 10 5 3 3 1
# #     for layer in source_layers:
#     upsampled_layer_1 = UpSampling2D(3)(merged_layer_0)
#     conv_layer_1 = BasicSepConv(128,kernel_size = 3,stride = 1,padding ='same')(upsampled_layer_1)
#     merged_layer_1 = Concatenate()([source_layers[-2],conv_layer_1])
#     conv_layer_2 = BasicSepConv(128,kernel_size = 3,stride = 1,padding ='same')(merged_layer_1)
#     merged_layer_2 = Concatenate(axis = 3)([source_layers[-3],conv_layer_2])
#     upsampled_layer_3 = UpSampling2D(2)(merged_layer_2)
#     cropped_layer_3 = Cropping2D(((0,1),(0,1)))(upsampled_layer_3)
#     conv_layer_3 = BasicSepConv(cropped_layer_3,128,kernel_size=3,stride=1,padding='same')
#     merged_layer_3 = Concatenate(axis = 3)([source_layers[-4],conv_layer_3])
#     upsampled_layer_4 = UpSampling2D(2)(merged_layer_3)
#     conv_layer_4 = BasicSepConv(upsampled_layer_4,128,kernel_size=3,stride=1,padding='same')
#     merged_layer_4 = Concatenate(axis = 3)([source_layers[-5],conv_layer_4])
#     upsampled_layer_5 = UpSampling2D(2)(merged_layer_4)
#     cropped_layer_5 = Cropping2D(((0,1),(0,1)))(upsampled_layer_5)
#     conv_layer_5 = BasicSepConv(cropped_layer_5,128,kernel_size=3,stride=1,padding='same')
#     merged_layer_5 = Concatenate(axis = 3)([source_layers[-6],conv_layer_5])
    
    
# #     merged_layers = [merged_layer_5,merged_layer_4,merged_layer_3,merged_layer_2,merged_layer_1,merged_layer_0]
#     return merged_layers

        

def multibox(features, prior_config, num_classes,softmax = True,lite = True):
    #Input a feature map layer , add locating layer and classificaton layer upon
    #Output shape should be (batch_size,#boxes,num_classes + 4)
    num_classes = num_classes + 1
    conv = BasicConv
    if lite:
        conv = BasicSepConv
    loc_all = []
    conf_all = []
    i = 0
    for x,prior_num in zip(features,prior_config):
        loc = conv(prior_num * 4,kernel_size = 3,name = 'loc_' + str(i))(x)
        loc = Reshape((-1,4),name = 'reshape_loc_'+ str(i) )(loc)
        conf = conv(prior_num * num_classes, kernel_size = 3,name = 'conf_' + str(i))(x)
        conf = Reshape((-1, num_classes),name = 'reshape_conf_' + str(i))(conf)
        loc_all.append(loc)
        conf_all.append(conf)
        i += 1
    loc_all = Concatenate(axis = 1,name = 'mbox_loc_merge')(loc_all)
    conf_all = Concatenate(axis = 1,name = 'mbox_conf_merge')(conf_all)
    if softmax:
        conf_all = Softmax()(conf_all)
    prediction = Concatenate(name = 'mbox_pred_merge')([conf_all,loc_all])
    return prediction

def build_RFBLite( input_shape,
            phase,
            source_layers,
            base_model,
            extra_config,
            source_config,
            include_base,
            prior_config,
            num_classes,
            source_expand_ratio = 6,
            source_dep_mul = 4,
            base_index = -4,
            lite=False,
            show_summary = False,
            l2_reg = 0.00005,
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
        def add_extra_layers(x, extra_cfg,lite, l2_reg=0.00005):
            source_layers = []
            flag = False
            conv = BasicConv
            if lite:
                conv = BasicSepConv
            for k, cfg in enumerate(extra_cfg):
                if cfg[0] == 'literfb_d':
                    filters,dil_base,dil_rate,only_bone,source = cfg [1:]
                    x = LiteRFB_d(x,
                             filters,
                             'rfb_extra_'+str(k),
                             l2_reg=l2_reg,
                             dilation_base=dil_base,
                             dilation_rate=dil_rate,
                             only_bone=only_bone)
                else:
                    print(cfg)
                    filters,stride,padding,source = cfg
                    x = conv(filters,kernel_size = 3,stride =stride,padding = padding, name = 'plain_extra_' + str(k) )(x)
                if source:
                    source_layers += [x]
            return source_layers
        
        features = list()

        # 1. Add feature map from mid model into sources
        for layer_name in source_layers:
            x = base_model.get_layer(layer_name).output
            features.append(x)
        
        # 2. Add extra layers from base_model output
        last_layer_output = base_model.layers[base_index].output
        if include_base:
            features.append(last_layer_output)
        features += add_extra_layers(last_layer_output,extra_config,l2_reg = l2_reg,lite=lite)
        # 3. Add feature pyramid structure
#         sources = add_up_sampling_layers(sources)
        
        # 4. Add RFBLite for layer from sources
        for k,cfg in enumerate(source_config):
            if cfg[0] == 'literfb_d':
                filters,only_bone,dil_base,dil_rate = cfg [1:]
                features[k] = LiteRFB_d(features[k],
                                   filters, 
                                   'rfb_source_'+str(k),
                                   l2_reg=l2_reg,
                                   dilation_base=dil_base,
                                   dilation_rate=dil_rate,
                                   only_bone=only_bone)  
            else:
                filters,kernel_size,stride = cfg
                features[k] = BasicSepConv(filters,kernel_size = 3,stride =stride,padding = padding, name = 'plain_extra_' + str(k) )(features[k])
            
        # 5. Add prediction layers 
        output = multibox(features,prior_config,num_classes,lite=True,softmax=True)
        model = Model(inputs=base_model.input, outputs=output)
        if return_predictor:
            print(np.array([feature.shape.as_list()[1:3] for feature in features]))
        return model

def build_tiny_dsod(input_shape,
              num_classes,
              prior_config,
              model_config,
              return_predictor = True):
    inp = Input(input_shape)
    stem = Sequential([BasicConv(64,kernel_size = 3,stride = 2),
                 BasicConv(64,kernel_size = 1,stride = 1),
                 BasicDepConv(kernel_size = 3,stride = 1),
                 BasicConv(128,kernel_size = 1,stride = 1),
                 BasicDepConv(kernel_size = 3,stride = 1),
                 MaxPooling2D(pool_size = 2,strides = 2,padding='same')],name = 'stem')
    x = stem(inp)
    def dense_stage(x, g, n):
        for i in range(0,n):
            x_conv = BasicConv(g,kernel_size = 1)(x)
            x_dw = BasicDepConv(kernel_size = 3)(x_conv)
            x =  Concatenate()([x,x_dw])
        return x 
    def transition(c, p):
        transition_layer = Sequential()
        transition_layer.add(BasicConv(c,kernel_size=1))
        if p:
            transition_layer.add(MaxPooling2D(pool_size = 2,strides = 2,padding='same'))
        return transition_layer
    def down_sample(x):
        x_0 = MaxPooling2D(pool_size = 2,strides = 2,padding = 'same')(x)
        x_0 = BasicConv(64, kernel_size = 1)(x_0)
        x_1 = BasicConv(64, kernel_size = 1)(x)
        x_1 = BasicDepConv(kernel_size = 3,stride = 2,padding = 'same')(x_1)
        return Concatenate()([x_0,x_1])
    def up_sample(x):
        x = UpSampling2D(2)(x)
        x = BasicDepConv(kernel_size = 3)(x)
        return x
    def pad(x):
        if int(x.shape[2])%2 != 0:
            x = ZeroPadding2D(((0,1),(0,1)))(x)
        return x
    def crop(x0,x1):
        if int(x0.shape[1]) != int(x1.shape[1]):
            x0 = Cropping2D(((0,1),(0,1)))(x0)
        return x0
    
    transition_layers = []
    for g,n,c,p in model_config:
        dense = dense_stage(x,g,n)
        x = transition(c,p)(dense)
        transition_layers.append(x)
    down_layers = [transition_layers[1],transition_layers[3]]
    x = Concatenate()([down_sample(transition_layers[1]), transition_layers[3]])
    for i in range(4):
        x = down_sample(x)
        down_layers.append(x)
    x = down_layers[-1]
    features = [x]
    for layer in down_layers[-2::-1]:
        x = up_sample(x)
        x = Add()([crop(x,layer),layer])
        features.append(x)
    features.reverse()
    prediction = multibox(features,prior_config,num_classes)
    model = Model(inputs=inp, outputs=prediction)
    if return_predictor:
        print(np.array([feature.shape.as_list()[1:3] for feature in features]))
    return model
def build_ssdlite(base_model,
            prior_config,
            source_layer_name_1,
            num_classes):
    base_model = load_mobilenetv2()
#     source_layer_name_1 = 'block_16_expansion'
    source_layer_1 = base_model.get_layer(source_layer_name_1).output
    source_layer_2 = base_model.layers[-1].output
    source_layer_3 = BasicSepConv(512,kernel_size = 3,stride = 2, padding='same',name = 'extra_3')(source_layer_2)
    source_layer_4 = BasicSepConv(256,kernel_size = 3,stride = 2, padding='same',name = 'extra_4')(source_layer_3)
    source_layer_5 = BasicSepConv(256,kernel_size = 3,stride = 1,padding='same',name = 'extra_5')(source_layer_4)
    source_layer_6 = BasicSepConv(128,kernel_size = 3,stride = 1, padding='valid',name = 'extra_6')(source_layer_5)
    source_layers = [source_layer_1,source_layer_2,source_layer_3,source_layer_4,source_layer_5,source_layer_6]
    prediction_layers = []
    prediction = multibox(source_layers,prior_config,num_classes,softmax=True,lite=True)
    model = Model(inputs=base_model.input, outputs=prediction)
    return model

def build_ssdlite_L(base_model,
            prior_config,
            source_layer_name_1,
            source_layer_name_2,
            num_classes):
    base_model = load_mobilenetv2()
#     source_layer_name_1 = 'block_16_expansion'
    source_layer_1 = base_model.get_layer(source_layer_name_1).output
    source_layer_2 = base_model.get_layer(source_layer_name_2).output
    base_layer = base_model.layers[-4].output
    source_layer_3 = BasicSepConv(512,kernel_size = 3,stride = 1, padding='same',name = 'extra_3')(base_layer)
    source_layer_4 = BasicSepConv(256,kernel_size = 3,stride = 2, padding='same',name = 'extra_4')(source_layer_3)
    source_layer_5 = BasicSepConv(256,kernel_size = 3,stride = 2,padding='same',name = 'extra_5')(source_layer_4)
    source_layer_6 = BasicSepConv(128,kernel_size = 3,stride = 1, padding='valid',name = 'extra_6')(source_layer_5)
    source_layers = [source_layer_1,source_layer_2,source_layer_3,source_layer_4,source_layer_5,source_layer_6]
    prediction_layers = []
    prediction = multibox(source_layers,prior_config,num_classes,softmax=True,lite=True)
    model = Model(inputs=base_model.input, outputs=prediction)
    return model


def preprocess(input_shape,model,mean_color,swap_channels):
        def input_mean_normalization(tensor):
            return tensor - np.array(mean_color)

        def input_channel_swap(tensor):
            if len(swap_channels) == 3:
                return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
            elif len(swap_channels) == 4:
                return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)


        # The following identity layer is only needed so that the subsequent lambda layers can be optional.
#         input_shape = x.get_shape().as_list()
      
        x = Input(input_shape)
        x1 = x
        if not (mean_color is None):
            x1 = Lambda(input_mean_normalization, output_shape= input_shape, name='input_mean_normalization')(x1)
        if not (swap_channels is None):
            x1 = Lambda(input_channel_swap, output_shape=input_shape, name='input_channel_swap')(x1)
        x2 = model(x1)
        new_model = Model(x,x2)
        return new_model