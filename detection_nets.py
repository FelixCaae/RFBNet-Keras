import module
from keras.layers import Conv2D,SeparableConv2D,Dense,Input,Flatten,Reshape,MaxPooling2D,AveragePooling2D,Activation,Softmax
from keras.layers import Concatenate,Add,Multiply,Lambda
from keras.layers import BatchNormalization,Dropout 
from keras.models import Model,Sequential

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
#    x = BasicFC(base_model.get_layer('out_relu').output, 4 + num_classes, 'relu')
#    x = Dense(4 + num_classes,activation = 'relu')(base_model.output)
    x = base_model.get_layer(source_layer).output
    x = AveragePooling2D(strides=2)(x)
    x = Flatten()(x)
#    x = Dense(2048,activation = 'relu')(x)
#    x = Dense(1024,activation = 'relu')(x)
    x = Dense(500,activation = 'relu')(x)
    x = Dropout(0.3)(x)
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
    x1 = Dropout(0.5)(x1)
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


def build_RFBNet(source_layers, base_model, extras_config,prior_config,num_classes):
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
        # self.base = nn.ModuleList(base)
        #self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        #self.extras = nn.ModuleList(extras)

        #self.loc = nn.ModuleList(head[0])
        #self.conf = nn.ModuleList(head[1])
        #if self.phase == 'test':
        #   self.softmax = nn.Softmax(dim=-1)
        
        sources = list()
        predictions = list()
        
        for layer_name in source_layers:
            source_layer = base_model.get_layer(layer_name)
            out = BasicRFB_a(source_layer.output,512,stride = 1,scale=1.0)            
            sources.append(out)
        
        base_output = base_model.get_layer("out_relu").output
        sources.append(base_output)

        # apply extra layers and cache source layer outputs
        #print(out.get_shape())
        sources += add_extras(base_output,extras_config)
        
        print("sources num", len(sources))
        for k,x in enumerate(sources):
            print(prior_config[k])
            prediction = multibox(x,prior_config[k],num_classes)
            predictions.append(prediction)
        
        output = Concatenate(axis = 1)(predictions)
        
        model = Model(inputs=base_model.input, outputs=output)
        return 'RFB_mobilet_net',model