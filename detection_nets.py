import modules
def build_simple_detection_net(base_model,num_classes):
    x = BasicFC(base_model.output, 4 + num_classes, 'sigmoid')
    model = Model(inputs = base_model.input, outputs = x)
    
    return 'simple_detection_net',model

def build_simple_detection_net2(base_model,source_layer,num_classes):
#    x = BasicFC(base_model.get_layer('out_relu').output, 4 + num_classes, 'relu')
#    x = Dense(4 + num_classes,activation = 'relu')(base_model.output)
    x = Flatten()(base_model.get_layer(source_layer).output)
    x = Dense(4096,activation = 'relu')(x)
    x = Dense(1024,activation = 'relu')(x)
    x = Dense(4 + num_classes,activation ='sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = x)
    
    return 'simple_detection_net_2',model


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