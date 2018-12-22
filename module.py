import keras
from keras.layers import Conv2D,SeparableConv2D,Dense,Input,Flatten,Reshape,MaxPooling2D,AveragePooling2D,Activation,Softmax
from keras.layers import Concatenate,Add,Multiply,Lambda
from keras.layers import BatchNormalization 
from keras.models import Model,Sequential
from matplotlib.image import imread
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras import optimizers
import keras.applications
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras import backend as K
from keras import regularizers

def BasicConv(x,out_planes, kernel_size, stride=1, padding='same', dilation=1, activation='relu',use_l2=False, use_bn=True, use_bias=False):
    '''
    Basic Convolution Layer
    '''
    l2_reg = kernel_regularizer=regularizers.l2(0.005) if use_l2 else None
    conv = Conv2D(out_planes, kernel_size=kernel_size, strides=stride, padding=padding,
              dilation_rate=dilation,activation=activation, use_bias=use_bias,
              kernel_regularizer = l2_reg,
              data_format='channels_last')

    x = conv(x)
    #x.add(conv)
    if  use_bn:
        bn = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.99)
        x = bn(x)
        #x.add(bn),
    return x

def BasicSepConv(x,out_planes, kernel_size, stride=1, padding='same', dilation=1, activation='relu',use_l2=False, use_bn=True, use_bias=False):
    '''
    Seperable Convolution Layer
    '''
    l2_reg = kernel_regularizer=regularizers.l2(0.005) if use_l2 else None
    conv = SeparableConv2D(out_planes, kernel_size=kernel_size, strides=stride, padding=padding, 
                           kernel_regularizer = l2_reg,
                           dilation_rate=dilation,activation=activation, 
                           use_bias=use_bias,data_format='channels_last')
    x = conv(x)
    #x.add(conv)
    if  use_bn:
        bn = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.01)
        x = bn(x)
        #x.add(bn)
    return x
    

def BasicFC(x,out_num,activation='softmax'):
    x = Flatten()(x)
    x = Dense(out_num,activation=activation)(x)
    return x

def BasicRFB(x, out_planes, stride=1, scale = 0.1):
    '''
    Basic RFB module
    Modified:
    1. All padding used same
    2. Add pooling to shortcut to match stride
    '''
    #scale = (scale,scale,scale,scales
    in_planes = x.get_shape().as_list()[3]
    inter_planes = in_planes // 8
    #original branch 0'
    x0 = BasicConv(x, 2*inter_planes, kernel_size=1, stride=stride)
    x0 = BasicConv(x0, 2*inter_planes, kernel_size=3, stride=1, padding='same', dilation=1, activation = None)
    
    #original branch 1'
    x1 = BasicConv(x,inter_planes, kernel_size=1, stride=1)
    x1 = BasicConv(x1,(inter_planes//2)*3, kernel_size=(1,3), stride=1, padding='same')
    x1 = BasicConv(x1,(inter_planes//2)*3, kernel_size=(3,1), stride=stride, padding='same')
    x1 = BasicSepConv(x1,(inter_planes//2)*3, kernel_size=3, stride=1, padding='same', dilation=3, activation= None)
    print(x0.shape, x1.shape)
    #original branch 2
    x2 = BasicConv(x,inter_planes, kernel_size=1, stride=1)
    x2 = BasicConv(x2,(inter_planes//2)*3, kernel_size=3, stride=1, padding='same')
    x2 = BasicConv(x2, (inter_planes//2)*3, kernel_size=3, stride=stride)
    x2 = BasicSepConv(x2,(inter_planes//2)*3, kernel_size=3, stride=1, dilation=5, activation= None)        
    
    out = Concatenate(axis=-1)([x0,x1,x2])
    #Original Conv Linear 
    out =  BasicConv(out, out_planes, kernel_size=1, stride=1, activation= None)
    out = Lambda(lambda x:x*scale )(out)
    if  in_planes != out_planes:
        x = BasicConv(x,out_planes,kernel_size=1,stride=1,activation = None,use_bn = False)
    if stride !=1 :
        x = MaxPooling2D(stride,padding='same')(x)
    out = Add()([out,x])
    out = Activation('relu')(out)
    return out

def BasicRFB_a(x, out_planes, stride=1, scale = 0.1):
    #|assert(in_planes==out_planes)
    in_planes = x.get_shape().as_list()[3]
    inter_planes = in_planes //4
    #print(x)
    x0 = BasicConv(x, inter_planes, kernel_size=1, stride=1)
    x0 = BasicSepConv(x0, inter_planes, kernel_size=3, stride=1, dilation=1, activation = None) #maybe should not use bn

    x1 = BasicConv(x, inter_planes, kernel_size=1, stride=1)
    x1 = BasicConv(x1, inter_planes, kernel_size=(3,1), stride=1)
    x1 = BasicSepConv(x1, inter_planes, kernel_size=3, stride=1, dilation=3, activation = None)


    x2 = BasicConv(x, inter_planes, kernel_size=1, stride=1)
    x2 = BasicConv(x2, inter_planes, kernel_size=(1,3), stride=1)
    x2 = BasicSepConv(x2, inter_planes, kernel_size=3, stride=1,  dilation=3, activation = None)

    x3 = BasicConv(x, inter_planes//2, kernel_size=1, stride=1)
    x3 = BasicConv(x3, (inter_planes//4)*3, kernel_size=(1,3), stride=1)
    x3 = BasicConv(x3, inter_planes, kernel_size=(3,1), stride=1)
    x3 = BasicSepConv(x3, inter_planes, kernel_size=3, stride=1,dilation=5, activation = None)

    out = Concatenate()([x0,x1,x2,x3])
    out = BasicConv(out,  out_planes, kernel_size=1, stride=stride, activation = None) #
    out = Lambda(lambda x:x * scale)(out)
    
    #Add extras pooling and point convolution to make it possible for different input and output shape
    if  in_planes != out_planes:
        x = BasicConv(x,out_planes,kernel_size=1,stride=1,activation = None,use_bn = False)
    if stride !=1 :
        x = MaxPooling2D(stride)(x)
   
    out = Add()([out,x])
    out = Activation('relu')(out)

    return out

def add_extras(x, cfg, batch_norm=False):
    source_layers = []
    flag = False
    for k, v in enumerate(cfg):
        if v == 'S':
            x = BasicRFB(x, cfg[k+1], stride=2, scale = 1.0)
            source_layers += [x]
        else:
            x = BasicRFB(x, v, scale = 1.0)
            
    x = BasicConv(x,128,kernel_size=1,stride=1) 
    source_layers += [x]
    x = BasicConv(x,256,kernel_size=3,stride=2)
    x = BasicConv(x,128,kernel_size=1,stride=1)
    source_layers += [x]
    x = BasicConv(x,256,kernel_size=3,stride=2)
    x = BasicConv(x,64,kernel_size=1,stride=1)
    source_layers += [x]
    #x = BasicConv(x,128,kernel_size=3,stride=2)

    return source_layers

def multibox(x, prior_num, num_classes):
    #Input a feature map layer , add locating layer and classificaton layer upon
    #Output shape should be (batch_size,#boxes,num_classes + 4)
    (batch_size,m_H,m_W,m_C) = x.get_shape().as_list()
    loc_pred = Conv2D(prior_num * 4,kernel_size = 1 ,padding= 'valid')(x)
    loc_pred = Reshape((m_H * m_W * prior_num,4))(loc_pred)
    conf = Conv2D(prior_num * num_classes, kernel_size = 1,padding = 'valid')(x)
    conf = Reshape((m_H * m_W * prior_num, num_classes))(conf)
    conf = Softmax()(conf)
    prediction = Concatenate()([conf,loc_pred])
    print('prediction shape',prediction.get_shape())
    return prediction