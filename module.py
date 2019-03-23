import keras
from keras.layers import Conv2D,SeparableConv2D,DepthwiseConv2D,Dense,Input,Flatten,Reshape,MaxPooling2D,AveragePooling2D,Activation,Softmax
from keras.layers import Concatenate,Add,Multiply,Lambda
from keras.layers import BatchNormalization 
from keras.models import Model,Sequential
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras import optimizers
from keras import backend as K
from keras import regularizers

def BasicConv(x,
          out_planes,
          kernel_size,
          stride=1,
          padding='same',
          dilation=1,
          kernel_initializer='he_normal',
          activation='relu',
          name =None,
          use_l2=False, 
          use_bn=True, 
          use_bias=False):
    '''
    Basic Convolution Layer
    '''
    l2_reg = kernel_regularizer=regularizers.l2(0.005) if use_l2 else None
    conv = Conv2D(out_planes,
              kernel_size=kernel_size,
              strides=stride,
              padding=padding,
              dilation_rate=dilation,
              activation=activation, use_bias=use_bias,
              kernel_initializer=kernel_initializer, 
              kernel_regularizer = l2_reg,
              name = name,
              data_format='channels_last')

    x = conv(x)
    #x.add(conv)
    if  use_bn:
        bn = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.99)
        x = bn(x)
        #x.add(bn),
    return x

def BasicDepConv(x, kernel_size, stride=1, padding='same', dilation=1, activation='relu',use_l2=False, use_bn=True, use_bias=False):
    '''
    Seperable Convolution Layer
    '''
    l2_reg = kernel_regularizer=regularizers.l2(0.005) if use_l2 else None
    conv = DepthwiseConv2D( kernel_size=kernel_size, strides=stride, padding=padding, 
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

def BasicSepConv(x, out_planes, kernel_size, stride=1, padding='same', dilation=1, activation='relu',use_l2=False, use_bn=True, use_bias=False):
    '''
    Seperable Convolution Layer
    '''
    l2_reg = kernel_regularizer=regularizers.l2(0.005) if use_l2 else None
    conv = SeparableConv2D( out_planes,kernel_size=kernel_size, strides=stride, padding=padding, 
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

def BasicRFB(x, out_planes, stride=1,dilation_base = 1, scale = 0.1,use_l2 = False,lite = False, pooling='maxpooling'):
    '''
    Basic RFB module
    Modified:
    1. All padding used same
    2. Add pooling to shortcut to match stride
    '''
    conv1 = BasicConv
    conv2 = BasicSepConv
    if lite:
        conv1 = BasicSepConv
    #scale = (scale,scale,scale,scales
    in_planes = x.get_shape().as_list()[3]
    inter_planes = in_planes // 8
    #original branch 0'
    x0 = conv1(x, 2*inter_planes, kernel_size=1, stride=stride, use_l2=use_l2)
    x0 = conv1(x0, 2*inter_planes, kernel_size=3, stride=1, padding='same', dilation=dilation_base, activation = None,use_l2 = use_l2)
    
    #original branch 1'
    x1 = conv1(x,inter_planes, kernel_size=1, stride=1,use_l2 = use_l2)
    x1 = conv1(x1,(inter_planes//2)*3, kernel_size=(1,3), stride=1, padding='same',use_l2 = use_l2)
    x1 = conv1(x1,(inter_planes//2)*3, kernel_size=(3,1), stride=stride, padding='same',use_l2 = use_l2)
    x1 = conv2(x1,(inter_planes//2)*3, kernel_size=3, stride=1, padding='same', dilation=dilation_base + 2, activation= None,use_l2 = use_l2)
    #print(x0.shape, x1.shape)
    #original branch 2
    x2 = conv1(x,inter_planes, kernel_size=1, stride=1,use_l2 = use_l2)
    x2 = conv1(x2,(inter_planes//2)*3, kernel_size=3, stride=1, padding='same',use_l2 = use_l2)
    x2 = conv1(x2, (inter_planes//2)*3, kernel_size=3, stride=stride,use_l2 = use_l2)
    x2 = conv2(x2,(inter_planes//2)*3, kernel_size=3, stride=1, dilation=dilation_base + 4, activation= None,use_l2 = use_l2)        
    
    out = Concatenate(axis=-1)([x0,x1,x2])
    #Original Conv Linear 
    out =  conv1(out, out_planes, kernel_size=1, stride=1, activation= None,use_l2 = use_l2)
    out = Lambda(lambda x:x*scale )(out)
    if  in_planes != out_planes:
        x = conv1(x,out_planes,kernel_size=1,stride=1,activation = None,use_bn = False,use_l2 = use_l2)
    if stride !=1 :
        if pooling == 'maxpooling':
            x = MaxPooling2D(stride,padding='same')(x)
        elif pooling == 'averagepooling':
            x = AveragePooling2D(stride,padding='same')(x)
    out = Add()([out,x])
    out = Activation('relu')(out)
    return out

def BasicRFB_a(x, out_planes, stride=1, dilation_base=1, scale = 0.1,use_l2 = False, lite = False,pooling='maxpooling'):
    #|assert(in_planes==out_planes)
    conv1 = BasicConv
    conv2 = BasicSepConv
    if lite:
        conv1 = BasicSepConv
    in_planes = x.get_shape().as_list()[3]
    inter_planes = in_planes //4
    #print(x)
    x0 = conv1(x, inter_planes, kernel_size=1, stride=1,use_l2 = use_l2)
    x0 = conv2(x0, inter_planes, kernel_size=3, stride=1, dilation=dilation_base, activation = None,use_l2 = use_l2) #maybe should not use bn

    x1 = conv1(x, inter_planes, kernel_size=1, stride=1,use_l2 = use_l2)
    x1 = conv1(x1, inter_planes, kernel_size=(3,1), stride=1,use_l2 = use_l2)
    x1 = conv2(x1, inter_planes, kernel_size=3, stride=1, dilation=dilation_base + 2, activation = None,use_l2 = use_l2)


    x2 = conv1(x, inter_planes, kernel_size=1, stride=1,use_l2 = use_l2)
    x2 = conv1(x2, inter_planes, kernel_size=(1,3), stride=1,use_l2 = use_l2)
    x2 = conv2(x2, inter_planes, kernel_size=3, stride=1,  dilation=dilation_base +2, activation = None,use_l2 = use_l2)

    x3 = conv1(x, inter_planes//2, kernel_size=1, stride=1,use_l2 = use_l2)
    x3 = conv1(x3, (inter_planes//4)*3, kernel_size=(1,3), stride=1,use_l2 = use_l2)
    x3 = conv1(x3, inter_planes, kernel_size=(3,1), stride=1,use_l2 = use_l2)
    x3 = conv2(x3, inter_planes, kernel_size=3, stride=1,dilation=dilation_base+4, activation = None,use_l2 = use_l2)

    out = Concatenate()([x0,x1,x2,x3])
    out = conv1(out,  out_planes, kernel_size=1, stride=stride, activation = None,use_l2 = use_l2) #
    out = Lambda(lambda x:x * scale)(out)
    
    #Add extras pooling and point convolution to make it possible for different input and output shape
    if  in_planes != out_planes:
        x = conv1 (x,out_planes,kernel_size=1,stride=1,activation = None,use_bn = False,use_l2 = use_l2)
    if stride !=1 :
        x = MaxPooling2D(stride)(x)
   
    out = Add()([out,x])
    out = Activation('relu')(out)

    return out

