import keras
from keras.layers import *
from keras.models import Model,Sequential
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras import optimizers
from keras import backend as K
from keras import regularizers
class BasicModule():
    def __call__(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
#         if self.avt is not None:
#             x = self.avt(x)
        return x
    def get_layers(self):
        layers = [self.conv]
        if self.bn is not None:
            layers.append(self.bn)
        if self.avt is not None:
            layers.append(self.avt)
        return layers
    
class BasicConv(BasicModule):
    def __init__(self,
          out_planes,
          kernel_size=3,
          stride=1,
          padding='same',
          dilation=1,
          kernel_initializer='he_normal',
          activation='relu',
          name = None,
          l2_reg=0.00005, 
          use_bn=True, 
          use_bias=False):
        l2 = regularizers.l2(l2_reg) if (l2_reg is not None) else None
        if name is not None:
            conv_name = name + '_conv'
            bn_name = name + '_bn'
            avt_name = name + '_' + activation
        else:
            conv_name = None
            bn_name = None
            avt_name = None
         
        self.conv = Conv2D( out_planes,
              kernel_size=kernel_size,
              strides=stride,
              padding=padding,
              dilation_rate=dilation,
              activation=activation, 
              use_bias=use_bias,
              kernel_initializer=kernel_initializer, 
              kernel_regularizer = l2,
              name = conv_name,
              data_format='channels_last')
        self.bn = None
        self.avt = None
        if use_bn:
            self.bn = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.99,name = bn_name) 
        if activation is not None:
            self.avt = Activation(activation,name = avt_name)
class BasicSepConv(BasicModule):
    def __init__(self,
          out_planes,
          kernel_size=3,
          stride=1,
          padding='same',
          dilation=1,
          kernel_initializer='he_normal',
          activation='relu',
          name = None,
          l2_reg=0.00005, 
          depth_multiplier = 1,
          use_bn=True, 
          use_bias=False):
        l2 = regularizers.l2(l2_reg) if (l2_reg is not None) else None
        if name is not None:
            conv_name = name + '_conv'
            bn_name = name + '_bn'
            avt_name = name + '_' + activation
        else:
            conv_name = None
            bn_name = None
            avt_name = None
         
        self.conv = SeparableConv2D( out_planes,
              kernel_size=kernel_size,
              strides=stride,
              padding=padding,
              dilation_rate=dilation,
              activation=activation, 
              use_bias=use_bias,
              kernel_initializer=kernel_initializer, 
              kernel_regularizer = l2,
              depth_multiplier = depth_multiplier,
              name = conv_name,
              data_format='channels_last')
        self.bn = None
        self.avt = None
        if use_bn:
            self.bn = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.99,name = bn_name) 
        if activation is not None:
            self.avt = Activation(activation,name = avt_name)
                      
class BasicDepConv(BasicModule):
    def __init__(self,
             kernel_size=3,
             stride=1, 
             padding='same',
             dilation=1,
             activation='relu',
             l2_reg=0.00005, 
             depth_multiplier=1,
             name = None,
             use_bn=True,
             use_bias=False):
    #Seperable Convolution Layer
        l2 = regularizers.l2(l2_reg) if (l2_reg is not None) else None
        if name is not None:
            conv_name = name + '_dwconv'
            bn_name = name + '_bn'
            avt_name = name + '_' + activation
        else:
            conv_name = None
            bn_name = None
            avt_name = None
    #     if name is None:
    #         name = 'BasicDepthwiseConv'
        self.conv = DepthwiseConv2D( kernel_size=kernel_size,
                           strides=stride,
                           padding=padding, 
                           kernel_regularizer = l2,
                           dilation_rate=dilation,
                           activation=activation, 
                           use_bias=use_bias,
                           depth_multiplier =1,
                           name = conv_name,
                           data_format='channels_last')
        self.bn = None
        self.avt = None
        if  use_bn:
            self.bn = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.99,name = bn_name) 
        if activation is not None:
            self.avt = Activation(activation,name = avt_name)
            
# class BasicSepConv(BasicModule):
#     def __init__(self,
#              out_planes, 
#              kernel_size = 3,
#              stride=1,
#              padding='same', 
#              dilation=1, 
#              activation='relu',
#              l2_reg=0.00005, 
#              use_bn=True, 
#              name =None,
#              use_bias=False):
#         '''
#          Seperable Convolution Layer
#         '''
#         l2 = regularizers.l2(l2_reg) if (l2_reg is not None) else None
#         if name is not None:
#             conv_1_name = name + '_dwconv'
#             bn_1_name = name + '_dw_bn'
#             avt_1_name = name + '_dw_' + activation
#             conv_2_name = name + '_1x1conv'
#             bn_2_name = name + '_1x1_bn'
#             avt_2_name = name + '_1x1_' + activation
#         else:
#             conv_1_name = None
#             bn_1_name = None
#             avt_1_name = None
#             conv_2_name = None
#             bn_2_name = None
#             avt_2_name = None
            
#         self.bn_1 = None
#         self.avt_1 = None
#         self.bn_2 = None
#         self.avt_2 = None
#         self.conv_1 = DepthwiseConv2D( 
#                            kernel_size=kernel_size, 
#                            strides=stride,
#                            padding=padding, 
#                            kernel_regularizer = l2,
#                            dilation_rate=dilation,
#                            activation=None, 
#                            use_bias=use_bias,
#                            name=conv_1_name,
#                            data_format='channels_last')
#         self.conv_2  =    Conv2D(out_planes, 
#                            kernel_size=1, 
#                            strides=1,
#                            padding=padding, 
#                            kernel_regularizer = l2,
#                            dilation_rate=dilation,
#                            activation=None, 
#                            use_bias=use_bias,
#                            name=conv_2_name,
#                            data_format='channels_last')
        
#         if use_bn:
#             self.bn_1 = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.01,name=bn_1_name)
#             self.bn_2 = BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.01,name=bn_2_name)
#         if activation is not None:
#             self.avt_1 = Activation(activation,name = avt_1_name)
#             self.avt_2 = Activation(activation,name = avt_2_name)
#     def __call__(self,x):
#         x = self.conv_1(x)
#         if self.bn_1 is not None:
#             x = self.bn_1(x)
#         if self.avt_1 is not None:
#             x = self.avt_1(x)
#         x = self.conv_2(x)
#         if self.bn_2 is not None:
#             x = self.bn_2(x)
#         if self.avt_2 is not None:
#             x = self.avt_2(x)
#         return x


# class BasicRFB():
#     def __init__(self,
#          out_planes, 
#          in_planes =None,
#          input_tensor = None,
#          stride=1,
#          dilation_base = 1,
#          scale = 0.1,
#          l2_reg = 0.00005,
#          name = None,
#          pooling='maxpooling'):
#     '''
#     Basic RFB module
#     Modified:
#     1. All padding used same
#     2. Add pooling to shortcut to match stride
#     '''
#     if in_planes is None and input_tensor is None:
#         raise Exception("Either in planes or input tenosr must not be None")
#     if input_tensor is not None:
#         in_planes = input_tensor.shape.as_list()[2]
#     #1. Prepare some names and input
# #     inp = Input((None,None,None))
#     conv = BasicConv
#     conv_d = BasicDepConv
#     inter_planes_br_0 = in_planes // 4
#     inter_planes_br_1 = in_planes // 4
#     inter_planes_br_2 = in_planes // 4
#     if name is not None:
#         br_name_0 = name + '_branch_0'
#         br_name_1 = name + '_branch_1'
#         br_name_2 = name + '_branch_2'
#         conc_name = name + '_merge'
#         pooling_name = name + '_pooling' 
#         project_name = name + '_project'
#         linear_shortcut_name = name + '_linear'
#     else:
#         br_name_0 = None
#         br_name_1 = None
#         br_name_2 = None
#         conc_name = None
#         pooling_name = None
#         project_name = None
#         linear_shortcut_name = None
#     #2. Make sequential models for each branch
#     #original branch 0'  kernel size 1>3 
#     self.branch_0 = Sequential(conv(inter_planes_br_0, kernel_size=1, l2_reg=l2_reg).get_layers() +
#                    conv_d( kernel_size=3, stride=stride,dilation=dilation_base, activation = None,l2_reg = l2_reg).get_layers(),name = br_name_0)
    
#     #original branch 1' kernel size 1>(1,3)>(3,1)
#     self.branch_1 = Sequential(conv(inter_planes_br_1, kernel_size=1,l2_reg = l2_reg).get_layers() +
#                 conv(inter_planes_br_1, kernel_size=(1,3), l2_reg = l2_reg).get_layers() +
#                 conv(inter_planes_br_1, kernel_size=(3,1), l2_reg = l2_reg).get_layers() +
#                 conv_d(kernel_size=3, stride=stride,  dilation=dilation_base + 2, activation= None,l2_reg = l2_reg).get_layers(),name=br_name_1)
    
#     #original branch 2
#     self.branch_2 = Sequential(conv(x,inter_planes_br_2, kernel_size=1,l2_reg = l2_reg).get_layers() +
#                 conv(inter_planes_br_2, kernel_size=3,  l2_reg = l2_reg).get_layers() +
#                 conv(inter_planes_br_2, kernel_size=3,  l2_reg = l2_reg).get_layers() + 
#                 conv_d(kernel_size=3, stride=stride, dilation=dilation_base + 4, activation= None,l2_reg = l2_reg).get_layers(),name=br_name_2)      
#     #3. Concatenate
#     x0,x1,x2 = branch_0(inp),branch_1(inp),branch_2(inp)
#     self.conc = Concatenate(axis=-1,name = conc_name)([x0,x1,x2])
    
#     #4. Do the residual work
#     #Original Conv Linear 
#     out = conv(out_planes, kernel_size=1, stride=1, activation= None,l2_reg = l2_reg)(out)
#     inp_pool = inp #Define this to make convenience for late layers 
#     if stride !=1 :
#         if pooling == 'maxpooling':
#             self.pooling = MaxPooling2D(stride,padding='same',name = pooling_name)
#         elif pooling == 'averagepooling':
#             self.pooling = AveragePooling2D(stride,padding='same',name = pooling_name)
#     inp_conv = inp_pool
#     if in_planes != out_planes:
#         inp_conv = conv(out_planes,kernel_size=1,stride=1,activation = None,use_bn = False,l2_reg = l2_reg,name='shortcut')(inp_pool)
#     out = Lambda(lambda x:x*scale )(out)
#     out = Add()([out,inp_conv])
#     out = Activation('relu')(out)
#     return out
# def BasicRFB_a(inp,
#          out_planes, 
#          dilation_base = 1,
#          scale = 1.0,
#          l2_reg = 0.00005,
#          name = None):
#     '''
#     Basic RFB_a module
#     Modified:
#     1. All padding used same
#     '''
#     #1. Prepare some names and input
# #     inp = Input((None,None,None))
#     conv = BasicConv
#     conv_d = BasicDepConv
#     in_planes = inp.get_shape().as_list()[3]
#     inter_planes_br_0 = in_planes // 4
#     inter_planes_br_1 = in_planes // 4
#     inter_planes_br_2 = in_planes // 4 
#     if name is None:
#         name = 'BasicRFB'
#     #2. Make sequential models for each branch
#     branch_0 = Sequential([ 
#                    conv(inter_planes_br_0, kernel_size=1, l2_reg = l2_reg),
#                    conv_d(kernel_size=3, dilation=dilation_base, activation = None,l2_reg = l2_reg)],name='branch_0')#maybe should not use bn
#     branch_1 = Sequential([
#                    conv( inter_planes_br_1, kernel_size=1,l2_reg = l2_reg),
#                    conv( inter_planes_br_1, kernel_size=(3,1), l2_reg = l2_reg),
#                    conv_d( kernel_size=3, dilation=dilation_base + 2, activation = None,l2_reg = l2_reg)],name='branch_1')

#     branch_2 = Sequential([conv( inter_planes_br_2, kernel_size=1, l2_reg = l2_reg),
#                    conv( inter_planes_br_2, kernel_size=(1,3), l2_reg = l2_reg),
#                    conv_d(kernel_size=3,dilation=dilation_base +2, activation = None,l2_reg = l2_reg)],name='branch_2')
#     branch_3 = Sequential([ conv(x, inter_planes_br_3, stride=1,l2_reg = l2_reg),
#                      conv(inter_planes_br_3, kernel_size=(1,3), stride=1,l2_reg = l2_reg),
#                      conv(inter_planes_br_3, kernel_size=(3,1), stride=1,l2_reg = l2_reg),
#                      conv_d( kernel_size=3, stride=1,dilation=dilation_base+4, activation = None,l2_reg = l2_reg)],name='branch_3')
    
#     out = Concatenate()([branch_0(inp),branch_1(inp),branch_2(inp),branch_3(inp)])
    
#     #4. Do the residual work
#     #Original Conv Linear 
#     out = conv(out_planes, kernel_size=1, stride=1, activation= None,l2_reg = l2_reg)(out)
# #     inp_pool = inp #Define this to make convenience for late layers 
# #     if stride !=1 :
# #         if pooling == 'maxpooling':
# #             inp_pool = MaxPooling2D(stride,padding='same')(inp)
# #         elif pooling == 'averagepooling':
# #             inp_pool = AveragePooling2D(stride,padding='same')(inp)
# #     inp_conv = inp_pool
# #     if in_planes != out_planes:
# #         inp_conv = conv(out_planes,kernel_size=1,stride=1,activation = None,use_bn = False,l2_reg = l2_reg,name='shortcut')(inp_pool)
#     out = Lambda(lambda x:x*scale )(out)
#     out = Add()([out,inp])
#     out = Activation('relu')(out)
#     return out
               
def LiteRFB( inp,
         out_planes, 
         stride = 1,
         expand_ratio = 1,
         depth_multiplier = 1,
         dilation_base = 1,
         scale = 1,
         l2_reg = 0.00005,
         pooling ='max',
         name = None):
    '''
    Lite RFB_a module
    Modified:
    1. Only two branches,each contains a 3x3 dw_conv with dilation rate 1 and 3.
    2. Do a inverted residual for input and output.So the output is not activated.
    '''
    #1. Prepare some names and input
    conv = BasicConv
    conv_d = BasicDepConv
    conv_sep = BasicSepConv
    branch_num = 3
    in_planes = inp.get_shape().as_list()[3]
    inter_planes = int(in_planes * expand_ratio)
    out_branch_planes = out_planes #// branch_num
#     if name is None:
#         name = 'LiteRFB'
    #2. Make sequential models for each branca
    if expand_ratio != 1:
        inp = conv(inter_planes,kernel_size = 1,l2_reg = l2_reg,name = name + '_expand')(inp)
#     inp = conv_d(kernel_size = 3,stride = 1,l2_reg =l2_reg,name = '_expand_dw')(inp)
    #branch 0
    branch_0 = conv_sep(out_planes,kernel_size=3, stride=stride,dilation=dilation_base,depth_multiplier =depth_multiplier ,l2_reg = l2_reg,name= name +'_conv1_1')(inp)
    branch_1 = conv_sep(kernel_size = 3,stride = 1,l2_reg = l2_reg,name = name + '_conv2_1')(inp)
    branch_1 = conv_sep(kernel_size = 3,stride = stride,dilation=dilation_base + 2,depth_multiplier=depth_multiplier,l2_reg = l2_reg,name = name + '_conv2_2')(branch_1)
    branch_2 = conv_sep(kernel_size = 3,stride = 1,l2_reg = l2_reg,name = name + '_conv3_1')(inp)
    branch_2 = conv_sep(kernel_size = 3,stride = 1,l2_reg = l2_reg,name = name + '_conv3_2')(branch_2)
    branch_2 = conv_sep(kernel_size = 3,stride = stride,dilation=dilation_base + 4,depth_multiplier=depth_multiplier,l2_reg = l2_reg,name = name + '_conv3_3')(branch_2)
#     #4. Do the residual work
#     #Original Conv Linear 
#     # out = conv(out_planes, kernel_size=1, stride=1, activation= None,l2_reg = l2_reg)(out)
#     inp_pool = inp #Define this to make convenience for late layers 
#     if stride !=1 :
#         if pooling == 'max':
#             inp_pool = MaxPooling2D(stride,padding='same',name = name + '_maxpool')(inp)
#         elif pooling == 'avg':
#             inp_pool = AveragePooling2D(stride,padding='same',name = name + '_avgpool')(inp)
#     inp_conv = conv(inter_planes,kernel_size = 1, l2_reg = l2_reg,name = name + '_conv3_1')(inp_pool)
    out = Add(name = name + '_merge')([branch_0,branch_1,branch_2])#branch_1(inp),
    out = conv(out_planes,kernel_size = 3,stride = 1,l2_reg = l2_reg,name = name + '_project')(out)
#     out = Activation('relu',name = name + '_out')(out)
    return out


