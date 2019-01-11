from keras.layers import Conv2D,SeparableConv2D,Dense,Input,Flatten,MaxPooling2D,AveragePooling2D,Activation
from keras.layers import Concatenate,Add,Multiply,Lambda
from keras.layers import BatchNormalization,Dropout
from keras.models import Model,Sequential
import keras
def build_simple_densenet():
    'for training set size = 4 can achieve training accuracy 1 under [200epochs 2batch_size  mse sgd] '
    'sigmoid: loss '
    x = Sequential() 
    x.add(Conv2D(3,(3,3),padding='same',strides=(2,2),data_format='channels_last',input_shape=(input_H,input_W,input_C)))
    x.add(Activation('tanh'))
    x.add(Flatten())
    x.add(Dense(3000,activation= 'tanh'))
    x.add(Dense(1000,activation= 'tanh'))
    x.add(Dense(500,activation= 'tanh'))
    x.add(Dense(40,activation= 'tanh'))
    x.add(Dense(2,activation= 'softmax'))
    return 'simple_densenet',x

def build_simple_convnet():
    x = Sequential()
    activation = 'relu'
    #l2_reg = kernel_regularizer=regularizers.l2(0.01)
    l2_reg = regularizers.l2(5e-4)
    x.add(Conv2D(32,(3,3),padding='same',strides=(1,1),data_format='channels_last',input_shape=(input_H,input_W,input_C)))
    x.add(Conv2D(64,(3,3),padding='same',strides=(1,1),kernel_regularizer=l2_reg,data_format='channels_last'))
    x.add(MaxPooling2D(strides=2,data_format='channels_last'))
    x.add(Activation(activation))
    x.add(Conv2D(128,  3,padding='same',strides=1,kernel_regularizer=l2_reg,data_format='channels_last'))
    x.add(Conv2D(128,  3,padding='same',strides=1,kernel_regularizer=l2_reg,data_format='channels_last'))
    x.add(MaxPooling2D(strides=2,data_format='channels_last'))
    x.add(Activation(activation))
    x.add(Conv2D(256,  3,padding='same',strides=2,kernel_regularizer=l2_reg,data_format='channels_last'))
    x.add(Conv2D(256,  3,padding='same',strides=1,kernel_regularizer=l2_reg,data_format='channels_last'))
    x.add(MaxPooling2D(strides=2,data_format='channels_last'))
    x.add(Activation(activation))
    x.add(Flatten())
    x.add(Dropout(0.5))
    x.add(Dense(1024,activation=activation))
    x.add(Dropout(0.5))
    x.add(Dense(2,activation=activation))
    x.add(Activation('softmax'))
    return 'simple_convnet',x

def build_mimic_mobilenet():
    inputs = Input(shape=(input_H,input_W,input_C))
    x = BasicConv(inputs,32,3,2)
    x = BasicSepConv(x,32,3,1)
    x = BasicConv(x,64,1,1)
    x = BasicSepConv(x,64,3,2)
    x = BasicConv(x,128,1)
    x = BasicSepConv(x,128,3)
    x = BasicConv(x,128,1)
    x = BasicSepConv(x,256,3)
    x = BasicConv(x,256,1)
    x = BasicSepConv(x,256,3,2)
    x = BasicConv(x,512,1)
    for i in range(0,5):
        x = BasicSepConv(x,512,3)
        x = BasicConv(x,512,1)
    x = BasicSepConv(x,512,3,2)
    x = BasicConv(x,1024,1)
    x = BasicSepConv(x,1024,3,2)
    x = BasicConv(x,1024,1)
    x = AveragePooling2D(3)(x)
    x = Flatten()(x)
    x = Dense(2,activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    return 'mimic_mobilenet',model

def build_simple_rfbnet():
    inputs = Input(shape=(input_H,input_W,input_C))
    x = BasicConv(inputs,32,3,2)
    x = BasicRFB(x,64,2)
    x = BasicRFB(x,256)
    for  i in range(0,3):
        x = BasicRFB_a(x,256,2)
    x = BasicFC(x,output_num)
    model = Model(inputs=inputs, outputs=x)
    return  'simple_rfbnet',model
def build_mobilenet_v2(input_shape,output_num):
    inputs = Input(shape=input_shape)
    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    for layer in base_model.layers[:-36]:
        layer.trainable = False
    x = base_model.get_layer('out_relu').output
    x = AveragePooling2D(strides = 7)(x)
    x = Flatten()(x)
    x = Dense(2000)(x)
    x = Dropout(0.35)(x)
    x = Dense(output_num, activation= 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return 'mobilenet_v2',model

def build_shufflenet_v2(input_shape):
    inputs = Input(shape=input_shape)
    base_model = ShuffleNetV2()
    out = base_model.output
    out = Dense(output_num, activation= 'softmax')(out)
    model = Model(inputs=base_model.input, outputs=out)  
    return 'shufflenet_v2',model
#multi_gpu_model(model, gpus=1, cpu_merge=True, cpu_relocation=False)
