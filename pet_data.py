import os
import random
import keras
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.image import imread
from keras.preprocessing import image
import scipy.io
from keras.preprocessing.image import ImageDataGenerator
import xml.etree.ElementTree as ET
import random
#import cv2
        
def read_list(list_path):
    name_list = open(list_path).readlines()
    class_list = np.ones(len(name_list))
    
    for k,name in enumerate(name_list):
        strings = name.split(' ')
        class_list[k] = int(strings[1])
        name_list[k] = strings[0]
  
    return name_list,to_categorical(class_list - 1)

def split_annotations(data_root,list_name,split = 0.75,seed = 1):
    images_path = os.path.join(data_root, "images")
    annot_path = os.path.join(data_root,"annotations")
    xml_path = os.path.join(annot_path,"xmls")
    list_path = os.path.join(annot_path,list_name)
    lines = open(list_path).readlines()
    name_list = []
    for k,line in enumerate(lines):
        strings = line.split(' ')
        name = strings[0]
        name_list.append(name)
    idx = np.arange(0,len(name_list))
    print('before ignore:',len(idx))
    idx = [i for i in idx if os.path.isfile(xml_path + '/' + name_list[i] + '.xml') and \
               os.path.isfile(images_path + '/' + name_list[i]  + '.jpg')]
    print('after ignore:',len(idx))
    
    random.seed(seed)
    random.shuffle(idx)
    train_size = int(len(idx) * split)
    train_data = [lines[i] for i in idx[:train_size]]
    test_data = [lines[i] for i in idx[train_size:]]
    train_file = open(os.path.join(annot_path,'train.txt'),'w')
    test_file = open(os.path.join(annot_path,'test.txt'),'w')
    train_file.writelines(train_data)
    test_file.writelines(test_data)

    
def get_bounding_boxes(name_list, xml_path):
    
    n = len(name_list)
    bboxes = np.ones((n,4))
    sizes = np.ones((n,2))
    for k,name in enumerate(name_list):
        path = os.path.join(xml_path,name + '.xml')
        root = ET.parse(path).getroot()
        bounding_box = np.array([int(ele.text) for ele in root[5][4]],dtype = 'float')
        size = [int(ele.text) for ele in root[3]] # w h
        #print(bounding_box,size)
        #print(bounding_box,size)
        
        bboxes[k] = bounding_box
        sizes[k] = size[:2]
    bboxes[:,(0,2)] /= sizes[:,0,np.newaxis]
    bboxes[:,(1,3)] /= sizes[:,1,np.newaxis]
    return bboxes


def read_imgs(name_list,images_path,input_shape,point_inter = 100):
    n = len(name_list)
    x = np.ones(((n,) + input_shape))
    for k,name in enumerate(name_list):
        if k % point_inter == 0:
            print('.', end = '')
        img = None
#         for t,ext in enumerate(supported_ext):
#             img_path= os.path.join(images_path,name) + ext
#             if os.path.isfile(img_path):
#                 img = read_function[t](img_path,input_shape)
#                 break
        img_path = os.path.join(images_path, name) + '.jpg'
        if os.path.isfile(img_path):
            img = image.load_img(img_path, target_size=input_shape[:2])
        if img == None:
            raise Exception('File format doesn`t support')

         #i = np.expand_dims(i, axis=0)
        x[k] = img
    x /= 255
    return x

def load_data(train_sample= 0.7,test_sample= 0.3,input_shape=(224,224,3),task = 'classification',root = "/home/cai/dataset/pet_images/",seed = 1):
    x_train,y_train,x_test,y_test = None,None,None,None
    
    images_path = os.path.join(root, "images")
    annot_path = os.path.join(root,"annotations")
    xml_path = os.path.join(annot_path,"xmls")
    
    #assert(train_sample + test_sample <= 1 )
    list_train,y_train = read_list(annot_path + '/train.txt')
    list_test,y_test = read_list(annot_path + '/test.txt')
    
    train_size = int(len(list_train)*train_sample)
    test_size = int(len(list_test)*test_sample)
    #Read boundingboxes firsly may help accelerate??
    if task == 'detection':    
        bboxes_train = get_bounding_boxes(list_train,xml_path)
        bboxes_test = get_bounding_boxes(list_test,xml_path)
        y_train = np.concatenate([y_train,bboxes_train],axis = 1)
        y_test = np.concatenate([y_test,bboxes_test], axis = 1)
    
    elif task != 'classification':
        raise RuntimeError("task argument should be either classification or detection")
    x_train = read_imgs(list_train[:train_size],images_path,input_shape)
    x_test = read_imgs(list_test[:test_size],images_path,input_shape)
    
    y_train = y_train[:train_size,:]
    y_test = y_test[:test_size,:]  
    print('done')
    return (x_train,y_train),(x_test,y_test)

def data_augment(x_train,y_train,batch_size,val_split =0.25):
    # Shuffle training data and split some for validation
    seed = 1
    m =len(x_train)
    np.random.seed(seed)
    index = np.arange(0,m)
    random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    split = int(m * (1 - val_split))
    x_train,x_val = x_train[:split],x_train[split:]
    y_train,y_val = y_train[:split],y_train[split:]
    
    #Use shift、flip and rotation to augment data 
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    train_flow = datagen.flow(x_train, y_train, batch_size=batch_size)
    datagen = ImageDataGenerator()
    val_flow = datagen.flow(x_val, y_val, batch_size=batch_size)
    
    return (train_flow,val_flow)
#show_all_data()

def show_data(x,y,index):
    if type(index) != type([]):
        index=[index]
    for i in index:
        if y[i][0] == 1:
            print('A cat')
        else:
            print('A dog')
        plt.imshow(x[i])
        plt.show()

def show_false_pics(model, x_test, y_test,limit=7):
    y_pred = model.predict(x_test)
    bias = np.argmax(y_pred,axis=1) - np.argmax(y_test,axis=1)
    bias = bias.squeeze()
    print(bias)
    false = np.where(bias != 0)[0]
    false = false[:limit]
    for index in false:
        print('index: ',index)
        show_data(x_test, y_pred,index)
        