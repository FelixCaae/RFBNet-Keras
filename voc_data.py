import os
import random
import keras
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.image import imread
from keras.preprocessing import image
import scipy.io
from importlib import reload
import xml.etree.ElementTree as ET
import random


def read_list(list_path):
    name_list = open(list_path).readlines()
    class_list = np.ones(len(name_list))
    
    for k,name in enumerate(name_list):
        strings = name.split(' ')
        class_list[k] = int(strings[1])
        name_list[k] = strings[0]
  
    return name_list,to_categorical(class_list)

def split_annotations(root_path,list_name,split = 0.75,seed = 1,task='detection'):
    images_path = os.path.join(root_path, "images")
    annot_path = os.path.join(root_path,"annotations")
    xml_path = os.path.join(annot_path,"xmls")
    list_path = os.path.join(annot_path,list_name)
    lines = open(list_path).readlines()
    name_list = []
    for k,line in enumerate(lines[:]):
        if line[0] == '#':
            lines.remove(line)
            continue
        strings = line.split(' ')
        name = strings[0]
        name_list.append(name)
    idx = np.arange(0,len(name_list))
    if task == 'detection':
        print('before ignore:',len(idx))
        idx = [i for i in idx if os.path.isfile(xml_path + '/' + name_list[i] + '.xml') and \
                   os.path.isfile(images_path + '/' + name_list[i]  + '.jpg')]
        print('after ignore:',len(idx))    
    random.seed(seed)
    random.shuffle(idx)
    train_size = int(len(idx) * split)
    train_data = [lines[i] for i in idx[:train_size]]
    test_data = [lines[i] for i in idx[train_size:]]
    t = [line for line in lines if line[0] == '#']
    print(t)
    train_file = open(os.path.join(annot_path,'train.txt'),'w')
    test_file = open(os.path.join(annot_path,'test.txt'),'w')
    train_file.writelines(train_data)
    test_file.writelines(test_data)
    
def generate_list_voc(root_path,train_lists = ["07train","07trainval","07val","12train","12trainval"],test_lists = ["12val"]):
    train_data = []
    test_data = []
    for data,lists in zip([train_data,test_data],[train_lists,test_lists]):
        for list_name in lists:
            if "07" in list_name:
                list_path = os.path.join(root_path,"VOC2007/ImageSets/Main")
            else:
                list_path = os.path.join(root_path,"VOC2012/ImageSets/Main")
            file_name = list_name[2:] + '.txt'
            file = open(os.path.join(list_path,file_name))
            file_data = file.readlines()
            data += file_data
    train_file = open(os.path.join(root_path,'train.txt'),'w')
    test_file = open(os.path.join(root_path,'test.txt'),'w')
    train_file.writelines(train_data)
    test_file.writelines(test_data)

def get_bounding_boxes(name_list, xml_path):
    
    n = len(name_list)
    bboxes = np.ones((n,4))
    sizes = np.ones((n,2))
    bbox_tag = ["xmin","ymin","xmax","ymax"]
    size_tag = ["width", "height"]
    for k,name in enumerate(name_list):
        path = os.path.join(xml_path,name + '.xml')
        root = ET.parse(path).getroot()
        bbox = np.array([int(root.find('./object/bndbox/' + tag).text) for tag in bbox_tag],dtype = 'float')
        size = np.array([int(root.find('./size/' + tag).text) for tag in size_tag],dtype = 'int') # w h
        bboxes[k] = bbox
        sizes[k] = size
    bboxes[:,(0,2)] /= sizes[:,0,np.newaxis]
    bboxes[:,(1,3)] /= sizes[:,1,np.newaxis]
    return bboxes


def read_imgs(name_list,root_path,input_shape,point_inter = 100):
    n = len(name_list)
    x = np.ones(((n,) + input_shape))
    for k,name in enumerate(name_list):
        if k % point_inter == 0:
            print('.', end = '')
        img = None
        if name[:2] != '20':
            img_path = os.path.join(root_path,'VOC2007/JPEGImages', name + '.jpg')
        else:
            img_path = os.path.join(root_path,'VOC2012/JPEGImages', name + '.jpg')
            
        if os.path.isfile(img_path):
            img = image.load_img(img_path, target_size=input_shape[:2])
        if img == None:
            print(img_path)
            raise Exception('File format doesn`t support')

         #i = np.expand_dims(i, axis=0)
        x[k] = img
    x /= 255
    return x

def read_labels(name_list,root_path):
    n = len(name_list)
    y = []
    for k,name in enumerate(name_list):
        label = None
        if name[:2] != '20':
            label_path = os.path.join(root_path,'VOC2007/labels', name + '.txt')
        else:
            label_path = os.path.join(root_path,'VOC2012/labels', name + '.txt')
        if os.path.isfile(label_path):
            with  open(label_path) as file:
                label_strings = file.readlines()
                label_context = [ [float(word) for word in string.split(' ') ] for string in label_strings ]
         #i = np.expand_dims(i, axis=0)
            y.append(label_context ) 
    return y

def load_data(train_sample= 0.7,test_sample= 0.3,input_shape=(224,224,3),task = 'classification',root_path = "/home/cai/dataset/VOCdevkit/",seed = 1):
    x_train,y_train,x_test,y_test = None,None,None,None
    
   # images_path = os.path.join(data_root, "JPEGImages")
   # annot_path = os.path.join(data_root,"labels")
   # xml_path = os.path.join(annot_path,"xmls")
    
    #assert(train_sample + test_sample <= 1 )
    with open(os.path.join(root_path, 'train.txt')) as file:
        list_train = [line.strip() for line in file.readlines()]
  
    with open(os.path.join(root_path, 'test.txt')) as file:
        list_test = [line.strip() for line in file.readlines()]
    
    train_size = int(len(list_train)*train_sample)
    test_size = int(len(list_test)*test_sample)
    
    x_train = read_imgs(list_train[:train_size],root_path,input_shape)
    x_test = read_imgs(list_test[:test_size],root_path,input_shape)
    y_train = read_labels(list_train[:train_size],root_path)
    y_test = read_labels(list_test[:test_size],root_path)
    print('done')
    return (x_train,y_train),(x_test,y_test)
#show_all_data()
def fit_generator(model,x_train,y_train,batch_size,epochs_per_stage,val_split = 0.25,task='classification'):
    (train_flow,val_flow) = data_augment(x_train,y_train,batch_size,val_split,task)
    history = model.fit_generator(train_flow,steps_per_epoch=len(x_train)/batch_size,validation_data=val_flow,validation_steps=300,epochs=epochs_per_stage)
    return history

        