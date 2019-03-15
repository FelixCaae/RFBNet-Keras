
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random

###RELOAD
from importlib import reload
import detector_help
import data_generator.object_detection_2d_data_generator
reload(detector_help)
reload(data_generator.object_detection_2d_data_generator)
from detector_help import process_y,coords_convert
from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
###
class LabelEncoder:
    def __init__(self,num_classes,priors,variances,input_H,input_W):
        self.num_classes = num_classes
        self.priors = priors
        self.input_H = input_H
        self.input_W = input_W
        self.variances = variances
        
    def __call__(self,labels,diagnostics = False):
        return process_y(self.priors, labels,self.variances,self.num_classes,self.input_H,self.input_W)
    
# class Extend:
#     def __init__(self,img_height,img_width):
#         self.img_height = img_height
#         self.img_width = img_width
        
#     def __call__(self,images,labels=None):
#         images = images.astype('float')
#         images *= 255
#         labels[:,1:] *= [self.img_width,self.img_height,self.img_width,self.img_height]
#         return images, labels
    
# class Normalize:
#     def __init__(self,img_height,img_width):
#         self.img_height = img_height
#         self.img_width = img_width
        
#     def __call__(self,images,labels=None):
#         images = images.astype('float')
#         images /= 255
#         labels[:,1:] /= [self.img_width,self.img_height,self.img_width,self.img_height]
#         return images, labels
    
# def data_augment(x_train,y_train,batch_size,val_split =0.25,task = 'classification',num_classes = None,priors = None):
#     h,w = x_train.shape[1:3]
    
#     # Shuffle training data and split some for validation
#     seed = 1
#     m =len(x_train)
#     np.random.seed(seed)
#     index = np.arange(0,m)
#     random.shuffle(index)
#     x_train = x_train[index]
#     y_train = [y_train[i] for i in index] #We can`t choose with method like y_train[index] because y_train is a list rather than np.array
    
#     split = int(m * (1 - val_split))
#     x_train,x_val = x_train[:split],x_train[split:]
#     y_train,y_val = y_train[:split],y_train[split:]
    
#     #Use shift、flip and rotation to augment data 
#     if task == 'classification':
#         datagen = ImageDataGenerator(
#             rotation_range=10,
#             width_shift_range=0.1,
#             height_shift_range=0.1,
#             validation_split = val_split,
#             horizontal_flip=True)
#         train_flow = datagen.flow(x_train, y_train, batch_size=batch_size)
#     elif task == 'detection':
#         ssd_data_augmentation = SSDDataAugmentation(img_height = h, img_width = w)
#         extender = Extend(h,w)
#         normalizer = Normalize(h,w)
#         label_encoder = LabelEncoder(num_classes,priors)
#         gen = DataGenerator()
#         gen.images = x_train 
#         gen.labels = y_train
#         gen.dataset_size = len(y_train)
#         gen.dataset_indices = np.arange(gen.dataset_size)
#         gen.filenames = ['x' for i in range(gen.dataset_size) ]
#         train_flow = gen.generate(batch_size=batch_size,
#                          shuffle=True,
#                          transformations=[extender, ssd_data_augmentation, normalizer],
#                          label_encoder=label_encoder,
#                          returns={'processed_images',
#                                'processed_labels'},
#                          keep_images_without_gt=False)
#         val_flow = gen.generate(batch_size=batch_size,
#                          shuffle=True,
#                          transformations=[],
#                          label_encoder=label_encoder,
#                          returns={'processed_images',
#                                'encoded_labels'},
#                          keep_images_without_gt=False)
#     return (train_flow,val_flow)