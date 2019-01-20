import cv2
import numpy as np
from matplotlib import pyplot as plt
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
def interpret(class_names,y):
    conf = y[:-4]
    loc = y[-4:]
    score = np.max(conf)
    class_name = class_names[np.argmax(conf)]
    return (class_name,score,loc)

def draw_detection(frame,prediction,class_names,box_color = (255,0,0),box_width = 2,text_color=(0,0,0),draw_label = True,water_mask=None,font = cv2.FONT_HERSHEY_COMPLEX,font_scale=0.5):
    '''
    Input a frame and prediction
    #each class
    '''
    if draw_label == True:
        class_name,score,bbox = interpret(class_names,prediction)
    else:
       # print('hello',prediction.shape)
        bbox = prediction
    #Transform from scale 0-1 to scale 0-255
    #h,w,c = frame.shape
    #bbox *= [w,h,w,h]
    bbox = bbox.astype('int32')
    
    #draw box
    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),box_color,box_width)
  
    if draw_label:
        offset_y = 5
        text = class_name #+ " " + str(round(float(score),3))
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        if text_width < bbox[2] - bbox[0]:
            text_width = bbox[2] - bbox[0]
        text_pos = [bbox[0],bbox[1]]
        if text_pos[1] <= 8:
            text_pos[1] = 8
        text_pos = tuple(text_pos)
        text_rect = [(text_pos[0],text_pos[1] + offset_y),(text_pos[0] + text_width , text_pos[1] - text_height)]
        cv2.rectangle(frame,text_rect[0],text_rect[1],box_color,cv2.FILLED)
        cv2.putText(frame,text,text_pos, font, font_scale,text_color,1,cv2.LINE_AA)
        
    if water_mask != None:
        cv2.putText(frame,water_mask, (20, 20), font, 1,text_color,1,cv2.LINE_AA)

def show_data(x,y,index,class_names):
    if type(index) != type([]):
        index=[index]
    for i in index:
        print(class_names[np.argmax(y[i])])
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