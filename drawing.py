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

def draw_detection(frame,prediction,class_names,size = 'medium',color = 'green',use_cm = False,width = 2,show = True,draw_label = True,draw_score =False, water_mask=None):
    '''
    Input an image and prediction(class,score,xmin,ymin,xmax,ymax)
    or label(class,xmin,ymin,xmax,ymax)  
    boungding boxes` coordinate number are between 0 and 1
    '''
    figsizes = [(5,3), (10,6), (15,9)]
    size = ['small','medium','big'].index(size)
    plt.figure(figsize=figsizes[size])
    if show:
        plt.imshow(frame)
    
    #colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
    current_axis = plt.gca()
    draw_color = color
    for box in prediction:
        class_id = int(box[0])
        if use_cm:
            draw_color = color[class_id]
        xmin,ymin,xmax,ymax = box[-4:]
        current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color= draw_color, fill=False, linewidth=2)) 
        if draw_score:
            score = box[1]
            label = "%s %.2f"%(class_names[class_id],score)
        else:
            label = class_names[class_id]
        if draw_label:
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':draw_color, 'alpha':1.0})
    if show:
        plt.show()
#     h,w = frame.shape[:2]
#     for obj in prediction:
#         bbox = obj[-4:]
#         if draw_label == True:
#             class_name = class_names[int(obj[0])]
#             #bbox[1:3] = [bbox[2], bbox[1]]
#         if draw_score == True:
#             score = obj[1]
#         if bbox.size == 0:
#             continue
#         #Get absolute coordinates and transform it into int
#         if not absolute:
#             bbox *= [w,h,w,h]
#         bbox = bbox.astype('int32')
#         cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),box_color,box_width)

#         if draw_label:
#             offset_y = 5   #The offset between the top line of bbox and text
#             text = class_name #+ " " + str(round(float(score),3)  whether or not to show score
#             (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
#             if text_width < bbox[2] - bbox[0]:
#                 text_width = bbox[2] - bbox[0]
#             text_pos = [bbox[0],bbox[1]]
#             if text_pos[1] <= 8:
#                 text_pos[1] = 8
#             text_pos = tuple(text_pos)
#             text_rect = [(text_pos[0],text_pos[1] + offset_y),(text_pos[0] + text_width , text_pos[1] - text_height)]
#             cv2.rectangle(frame,text_rect[0],text_rect[1],box_color,cv2.FILLED)
#             cv2.putText(frame,text,text_pos, font, font_scale,text_color,1,cv2.LINE_AA)

#         if water_mask != None:
#             cv2.putText(frame,water_mask, (20, 20), font, 1,text_color,1,cv2.LINE_AA)

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