import cv2
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def draw_detection(frame,prediction,class_names,box_color = (255,0,0),box_width = 2,text_color=(0,0,0),draw_label = True,water_mask=None,font = cv2.FONT_HERSHEY_COMPLEX,font_scale=0.5):
    '''
    Input a frame and prediction
    #each class
    '''
    conf = prediction[:-4]
    bbox = prediction[-4:].copy() #* 224).astype('int32')
    h,w,c = frame.shape
    bbox[[0,2]] *= w
    bbox[[1,3]] *= h
    bbox = bbox.astype('int32')
    class_name = class_names[np.argmax(conf)] 
    score = np.max(conf)
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