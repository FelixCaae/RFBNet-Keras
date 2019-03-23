import numpy as np
from cv2.dnn import NMSBoxes
from copy import deepcopy
#import cv2
def prior_box(feature_map, aspect_ratios,scale = None):
    '''
    Calculate the coordinates of prior boxes according to shape of feature maps、aspect ratios
    Input:
       feature_maps: list of feature_map size (assuming width is same as height)
       aspect_ratios: list of tuples  it dosen`t need to include aspect ratio 1 eg. [[2,3],[2]]  
    Output:
       prior_boxes: tensor of shape (#box,4) each row contains coords in format (cx,cy,w,h)
    '''
    prior_boxes = []
    m = len(feature_map)
    if  scale is None:
        s_min = 0.2
        s_max = 0.9
        s = [s_min + (k - 1)*(s_max - s_min)/(m - 1) for k in range(1, m + 2)]
    else:
        s = scale
    for k,f in enumerate(feature_map):
        for i in range(f):
            for j in range(f):
                cx = (j + 0.5)/f
                cy = (i + 0.5)/f
                prior_boxes.append([cx,cy,s[k],s[k]])
                prior_boxes.append([cx,cy,np.sqrt(s[k] * s[k+1]),np.sqrt(s[k]*s[k+1])])
                for ar in aspect_ratios[k]:
                    prior_boxes.append([cx,cy,s[k] * np.sqrt(ar), s[k] / np.sqrt(ar)])
                    prior_boxes.append([cx,cy,s[k] / np.sqrt(ar), s[k] * np.sqrt(ar)])
    
    return np.array(prior_boxes)
def coords_convert(input_boxes,coding = "centroids2corners"):
    '''
    Convert a group of boxes in format of corners(x_min,y_min,x_max,y_max) to centroids(cx,cy,wx,wy)
    There are three supported box format:
        centroids: (cx,cy,w,h)
        corners: (x_min,y_min,x_max,y_max)
        minmax: (x_min,x_max,y_min,y_max)
        
    Input:
        input_box: numpy array of shape (#boxes,4) 
        
    Output:
        output_box: numpy array of shape(#boxes,4)
    '''
    
    output_boxes = np.ones(input_boxes.shape)
    if coding is "centroids2corners":
        output_boxes[:,0] = input_boxes[:,0] - input_boxes[:,2] / 2
        output_boxes[:,1] = input_boxes[:,1] - input_boxes[:,3] / 2
        output_boxes[:,2] = input_boxes[:,0] + input_boxes[:,2] / 2
        output_boxes[:,3] = input_boxes[:,1] + input_boxes[:,3] /2
    elif coding is "corners2centroids":
        output_boxes[:,0] = (input_boxes[:,2] + input_boxes[:,0]) / 2
        output_boxes[:,1] = (input_boxes[:,3] + input_boxes[:,1]) / 2
        output_boxes[:,2] = input_boxes[:,2] - input_boxes[:,0]
        output_boxes[:,3] = input_boxes[:,3] - input_boxes[:,1]
    elif coding is "corners2minmax" or coding is "minmax2corners":
        output_boxes[:,0] = input_boxes[:,0]
        output_boxes[:,1] = input_boxes[:,2]
        output_boxes[:,2] = input_boxes[:,1]
        output_boxes[:,3] = input_boxes[:,3]
        
    return output_boxes
def calculate_iou(boxes1, boxes2,mode='element_wise'):
    '''
    Input should be corner format.Support two modes of iou calculating method:elementwise and outerproduct.
    '''

    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])

    if mode == 'element_wise':
        max_xy = np.minimum(boxes1[:,2:],boxes2[:,2:])
        min_xy = np.maximum(boxes1[:,:2],boxes2[:,:2])
        inner_area = (max_xy[:,0] - min_xy[:,0]) * (max_xy[:,1] - min_xy[:,1])
        inner_area = np.where(inner_area < 0 ,0,inner_area)
        iou = inner_area / (area1 + area2 - inner_area)
        return iou
    
    elif mode == 'outer_product':
        m = len(boxes1)
        n = len(boxes2)
        
        boxes1 = np.expand_dims(boxes1,2)
        boxes2 = np.expand_dims(boxes2,2)
        
        area1 = np.tile(np.expand_dims(area1, axis = 1), reps = (1,n))
        area2 = np.tile(np.expand_dims(area2, axis = 0), reps = (m,1))
        max_xy = np.stack([np.minimum(boxes1[:,2] , boxes2[:,2].T), 
                     np.minimum(boxes1[:,3] , boxes2[:,3].T)], axis = 2)
        min_xy = np.stack([np.maximum(boxes1[:,0] , boxes2[:,0].T),
                     np.maximum(boxes1[:,1] , boxes2[:,1].T)], axis = 2)
        
        wh = max_xy - min_xy
        inner_area = wh[...,0] * wh[...,1]
        negative_mask = np.logical_or(wh[...,0]<0, wh[...,1]<0)
        inner_area[negative_mask] = 0
        bug = np.where(area1 + area2 - inner_area == 0)
        if np.any(np.where(area1 + area2 - inner_area == 0)):
            print("Inner area bug",bug)
            print(boxes1[bug[0]],boxes2[bug[1]])
            print(area1[bug],area2[bug],inner_area[bug])
        iou = inner_area / (area1 + area2 - inner_area)
        
        return iou
            
def compute_ghat(g,matched_priors,variances):
    '''
    Compute ghat from the input prior and ground_truth.
    The prior and ground_truth should be a single box
    Input:
        prior:Numpy array of matched priors(centroid).(#matched_priors,4)
        g: A label(centroid)
    Output:
        ghat: numpy array of shape (#matched_priors,4)
    '''
    #print('g shape',g.shape,'priors shape',priors.shape)
    g_hat = np.ones((len(matched_priors),4))
    g_hat[:,0] = (g[0] - matched_priors[:,0]) / matched_priors[:,2] / variances[0]
    g_hat[:,1] = (g[1] - matched_priors[:,1]) / matched_priors[:,3] / variances[1]
    g_hat[:,2] = np.log(g[2] / matched_priors[:,2]) / variances[2]
    g_hat[:,3] = np.log(g[3] / matched_priors[:,3]) / variances[3]
    
    return g_hat
def compute_coords(loc,priors, variances):
    '''
    This function calculate the absolute coordinates according to the predicted location and priors.
    The loc and priors must match the shape.
    Input:
         loc: Numpy array of predicted location(centroid)(#priors,4)
         priors: numpy array of centroid boxes(centroid)(#priors,4)
    Output:
         boxeses: numpy array of shape(#priors,4)
    '''
    boxeses = np.zeros(priors.shape)
    #TODO: we don`t need to use all loc to decode ,just those has scores > score_threshold
    #print("input_boxeses shape",input_boxeses.shape,"priors shape",priors.shape,"loc shape",loc.shape)
    boxeses[:,0] = loc[:,0] * priors[:,2] * variances[0] + priors[:,0]
    boxeses[:,1] = loc[:,1] * priors[:,3] * variances[1] + priors[:,1]
    boxeses[:,2] =  np.exp(loc[:,2] * variances[2] ) * priors[:,2] 
    boxeses[:,3] =  np.exp(loc[:,3] * variances[3] ) * priors[:,3] 
    
    return boxeses

def process_y(priors,ground_truth,variances,num_classes,input_H,input_W,iou_threshold = 0.5):
    ''' 
    Match prior boxes with ground truth.
    This function should return a Numpy array as y_true in training process.It is suppposed 
    to be a tensor with three dims (#samples,#priors,#classes+4).
    Priorbox which has a default box overlapping the ground truth of IOU more than 0.5
    is responsible for this prediction.That`s to say,more than one priorbox can be matched to one detection
    target.But each priorbox should only be responsible for one target.We use out-product iou calculating
    and masking for priors and target boxes.
    When matching is done we still need to calculate the offsets of targets` relative
    to default boxes .These offset are what our model is expected to predict.And we also need to 
    set proper class value for classification.
    Input:
        priors: numpy array of shape (#priors,4) 
        ground_truth: numpy array of processed samples (#samples,#objects,#classes + 1 + 4)
    Output:
        y:list of shape [#samples, #priors ,#classes + 1 + 4]
    '''
    
    num_priors = len(priors)
    num_samples = len(ground_truth)
    y = np.zeros((num_samples, num_priors, num_classes + 5))
   
    # Convert priors and targets from centroid to corner to calculate iou
    corner_priors = coords_convert(priors,"centroids2corners") #get corner format of priors to calculate iou
    for k in range(num_samples):      
        
        
        # 0. Extract targets and classes for k sample and convert to np.array
        gt = np.array(ground_truth[k])
        
        if gt.size == 0:
            y[k,:,0] = 1
            continue 
        
        # 1. Calulate iou between  ground_truth and all priors
        
        # Normalize labels     
        targets = gt[:,-4:] / [input_W,input_H,input_W,input_H] 
        classes = gt[:,0].astype('int')
       
        
        centroid_targets = coords_convert(targets,"corners2centroids") #get centroid format of boxes
        iou = calculate_iou(corner_priors, targets,mode = 'outer_product')
        
        #2. Find a matched target for each prior 
        max_iou = np.max(iou,axis = 1)
        #print(np.where(max_iou > iou_threshold))
        unmatched_priors = max_iou < iou_threshold
        #Set 1 for first element in unmatched rows 
        y[k,unmatched_priors,0] = 1
        match = np.argmax(iou,axis = 1)
        match[unmatched_priors] = -1
        
        
        #3. Compute offsets for each matched default box.This offset are called g_hat in original papers
        #    We should use centroid-boxes for both priors and target format
        for i in range(len(targets)):
        #Input classes are single number ranged from 1 - num_classes 
            
            #For each target, calculate matched ghat for them
            matched_priors = np.where(match == i)[0] 
            y[k,matched_priors,-4:] = compute_ghat(centroid_targets[i], priors[matched_priors], variances)
            y[k,matched_priors,classes[i]] = 1
        
    return y

def post_process(y_pred_no_process,priors,variances,num_classes,input_W,input_H,top_k = 200,score_thresh = 0.01,iou_thresh = 0.5):
    '''
    A for-loop implemention
    Input: y_pred_no_process (batch_size,#priors,#classes + 4)
    Output: y_pred: a list
         * for each element in y_pred, it has a shape of (#targets,6)
         * (class,score,xmin,ymin,xmax,ymax)
    '''
    batch_size = y_pred_no_process.shape[0]
    y_pred = [] #np.zeros((batch_size,num_classes,top_k,1 + 4))
    
#     bboxes = y_pred_no_process[:,:,-4:]
#     classes = np.argmax(y_pred_no_process[:,:,:-4], axis = 2)
#     conf = np.max(y_pred_no_process[:,:,:-4], axis = 2)
    i = 0
    for pred in y_pred_no_process:
        #Do three things for boxes processing 
        #1. Take the offsets 
        #2. Convert the format from centroids to corners
        #3. Multiply img size to get the absolute coordinates 
        loc = pred[:,-4:]       
        bboxes = coords_convert(compute_coords(loc,priors,variances),"centroids2corners")    
        bboxes *= [input_W,input_H,input_W,input_H]
        
        #Do two things for conf processing
        #1. Take confidence scores
        #2. Calculate which class has highest score
        #3. Collect highest scores for each boungding box
        conf_raw = pred[:,:-4]   
        classes = np.argmax(conf_raw[:,1:], axis= 1) + 1
        conf = conf_raw[:,1:].max(axis= 1)
        #classes[conf > 1 - score_thresh] = 0
        #print(classes)
        
        detections = []
        for c in range(1,num_classes + 1):
            #Do NMS for all predictions with same predicted class and add them into y_pred
           
            #c_ prefix means this variable is belong to some certain class. 
            c_mask = classes==c
            c_conf = conf[c_mask]
            c_boxes = bboxes[c_mask]
            c_classes = deepcopy(classes[c_mask]) 
            #print('c_classes', c_classes)
            #This is an important step which may cost remarkable time
            keep = np.arange(0,len(c_classes))
            c_boxes_copy = deepcopy(c_boxes)
            c_boxes_copy[:,2:] -= c_boxes_copy[:,:2]
            keep = NMSBoxes(c_boxes_copy.tolist(), c_conf.tolist(),score_thresh,iou_thresh,top_k = top_k)
            
            if len(keep) == 0: #keep might be () if no element is kept
                continue
            keep = keep[:,0]
       
            #keep = np.arange(0,len(c_classes))
            detections.append(np.concatenate([c_classes[keep,np.newaxis],c_conf[keep,np.newaxis],c_boxes[keep]],axis=1))
        if len(detections) > 0:
            y_pred.append(np.concatenate(detections))
        else:
            y_pred.append([])
            
    return y_pred 



## Calculate accuracy. 
def evaluate(y_pred,y_test,iou_thresh = 0.5,metrics = ['precision','recall']):
    '''
    Given process prediction,ground truth and iou thresh. Do some evaluation according to metrics 
    Possible metrics:
      mAP_07,
      mAP_12,
      precision,
      recall
    '''
    true_pos = 0  #true positive
    pred_pos = 0   #all predicted positive
    truth_pos = 0 #all true positive
    for pred,truth in zip(y_pred,y_test):
        pred_pos += len(pred)
        truth_pos += len(truth)
        truth = np.array(truth)
        pred = np.array(pred)
        if len(pred) == 0:
            continue
        #print(truth.shape,pred.shape)
        truth_boxes = truth[:,-4:]
        truth_class = np.expand_dims(truth[:,0],0).T
        pred_boxes = pred[:,-4:]
        pred_class = np.expand_dims(pred[:,0],1)
        iou = calculate_iou(pred_boxes,truth_boxes,mode = 'outer_product')
        iou_match = iou > iou_thresh
        classes_match = pred_class == truth_class 
        match = np.logical_and(iou_match, classes_match)
        print(np.sum(match))
        matched_pred_list = []
        for i in range(len(truth)):
            matched_pred = np.where(match[:,i] == True)[0]
            # To be simple, we select first element as matched pred
            matched_pred = list(set(matched_pred) - set(matched_pred_list))  
            if len(matched_pred) != 0:
                matched_pred_list.append(matched_pred[0])
        true_pos += len(matched_pred_list)
    
    result = []
    if 'precision' in metrics:
        result.append(true_pos / pred_pos)
    if 'recall' in metrics:
        result.append(true_pos / truth_pos)
        
    return result