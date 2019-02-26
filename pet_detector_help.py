import numpy as np
#from cv2.dnn import NMSBoxes
#import cv2
def prior_box(feature_map, aspect_ratios):
    '''
    Calculate the coordinates of prior boxes according to shape of feature maps、aspect ratios
    Input:
       feature_maps: list of feature_map size (assuming width is same as height)
       aspect_ratios: list of tuples  it dosen`t need to include aspect ratio 1 eg. [[2,3],[2]]  
    Output:
       prior_boxes: tensor of shape (#box,4) each row contains coords in format (cx,cy,w,h)
    '''
    prior_boxes = []
    s_min = 0.2
    s_max = 0.9
    m = len(feature_map)
    s = [s_min + (k - 1)*(s_max - s_min)/(m - 1) for k in range(1, m + 2)]
    for k,f in enumerate(feature_map):
        for i in range(f):
            for j in range(f):
                cx = (i + 0.5)/f
                cy = (j + 0.5)/f
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
def iou(boxes1, boxes2,mode='elementwise'):
    '''
    Input should be corner format.Support two modes of iou calculating method:elementwise and outerproduct.
    '''
     
    if mode == 'elementwise':
        area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
        area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
        max_xy = np.minimum(boxes1[:,2:],boxes2[:,2:])
        min_xy = np.maximum(boxes1[:,:2],boxes2[:,:2])
        inner_area = (max_xy[:,0] - min_xy[:,0]) * (max_xy[:,1] - min_xy[:,1])
        inner_area = np.where(inner_area < 0 ,0,inner_area)
        iou = inner_area / (area1 + area2 - inner_area)
        return iou
    elif mode == 'group':
        area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
        area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
        

def compute_ghat(g,priors):
    '''
    Compute ghat from the input prior and ground_truth.
    The prior and ground_truth should be a single box
    Input:
        prior:Numpy array of  centroids boxes.(#matched_priors,4)
        g: numpy array of shape (4,)
    Output:
        ghat: numpy array of shape (#matched_priors,4)
    '''
    #print('g shape',g.shape,'priors shape',priors.shape)
    g_hat = np.ones((len(priors),4))
    g_hat[:,0] = (g[0] - priors[:,0]) / priors[:,2]
    g_hat[:,1] = (g[1] - priors[:,1]) / priors[:,3]
    g_hat[:,2] = np.log(g[2] / priors[:,2])
    g_hat[:,3] = np.log(g[3] / priors[:,3])
    
    return g_hat
def compute_coords(loc,priors):
    '''
    This function calculate the absolute coordinates according to the predicted location and priors.
    The loc and priors must match the shape.
    Input:
         loc: numpy array of shape(#priors,4)
         priors: numpy array of shape(#priors,4)
    Output:
         boxeses: numpy array of shape(#priors,4)
    '''
    boxeses = np.zeros(priors.shape)
    #TODO: we don`t need to use all loc to decode ,just those has scores > score_threshold
    #print("input_boxeses shape",input_boxeses.shape,"priors shape",priors.shape,"loc shape",loc.shape)
    boxeses[:,0] = loc[:,0] * priors[:,2] + priors[:,0]
    boxeses[:,1] = loc[:,1] * priors[:,3] + priors[:,1]
    boxeses[:,2] =  np.exp(loc[:,2]) * priors[:,2]
    boxeses[:,3] =  np.exp(loc[:,3]) * priors[:,3]
    
    return boxeses

def process_y(priors, ground_truth,iou_threshold = 0.5):
    ''' 
    Match prior boxes with ground truth.
    This function should return a Numpy array as y_true in training process.It is suppposed 
    to be a tensor with three dims (#samples,#priors,#classes+4).
    In this implemention each image only has one GT to predict.So it can safely just pick matched 
    grids through IOU.Grid which has a default box overlapping the ground truth of IOU more than 0.5
    is responsible for this prediction.That`s to say,more than one grid can be matched to one detection
    target.As we have mentioned,because there is only one gt in each image it can be safe to do this match
    by a single loop through all samples.We use vectorized iou calculating and masking for priors and 
    target box.When matched grids are found we still need to calculate the offsets of coordination relative
    to default boxes for them.These offset are what our model is expected to predict.And we also need to 
    set proper class value for these matched grid.
    Input:
        priors: numpy array of shape (#priors,4) 
        ground_truth: numpy array of unprocessed samples (#samples,#objects,#classes + 4)
    Output:
        y:list of shape [#samples, #priors ,#classes + 4]
    '''
    num_priors = len(priors)
    num_samples = len(ground_truth)
    num_classes = ground_truth.shape[1] + 1 # plus 1 because ground_truth doesn`t have background representation 
    y = np.zeros((num_samples, num_priors, num_classes))
    
    classses = ground_truth[:,:-4]
    input_boxes = ground_truth[:,-4:]  # box format is corner
    
    
    # Convert priors from centroid to corner and input from corner to centroid
    corner_priors = coords_convert(priors,"centroids2corners") #get corner format of priors to calculate iou
    center_input_boxes = coords_convert(input_boxes,"corners2centroids") #get centroids format of boxes
    # Precalculate areas to store for iou calculating
    prior_areas = priors[:,2] * priors[:,3]
    input_boxes_areas = center_input_boxes[:,2] * center_input_boxes[:,3]
    for k in range(len(ground_truth)):      
        
        # 1. Calulate iou between one ground_truth and all priors
        #    In this step, we should use corner representation for all related boxes 
        input_boxes_area = input_boxes_areas[k]
        # min of x_max,y_max
        max_xy = np.minimum(corner_priors[:,2:],input_boxes[k,2:])
        # max of x_min,y_min
        min_xy = np.maximum(corner_priors[:,:2],input_boxes[k,:2])
        iner_areas = (max_xy[:,1] - min_xy[:,1]) * (max_xy[:,0] - min_xy[:,0])
        iou = iner_areas / ( input_boxes_area + prior_areas - iner_areas )
        # Set iou[k]  to zero if iner_area[k] < 0
        iou = np.where(iner_areas < 0,0,iou)

        #2. Create masks for matched grids and unmatched grids
        # If there is only one input for np.where, the output are wrapped by a tuple
        matched_grids = np.where(iou > iou_threshold)[0]
        # Do a diff operation to get the unmatched grids
        unmatched_grids = np.setdiff1d(np.arange(0,num_priors),matched_grids)
        #print("num of matched grids",len(matched_grids))
       # print("num of unmatched grids",len(unmatched_grids))
        #3. Compute offsets for each matched default box.This offset are called g_hat in original papers
        #    We should use centroid-boxes for both priors and target presentation
        y[k,matched_grids,-4:] = compute_ghat(center_input_boxes[k], priors[matched_grids,:])
        y[k,matched_grids,1:-4] = classses[k]
        #4. Set 1 for first element in unmatched rows 
        y[k,unmatched_grids,0] = 1
        
    return y

def post_process(y_pred_no_process,priors,top_k = 200,score_thresh = 0.1,iou_thresh = 0.5):
    '''
    A for-loop implemention
    Input: y_pred_no_process (batch_size,#priors,#classes + 4)
    Output: y_pred (batch_size,#result)
    '''
    batch_size = y_pred_no_process.shape[0]
    num_classes = y_pred_no_process.shape[-1] - 4
    y_pred = [] #np.zeros((batch_size,num_classes,top_k,1 + 4))
    
    for k,pred in enumerate(y_pred_no_process):
        
        detections = []
        
        #p#rint(pred.shape,priors.shape)
        # This do three things: 1. Take the offsets 2. Compute the absolute coordinates 3. Convert the format
        bboxes = coords_convert(compute_coords(pred[:,-4:],priors),"centroids2corners")
        bboxes = compute_coords(pred[:,-4:],priors)
        classes = np.argmax(pred[:,:-4], axis= 1)
        scores = pred[:,:-4].max(axis= 1)

        for c in range(1,num_classes):
            c_mask = classes==c
            c_scores = scores[c_mask]
            c_boxes = bboxes[c_mask]
            if len(c_scores) == 0:
                detections.append([])
                continue
            #print('c boxes shape',c_boxes.shape,'\nc scores shape',c_scores.shape)
            keep = NMSBoxes(c_boxes.tolist(), c_scores.tolist(),score_thresh, iou_thresh,top_k = top_k)
            keep = keep.squeeze()
            if keep.shape == ():
                detections.append([])
                continue
            detections.append(np.concatenate([c_scores[keep,np.newaxis],c_boxes[keep]],axis=1)) 
            #print('c_scores.shape',c_scores.shape,'c_boxes.shape',c_boxes.shape)
        y_pred.append(detections)
    return y_pred 



## Calculate accuracy. 
def evaluate(model,x_test,y_test,iou_thresh = 0.5):
    y_pred = model.predict(x_test)
    correct = np.ones((len(y_test),))
    #1. Mask all wrong classification  incorrect
    classes_gt = np.argmax(y_test[:,:-4],axis=1)
    classes_pred = np.argmax(y_pred[:,:-4],axis=1)
    correct_class = np.where(classes_gt == classes_pred,1,0)
    #2. Mask all box with iou < 0.5 incorrect
    correct_loc = np.where(iou(y_pred[:,-4:],y_test[:,-4:]) >= iou_thresh,1,0)
    correct = np.logical_and(correct_class,correct_loc)
    accuracy={
          'class accuracy':round(np.sum(correct_class) / len(correct),4),
          'loc accuracy':round(np.sum(correct_loc) / len(correct),4),
          'accuracy ':round(np.sum(correct) / len(correct),4)
     }
    return accuracy