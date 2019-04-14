import numpy as np
import tensorflow as tf
import keras.backend as K
from matplotlib import pyplot as plt
from copy import deepcopy
def model_analysis(model,start_layer,end_layer,options = ["params","mac","flops"]):
    if "params" in options:
        model_params(model,start_layer,end_layer)
def model_params_bar(model,start_layer,end_layer,count_limit,keys = ['rfb_source','rfb_extras','extra_','conf','loc','conv','sep','bn','dep']):
    plt.figure(figsize=(10,5))
    params = np.array([[layer.name,int(layer.count_params())]  for layer in model.layers[-1].layers[16*9+3:]])
    params_count = [np.sum([int(param[1]) for param in params if key in param[0]]) for key in keys]
    plt.bar(keys,params_count)

def model_params_of_layer(model,start_layer,end_layer,count_limit,key ):
    params = np.array([[layer.name,int(layer.count_params()),layer.input,layer.output.shape.as_list()]  for layer in model.layers[-1].layers[16*9+3:] if key in layer.name])
#     params_count = [np.sum([int(param[1]) for param in params if key in param[0]]) for key in keys]
    print(params)

    
def model_flops(model,start_layer,end_layer,count_limit):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


def model_mac(model,start_layer,end_layer,count_limit):
    pass

def model_speed(model,input_shape,batch_size_range=[1,2,4,8],rounds = 100):
    batch_n = rounds
    img_height = input_shape[0]
    img_width = input_shape[1]
    img_channels = input_shape[2]
    # detector = Detect(num_classes,0,cfg)
#     batch_size_range = [1,2,4,8,16,32]#[1,2,4,8]
    print('Start testing...')
    for batch_size in batch_size_range:
        time_total = 0
        time_net = 0
        time_post = 0
        print('batch size',batch_size)
        for i in range(0,batch_n):
            test_batch = np.random.rand(batch_size,img_height,img_width,img_channels)
            click = time.time()
            y_pred = model.predict(test_batch)
            time_net += (time.time() - click)
            click = time.time()
            y_pred_decoded = decode_detections(y_pred,
                              priors,variances,
                              img_height = input_H,
                              img_width = input_W,
                              confidence_thresh = 0.5,
                              iou_threshold = 0.45)
            time_post += (time.time() - click)
        time_total = time_net + time_post
        print('Time cost per batch: %.3f FPS: %.1f'%(time_total / batch_n,batch_n*batch_size/ time_total))
        print('Time(pure forward) cost per batch: %.3f FPS: %.1F'%(time_net / batch_n, batch_n*batch_size/ time_net))


def mask_prediction(prediction,select_features,predictor_sizes,prior_config,num_classes):
    prediction = deepcopy(np.array(prediction))
    predictor_sizes = [size[0] * size[1] * prior_config[k] for k,size in enumerate(predictor_sizes)]
    size_pyramid = [np.sum(predictor_sizes[: k + 1]) for k in range(len(predictor_sizes))]
    size_pyramid.insert(0,0)
    for i in range(len(prior_config)):
        if i in select_features:
            continue
        prediction[:,size_pyramid[i]:size_pyramid[i+1],0] = 1
        
    return prediction
        