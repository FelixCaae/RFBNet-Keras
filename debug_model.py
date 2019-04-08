import numpy as np
import tensorflow as tf
import keras.backend as K
def model_analysis(model,start_layer,end_layer,options = ["params","mac","flops"]):
    if "params" in options:
        model_params(model,start_layer,end_layer)
def model_params(model,start_layer,end_layer,count_limit):
    params = np.array([[layer.name,int(layer.count_params())]  for layer in model.layers[start_layer:end_layer] if layer.count_params() > count_limit])
    print(params)
    params = {'name':params[:,0],'params':params[:,1].astype('int')}
    # plt.xticks(range(len(params)), params[:,0])
    plt.bar(range(len(params["params"])),params["params"])

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

