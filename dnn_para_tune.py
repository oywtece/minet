# this script tunes model paras

import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import config_amazon as cfg
import os
import shutil

# data format
# label, tar
# if your data contain more than the required, the remaining is simply ignored
# example: your data: "label, tar, tar clk, src clk"
# then this code only uses "label, tar"

# config
str_txt = cfg.output_file_name
base_path = './tmp'
model_saving_addr = base_path + '/dnn_' + str_txt + '/'
save_model_ind = cfg.save_model_ind
num_csv_col = cfg.num_csv_col
train_file_name = cfg.train_file_name
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
n_ft = cfg.n_ft
k = cfg.k
kp_prob = cfg.kp_prob
n_epoch = cfg.n_epoch
# max_num_lower_ct = cfg.max_num_lower_ct
record_step_size = cfg.record_step_size
layer_dim = cfg.layer_dim
opt_alg = cfg.opt_alg
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
max_len_per_slot = cfg.max_len_per_slot
input_format = cfg.input_format
rnd_seed = cfg.rnd_seed

label_col_idx = 0
record_defaults = [[0]]*num_csv_col
record_defaults[0] = [0.0]
total_num_ft_col = num_csv_col - 1

eta_range = cfg.eta_range

## create para list
para_list = []
for i in range(len(eta_range)):
    para_list.append([eta_range[i]])

## record results
result_list = []

# loop over para_list
for item in para_list:
    tf.reset_default_graph()
    eta = item[0]
        
    # create dir
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    # remove dir
    if os.path.isdir(model_saving_addr):
        shutil.rmtree(model_saving_addr)
    
    ###########################################################
    ###########################################################
    
    print('Loading data start!')
    tf.set_random_seed(rnd_seed)
    
    if input_format == 'csv':
        train_ft, train_label = func.tf_input_pipeline(train_file_name, batch_size, n_epoch, label_col_idx, record_defaults)
        test_ft, test_label = func.tf_input_pipeline_test(test_file_name, batch_size, 1, label_col_idx, record_defaults)
    elif input_format == 'tfrecord':
        train_ft, train_label = func.tfrecord_input_pipeline(train_file_name, num_csv_col, batch_size, n_epoch)    
        test_ft, test_label = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col, batch_size, 1)
    print('Loading data done!')
    
    ########################################################################
    # add mask
    def get_masked_one_hot(x_input_one_hot):
        data_mask = tf.cast(tf.greater(x_input_one_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 2)
        data_mask = tf.tile(data_mask, (1,1,k))
        # output: (?, n_one_hot_slot, k)
        data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot)
        data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
        return data_embed_one_hot_masked
    
    def get_masked_mul_hot(x_input_mul_hot):
        data_mask = tf.cast(tf.greater(x_input_mul_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 3)
        data_mask = tf.tile(data_mask, (1,1,1,k))
        # output: (?, n_mul_hot_slot, max_len_per_slot, k)
        data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot)
        data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
        # move reduce_sum here
        data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 2)
        return data_embed_mul_hot_masked
    
    # output: (?, n_one_hot_slot + n_mul_hot_slot, k)
    def get_concate_embed(x_input_one_hot, x_input_mul_hot):
        data_embed_one_hot = get_masked_one_hot(x_input_one_hot)
        data_embed_mul_hot = get_masked_mul_hot(x_input_mul_hot)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 1)
        return data_embed_concat
    
    # input: (?, n_slot, k)
    # output: (?, 1)
    def get_dnn_output(data_embed_concat):
        # include output layer
        n_layer = len(layer_dim)
        data_embed_dnn = tf.reshape(data_embed_concat, [-1, (n_one_hot_slot + n_mul_hot_slot)*k])
        cur_layer = data_embed_dnn
        # loop to create DNN struct
        for i in range(0, n_layer):
            # output layer, linear activation
            if i == n_layer - 1:
                cur_layer = tf.matmul(cur_layer, weight_dict[i]) #+ bias_dict[i]
            else:
                cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict[i])) # + bias_dict[i])
                cur_layer = tf.nn.dropout(cur_layer, keep_prob)
        
        y_hat = cur_layer
        return y_hat
    
    ###########################################################
    # only use part of data
    idx_1 = n_one_hot_slot
    idx_2 = idx_1 + n_mul_hot_slot*max_len_per_slot
    
    x_input = tf.placeholder(tf.int32, shape=[None, total_num_ft_col])
    # shape=[None, n_one_hot_slot]
    x_input_one_hot = x_input[:, 0:idx_1]
    x_input_mul_hot = x_input[:, idx_1:idx_2]
    # shape=[None, n_mul_hot_slot, max_len_per_slot]
    x_input_mul_hot = tf.reshape(x_input_mul_hot, (-1, n_mul_hot_slot, max_len_per_slot))
    
    y_target = tf.placeholder(tf.float32, shape=[None, 1])
    # dropout keep prob
    keep_prob = tf.placeholder(tf.float32)
    
    # emb_mat dim add 1 -> for padding (idx = 0)
    with tf.device('/cpu:0'):
        emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))
    
    ################################
    # include output layer
    n_layer = len(layer_dim)
    in_dim = (n_one_hot_slot + n_mul_hot_slot)*k
    weight_dict={}
#     bias_dict={}
    
    # loop to create DNN vars
    for i in range(0, n_layer):
        out_dim = layer_dim[i]
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        weight_dict[i] = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
#         bias_dict[i] = tf.Variable(tf.constant(0.0, shape=[out_dim]))
        in_dim = layer_dim[i]
    
    ####### DNN ########
    data_embed_concat = get_concate_embed(x_input_one_hot, x_input_mul_hot)
    y_hat = get_dnn_output(data_embed_concat)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target))
    
    #############################
    # prediction
    #############################
    pred_score = tf.sigmoid(y_hat)
    
    if opt_alg == 'Adam':
        optimizer = tf.train.AdamOptimizer(eta).minimize(loss)
    else:
        # default
        optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)
    
    ########################################
    # Launch the graph
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
      
        train_loss_list = []

        func.print_time()
        print('Start train loop')
        
        epoch = -1
        try:
            while not coord.should_stop():           
                epoch += 1  
                train_ft_inst, train_label_inst = sess.run([train_ft, train_label])
                if input_format == 'csv':
                    train_label_inst = np.transpose([train_label_inst])
                sess.run(optimizer, feed_dict={x_input:train_ft_inst, \
                                               y_target:train_label_inst, keep_prob:kp_prob})
        
                # record loss and accuracy every step_size generations
                if (epoch+1)%record_step_size == 0:
                    train_loss_temp = sess.run(loss, feed_dict={ \
                                               x_input:train_ft_inst, \
                                               y_target:train_label_inst, keep_prob:1.0})
                    train_loss_list.append(train_loss_temp)
                    
                    auc_and_loss = [epoch+1, train_loss_temp]
                    auc_and_loss = [np.round(xx,4) for xx in auc_and_loss]
                    func.print_time() 
                    print('Generation # {}. Train Loss: {:.4f}.'\
                          .format(*auc_and_loss))
                    
        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done training -- epoch limit reached')
            # whether to save the model
            if save_model_ind == 1:                          
                saver = tf.train.Saver() 
                save_path = saver.save(sess, model_saving_addr)
                print("Model saved in file: %s" % save_path)
         
        # load test data
        test_pred_score_all = []
        test_label_all = []
        test_loss_all = []
        
        try:
            while True:
                test_ft_inst, test_label_inst = sess.run([test_ft, test_label])
                if input_format == 'csv':
                    test_label_inst = np.transpose([test_label_inst])
                cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                        x_input:test_ft_inst, keep_prob:1.0})
                test_pred_score_all.append(cur_test_pred_score.flatten())
                test_label_all.append(test_label_inst)
                
                cur_test_loss = sess.run(loss, feed_dict={ \
                                        x_input:test_ft_inst, \
                                        y_target:test_label_inst, keep_prob:1.0})
                test_loss_all.append(cur_test_loss)
    
        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done testing -- epoch limit reached')    
        finally:
            coord.request_stop()
        coord.join(threads)
             
        # calculate auc
        test_pred_score_re = func.list_flatten(test_pred_score_all)
        test_label_re = func.list_flatten(test_label_all)
        test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
        test_rmse = func.cal_rmse(test_pred_score_re, test_label_re)
        test_loss = np.mean(test_loss_all)
        
        # rounding
        test_auc = np.round(test_auc, 4)
        test_rmse = np.round(test_rmse, 4)
        test_loss = np.round(test_loss, 5)
        train_loss_list = [np.round(xx,4) for xx in train_loss_list]
          
        print('test_auc = ', test_auc)
        print('test_rmse =', test_rmse)
        print('test_loss =', test_loss)
        print('train_loss_list =', train_loss_list)

        # append to result_list
        result_list.append([eta, test_auc, test_loss])

fmt_str = '{:<6}\t{:<6}\t{:<6}'
header_row = ['eta', 'auc', 'loss']
print(fmt_str.format(*header_row))

for i in range(len(result_list)):
    tmp = result_list[i]
    print(fmt_str.format(*tmp))

