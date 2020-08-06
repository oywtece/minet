'''
Tensorflow implementation of MiNet described in:
[CIKM 2020] MiNet: Mixed Interest Network for Cross-Domain Click-Through Rate Prediction

Code logic: read_config -> load data -> define functions -> define placeholders and variables 
-> define computation graph -> launch computation graph (run session)
-> train -> test -> print results
'''

import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import config_amazon as cfg
import os
import shutil

# data format
# label, tar, tar clk, src clk

###############################################
# config
str_txt = cfg.output_file_name
save_model_ind = cfg.save_model_ind
base_path = './tmp'
model_saving_addr = base_path + '/minet_' + str_txt + '/'
input_format = cfg.input_format
rnd_seed = cfg.rnd_seed

n_ft = cfg.n_ft
k = cfg.k
kp_prob = cfg.kp_prob
pool_mode = cfg.pool_mode
n_epoch = cfg.n_epoch
record_step_size = cfg.record_step_size
opt_alg = cfg.opt_alg
item_att_hidden_dim = cfg.item_att_hidden_dim
interest_att_hidden_dim = cfg.interest_att_hidden_dim

## dataset 1
train_file_name_1 = cfg.train_file_name_1
test_file_name_1 = cfg.test_file_name_1
batch_size_1 = cfg.batch_size_1
layer_dim_1 = cfg.layer_dim_1
n_one_hot_slot_1 = cfg.n_one_hot_slot_1
n_mul_hot_slot_1 = cfg.n_mul_hot_slot_1
max_len_per_slot_1 = cfg.max_len_per_slot_1
label_col_idx_1 = 0
num_csv_col_1 = cfg.num_csv_col_1
record_defaults_1 = [[0]]*num_csv_col_1
record_defaults_1[0] = [0.0]
total_num_ft_col_1 = num_csv_col_1 - 1
total_embed_dim_1 = (n_one_hot_slot_1 + n_mul_hot_slot_1)*k
inter_dim = cfg.inter_dim
user_b_ini = cfg.user_b_ini
tar_clk_b_ini = cfg.tar_clk_b_ini
src_clk_b_ini = cfg.src_clk_b_ini

# ori val in dataset
max_n_clk_ori_1 = cfg.max_n_clk_ori_1
max_n_clk_ori_2 = cfg.max_n_clk_ori_2 # in dataset 1
# actual val used in experiment
max_n_clk_1 = cfg.max_n_clk_1
max_n_clk_2 = cfg.max_n_clk_2

# n_user_slot_1 == n_user_slot_2
n_user_slot = cfg.n_user_slot
total_embed_dim_user = n_user_slot*k

## dataset 2
train_file_name_2 = cfg.train_file_name_2
test_file_name_2 = cfg.test_file_name_2
batch_size_2 = cfg.batch_size_2
layer_dim_2 = cfg.layer_dim_2
n_one_hot_slot_2 = cfg.n_one_hot_slot_2
n_mul_hot_slot_2 = cfg.n_mul_hot_slot_2
max_len_per_slot_2 = cfg.max_len_per_slot_2
label_col_idx_2 = 0
num_csv_col_2 = cfg.num_csv_col_2
record_defaults_2 = [[0]]*num_csv_col_2
record_defaults_2[0] = [0.0]
total_num_ft_col_2 = num_csv_col_2 - 1
total_embed_dim_2 = (n_one_hot_slot_2 + n_mul_hot_slot_2)*k

wgt_1 = cfg.wgt_1
wgt_2_range = cfg.wgt_2_range
eta_range = cfg.eta_range
###############################################
## create para list
para_list = []
for i in range(len(wgt_2_range)):
    for ii in range(len(eta_range)):
        para_list.append([wgt_2_range[i], eta_range[ii]])

## record results
result_list = []

# loop over para_list
for item in para_list:
    tf.reset_default_graph()
    
    wgt_2 = item[0]
    eta = item[1]

    # create dir
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    # remove dir
    if os.path.isdir(model_saving_addr):
        shutil.rmtree(model_saving_addr)
    
    ###########################################################
    ###########################################################
    # if input is tfrecord format
    print('Loading data start!')
    tf.set_random_seed(rnd_seed)
    
    if input_format == 'csv':
        # load data set 1
        train_ft_1, train_label_1 = func.tf_input_pipeline(train_file_name_1, batch_size_1, n_epoch, \
                                                           label_col_idx_1, record_defaults_1)
        test_ft_1, test_label_1 = func.tf_input_pipeline_test(test_file_name_1, batch_size_1, 1, \
                                                              label_col_idx_1, record_defaults_1)
        # load data set 2
        train_ft_2, train_label_2 = func.tf_input_pipeline(train_file_name_2, batch_size_2, n_epoch, \
                                                           label_col_idx_2, record_defaults_2)
        test_ft_2, test_label_2 = func.tf_input_pipeline_test(test_file_name_2, batch_size_2, 1, \
                                                              label_col_idx_2, record_defaults_2)       
    
    elif input_format == 'tfrecord':
        train_ft_1, train_label_1 = func.tfrecord_input_pipeline(train_file_name_1, num_csv_col_1, \
                                    batch_size_1, n_epoch)
        test_ft_1, test_label_1 = func.tfrecord_input_pipeline_test(test_file_name_1, num_csv_col_1, \
                                    batch_size_1, 1)
        train_ft_2, train_label_2 = func.tfrecord_input_pipeline(train_file_name_2, num_csv_col_2, \
                                    batch_size_2, n_epoch)
        test_ft_2, test_label_2 = func.tfrecord_input_pipeline_test(test_file_name_2, num_csv_col_2, \
                                    batch_size_2, 1)
    
    ########################################################################
    # data format (label is removed from x_input)
    # tar, tar clk, src clk
    def partition_input_1(x_input):
        # generate idx_list
        len_list = []
        # tar
        len_list.append(n_user_slot + n_one_hot_slot_1)
        len_list.append(n_mul_hot_slot_1*max_len_per_slot_1)
        
        # tar domain clk
        for _ in range(max_n_clk_ori_1):
            len_list.append(n_one_hot_slot_1)
            len_list.append(n_mul_hot_slot_1*max_len_per_slot_1)
        
        # src domain clk
        for _ in range(max_n_clk_ori_2):
            len_list.append(n_one_hot_slot_2)
            len_list.append(n_mul_hot_slot_2*max_len_per_slot_2)
        
        len_list = np.array(len_list)
        idx_list = np.cumsum(len_list)
        
        # if do not differentiate user fts vs. item fts, x_input_one_hot = x_input[:, 0:idx_list[0]]
        x_input_first_part = x_input[:, 0:idx_list[0]]
        # shape=[None, n_user_slot]
        x_input_user = x_input_first_part[:, 0:n_user_slot]
        # shape=[None, n_one_hot_slot]
        x_input_one_hot = x_input_first_part[:, n_user_slot:]
        
        x_input_mul_hot = x_input[:, idx_list[0]:idx_list[1]]
        # shape=[None, n_mul_hot_slot, max_len_per_slot]
        x_input_mul_hot = tf.reshape(x_input_mul_hot, [-1, n_mul_hot_slot_1, max_len_per_slot_1])
        
        #######################
        # tar domain clk
        concat_one_hot_1 = x_input[:, idx_list[1]:idx_list[2]]
        concat_mul_hot_1 = x_input[:, idx_list[2]:idx_list[3]]
        for i in range(1, max_n_clk_ori_1):
            # one_hot
            temp_1 = x_input[:, idx_list[2*i+1]:idx_list[2*i+2]]
            concat_one_hot_1 = tf.concat([concat_one_hot_1, temp_1], 1)
            
            # mul_hot
            temp_2 = x_input[:, idx_list[2*i+2]:idx_list[2*i+3]]
            concat_mul_hot_1 = tf.concat([concat_mul_hot_1, temp_2], 1)
        
        # shape=[None, max_n_clk, n_one_hot_slot]
        concat_one_hot_1 = tf.reshape(concat_one_hot_1, [-1, max_n_clk_ori_1, n_one_hot_slot_1])
    
        # shape=[None, max_n_clk, n_mul_hot_slot, max_len_per_slot]
        concat_mul_hot_1 = tf.reshape(concat_mul_hot_1, [-1, max_n_clk_ori_1, n_mul_hot_slot_1, \
                max_len_per_slot_1])
        
        x_input_one_hot_1 = concat_one_hot_1[:, 0:max_n_clk_1, :]
        x_input_mul_hot_1 = concat_mul_hot_1[:, 0:max_n_clk_1, :, :]
        
        #######################
        # src domain clk
        base_idx = 2*(max_n_clk_ori_1+1) - 1
        concat_one_hot_2 = x_input[:, idx_list[base_idx]:idx_list[base_idx+1]]
        concat_mul_hot_2 = x_input[:, idx_list[base_idx+1]:idx_list[base_idx+2]]
        
        for i in range(1, max_n_clk_ori_2):
            # one_hot
            temp_3 = x_input[:, idx_list[2*i+base_idx]:idx_list[2*i+base_idx+1]]
            concat_one_hot_2 = tf.concat([concat_one_hot_2, temp_3], 1)
            
            # mul_hot
            temp_4 = x_input[:, idx_list[2*i+base_idx+1]:idx_list[2*i+base_idx+2]]
            concat_mul_hot_2 = tf.concat([concat_mul_hot_2, temp_4], 1)        
    
        # shape=[None, max_n_clk, n_one_hot_slot]
        concat_one_hot_2 = tf.reshape(concat_one_hot_2, [-1, max_n_clk_ori_2, n_one_hot_slot_2])
    
        # shape=[None, max_n_clk, n_mul_hot_slot, max_len_per_slot]
        concat_mul_hot_2 = tf.reshape(concat_mul_hot_2, [-1, max_n_clk_ori_2, n_mul_hot_slot_2, \
                max_len_per_slot_2])
        
        x_input_one_hot_2 = concat_one_hot_2[:, 0:max_n_clk_2, :]
        x_input_mul_hot_2 = concat_mul_hot_2[:, 0:max_n_clk_2, :, :]
            
        return x_input_user, x_input_one_hot, x_input_mul_hot, x_input_one_hot_1, x_input_mul_hot_1, \
            x_input_one_hot_2, x_input_mul_hot_2 

    def partition_input_2(x_input):
        idx_a = n_user_slot
        idx_b = idx_a + n_one_hot_slot_2
        idx_c = idx_b + n_mul_hot_slot_2*max_len_per_slot_2
        x_input_user = x_input[:, 0:idx_a]
        x_input_one_hot = x_input[:, idx_a:idx_b]
        x_input_mul_hot = x_input[:, idx_b:idx_c]
        # shape=[None, n_mul_hot_slot, max_len_per_slot]
        x_input_mul_hot = tf.reshape(x_input_mul_hot, [-1, n_mul_hot_slot_2, max_len_per_slot_2])
        return x_input_user, x_input_one_hot, x_input_mul_hot
    
    # count number of valid (i.e., not padded with all 0) clicked ads
    # output: none*1
    def count_n_valid_clk(x_input_one_hot_clk):
        # none * max_n_clk * total_embed_dim
        data_mask_a = tf.cast(tf.greater(x_input_one_hot_clk, 0), tf.float32)
        # none * max_n_clk
        data_mask_a_reduce_sum = tf.reduce_sum(data_mask_a, 2)
        data_mask_b = tf.cast(tf.greater(data_mask_a_reduce_sum, 0), tf.float32)
        # none * 1
        n_valid = 1.0*tf.reduce_sum(data_mask_b, 1, keep_dims=True)
        return n_valid
    
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
    
    # output: (?, (n_one_hot_slot + n_mul_hot_slot)*k)
    def get_concate_embed(x_input_one_hot, x_input_mul_hot, total_embed_dim):
        data_embed_one_hot = get_masked_one_hot(x_input_one_hot)
        data_embed_mul_hot = get_masked_mul_hot(x_input_mul_hot)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 1)
        data_embed_concat = tf.reshape(data_embed_concat, [-1, total_embed_dim])
        return data_embed_concat
    
    def get_user_embed(x_input_user, total_embed_dim_user):
        data_embed_user = get_masked_one_hot(x_input_user)
        data_embed_user = tf.reshape(data_embed_user, [-1, total_embed_dim_user])
        return data_embed_user

    def get_masked_one_hot_clk(x_input_one_hot_clk):
        data_mask = tf.cast(tf.greater(x_input_one_hot_clk, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 3)
        data_mask = tf.tile(data_mask, (1,1,1,k))
        # output: (?, max_n_clk, n_one_hot_slot, k)
        data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot_clk)
        data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
        return data_embed_one_hot_masked
    
    def get_masked_mul_hot_clk(x_input_mul_hot_clk):
        data_mask = tf.cast(tf.greater(x_input_mul_hot_clk, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 4)
        data_mask = tf.tile(data_mask, (1,1,1,1,k))
        # output: (?, max_n_clk, n_mul_hot_slot, max_len_per_slot, k)
        data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot_clk)
        data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
        # output: (?, max_n_clk, n_mul_hot_slot, k)
        data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 3)
        return data_embed_mul_hot_masked
    
    # output: (?, max_n_clk, (n_one_hot_slot + n_mul_hot_slot)*k)
    def get_concate_embed_clk(x_input_one_hot_clk, x_input_mul_hot_clk, max_n_clk, total_embed_dim):
        data_embed_one_hot = get_masked_one_hot_clk(x_input_one_hot_clk)
        data_embed_mul_hot = get_masked_mul_hot_clk(x_input_mul_hot_clk)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 2)
        data_embed_concat = tf.reshape(data_embed_concat, [-1, max_n_clk, total_embed_dim])
        return data_embed_concat

    def reshape_data_user_embed(data_embed_1, data_embed_user, max_n_clk):
        data_embed_1_exp = tf.expand_dims(data_embed_1, 1)
        # tile, dim: none * max_n_clk * total_embed_dim_1
        data_embed_1_tile = tf.tile(data_embed_1_exp, [1, max_n_clk, 1])
        data_embed_1_re = tf.reshape(data_embed_1_tile, [-1, total_embed_dim_1])
        
        data_embed_user_exp = tf.expand_dims(data_embed_user, 1)
        # tile, dim: none * max_n_clk * total_embed_dim_user
        data_embed_user_tile = tf.tile(data_embed_user_exp, [1, max_n_clk, 1])
        data_embed_user_re = tf.reshape(data_embed_user_tile, [-1, total_embed_dim_user])
        return data_embed_1_re, data_embed_user_re

    # input: (?, max_n_clk, total_embed_dim)
    # output: (?, total_embed_dim)
    # n_valid: (?, 1)
    def item_level_att_1(data_embed_clk_1, data_embed_1_re, data_embed_user_re, \
                                pool_mode, x_input_one_hot_clk_1):
        # convert to 2D
        data_embed_clk_1_re = tf.reshape(data_embed_clk_1, [-1, total_embed_dim_1])
        cur_input = tf.concat([data_embed_clk_1_re, data_embed_1_re, data_embed_user_re, \
                               tf.multiply(data_embed_clk_1_re, data_embed_1_re)], 1)  
        hidden = tf.nn.relu(tf.matmul(cur_input, V_1))
        weight = tf.matmul(hidden, vv_1)
        weight = tf.reshape(weight, [-1, max_n_clk_1, 1])
        
        nlz_wgt_clk_1 = tf.nn.softmax(weight, dim=1)
        data_embed_clk_1_ig = data_embed_clk_1 * nlz_wgt_clk_1
        
        # output: (?, total_embed_dim_1)
        if pool_mode == 'sum':
            data_embed_agg_1_ig = tf.reduce_sum(data_embed_clk_1_ig, 1)
        elif pool_mode == 'avg':
            n_valid_1 = count_n_valid_clk(x_input_one_hot_clk_1)
            data_embed_agg_1_ig = tf.reduce_sum(data_embed_clk_1_ig, 1) / (n_valid_1 + 1e-5)
        elif pool_mode == 'max':
            data_embed_agg_1_ig = tf.reduce_max(data_embed_clk_1_ig, 1)
        return data_embed_agg_1_ig

    def item_level_att_2(data_embed_clk_2, data_embed_1_re, data_embed_user_re, \
                                pool_mode, x_input_one_hot_clk_2):
        data_embed_clk_2_re = tf.reshape(data_embed_clk_2, [-1, total_embed_dim_2])
        # ?, total_embed_dim_2 -> ?, total_embed_dim_1
        temp_mat_ab = tf.matmul(tf.matmul(data_embed_clk_2_re, H_a), H_b)
        cur_input = tf.concat([data_embed_clk_2_re, data_embed_1_re, data_embed_user_re, \
                               tf.multiply(temp_mat_ab, data_embed_1_re)], 1)
        hidden = tf.nn.relu(tf.matmul(cur_input, V_2))
        weight = tf.matmul(hidden, vv_2)
        weight = tf.reshape(weight, [-1, max_n_clk_2, 1])
        
        nlz_wgt_clk_2 = tf.nn.softmax(weight, dim=1)
        data_embed_clk_2_ig = data_embed_clk_2 * nlz_wgt_clk_2
        
        # output: (?, total_embed_dim_2)
        if pool_mode == 'sum':
            data_embed_agg_2_ig = tf.reduce_sum(data_embed_clk_2_ig, 1)
        elif pool_mode == 'avg':
            n_valid_2 = count_n_valid_clk(x_input_one_hot_clk_2)
            data_embed_agg_2_ig = tf.reduce_sum(data_embed_clk_2_ig, 1) / (n_valid_2 + 1e-5)
        elif pool_mode == 'max':
            data_embed_agg_2_ig = tf.reduce_max(data_embed_clk_2_ig, 1)        
        return data_embed_agg_2_ig

    def interest_level_att(data_embed_1, data_embed_user_1, data_embed_clk_1_agg, \
                            data_embed_clk_2_agg):
        data_embed_z = tf.concat([data_embed_1, data_embed_user_1, data_embed_clk_1_agg, \
                                  data_embed_clk_2_agg], 1)
        hidden_0 = tf.nn.relu(tf.matmul(data_embed_z, dw_0))
        weight_0 = tf.exp(tf.matmul(hidden_0, dh_0) + db_0)
        hidden_1 = tf.nn.relu(tf.matmul(data_embed_z, dw_1))
        weight_1 = tf.exp(tf.matmul(hidden_1, dh_1) + db_1)
        hidden_2 = tf.nn.relu(tf.matmul(data_embed_z, dw_2))
        weight_2 = tf.exp(tf.matmul(hidden_2, dh_2) + db_2)
        data_embed_dg = tf.concat([data_embed_1, weight_0*data_embed_user_1, \
                                   weight_1*data_embed_clk_1_agg, weight_2*data_embed_clk_2_agg], 1)
        return data_embed_dg

    # input: (?, total_embed_dim); output: (?, 1)
    def get_y_hat(data_embed, weight_dict, layer_dim):
        # include output layer
        n_layer = len(layer_dim)
        cur_layer = data_embed
        # loop to create DNN struct
        for i in range(0, n_layer):
            # output layer, linear activation
            if i == n_layer - 1:
                cur_layer = tf.matmul(cur_layer, weight_dict[i])
            else:
                cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict[i]))
                cur_layer = tf.nn.dropout(cur_layer, keep_prob)
        
        y_hat = cur_layer
        return y_hat

    ##########################################################    
    ##########################################################
    # Define placeholders and variables
    x_input_1 = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_1])
    y_target_1 = tf.placeholder(tf.float32, shape=[None, 1])
    
    x_input_2 = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_2])
    y_target_2 = tf.placeholder(tf.float32, shape=[None, 1])
    
    # dropout keep prob
    keep_prob = tf.placeholder(tf.float32)
    
    # emb_mat dim add 1 -> for padding (idx = 0)
    with tf.device('/cpu:0'):
        emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))
    
    # parameters - item att
    in_dim = 3*total_embed_dim_1 + total_embed_dim_user
    out_dim = item_att_hidden_dim
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))
    V_1 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

    in_dim = item_att_hidden_dim
    out_dim = 1
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))
    vv_1 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    
    in_dim = 2*total_embed_dim_1 + total_embed_dim_2 + total_embed_dim_user
    out_dim = item_att_hidden_dim
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))
    V_2 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

    in_dim = item_att_hidden_dim
    out_dim = 1
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))
    vv_2 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

    in_dim = total_embed_dim_2
    out_dim = inter_dim
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))
    H_a = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    
    in_dim = inter_dim
    out_dim = total_embed_dim_1
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))
    H_b = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))    
        
    # parameters - interest att
    total_embed_dim_z = 2*total_embed_dim_1 + total_embed_dim_2 + total_embed_dim_user
    
    in_dim = total_embed_dim_z
    out_dim = interest_att_hidden_dim
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))
    dw_0 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    dw_1 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    dw_2 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    
    in_dim = interest_att_hidden_dim
    out_dim = 1
    cur_range = np.sqrt(6.0 / (in_dim + out_dim))    
    dh_0 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    dh_1 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    dh_2 = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
    db_0 = tf.Variable(tf.constant(user_b_ini, shape=[out_dim]))
    db_1 = tf.Variable(tf.constant(tar_clk_b_ini, shape=[out_dim]))
    db_2 = tf.Variable(tf.constant(src_clk_b_ini, shape=[out_dim]))

    ################################
    # include output layer
    n_layer_1 = len(layer_dim_1)
    in_dim_1 = total_embed_dim_z
    weight_dict_1={}

    # loop to create DNN vars
    for i in range(0, n_layer_1):
        out_dim_1 = layer_dim_1[i]
        cur_range = np.sqrt(6.0 / (in_dim_1 + out_dim_1))
        weight_dict_1[i] = tf.Variable(tf.random_uniform([in_dim_1, out_dim_1], -cur_range, cur_range))
        in_dim_1 = layer_dim_1[i]
    
    n_layer_2 = len(layer_dim_2)
    in_dim_2 = total_embed_dim_2 + total_embed_dim_user
    weight_dict_2={}
    
    for i in range(0, n_layer_2):
        out_dim_2 = layer_dim_2[i]
        cur_range = np.sqrt(6.0 / (in_dim_2 + out_dim_2))
        weight_dict_2[i] = tf.Variable(tf.random_uniform([in_dim_2, out_dim_2], -cur_range, cur_range))
        in_dim_2 = layer_dim_2[i]
    
    ##########################################################    
    ##########################################################
    # MiNet
    x_input_user_1, x_input_one_hot, x_input_mul_hot, x_input_one_hot_1, x_input_mul_hot_1, \
        x_input_one_hot_2, x_input_mul_hot_2 \
        = partition_input_1(x_input_1)
    data_embed_user_1 = get_user_embed(x_input_user_1, total_embed_dim_user)
    data_embed_1 = get_concate_embed(x_input_one_hot, x_input_mul_hot, total_embed_dim_1)
    data_embed_clk_1 = get_concate_embed_clk(x_input_one_hot_1, x_input_mul_hot_1, max_n_clk_1, total_embed_dim_1)
    data_embed_clk_2 = get_concate_embed_clk(x_input_one_hot_2, x_input_mul_hot_2, max_n_clk_2, total_embed_dim_2)

    data_embed_1_re_1, data_embed_user_1_re_1 = reshape_data_user_embed(data_embed_1, data_embed_user_1, max_n_clk_1)
    data_embed_1_re_2, data_embed_user_1_re_2 = reshape_data_user_embed(data_embed_1, data_embed_user_1, max_n_clk_2)
    
    # item-level att
    data_embed_agg_1_ig = item_level_att_1(data_embed_clk_1, data_embed_1_re_1, data_embed_user_1_re_1, \
                        pool_mode, x_input_one_hot_1)
    data_embed_agg_2_ig = item_level_att_2(data_embed_clk_2, data_embed_1_re_2, data_embed_user_1_re_2, \
                        pool_mode, x_input_one_hot_2)

    # interest-level att
    data_embed_1_final = interest_level_att(data_embed_1, data_embed_user_1, data_embed_agg_1_ig,\
                            data_embed_agg_2_ig)
    
    x_input_user_s, x_input_one_hot_s, x_input_mul_hot_s = partition_input_2(x_input_2)
    data_embed_user_2 = get_user_embed(x_input_user_s, total_embed_dim_user)
    data_embed_2 = get_concate_embed(x_input_one_hot_s, x_input_mul_hot_s, total_embed_dim_2)
    # final input
    data_embed_2_final = tf.concat([data_embed_2, data_embed_user_2], 1)
    
    y_hat_1 = get_y_hat(data_embed_1_final, weight_dict_1, layer_dim_1)
    y_hat_2 = get_y_hat(data_embed_2_final, weight_dict_2, layer_dim_2)
    loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_1, labels=y_target_1))
    loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_2, labels=y_target_2))
    loss = wgt_1*loss_1 + wgt_2*loss_2
    
    #############################
    # prediction
    #############################
    pred_score_1 = tf.sigmoid(y_hat_1)
    pred_score_2 = tf.sigmoid(y_hat_2)
    
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
    
        func.print_time()
        print('Start train loop')
        try:
            while not coord.should_stop():           
                train_ft_inst_1, train_label_inst_1 = sess.run([train_ft_1, train_label_1])  
                train_ft_inst_2, train_label_inst_2 = sess.run([train_ft_2, train_label_2])
                if input_format == 'csv':
                    train_label_inst_1 = np.transpose([train_label_inst_1])
                    train_label_inst_2 = np.transpose([train_label_inst_2])
                            
                sess.run(optimizer, feed_dict={x_input_1:train_ft_inst_1, \
                                               y_target_1:train_label_inst_1, \
                                               x_input_2:train_ft_inst_2, \
                                               y_target_2:train_label_inst_2, \
                                               keep_prob:kp_prob})
        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done training -- epoch limit reached')
            # whether to save the model
            if save_model_ind == 1:                          
                saver = tf.train.Saver() 
                save_path = saver.save(sess, model_saving_addr)
                print("Model saved in file: %s" % save_path)

        # load test data
        test_pred_score_all_1 = []
        test_label_all_1 = []
        test_loss_all_1 = []
        try:
            while True:
                test_ft_inst_1, test_label_inst_1 = sess.run([test_ft_1, test_label_1])
                if input_format == 'csv':
                    test_label_inst_1 = np.transpose([test_label_inst_1])
                cur_test_pred_score_1 = sess.run(pred_score_1, feed_dict={ \
                                        x_input_1:test_ft_inst_1, keep_prob:1.0})
                test_pred_score_all_1.append(cur_test_pred_score_1.flatten())
                test_label_all_1.append(test_label_inst_1)
                
                # y_target_1: np.transpose([test_label_inst_1])
                cur_test_loss_1 = sess.run(loss_1, feed_dict={ \
                                        x_input_1:test_ft_inst_1, \
                                        y_target_1: test_label_inst_1, keep_prob:1.0})
                test_loss_all_1.append(cur_test_loss_1)
    
        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done testing 1 -- epoch limit reached')
    
        test_pred_score_all_2 = []
        test_label_all_2 = []
        test_loss_all_2 = []
        try:
            while True:
                test_ft_inst_2, test_label_inst_2 = sess.run([test_ft_2, test_label_2])
                if input_format == 'csv':
                    test_label_inst_2 = np.transpose([test_label_inst_2])
                cur_test_pred_score_2 = sess.run(pred_score_2, feed_dict={ \
                                        x_input_2:test_ft_inst_2, keep_prob:1.0})
                test_pred_score_all_2.append(cur_test_pred_score_2.flatten())
                test_label_all_2.append(test_label_inst_2)
                
                # y_target_2: np.transpose([test_label_inst_2])
                cur_test_loss_2 = sess.run(loss_2, feed_dict={ \
                                        x_input_2:test_ft_inst_2, \
                                        y_target_2: test_label_inst_2, keep_prob:1.0})
                test_loss_all_2.append(cur_test_loss_2)
    
        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done testing 2 -- epoch limit reached')
        finally:
            # whether to save the model
            if save_model_ind == 1:                
                saver = tf.train.Saver() 
                save_path = saver.save(sess, model_saving_addr)
                print("Model saved in file: %s" % save_path)
                
            coord.request_stop()
        coord.join(threads) 
        
        #################################
        # calculate auc
        test_pred_score_re_1 = func.list_flatten(test_pred_score_all_1)
        test_label_re_1 = func.list_flatten(test_label_all_1)
        test_auc_1, _, _ = func.cal_auc(test_pred_score_re_1, test_label_re_1)
        test_rmse_1 = func.cal_rmse(test_pred_score_re_1, test_label_re_1)
        test_loss_1 = np.mean(test_loss_all_1)
        
        test_pred_score_re_2 = func.list_flatten(test_pred_score_all_2)
        test_label_re_2 = func.list_flatten(test_label_all_2)
        test_auc_2, _, _ = func.cal_auc(test_pred_score_re_2, test_label_re_2)
        test_rmse_2 = func.cal_rmse(test_pred_score_re_2, test_label_re_2)
        test_loss_2 = np.mean(test_loss_all_2)
        
        # rounding
        test_auc_1 = np.round(test_auc_1, 4)
        test_rmse_1 = np.round(test_rmse_1, 4)
        test_loss_1 = np.round(test_loss_1, 5)
        
        test_auc_2 = np.round(test_auc_2, 4)
        test_rmse_2 = np.round(test_rmse_2, 4)
        test_loss_2 = np.round(test_loss_2, 5)
          
        print('test_auc_1 = ', test_auc_1)
        print('test_rmse_1 =', test_rmse_1)
        print('test_loss_1 =', test_loss_1)
        print('test_auc_2 = ', test_auc_2)
        print('test_rmse_2 =', test_rmse_2)
        print('test_loss_2 =', test_loss_2)    
 
        # append to result_list
        result_list.append([wgt_2, eta, test_auc_1, test_loss_1])

# print results
fmt_str = '{:<6}\t{:<6}\t{:<6}\t{:<6}'
header_row = ['wgt_2', 'eta', 'auc_1', 'loss_1']
print(fmt_str.format(*header_row))

for i in range(len(result_list)):
    tmp = result_list[i]
    print(fmt_str.format(*tmp))

