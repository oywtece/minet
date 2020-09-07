'''
config file
'''
n_user_slot = 1 # user fts
n_one_hot_slot_1 = 3 # only item fts; no user fts; id, main_cate, brand 
n_mul_hot_slot_1 = 2 # title, categories
max_len_per_slot_1 = 10
num_csv_col_1 = 715 # num of cols in the csv file = 1 (label) + n_user_slot + (n_one_hot_slot_1 + n_mul_hot_slot_1*max_len_per_slot_1)*(1 + max_n_clk_ori_1) + (n_one_hot_slot_2 + n_mul_hot_slot_2*max_len_per_slot_2)*max_n_clk_ori_2
batch_size_1 = 32
layer_dim_1 = [256, 128, 1]

max_n_clk_ori_1 = 10 # ori val in dataset
max_n_clk_ori_2 = 20 # in dataset 1
max_n_clk_1 = 10 # actual val used in experiment; max_n_clk_1 <= max_n_clk_ori_1
max_n_clk_2 = 20

# n_one_hot_slot_2 may be different from n_one_hot_slot_1
# because the 1st dataset removes redundant user features in the src domain data
n_one_hot_slot_2 = 3 # only item fts; no user fts
n_mul_hot_slot_2 = 2
max_len_per_slot_2 = 10
num_csv_col_2 = 25  # num of cols in the csv file = 1 (label) + n_user_slot + n_one_hot_slot_2 + n_mul_hot_slot_2*max_len_per_slot_2
batch_size_2 = 2*batch_size_1
layer_dim_2 = [256, 128, 1]

input_format = 'tfrecord' # 'csv' or 'tfrecord'
pre = './data/Movies_Books_'
suf = '.tfrecord' # '.csv'
train_file_name_1 = [pre+'train'+suf, pre+'val'+suf]
test_file_name_1 = [pre+'test'+suf]

pre = './data/Books_'
suf = '.tfrecord' # '.csv'
train_file_name_2 = [pre+'train'+suf, pre+'val'+suf]
test_file_name_2 = [pre+'test'+suf]

save_model_ind = 0

## for DNN
num_csv_col = num_csv_col_1
train_file_name = train_file_name_1
test_file_name = test_file_name_1
batch_size = batch_size_1
n_one_hot_slot = n_one_hot_slot_1 + n_user_slot
n_mul_hot_slot = n_mul_hot_slot_1
max_len_per_slot = max_len_per_slot_1
num_csv_col = num_csv_col_1
layer_dim = layer_dim_1

n_ft = 841927
time_style = '%Y-%m-%d %H:%M:%S'
rnd_seed = 123
user_b_ini = 1.0
tar_clk_b_ini = 0.0
src_clk_b_ini = 0.0
wgt_1 = 1.0
wgt_2_range = [1.0] # [0.1, 0.3, 0.5, 0.7, 1.0]
eta_range = [0.02] # [0.01, 0.02, 0.05, 0.1]
inter_dim = 10
k = 10 # embedding size / number of latent factors
opt_alg = 'Adagrad' # 'Adam'
kp_prob = 1.0
output_file_name = '0728_1629' # for model saving
item_att_hidden_dim = 64
interest_att_hidden_dim = 64
n_epoch = 2 # number of times to loop over the whole data set
record_step_size = 5000 # record the loss and auc after xx mini-batches

