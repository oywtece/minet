# MiNet

Mixed Interest Network (MiNet) is a model for **cross-domain** click-through rate (CTR) prediction.

A major advantage of cross-domain CTR prediction is that by enriching data across domains, the data sparsity and the cold-start problem in the target domain can be alleviated, which leads to improved prediction performance.

MiNet jointly models **three types of user interest**: 1) long-term interest across domains, 2) short-term interest from the source domain and 3) short-term interest in the target domain. MiNet contains **two levels of attentions,** where the **item-level** attention can adaptively distill useful information from clicked news / ads and the **interest-level** attention can adaptively fuse different interest representations. 

If you use this code, please cite the following paper:
* **MiNet: Mixed Interest Network for Cross-Domain Click-Through Rate Prediction. In CIKM, ACM, 2020.**

arXiv: https://arxiv.org/abs/2008.02974

#### Bibtex
```
@inproceedings{ouyang2020minet,
  title={MiNet: Mixed Interest Network for Cross-Domain Click-Through Rate Prediction},
  author={Ouyang, Wentao and Zhang, Xiuwu and Zhao, Lei and Luo, Jinmei and Zhang, Yu and Zou, Heng and Liu, Zhaojie and Du, Yanlong},
  booktitle={CIKM},
  year={2020}
}
```

#### TensorFlow (TF) version
1.12.0

#### Abbreviation
ft - feature, src - source, tar - target, slot == field

## Data Preparation
Data is in the "csv" or the "tfrecord" format.
* Each csv row contains the label, the user, the target item, clicked items in the target domain and clicked items in the source domain.
* In the Amazon example here, the target domain is the Movies&TV and the source domain is the Books.
* In UC Toutiao, the target domain is the Ads and the source domain is the News.

Assume there are N unique fts. Fts need to be indexed from 1 to N. Use 0 for missing values or for padding.

We categorize fts as i) **one-hot** or **univalent** (e.g., user id, city) and ii) **mul-hot** or **multivalent** (e.g., words in ad title).

We need to prepare two data sets. One for the target domain, and the other for the source domain.

We need to specify the following parameters for data partitioning after the data are loaded:
* n_user_slot [num of user ft slots; it must be the same in 2 data sets]
* n_one_hot_slot_1 [num of one-hot fts in the items in the target domain]
* n_mul_hot_slot_1 [num of mul-hot fts in the items in the target domain]
* max_len_per_slot_1 [max num of fts in each mul-hot slot in the target domain]
* max_n_clk_ori_1 [max num of clicked items in the target domain]
* n_one_hot_slot_2 [num of one-hot fts in the items in the source domain]
* n_mul_hot_slot_2 [num of mul-hot fts in the items in the source domain]
* max_len_per_slot_2 [max num of fts in each mul-hot slot in the source domain]
* max_n_clk_ori_2 [max num of clicked items in the source domain]

### Target domain dataset
Assume we have at most 2 clicked items in the target domain (tar_clk) and 2 clicked items in the source domain (src_clk), then one row of the csv data looks like:
* \<label\>\<user fts\>\<tar one-hot fts, tar mul-hot fts\>\<tar_clk_1 one-hot fts, tar_clk_1 mul-hot fts\>\<tar_clk_2 one-hot fts, tar_clk_2 mul-hot fts\>\<src_clk_1 one-hot fts, src_clk_1 mul-hot fts\>\<src_clk_2 one-hot fts, src_clk_2 mul-hot fts\>

#### Example 1 (target item)
1) original fts (ft_name:ft_value)
* label:0, user_id:u_1, item_id:i_1, title:apple, title:fruit
2) csv fts
* 0, u_1, i_1, apple, fruit, 0

#### Explanation 1
csv format:\
\<label\>\<user fts\>\<tar one-hot fts, tar mul-hot fts\>

csv format settings:\
n_user_slot = 1 # user_id \
n_one_hot_slot_1 = 1 # item_id \
n_mul_hot_slot_1 = 1 # title \
max_len_per_slot_1 = 3

For the mul-hot ft slot "title", we have 2 fts, which are "apple" and "fruit". Terefore, we pad 1 zero (because max_len_per_slot_1 = 3).
If there are more than max_len_per_slot_1 fts, we keep only the first max_len_per_slot_1.

#### Example 2 (target item, target domain clicked item 1, target domain clicked item 2)
1) original fts (ft_name:ft_value)
* label:0, user_id:u_1, item_id:i_1, title:apple, title:fruit; item_id:i_2, title:a, title:b, title:c, title:d; item_id:i_3, title:e
2) csv fts
* 0, u_1, i_1, apple, fruit, 0, i_2, a, b, c, i_3, e, 0, 0

#### Explanation 2
csv format:\
\<label\>\<user fts\>\<tar one-hot fts, tar mul-hot fts\><tar_clk_1 one-hot fts, tar_clk_1 mul-hot fts\>\<tar_clk_2 one-hot fts, tar_clk_2 mul-hot fts\>

### Source domain dataset
One row of the csv data looks like:
* \<label\>\<user fts\>\<src one-hot fts, src mul-hot fts\>

## Sample Data
In the **data** folder.\
Reformatted Amazon data (csv/tfrecord format with ft index). \
The scripts run **much faster** with the **tfrecord** data files.
We provide **tfrecord_writer.py** which can easily convert csv files to tfrecord files.

Target domain: Movies&TV, Source domain: Books. \
Target domain data files: Movies_Books_xx.tfrecord (xx is in {train, val, test})\
Source domain data files: Books_xx.tfrecord (xx is in {train, val, test}) \
Please refer to the paper for more details about the datasets.

Original Amazon datasets:
* [Original Amazon datasets](http://jmcauley.ucsd.edu/data/amazon/index_2014.html)

## Config
### Validation and hyperparameter tunining
train file: xx_train.tfrecord, test file: xx_val.tfrecord \
Set multiple values for hyperparameters. \
Example: eta_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

### Testing
train files: xx_train.tfrecord & xx_val.tfrecord, test file: xx_test.tfrecord \
Set only 1 value for the optimal hyperparameter found in the validation step. \
Example: eta_range = [0.02]

## Source Code
* config_amazon.py -- config file
* ctr_funcs.py -- functions
* dnn_para_tune.py -- DNN model
* minet_para_tune.py -- MiNet model

## Run the Code
x_para_tune.py files perform hyperparameter tuning.
Config the hyperparameter ranges in config_amazon.py.

Run the code
```bash
nohup python dnn_para_tune.py > dnn_[output_file_name].out 2>&1 &
nohup python minet_para_tune.py > minet_[output_file_name].out 2>&1 &
```
