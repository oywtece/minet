## Data processing pipeline

- **Original Amazon data**: ratings_Books.csv, ratings_Movies_and_TV.csv, meta_Books.txt, meta_Movies_and_TV.txt

- **Target domain**: Movies_and_TV; **Source domain**: Books

- **Features used in ratings_Books.csv**: user_id, item_id, rating, [**time_stamp - for differentiating target / history and for splitting the dataset as train / val / test**]

- **Features used in meta_Books.txt**: asin (item_id), main_cat, brand, title, category

- Join meta data with rating data based on **item_id**


### 1) filter out users with enough data (e.g., at least K records; each interacted item should have meta feature; both Books and Movies datasets)
* input: ratings_Books.csv, ratings_Movies_and_TV.csv, meta_Books.txt, meta_Movies_and_TV.txt
* output: Books_user_w_enough_data.txt, Movies_user_w_enough_data.txt [each line contains one user_id]

### 2) find shared users in the above two output files
* input: Books_user_w_enough_data.txt, Movies_user_w_enough_data.txt 
* output: Books_Movies_shared_users.txt [each line contains one user_id]

### 3) filter out samples for shared users in the Books dataset
* input: ratings_Books.csv, Books_Movies_shared_users.txt
* output: ratings_Books_shared_users.csv
  * one example line: time_stamp, rating, user_id, item_id

### 4) filter out samples for shared users in the Movies dataset and append each sample with historically clicked samples according to the timestamp
* input: ratings_Books.csv, ratings_Movies_and_TV.csv, Books_Movies_shared_users.txt
* output: ratings_Movies_and_TV_shared_users_w_hist.csv
  * one example line: time_stamp, rating, user_id, item_id, clk_item_id_1_Movie, clk_item_id_2_Movie, ..., clk_item_id_1_Book, clk_item_id_2_Book, ...

### 5) add item meta features for the Books dataset
* input: ratings_Books_shared_users.csv, meta_Books.txt
* output: ratings_Books_shared_users_w_meta.csv
  * one example line: time_stamp, rating, user_id, item_id, main_cat, brand, title, category

### 6) add item meta features for the Movies dataset
* input: ratings_Movies_and_TV_shared_users_w_hist.csv, meta_Movies_and_TV.txt
* output: ratings_Movies_and_TV_shared_users_w_hist_meta.csv
  * one example line: time_stamp, rating, user_id, item_id, main_cat, brand, title, category, clk_item_id_1_Movie, main_cat_1_Movie, brand_1_Movie, title_1_Movie, category_1_Movie, 
clk_item_id_2_Movie, main_cat_2_Movie, brand_2_Movie, title_2_Movie, category_2_Movie, ..., 
clk_item_id_1_Book, main_cat_1_Book, brand_1_Book, title_1_Book, category_1_Book, 
clk_item_id_2_Book, main_cat_2_Book, brand_2_Book, title_2_Book, category_2_Book, ...

### 7) encode the raw features as index (except time_stamp)
* input: ratings_Books_shared_users_w_meta.csv, ratings_Movies_and_TV_shared_users_w_hist_meta.csv
* output: ratings_Books_shared_users_w_meta_idx_format.csv, ratings_Movies_and_TV_shared_users_w_hist_meta_idx_format.csv
  * one example line: 1019865600, 1, 185, 430, 578, 602, 229, ...

### 8) split the dataset according to time_stamp as train / val / test files (remove time_stamp after data splitting)
* input: ratings_Books_shared_users_w_meta_idx_format.csv, ratings_Movies_and_TV_shared_users_w_hist_meta_idx_format.csv
* output: Books_train.csv, Books_val.csv. Books_test.csv, Movies_Books_train.csv, Movies_Books_val.csv, Movies_Books_test.csv
  * one example line: 1, 185, 430, 578, 602, 229, ...
