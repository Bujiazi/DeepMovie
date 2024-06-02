from utils import Data_Factory

# load movielens dataset
raw_rating_data_path = "data/movielens/ratings.dat"
raw_item_document_data_path = "data/movielens/ml_plot.dat"

# preprocessed data destination path
data_path = "data/preprocessed/ml-1m/0.2/"
aux_path = "data/preprocessed/ml-1m/"

assert data_path is not None, "data_path is required for data preprocessing"
assert aux_path is not None, "aux_path is required for data preprocessing"

# dataset paramters, please do not modify this part
min_rating = 1
max_length_document = 300
max_df = 0.5
vocab_size = 8000
split_ratio = 0.2 # for train set and test set split
df = Data_Factory()

# data preprocess stage
R, D_all = df.preprocess(raw_rating_data_path, raw_item_document_data_path, min_rating, max_length_document, max_df, vocab_size)
df.save(aux_path, R, D_all)
df.generate_train_valid_test_file_from_R(data_path, R, split_ratio)



