import torch
import numpy as np
from utils import Data_Factory
from models.deepmovie import DeepMovie

# if cuda is available for training
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("device: ", device)

# preprocessed data destination path
data_path = "data/preprocessed/ml-10m/0.2/"
aux_path = "data/preprocessed/ml-10m/"

# result save direction
save_path = "result_ml-10m/CNN_KAN/"

# hyper parameters for training 
max_epochs = 200 # not that much, early stop is enabled
embed_dim = 50
latent_dim = 50
lambda_u = 100
lambda_v = 10
give_item_weight = True

# select model type
model_type = "CNN_KAN" # you need to specify a model type from ["CNN", "CNN_KAN", "LSTM", "ResNet", "Transformer"]

# corresponding model arguments
if model_type == "CNN":
    model_args = {'num_kernel_per_ws': 100}
elif model_type == "CNN_KAN":
    model_args = {'num_kernel_per_ws': 100,
                "hidden_dim": 128}
elif model_type == "LSTM":
    model_args = {'n_filters': 100,
                'hidden_dim': 128,
                'n_layer': 4}
elif model_type == "ResNet":
    model_args = {'n_filters': 100,
                'hidden_dim': 128}
elif model_type == "Transformer":
    model_args = {'nhead': 2,
                'n_layer': 2}


assert data_path is not None, "data_path is required for training"
assert aux_path is not None, "aux_path is required for training"
assert save_path is not None, "save_path is required for training"
assert model_type is not None, "model_type is required for training"

# load preprocessed data (if error, please run data preprocessing first)
df = Data_Factory()
R, D_all = df.load(aux_path)
X = D_all['X_sequence']
vocab_size = len(D_all['X_vocab']) + 1
init_W = None # since we don't have a pretrained word embedding model

# padding X
X_pad = np.full((len(X), 300), 8000)
for i in range(len(X)):
    for j in range(len(X[i])):
        X_pad[i][j] = X[i][j]

# form datasets
train_user = df.read_rating(data_path + '/train_user.dat')
train_item = df.read_rating(data_path + '/train_item.dat')
valid_user = df.read_rating(data_path + '/valid_user.dat')
test_user = df.read_rating(data_path + '/test_user.dat')

# train and test DeepMovie model
model = DeepMovie(max_iter=max_epochs, 
                res_dir=save_path,
                lambda_u=lambda_u, 
                lambda_v=lambda_v, 
                dimension=latent_dim, 
                vocab_size=vocab_size, 
                init_W=init_W,
                give_item_weight=give_item_weight, 
                X=X_pad, 
                emb_dim=embed_dim, 
                R=R, 
                cuda=cuda,
                model_type = model_type,
                model_args = model_args)

model.train(train_user=train_user, 
            train_item=train_item, 
            valid_user=valid_user, 
            test_user=test_user)






