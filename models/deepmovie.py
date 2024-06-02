import os
import time
import torch
from torch.autograd import Variable
import math
import numpy as np
from utils import eval_RMSE
from models.cnn import CNN
from models.cnn_kan import CNN_KAN
from models.lstm import LSTM
from models.resnet import ResNet
from models.transformer import Transformer


class DeepMovie:
    def __init__(self, res_dir, R, X, vocab_size, cuda, init_W=None, 
                 give_item_weight=True, max_iter=50, lambda_u=1, lambda_v=100, 
                 dimension=50, dropout_rate=0.2, emb_dim=200, max_len=300, 
                 model_type = "CNN", model_args = None):

        self.res_dir = res_dir
        self.R = R
        self.X = X
        self.vocab_size = vocab_size
        self.cuda = cuda
        self.init_W = init_W
        self.give_item_weight = give_item_weight
        self.max_iter = max_iter
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.dimension = dimension
        self.dropout_rate = dropout_rate
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.model_type = model_type
        self.model_args = model_args

        self.a = 1
        self.b = 0
        self.prev_loss = 1e-50
        self.num_user = R.shape[0]
        self.num_item = R.shape[1]
        
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        
        self.f1 = open(res_dir + '/state.log', 'w')
        self.f1.write("lambda_u: %.2f, lambda_v: %.2f, dimension: %d, dropout_rate: %.2f, emb_dim: %d, max_len: %d, model_type: %s\n" % (
            lambda_u, lambda_v, dimension, dropout_rate, emb_dim, max_len, model_type))
        for key, value in model_args.items():
            self.f1.write("%s: %s\n" % (key, value))

        if model_type == "CNN":
            num_kernel_per_ws = self.model_args['num_kernel_per_ws']
            self.module = CNN(dimension, vocab_size, dropout_rate, emb_dim, max_len, num_kernel_per_ws, cuda, init_W)
        elif model_type == "CNN_KAN":
            num_kernel_per_ws = self.model_args['num_kernel_per_ws']
            hidden_dim = self.model_args['hidden_dim']
            self.module = CNN_KAN(dimension, vocab_size, dropout_rate, emb_dim, max_len, num_kernel_per_ws, hidden_dim, cuda, init_W)
        elif model_type == "LSTM":
            n_filters = self.model_args['n_filters']
            hidden_dim = self.model_args['hidden_dim']
            n_layer = self.model_args['n_layer']
            self.module = LSTM(dimension, vocab_size, dropout_rate, emb_dim, max_len, n_filters, n_layer, hidden_dim, cuda)
        elif model_type == "ResNet":
            n_filters = self.model_args['n_filters']
            hidden_dim = self.model_args['hidden_dim']
            self.module = ResNet(dimension, vocab_size, dropout_rate, emb_dim, max_len, n_filters, hidden_dim, cuda)
        elif model_type == "Transformer":
            nhead = self.model_args['nhead']
            n_layer = self.model_args['n_layer']
            self.module = Transformer(dimension, vocab_size, dropout_rate, emb_dim, max_len, nhead, n_layer, cuda)
        else:
            raise NotImplementedError("We only implement CNN, LSTM and Transformer as our base module")

        if cuda:
            self.module = self.module.cuda()
        
        self.theta = self.module.get_projection_layer(X)
        self.U = np.random.uniform(size=(self.num_user, dimension))
        self.V = self.theta
        
        self.endure_count = 5
        self.count = 0

    def prepare_item_weight(self, Train_R_J):
        if self.give_item_weight:
            item_weight = np.array([math.sqrt(len(i)) for i in Train_R_J], dtype=float)
            item_weight *= (float(self.num_item) / item_weight.sum())
        else:
            item_weight = np.ones(self.num_item, dtype=float)
        return item_weight

    # adapted from https://github.com/cartopy/ConvMF/tree/master
    def train(self, train_user, train_item, valid_user, test_user):
        Train_R_I = train_user[1]
        Train_R_J = train_item[1]
        Test_R = test_user[1]
        Valid_R = valid_user[1]

        item_weight = self.prepare_item_weight(Train_R_J)
        pre_val_eval, best_tr_eval, best_val_eval, best_te_eval = 1e10, 1e10, 1e10, 1e10

        for iteration in range(self.max_iter):
            loss = 0
            tic = time.time()
            print("%d iteration\t(patience: %d)" % (iteration, self.count))

            VV = self.b * (self.V.T.dot(self.V)) + self.lambda_u * np.eye(self.dimension)
            sub_loss = np.zeros(self.num_user)

            for i in range(self.num_user):
                idx_item = train_user[0][i]
                V_i = self.V[idx_item]
                R_i = Train_R_I[i]
                A = VV + (self.a - self.b) * (V_i.T.dot(V_i))
                B = (self.a * V_i * np.tile(R_i, (self.dimension, 1)).T).sum(0)

                self.U[i] = np.linalg.solve(A, B)
                sub_loss[i] = -0.5 * self.lambda_u * np.dot(self.U[i], self.U[i])

            loss += np.sum(sub_loss)

            sub_loss = np.zeros(self.num_item)
            UU = self.b * (self.U.T.dot(self.U))
            for j in range(self.num_item):
                idx_user = train_item[0][j]
                U_j = self.U[idx_user]
                R_j = Train_R_J[j]

                tmp_A = UU + (self.a - self.b) * (U_j.T.dot(U_j))
                A = tmp_A + self.lambda_v * item_weight[j] * np.eye(self.dimension)
                B = (self.a * U_j * np.tile(R_j, (self.dimension, 1)).T).sum(0) + self.lambda_v * item_weight[j] * self.theta[j]
                self.V[j] = np.linalg.solve(A, B)

                sub_loss[j] = -0.5 * np.square(R_j * self.a).sum()
                sub_loss[j] += self.a * np.sum((U_j.dot(self.V[j])) * R_j)
                sub_loss[j] -= 0.5 * np.dot(self.V[j].dot(tmp_A), self.V[j])

            loss += np.sum(sub_loss)

            # train deep learning model
            self.module.train(self.X, self.V)

            self.theta = self.module.get_projection_layer(self.X)

            tr_eval = eval_RMSE(Train_R_I, self.U, self.V, train_user[0])
            val_eval = eval_RMSE(Valid_R, self.U, self.V, valid_user[0])
            te_eval = eval_RMSE(Test_R, self.U, self.V, test_user[0])

            toc = time.time()
            elapsed = toc - tic

            converge = abs((loss - self.prev_loss) / self.prev_loss)

            if val_eval < pre_val_eval:
                torch.save(self.module, self.res_dir + f'{self.model_type}_model.pt')
                best_tr_eval, best_val_eval, best_te_eval = tr_eval, val_eval, te_eval
                np.savetxt(self.res_dir + '/U.dat', self.U)
                np.savetxt(self.res_dir + '/V.dat', self.V)
                np.savetxt(self.res_dir + '/theta.dat', self.theta)
            else:
                self.count += 1

            pre_val_eval = val_eval

            print("Elapsed: %.4fs Converge: %.6f Train: %.5f Valid: %.5f Test: %.5f" % (
                elapsed, converge, tr_eval, val_eval, te_eval))
            self.f1.write("Elapsed: %.4fs Converge: %.6f Train: %.5f Valid: %.5f Test: %.5f\n" % (
                elapsed, converge, tr_eval, val_eval, te_eval))

            if self.count == self.endure_count:
                print("\n\nBest Model: Train: %.5f Valid: %.5f Test: %.5f" % (
                    best_tr_eval, best_val_eval, best_te_eval))
                self.f1.write("\n\nBest Model: Train: %.5f Valid: %.5f Test: %.5f\n" % (
                    best_tr_eval, best_val_eval, best_te_eval))
                break

            self.prev_loss = loss

        self.f1.close()