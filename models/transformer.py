import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


class Transformer(nn.Module):
    batch_size = 128
    nb_epoch = 5
    
    def __init__(self, output_dimension, vocab_size, dropout_rate, emb_dim, max_len, n_head, n_layer, if_cuda, init_W=None):
        super(Transformer, self).__init__()

        self.max_len = max_len
        self.emb_dim = emb_dim
        self.if_cuda = if_cuda
        vanila_dimension = 2 * output_dimension
        projection_dimension = output_dimension

        '''Embedding Layer'''
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim)

        '''Transformer Blocks'''
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)

        '''Fully Connected Layers with Dropout'''
        self.fc1 = nn.Linear(emb_dim, vanila_dimension)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(vanila_dimension, projection_dimension)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        x = self.transformer_encoder(embeds)
        x = x[:, -1, :]
        
        y = self.fc1(x)
        y = self.dropout(y)
        y = self.fc2(y)
        return y
    
    def train(self, X_train, V):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(1, self.nb_epoch + 1):
            n_batch = len(X_train) // self.batch_size

            for i in range(n_batch + 1):
                begin_idx, end_idx = i * self.batch_size, (i + 1) * self.batch_size

                if i < n_batch:
                    feature = X_train[begin_idx:end_idx][...]
                    target = V[begin_idx:end_idx][...]
                else:
                    feature = X_train[begin_idx:][...]
                    target = V[begin_idx:][...]

                feature = Variable(torch.from_numpy(feature.astype('int64')).long())
                target = Variable(torch.from_numpy(target))
                if self.if_cuda:
                    feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                logit = self(feature)

                loss = F.mse_loss(logit, target)
                loss.backward()
                optimizer.step()

    def get_projection_layer(self, X_train):
        # inputs = Variable(torch.from_numpy(X_train.astype('int64')).long())
        # if self.if_cuda:
        #     inputs = inputs.cuda()
        # with torch.no_grad():
        #     outputs = self(inputs)
        out_batches = []
        n_batch = len(X_train) // self.batch_size
        
        for i in range(n_batch + 1):
            begin_idx, end_idx = i * self.batch_size, (i + 1) * self.batch_size
            
            if i < n_batch:
                feature = X_train[begin_idx:end_idx][...]
            else:
                feature = X_train[begin_idx:][...]
                
            feature = Variable(torch.from_numpy(feature.astype('int64')).long())
            if self.if_cuda:
                feature = feature.cuda()
                
            output = self(feature)
            out_batches.append(output.detach().cpu())
            
        outputs = torch.cat(out_batches, 0)
        return outputs.cpu().data.numpy()