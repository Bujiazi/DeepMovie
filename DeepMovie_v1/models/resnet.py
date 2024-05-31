import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_len, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=max_len - 3 + 1)
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.pool(out)
        return out



class ResNet(nn.Module):
    batch_size = 128
    nb_epoch = 5

    def __init__(self, output_dimension, vocab_size, dropout_rate, emb_dim, max_len, n_filters, hidden_dim, if_cuda, init_W=None):
        super(ResNet, self).__init__()

        self.max_len = max_len
        self.emb_dim = emb_dim
        self.if_cuda = if_cuda
        self.hidden_dim = hidden_dim
        vanila_dimension = 2 * n_filters + hidden_dim
        projection_dimension = output_dimension

        '''Embedding Layer'''
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim)

        '''Residual Blocks'''
        self.res_block1 = ResidualBlock(emb_dim, n_filters, kernel_size=3, max_len = max_len)
        self.res_block2 = ResidualBlock(emb_dim, n_filters, kernel_size=3, max_len = max_len)
        self.res_block3 = ResidualBlock(emb_dim, n_filters, kernel_size=3, max_len = max_len)

        '''Dropout Layer'''
        self.layer = nn.Linear(n_filters * 3, vanila_dimension)
        self.dropout = nn.Dropout(dropout_rate)

        '''Projection Layer & Output Layer'''
        self.output_layer = nn.Linear(vanila_dimension, projection_dimension)




    def forward(self, inputs):
        size = len(inputs)
        embeds = self.embedding(inputs)
        embeds = embeds.transpose(1, 2)  # (batch_size, emb_dim, max_len)

        
        x = self.res_block1(embeds)
        y = self.res_block2(embeds)
        z = self.res_block3(embeds)

        flatten = torch.cat((x.view(size, -1), y.view(size, -1), z.view(size, -1)), 1)


        out = F.tanh(self.layer(flatten))
        out = self.dropout(out)
        out = F.tanh(self.output_layer(out))

        return out

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
        # outputs = self(inputs)
        # return outputs.cpu().data.numpy()
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



