import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


class LSTM(nn.Module):
    batch_size = 128
    nb_epoch = 5

    def __init__(self, output_dimension, vocab_size, dropout_rate, emb_dim, max_len, n_filters, n_layer, hidden_dim, if_cuda, init_W=None):
        super(LSTM, self).__init__()

        self.max_len = max_len
        self.emb_dim = emb_dim
        self.if_cuda = if_cuda
        self.hidden_dim = hidden_dim
        # vanila_dimension = 2 * n_filters + hidden_dim
        vanila_dimension = hidden_dim // 2
        projection_dimension = output_dimension

        '''Embedding Layer'''
        self.embedding = nn.Embedding(vocab_size + 1, emb_dim)

        '''Convolutional Layers with Batch Normalization and LeakyReLU'''
        self.conv1 = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters, kernel_size=3),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=max_len - 3 + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters, kernel_size=4),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=max_len - 4 + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dim, n_filters, kernel_size=5),
            nn.BatchNorm1d(n_filters),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(kernel_size=max_len - 5 + 1)
        )

        '''BiLSTM Layer'''
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, num_layers=n_layer, bidirectional=True, batch_first=True)

        '''Fully Connected Layers with Dropout'''
        # self.fc1 = nn.Linear(n_filters * 3 + hidden_dim, vanila_dimension)
        self.fc1 = nn.Linear(hidden_dim, vanila_dimension)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(vanila_dimension, projection_dimension)

    def forward(self, inputs):
        size = len(inputs)
        embeds = self.embedding(inputs)
        embeds = embeds.view([len(embeds), self.emb_dim, -1])

        # x = self.conv1(embeds)
        # y = self.conv2(embeds)
        # z = self.conv3(embeds)
        # conv_out = torch.cat((x.view(size, -1), y.view(size, -1), z.view(size, -1)), 1)

        '''LSTM Layer'''
        lstm_out, _ = self.lstm(embeds.transpose(1, 2))
        lstm_out = lstm_out[:, -1, :]

        '''Concatenate Conv and LSTM outputs'''
        # combined = torch.cat((conv_out, lstm_out), 1)

        # out = F.relu(self.fc1(combined))
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        out = self.fc2(out)

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


# # Example Usage:
# output_dimension = 10
# vocab_size = 5000
# dropout_rate = 0.5
# emb_dim = 300
# max_len = 100
# n_filters = 100
# hidden_dim = 128
# if_cuda = torch.cuda.is_available()

# model = EnhancedCNNLSTM(output_dimension, vocab_size, dropout_rate, emb_dim, max_len, n_filters, hidden_dim, if_cuda)
# if if_cuda:
#     model.cuda()

# # Assume X_train and V are your training data and labels
# # model.train_model(X_train, V)

# # Get the output from the projection layer
# # projections = model.get_projection_layer(X_train)
