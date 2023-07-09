import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, num_filters, k_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters * 2, kernel_size=k_size, stride=1, padding=k_size // 2),

        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size // 2),

        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size // 2),

        )

        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(num_filters * 3, 96),
            nn.ReLU()
        )
        

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_(0, 0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        output = self.out(x)
        output = output.squeeze()

        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        output = self.layer3(output)

        return output, output1, output2
    
class attention_pooling_layer(nn.Module):
    def __init__(self, dropout_rate=0.1,  feature_dim=96, h_= 3):
        super(attention_pooling_layer,self).__init__()

        self.dropout_rate = dropout_rate

        self.pool   = nn.AdaptiveAvgPool1d(feature_dim)
        self.h_mat  = nn.Parameter(torch.Tensor(self.heads, 3, feature_dim//self.heads, 1).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(self.heads, 3, 1, 1).normal_())
        self.h_ = h_
        self.bn     = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self,q,v):


        a = v.view(v.shape[0], 1, v.shape[1], -1)
        b = q.view(q.shape[0], 1, q.shape[-1],-1)
        a1 = a.transpose(2,3)
        b1 = b.transpose(2,3)
        att_maps = torch.einsum('bijk,hcxk,bqiv->bjck', (a , self.h_mat, b)).transpose(1,2) + self.h_bias 
        norm_att_maps = att_maps / torch.norm(att_maps, dim=2, keepdim=True)
        att_maps = torch.softmax(norm_att_maps, dim=2)
        logits = torch.einsum('bikj,bhi,bqvk->bivq', (a1, att_maps[:, :, 0, :],b1))
        logits = self.pool(logits.squeeze(-1)).transpose(1,2)
        
        for i in range(1, self.h_):
            logits_i = torch.einsum('bihc,bxv,bqkv->bi', (a1, att_maps[:, :, i, :], b1))
            logits_i = self.pool(logits_i.unsqueeze(-1)).transpose(1,2)
            logits += logits_i
        logits = self.bn(logits)

        return logits 



class net_reg(nn.Module):
    def __init__(self, num_filters):
        super(net_reg, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(96, 1024),            
            nn.ReLU(),              
            nn.Dropout(0.1),
            nn.Linear(1024,512),            
            nn.ReLU(),            
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )



        self.attention_pooling = attention_pooling_layer()


    def forward(self, A, B):
        
        B = F.avg_pool1d(B, kernel_size=B.shape[-1] // 96, stride=None, padding=0, ceil_mode=False,
                          count_include_pad=True)  # 平均池化
     
        B = B[:, :, 0].unsqueeze(-1)
        B = B[:, :96, :]

        out = self.attention_pooling(A, B)
        out = out.squeeze(-1)
        out = self.reg(out)

        return out



class net(nn.Module):
    def __init__(self, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(FLAGS.charsmiset_size, 128).cuda()
        self.embedding2 = nn.Embedding(FLAGS.charseqset_size, 128).cuda()
        self.cnn1 = CNN(NUM_FILTERS, FILTER_LENGTH1).cuda()

        self.reg = net_reg(NUM_FILTERS)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 8, 2048, dropout=0.1), num_layers=3).cuda()

    def forward(self, x, y, FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
        
        y_init = Variable(y.long()).cuda()
        x_init = Variable(x.long()).cuda()
        device1 = x_init.device
        device2 = y_init.device
        x = self.embedding1(x_init).cuda()
        x_embedding = x.permute(0, 2, 1).cuda()
        x, mu_x, logvar_x = self.cnn1(x_embedding) 
        x = x.unsqueeze(-1)    
        y = self.embedding2(y_init)
        #[batchsize/4 ,1000,128]
        y = self.transformer_encoder(y).transpose(1, 2).cuda()  # ([batchsize/4, 128, 1000])
        #before out y [,96,1]
        out = self.reg(x, y).squeeze().cuda()
        
        return out







