import math
import torch.nn as nn
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

#位置编码，对序列顺序进行约束（word position编码为d_madel维向量）
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        #词语在序列中的位置position
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
                 ntoken=100,
                 ninp=512,
                 nhead=8,
                 nhid=2048,
                 nlayers=6,
                 dropout=0.5):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout) #ninp 词向量维度
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout) #d_model,nhead,dim_feedforward
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        #output = self.decoder(output)
        return output



def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class SIONR(nn.Module):
    def __init__(self, inplace=True):  #low level feature
        super(SIONR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(inplace=inplace),
        )

        self.high = nn.Sequential(     #high level feature 2048-128
            nn.Linear(in_features=2048, out_features=1024), #FC1
            nn.LeakyReLU(inplace=inplace),
            #nn.Linear(in_features=1024, out_features=128), #FC2
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(inplace=inplace),
        )

        self.spatial = nn.Sequential(
            nn.Linear(in_features=64, out_features=1),
            nn.LeakyReLU(inplace=inplace),
        )

        self.temporal = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(inplace=inplace),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features= 128, out_features=64), # FC5 high + low
            nn.LeakyReLU(inplace=inplace),
            nn.Linear(in_features=64, out_features=1),  #FC6 score
            nn.LeakyReLU(inplace=inplace),
        )

        self.weight_init()

    def weight_init(self):
        initializer = kaiming_init
        for block in self._modules:
            for m in self._modules[block]:
                    initializer(m)


    def forward(self, video, feature):
        # batch_size, channel, depth, height, width
        out_tensor = self.conv1(video)
        out_tensor = self.conv2(out_tensor)
        out_tensor = self.conv3(out_tensor)
        out_tensor = self.conv4(out_tensor)

        # low-level temporal variation
        diff_tensor = torch.abs(out_tensor[:, :, 0::2, :, :] - out_tensor[:, :, 1::2, :, :])

        # temporal factor
        out_feature1 = torch.mean(diff_tensor, dim=[3, 4])
        # spatial factor
        out_feature2 = torch.mean(out_tensor[:, :, 1::2, :, :], dim=[3, 4])

        # batch_size, channel, depth
        out_feature1 = out_feature1.permute([0, 2, 1])  #permute： tensor的维度换位
        out_feature2 = out_feature2.permute([0, 2, 1])

        # spatiotemporal feature fusion
        out_feature_L = self.temporal(out_feature1) * self.spatial(out_feature2)

        # high-level temporal variation
        feature_abs = torch.abs(feature[:, 0::2] - feature[:, 1::2])
        out_feature_H = self.high(feature_abs)

        # hierarchical feature fusion
        # score = self.fc(torch.cat((out_feature_L, out_feature_H), dim=2)) #dim=0,行；dim=1，列；dim=3，第三维度

        out_feature = TransformerModel(out_feature_H)
        score = self.fc(out_feature)

        # mean pooling
        score = torch.mean(score, dim=[1, 2])

        return score








