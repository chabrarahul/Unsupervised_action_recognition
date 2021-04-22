import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerEncoderPartSeq(nn.Module):

    def __init__(self, d_model_part, n_head_part,
                 d_model_seq, n_head_seq, n_layers_part ,n_layers_seq ,dropout=0.5):
        super(TransformerEncoderPartSeq, self).__init__()
        self.encoder_layer_part = TransformerEncoderLayer(d_model= d_model_part, nhead=n_head_part)
        self.encoder_layer_seq = TransformerEncoderLayer(d_model= d_model_seq, nhead=n_head_seq)
        self.transformer_encoder_part = TransformerEncoder(self.encoder_layers_part, num_layers =n_layers_part)
        self.transformer_encoder_seq = TransformerEncoder(self.encoder_layers_seq, num_layers =n_layers_seq)

    def forward(self, in_part, in_seq, mask=None, src_key_padding_mask=None):
        out_part = self.transformer_encoder_part(in_part)
        out_seq = self.transformer_encoder_seq(in_seq)

    return out_part, out_seq

class LstmNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, drop_prob=0.2):
        super(LstmNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = n_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias = True,
            batch_first = True)
        
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, h):
        output, state = self.lstm(x, h)
        out = self.dropout(output[:,-1,:])
        logits = self.fc(out)

        return logits, state
    

class AutoEncoderDecoder(nn.Module):

    def __init__(self, input_size, drop_prob=0.2):
        super(AutoEncoderDecoder, self).__init__()  
        self.input_size = input_size
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, self.input_size)
        

    def forward(self, input):
        out1 = self.fc1(input)
        out2 = self.fc2(input)
        out3 = self.fc3(input)
        out4 = self.fc4(input)

        return out2, out4 
