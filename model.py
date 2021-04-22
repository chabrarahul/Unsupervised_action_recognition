import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *


class Model(nn.Module)

    def __init__(self, input_size, drop_prob=0.2):
         super(Model, self).__init__()
         self.transformer_encoder = TransformerEncoderPartSeq()
         self.encoder = LstmNet() # number of layer should be given 
         self.auto_encoder = AutoEncoderDecoder()
         self.decoder = LstmNet() # one layer lstm network as the decoder

    
    def forward(self, input): 
         output = self.transformer(input)
         output = self.encoder(output)
         out_for_prediction, output = self.auto_encoder(output)
         output = self.decoder(output)

         return output_for_prediction, output  
