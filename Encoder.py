import torch
import torch.nn as nn

from ConvS2S import ConvEncoder
from Attention import MultiHeadAttention, PositionFeedforward

class Encoder(nn.Module):
    def __init__(self, conv_layers, hidden_dim, feed_forward_dim=2048):
        super(Encoder, self).__init__()
        # The classes are ConvEncoder,MultiheadAttention and PositionFeedForward.
        self.conv=ConvEncoder(input_dim = hidden_dim, num_layers = conv_layers)
        self.feed_forward=PositionFeedforward(hidden_dim,feed_forward_dim)
        self.attention=MultiHeadAttention(hid_dim = hidden_dim, n_heads = 16)

    def forward(self, input):
        """
        Forward Pass of the Encoder Class
        :param input: Input Tensor for the forward pass. 
        """
        result=self.conv(input)
        result=self.attention(result,result,result)
        final_result=self.feed_forward(result)
        return final_result

