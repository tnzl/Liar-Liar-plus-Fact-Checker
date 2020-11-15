import torch
import torch.nn as nn

from Attention import MultiHeadAttention, PositionFeedforward
from Encoder import Encoder

class arePantsonFire(nn.Module):

    def __init__(self, sentence_encoder: Encoder, explanation_encoder: Encoder, multihead_Attention: MultiHeadAttention,
                 position_Feedforward: PositionFeedforward, hidden_dim: int, max_length_sentence,
                 max_length_justification, input_dim, device='cuda:0'):
        
        #If you wish to shift on cpu pass device as 'cpu'

        super(arePantsonFire, self).__init__()
        self.device = device

        self.sentence_pos_embedding = nn.Embedding(max_length_sentence, hidden_dim)
        self.justification_pos_embedding = nn.Embedding(max_length_justification, hidden_dim)

        self.sentence_encoder = sentence_encoder
        self.explanation_encoder = explanation_encoder
        self.attention = multihead_Attention
        self.position_feedforward = position_Feedforward

        self.upscale_conv, self.first_conv, self.flatten_conv = self.get_convolutions(input_dim=input_dim, hidden_dim=hidden_dim)
        self.linear1, self.linear2, self.bilinear, self.classifier = self.get_linears_layers(max_length_sentence=max_length_sentence)

    def forward(self, sentence, justification, credit_history):

        sentence_pos = torch.arange(0, sentence.shape[2]).unsqueeze(0).repeat(sentence.shape[0],1).to(self.device).long()
        justification_pos = torch.arange(0, justification.shape[2]).unsqueeze(0).repeat(justification.shape[0], 1).to(self.device).long()

        sentence = self.upscale_conv(sentence)
        sentence = sentence + self.sentence_pos_embedding(sentence_pos).permute(0, 2, 1)

        justification = self.upscale_conv(justification)
        justification = justification + self.justification_pos_embedding(justification_pos).permute(0, 2, 1)

        #encode the sentence and justification

        encoded_s = self.sentence_encoder(sentence)
        encoded_j = self.explanation_encoder(justification)

        #define attention_output as the output from the attention layer
        att = self.attention(encoded_s, encoded_j, encoded_j)

        #apply positional feed forward layer on the attention_output.
        x = self.position_feedforward(att)

        #apply first_conv followed by relu activation on the output of positional feed forward layer
        x = self.first_conv(x)
        x = nn.ReLU()(x)
        
	#apply flattened convolution of the output of step 4.
        x = self.flatten_conv(x)
        x = nn.ReLU()(x)

        #Flatten the output from the last layer so that we can pass the output to a linear layer.
        x = torch.flatten(x)

        #apply linear1, linear2 and bilinear layers in this order on the final output of
	#For bilinear layer, the second input is the credit history.
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.bilinear(x, credit_history)

        #apply the classifier layer
        x = self.classifier(x)

        return x

    def get_convolutions(self, input_dim, hidden_dim): 
        
        upscale_conv = nn.Conv1d(input_dim, hidden_dim, 1, stride=1)
        first_conv = nn.Conv1d(hidden_dim, input_dim//2, 3, stride=1, padding=1)
        flatten_conv = nn.Conv1d(input_dim//2, 1, kernel_size=5, stride=1, padding=2)
        return upscale_conv, first_conv, flatten_conv


    def get_linears_layers(self, max_length_sentence):

        linear1 = nn.Linear(max_length_sentence, max_length_sentence//4)
        linear2 = nn.Linear(max_length_sentence//4, 6)
        bilinear = nn.Bilinear(6, 5, 12, bias=True)
        classifier = nn.Linear(12, 6)
        return linear1, linear2, bilinear, classifier
    

