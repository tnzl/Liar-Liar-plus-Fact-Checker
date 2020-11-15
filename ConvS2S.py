import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        """
        The convolutional Encoder layer.
        Model based on the paper "Convolutional Sequence to Sequence Learning - https://arxiv.org/pdf/1705.03122.pdf"
        :param input_dim: input dimension of the tensor
        :param num_layers: Number of convolutional you desire to stack on top of each other
        """
        super(ConvEncoder, self).__init__()
        self.convolutions = self.get_convolutions(input_dim=input_dim, num_layers=num_layers)

    def forward(self, source):
        """
        Remember, we have multiple convolutional layers, for iterate over them.
        :param source: Input tensor for the forward pass. 
        """
        input_source = source.clone() #Cloning source to avoid error (had to google this)
        for conv in self.convolutions:
            o = conv(input_source)
            o = nn.functional.glu(o, dim=1)
            obtained_output = input_source + o 
        return obtained_output

    def get_convolutions(self, input_dim, num_layers=3): 
        """
        :param input_dim: input dimension
        :param num_layers: Number of convolutional you desire to stack on top of each other
        :return: nn.ModuleList()
        """
        param = nn.ModuleList([nn.Conv1d(
            (i+1) * input_dim
            , 2 * (i+1) * input_dim
            , 3
            , stride=1
            , padding=1
            ) for i in range(num_layers)])
        return param
