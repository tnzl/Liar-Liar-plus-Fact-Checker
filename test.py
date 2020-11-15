from utils import *
import pickle
import torch
import pandas as pd
from datasets import *
from ConvS2S import *
from Attention import *
from Encoder import *

def test_word_embeddings(path_to_csv):
  #File: utils.py
  #Function: create_glove_dict()
  word_embeddings = create_glove_dict(path_to_csv)
  with open('utils/sample_test_cases_embeddings.pkl', 'rb') as f:
    correct_embeddings = pickle.load(f)
    assert (word_embeddings['spelled'] == correct_embeddings[0]).all()
    assert (word_embeddings['witch'] == correct_embeddings[1]).all()
    assert (word_embeddings['defencemen'] == correct_embeddings[2]).all()
  print('Sample test cases passed')


def test_get_max_length(path_to_tsv = 'train2.tsv'):
  #File: utils.py
  #Function: get_max_length()
  df = pd.read_csv(path_to_tsv, sep = '\t', header = None)
  print(get_max_length(df.iloc[:100], 3))
  assert get_max_length(df.iloc[:100], 3) == 39
  print('Sample test case passed')

def test_dataset(): 
  #File: datasets.py
  #Class: dataset
  #Function: __get_item__()
  
  sample_dataset = dataset(purpose = 'test_class')
  with open('dataset/statements.pkl', 'rb') as f:
    correct_statements = pickle.load(f)
  with open('dataset/justifications.pkl', 'rb') as f:
    correct_justifications = pickle.load(f)
  with open('dataset/labels.pkl', 'rb') as f:
    correct_labels = pickle.load(f)
  i = 0
  for _, data in enumerate(sample_dataset):
    print(i)
    assert (data['statement'].numpy() == correct_statements[i]).all()
    assert (data['justification'].numpy() == correct_justifications[i]).all()
    assert (data['label'].numpy() == correct_labels[i]).all()  
    i = i + 1
  print('Sample test cases passed')

def test_get_convolutions():
  #File: ConvS2S.py
  #Class: ConvEncoder
  #Function: get_convolutions()
  conv_block = ConvEncoder(input_dim = 10, num_layers = 20)
  with open('convs2s/conv_layers_in_channels.pkl', 'rb') as f:
    input_specs = pickle.load(f)
  with open('convs2s/conv_layers_out_channels.pkl', 'rb') as f:
    output_specs = pickle.load(f)
  with open('convs2s/conv_layers_paddings.pkl', 'rb') as f:
    padding_specs = pickle.load(f)
  i= 0
  for layer in conv_block.convolutions:
    assert layer.in_channels == input_specs[i]
    assert layer.out_channels == output_specs[i]
    assert layer.padding == padding_specs[i]
    i = i  + 1
  print('Sample test cases passed')

def test_conv_block():
  #File: ConvS2S.py
  #Class: COnvEncoder
  #Function: forward()
  conv_block = ConvEncoder(input_dim = 10, num_layers = 5)
  for param in conv_block.parameters():
    torch.nn.init.constant(param, 1.)
  correct_model = torch.load('convs2s/conv_encoder_block.pt')
  input = torch.rand((20, 10, 3))
  assert (conv_block(input).detach().numpy() == correct_model(input).detach().numpy()).all()
  print('Sample Test Case passed.')

def test_get_layers_pff():
  #File: Attention.py
  #Class: PositionFeedforward
  #Function: get_layers()
  pff = PositionFeedforward(5, 20)
  correct_pff = torch.load('attention/pff.pt')
  assert str(pff.conv1) == str(correct_pff.conv1)
  assert str(pff.conv2) == str(correct_pff.conv2)
  print(pff.conv2)
  print('Sample test case Passed')

def test_pff_forward():
  #File: Attention.py
  #Class: PositionFeedForward
  #Function: forward()
  pff = PositionFeedforward(5, 20)
  for param in pff.parameters():
    torch.nn.init.constant(param, 1.)
  correct_pff = torch.load('attention/pff.pt')
  input = torch.rand((20, 5, 3))
  assert (pff(input).detach().numpy() == correct_pff(input).detach().numpy()).all()
  print('Sample test cases passed')

def test_get_layers_mha():
  #File: Attention.py
  #Class: MultiHeadAttention
  #Function: get_layers()
  mha = MultiHeadAttention(hid_dim = 256, n_heads = 16)
  correct_mha = torch.load('attention/mha.pt')
  assert str(correct_mha.conv_query) == str(mha.conv_query)
  assert str(correct_mha.conv_key) == str(mha.conv_key)
  assert str(correct_mha.conv_value) == str(mha.conv_value)
  assert str(correct_mha.conv_output) == str(mha.conv_output)
  print('Sample Test Cases Passed')

def test_encoder_forward():
  #File: Encoder.py
  #Class: Encoder
  #Function: forward() and __init__()
  enc = Encoder(hidden_dim=32, conv_layers = 5)
  enc_param_list = []
  for param in enc.parameters():
    torch.nn.init.constant(param, 1.)
  correct_enc = torch.load('encoder/enc.pt')
  input = torch.rand((64, 32, 3))
  assert (enc(input).detach().numpy() == correct_enc(input).detach().numpy()).all()
  print('Sample Test Cases Passed')
