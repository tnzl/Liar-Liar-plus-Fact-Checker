import pandas
import numpy
from LiarLiar import arePantsonFire

import seaborn
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

from torch.utils.data import DataLoader
import torch

def create_glove_dict(path_to_text:str): 
    """
    Create the dictionary containing word and corresponding vector. 
    :param path_to_text: Path to Glove embeddings.      
    """
    embeddings = {}
    '''
    Your code goes here. 
    '''

    lines = open(path_to_text, 'r').readlines()
    for line in lines:
        tokens = line.strip().split()
                    
        embeddings[tokens[0]] = map(int, tokens[1:])

    return embeddings

def get_max_length(dataframe: pandas.DataFrame, column_number: int): 							
    """
    :param dataframe: Pandas Dataframe
    :param column_number: Column number you want to get max value from
    :return: max_length: int
    """
    max_length = 0
    '''
    Your code goes here
    '''

    return dataframe[:,[column_number]].applymap(lambda x: len(str(x))).max()
            
def visualize_Attenion(attention_matrix: numpy.ndarray):
    """
    Visualizes multihead attention. Expected input shape to [n_heads, query_len, key_len]
    :param attention_matrix:
    :return:
    """
    assert len(attention_matrix.shape) == 3

    for head in range(attention_matrix.shape[0]):
        seaborn.heatmap(attention_matrix[head])
    plt.show()

def infer(model: arePantsonFire, dataloader:DataLoader):
    """
    Use for inferencing on the trained model. Assumes batch_size is 1.
    :param model: trained model.
    :param dataloader: Test Dataloader
    :return:
    """
    labels = {0: "true", 1: "mostly true", 2: "half true", 3: "barely true" , 4: "false", 5: "pants on fire"}
    model.eval()
    correct = 0
    wrong = 0
    for _, data in enumerate(dataloader):
        statement = data['statement']
        justification = data['justification']
        credit_history = data['credit_history']
        label = data['label']

        prediction = model(statement, justification, credit_history)
        if torch.argmax(prediction).item() == label.item():
            print("Correct Prediction")
            correct+=1
        else:
            print("wrong prediction")
            wrong+=1

        print(labels[torch.argmax(prediction, dim=1).item()])

        print('-------------------------------------------------------------------------------------------------------')
    print(correct/_)
    print(wrong/_)
    
