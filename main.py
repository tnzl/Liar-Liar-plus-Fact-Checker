from torch.utils.data import DataLoader

from datasets import dataset
from encoder import Encoder
from Attention import MultiHeadAttention, PositionFeedforward
from LiarLiar import arePantsonFire
path_to_glove = None

#liar_dataset_train and liar_dataset_val defined as datasets.dataset() with appropriate
#value in prep_data_from argument to prepare data. sentence and justification length are both
#defined as liar_dataset_train.get_max_length(). Instantiate dataloader_train and dataloader_val
#on train and val dataset
liar_dataset_train = dataset()
liar_dataset_val = dataset(prep_Data_from='val', purpose='test_class')

batch_size = 25
dataloader_train = DataLoader(liar_dataset_train, batch_size)
dataloader_val = DataLoader(liar_dataset_val, batch_size)

#statement_encoder and justification_encoder defined as instances of Encoder class
statement_encoder = Encoder(5,512)
justification_encoder = Encoder(5,512)

#multiHeadAttention and positionFeedForward are instances of the respective classes
multiHeadAttention = MultiHeadAttention(512, 32)
positionFeedForward = PositionFeedforward(512, 2048)

#model is an instance of arePantsOnFire class
model = arePantsonFire(
    statement_encoder
    , justification_encoder
    , multihead_Attention
    , position_Feedforward
    , 512
    , max_length_sentence
    )

#call to the trainer function
from trainer import trainer
path_to_save = None #Define it
path_to_checkpoint = None #Define it
trainer(
    model
    , dataloader_train
    , dataloader_val
    , -1
    , path_to_save
    , checkpoint_path)
#define liar_data_test as datasets.dataset with test data and test_dataloader on this
#dataset with batch_size = 1
liar_dataset_test = liar_dataset_val = dataset(prep_Data_from='test', purpose='test_class')
test_dataloader = DataLoader(liar_dataset_test, batch_size)

#function call to the infer function from utils.
from utils import infer
infer(model, test_dataloader)

module_list = [liar_dataset_train, liar_dataset_val, dataloader_train, dataloader_val, statement_encoder, justification_encoder, multiheadAttention, positionFeedForward, model]
del  liar_dataset_val, liar_dataset_train, dataloader_train, dataloader_val


liar_dataset_test = dataset(prep_Data_from='test')
test_dataloader = DataLoader(dataset=liar_dataset_test, batch_size=1)
infer(model=model, dataloader=test_dataloader)

