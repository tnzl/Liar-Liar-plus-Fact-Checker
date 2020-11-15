import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def trainer(model, train_dataloader, val_dataloader, num_epochs, path_to_save='/home/atharva',
          checkpoint_path='/home/atharva',
          checkpoint=100, train_batch=1, test_batch=1, device='cuda:0'): # 2 Marks. 
    """
    Everything by default gets shifted to the GPU. Select the device according to your system configuration
    If you do no have a GPU, change the device parameter to "device='cpu'"
    :param model: the Classification model..
    :param train_dataloader: train_dataloader
    :param val_dataloader: val_Dataloader
    :param num_epochs: num_epochs
    :param path_to_save: path to save model
    :param checkpoint_path: checkpointing path
    :param checkpoint: when to checkpoint
    :param train_batch: 1
    :param test_batch: 1
    :param device: Defaults on GPU, pass 'cpu' as parameter to run on CPU. 
    :return: None
    """
    #torch.backends.cudnn.benchmark = True #Comment this if you are not using a GPU...
    # set the network to training mode.
      model.cuda()  # if gpu available otherwise comment this line. 
    # your code goes here. 
      def accuracy(y1,y2):
            aa = list((y1==y2).astype('int'))
            acc = sum(aa) / len(aa)
            del aa
            return acc
      training_acc = []
      training_loss = []
      val_acc = []
      val_loss = []

      #Train the model on the train_dataloader.
      from torch.nn import CrossEntropyLoss
      criterion = CrossEntropyLoss()
      for epoch in range(num_epochs):  # loop over the dataset multiple times
            preds = []
            labels = []
            
            for i in range(len(train_dataloader)):
                  # get the inputs; data is a list of [inputs, labels]
                  data_dict = train_dataloader[i]

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # forward + backward + optimize
                  output = model(data_dict['statement'], data_dict['justification'], data_dict['credit_history'])
                  loss = criterion(outputs, data_dict['label'])
                  loss.backward()
                  optimizer.step()
                  preds.append(output)
                  labels.append(data_dict['label'])

            #Calculate the metrics, that is the loss and accuracy for the training phase per epoch and store them in a list.
            training_acc.append(accuracy(preds.numpy(), labels.numpy()))
            training_loss.append(criterion(preds, label))

            #Validating
            preds = []
            labels = []
            for i in range(len(val_dataloader)):
                  # get the inputs; data is a list of [inputs, labels]
                  data_dict = val_dataloader[i]

                  # forward + backward + optimize
                  outputs = model(data_dict['statement'], data_dict['justification'], data_dict['credit_history'])
                  loss = criterion(outputs, labels)
                  preds.append(output)
                  labels.append(data_dict['label'])

            val_acc.append(accuracy(preds.numpy(), labels.numpy()))
            val_loss.append(criterion(preds, label))

            #Save your model at the maximum validation accuracy obtained till the latest epoch.
            if val_acc[-1] > max(val_acc[:-1]):
                  #Save model
                  torch.save(model.state_dict(), save_path)

            #Checkpoint at the 100th epoch
            if epoch%100 == 0:
                  #make a checkpoint
                  torch.save(model.state_dict(), save_path)



            


    plt.plot(training_acc)
    plt.plot(val_acc)
    plt.plot(training_loss)
    plt.plot(val_loss)
    plt.show()

