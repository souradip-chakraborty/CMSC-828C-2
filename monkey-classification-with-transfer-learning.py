#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch 
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import random
from tqdm import tqdm
from matplotlib import pyplot as plt


# In[ ]:


tr_path = '../input/10-monkey-species/training/training/'
te_path = '../input/10-monkey-species/validation/validation/'


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


img_crop = 224


# In[ ]:


#Transform that needs to be done for Transfer Learning

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize(256), 
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# In[ ]:


train_set = torchvision.datasets.ImageFolder(tr_path, transform=transform)
test_set = torchvision.datasets.ImageFolder(te_path, transform=transform)


# In[ ]:


plt.imshow(train_set[12][0].permute(1,2,0))


# In[ ]:


BATCH_SIZE = 64

torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
torch.manual_seed(0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# In[ ]:


# pretrained resnet

net = models.resnet18(pretrained=True).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


# In[ ]:


EPOCHS = 15


# In[ ]:


net


# In[ ]:


for name, param in net.named_parameters():
    print(name)


# In[ ]:


# freeze parameters

for name, param in net.named_parameters():
    if 'fc.' not in name:
        param.requires_grad = False
    else:
        print(name)


# In[ ]:


def train(EPOCHS):
    
    train_loss = []
    test_loss = []
    test_accuracy = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_test_loss = 0.0
        correct_predictions = 0.0

        net.train()
        for (data, target) in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data.view(BATCH_SIZE, 3, img_crop, img_crop))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()

        
        with torch.no_grad():
            net.eval()
            for (data, target) in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = net(data.view(BATCH_SIZE, 3, img_crop, img_crop))
                loss = criterion(output, target)
                running_test_loss += loss.detach().item()

                index = output.max(dim=1)[1]
                correct_predictions = correct_predictions + (index == target).sum().detach().item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        avg_test_loss = running_test_loss / len(test_loader)
        test_loss.append(avg_test_loss)

        accuracy = 100*(correct_predictions / (len(test_loader)*BATCH_SIZE))

        print('Epoch {}, Train Loss: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.3f}'.format(epoch+1, avg_train_loss, avg_test_loss, accuracy))
        test_accuracy.append(accuracy)

#         if accuracy > 75:
#             return train_loss, test_loss

    return train_loss, test_loss, test_accuracy


# In[ ]:





# In[ ]:


EPOCHS = 15
train_loss, test_loss, test_accuracy = train(EPOCHS)


# In[ ]:


plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.ylabel('Train and Test Loss')
plt.xlabel('Epochs')
plt.title('Resnet-18 Finetune Linear Layer')
plt.show()


# In[ ]:


plt.plot(test_accuracy, label='Test Accuracy')
plt.legend()
plt.ylabel('Test accuracy')
plt.xlabel('Epochs')
plt.title('Resnet-18 Finetune Linear Layer')
plt.show()


# In[ ]:


# unfreeze and fine-tune
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.00001)

for name, param in net.named_parameters():
        param.requires_grad = True


# In[ ]:


#Finetuned accuracy
EPOCHS = 15
train_loss, test_loss, test_accuracy = train(EPOCHS)


# In[ ]:


plt.plot(test_accuracy, label='Test Accuracy')
plt.legend()
plt.ylabel('Test accuracy')
plt.xlabel('Epochs')
plt.title('Resnet-18 Finetune Full Network')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




