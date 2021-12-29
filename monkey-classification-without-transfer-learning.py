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


img_size = 380
img_crop = 256


# In[ ]:


# normalization parameters
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

#Applying Transformation
train_transforms = transforms.Compose([transforms.Resize((img_size,img_size)),
                                transforms.CenterCrop(img_crop),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize((img_size,img_size)),
                                      transforms.CenterCrop(img_crop),
                                      transforms.ToTensor()])


# In[ ]:


train_set = torchvision.datasets.ImageFolder(tr_path, transform=train_transforms)
test_set = torchvision.datasets.ImageFolder(te_path, transform=test_transforms)


# In[ ]:


plt.imshow(train_set[0][0].permute(1,2,0))


# In[ ]:


BATCH_SIZE = 64

torch.manual_seed(0)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
torch.manual_seed(0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.batch_32 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)
        self.batch_64 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=5, padding=2)
        self.batch_128 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256, kernel_size=5, padding=2)
        self.batch_256 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, padding=3)
        self.batch_512 = nn.BatchNorm2d(512)

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.45)

        self.fc1 = nn.Linear(in_features=512*8*8, out_features=64)
        self.batch_fc = nn.BatchNorm1d(64)

        self.out = nn.Linear(in_features=64, out_features=10)

    
    def forward(self, x):

        x = self.conv1(x)
        x = self.batch_32(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batch_64(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_128(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.batch_256(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = self.batch_512(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.fc1(x.view(-1, 512*8*8))
        x = self.batch_fc(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.out(x)

        return x


# In[ ]:


torch.manual_seed(0)
net = Net()
net.to(device)
print(net)


# In[ ]:


EPOCHS = 35
LEARNING_RATE = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()


# In[ ]:


def train():
    
    train_loss = []
    test_loss = []

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

        if accuracy > 75:
            return train_loss, test_loss

    return train_loss, test_loss


# In[ ]:


train_loss, test_loss = train()


# In[ ]:


plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.legend()
plt.ylabel('Train and Test Loss')
plt.xlabel('Epochs')
plt.show()


# In[ ]:


dataset = transforms.Compose([transforms.Resize((img_size,img_size)),
                                transforms.CenterCrop(img_crop),
                                transforms.ToTensor()])

dataset = torchvision.datasets.ImageFolder('../input/10-monkey-species/validation/validation', transform=dataset)


# In[ ]:


animal = {0: 'mantled_howler', 1: 'patas_monkey', 2: 'bald_uakari ', 3: 'japanese_macaque', 4: 'pygmy_marmoset', 5: 'white_headed_capuchin', 6: 'silvery_marmoset', 7: 'common_squirrel_monkey', 8: 'black_headed_night_monkey', 9: 'nilgiri_langur'}

n = 10

plt.figure()
fig, ax = plt.subplots(n, 1, figsize=(50, 50))
fig.subplots_adjust(hspace=0.5)

indices = list(range(272))
random.shuffle(indices)
indices = indices[:10]

for j, i in enumerate(indices):
    truth_label = dataset[i][1]

    prediction = net(dataset[i][0].view(1, 3, img_crop, img_crop).to(device))
    index = prediction.argmax().item()

    ax[j].set_title('Prediction: {} \nTruth: {}'.format(animal[index], animal[truth_label]))
    ax[j].imshow(dataset[i][0].permute(1,2,0), cmap='gray_r')


# In[ ]:




