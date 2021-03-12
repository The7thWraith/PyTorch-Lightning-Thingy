#!pip install update torch
#!pip install update cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8-cp37-cp37m-linux_x86_64.whl
#!pip install pytorch_lightning
#!pip install DataLoader
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader

#print(device_lib.list_local_devices())

#!git clone https://github.com/NVIDIA/apex
#%cd apex
#!pip install -v --no-cache-dir  ./ --cuda_ext --cpp_ext

if __name__ == '__main__':

    #torch.backends.cudnn.benchmark = True

    #torch.cuda.set_device(0)
    # functions to show an image

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        #    download=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          #    shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('beaver', 'dolphin', 'otter', 'seal', 'whale', 'fish', 'aquarium', 'fish', 'flatfish', 'ray', 'shark',
               'trout', 'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 'containers', 'bottles', 'bowls', 'cans', 
               'cups', 'plates', 'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 'clock', 'computer keyboard', 'lamp', 'telephone', 
               'television', 'bed', 'chair', 'couch', 'table', 'wardrobe', 'bee', 'beetle',
               'butterfly', 'caterpillar', 'cockroach', 'bear', 'leopard', 'lion', 
               'tiger', 'wolf', 'bridge', 'castle', 'house', 'road', 'skyscraper', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'camel', 'cattle', 
               'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'crab', 'lobster', 'snail', 'spider', 'worm', 
               'baby', 'boy', 'girl', 'man', 'woman', 'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', 'hamster', 'mouse', 
               'rabbit', 'shrew', 'squirrel', 'maple', 'oak', 'palm', 
               'pine', 'willow', 'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')
    # functions to show an image
    
#    def imshow(img):
#        img = img / 2 + 0.5     # unnormalize
#        npimg = img.numpy()
#        plt.imshow(np.transpose(npimg, (1, 2, 0)))
#        plt.show()


    # get some random training images
#    dataiter = iter(trainloader)
#    images, labels = dataiter.next()

    # show images
#    imshow(torchvision.utils.make_grid(images))
    # print labels
#    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    # get some random training images
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    

   
#    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#    inputs, labels = data

    trainer = pl.Trainer(gpus=1)

    class LitModel(pl.LightningModule):
        def __init__(self):
            super(LitModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 100)
        

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def train_dataloader(self):
          trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
          trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
          return trainloader
          
        
        def configure_optimizers(self):
          optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
          return optimizer
        
        def training_step(self, trainloader, dataiter):
          dataiter = iter(trainloader)
          print(next(dataiter))
          images, labels = dataiter.next()
          outputs = self(inputs)
          criterion = nn.CrossEntropyLoss()
          loss = criterion(outputs, labels)
          return loss
        
        def test_step(self, trainloader, dataiter):
          inputs, labels = data
          outputs = self(inputs)
          criterion = nn.CrossEntropyLoss()
          loss = criterion(outputs, labels)
          return loss


    net = LitModel()
    criterion = nn.CrossEntropyLoss()
    trainer.fit(net)
    trainer.train()
    trainer.test()



