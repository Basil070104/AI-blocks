import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

rand_tensor = torch.rand(5,2)
simple_model = nn.Sequential(nn.Linear(2,10), nn.ReLU(), nn.Linear(10,1))
print(f'input is on {rand_tensor.device}')
print(f'model parameters are on {[param.device for param in simple_model.parameters()]}')
print(f'output is on {simple_model(rand_tensor).device}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------- <Your code> ---------------
# Move rand_tensor and model onto the GPU device

rand_tensor = torch.Tensor.to(rand_tensor, device)
simple_mode = torch.nn.Module.to(simple_model, device)

# --------- <End your code> -------------
print(f'input is on {rand_tensor.device}')
print(f'model parameters are on {[param.device for param in simple_model.parameters()]}')
print(f'output is on {simple_model(rand_tensor).device}')

def train(model: nn.Module,
          loss_fn: nn.modules.loss._Loss,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          epoch: int=0)-> List:
    train_loss = list()
    model.train(True)

    for batch_idx, (images, targets) in enumerate(train_loader):
      images = images.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()
      output = model(images)
      loss = loss_fn(output, targets)
      loss.backward()
      optimizer.step()

      train_loss.append(loss.item())
      if batch_idx % 100 == 0:
        print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item():.3f}')

    assert len(train_loss) == len(train_loader)
    return train_loss

def test(model: nn.Module, loss_fn: nn.modules.loss._Loss,
        test_loader: torch.utils.data.DataLoader,
        epoch: int=0)-> Dict:
    test_stat = {"loss" : None, "accuracy" : None, "prediction" : None}
    model.eval()

    test_loss = 0
    correct = 0
    all_pred = []

    total_samples = len(test_loader.dataset)

    with torch.no_grad():
      for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)

        output = model(images)
        test_loss += loss_fn(output, targets).item()
        pred = output.data.max(1, keepdim=True)[1]
        all_pred.append(pred)
        correct += pred.eq(targets.data.view_as(pred)).sum().item()

    test_stat["loss"] = test_loss / total_samples
    test_stat["accuracy"] = correct / total_samples
    test_stat["prediction"] = torch.cat(all_pred, dim=0)
    total_num = total_samples

    print(f"Test result on epoch {epoch}: total sample: {total_num}, Avg loss: {test_stat['loss']:.3f}, Acc: {100*test_stat['accuracy']:.3f}%")
    # dictionary should include loss, accuracy and prediction
    assert "loss" and "accuracy" and "prediction" in test_stat.keys()
    # "prediction" value should be a 1D tensor
    assert len(test_stat["prediction"]) == len(test_loader.dataset)
    assert isinstance(test_stat["prediction"], torch.Tensor)
    return test_stat
  
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Normalize((0.1307,),(0.3081,))])

train_dataset = torchvision.datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('data', train=False, download=True, transform=transform)

batch_size_train, batch_size_test = 64, 1000

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

print(train_dataset)
batch_idx, (images, targets) = next(enumerate(train_loader))
print(f'current batch index is {batch_idx}')
print(f'images has shape {images.size()}')
print(f'targets has shape {targets.size()}')

class OurFC(nn.Module):

  def __init__(self):
    super(OurFC, self).__init__()
    self.first= nn.Linear(28*28, 128)
    self.second = nn.Linear(128, 64)
    self.third = nn.Linear(64, 10)

  def forward(self, x):
    x = x.view(-1, 784)
    x = self.first(x)
    x = F.relu(x)
    x = self.second(x)
    x = F.relu(x)
    x = self.third(x)
    return F.log_softmax(x, -1)

class OurCNN(nn.Module):

  def __init__(self):
    super(OurCNN, self).__init__()
    self.conv = nn.Conv2d(1, 3, kernel_size=5)
    self.fc = nn.Linear(432, 10)

  def forward(self, x):
    x = self.conv(x)        # x now has shape (batchsize x 3 x 24 x 24)
    x = F.relu(F.max_pool2d(x,2))  # x now has shape (batchsize x 3 x 12 x 12)
    x = x.view(-1, 432)      # x now has shape (batchsize x 432)
    x = F.relu(self.fc(x))     # x has shape (batchsize x 10)
    return F.log_softmax(x,-1)
  
criterion = nn.CrossEntropyLoss()

start = time.time()
max_epoch = 3

model = OurFC()
optimize = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

for epoch in range(1, max_epoch+1):
  train(model,F.nll_loss, optimize, train_loader, epoch)
  test(model, F.nll_loss, test_loader, epoch)

end = time.time()
print(f'Finished Training after {end-start} s ')

# Let's then train the OurCNN model.
start = time.time()

model = OurCNN()
optimize = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

for epoch in range(1, max_epoch+1):
  train(model,F.nll_loss, optimize, train_loader, epoch)
  test(model, F.nll_loss, test_loader, epoch)

end = time.time()
print(f'Finished Training after {end-start} s ')

ourfc = OurFC()
total_params = sum(p.numel() for p in ourfc.parameters())
print(f'OurFC has a total of {total_params} parameters')

ourcnn = OurCNN()
total_params = sum(p.numel() for p in ourcnn.parameters())
print(f'OurCNN has a total of {total_params} parameters')