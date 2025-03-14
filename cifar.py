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
# Move rand_tensor and model onto the GPU device

rand_tensor = torch.Tensor.to(rand_tensor, device)
simple_mode = torch.nn.Module.to(simple_model, device)
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

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Create the appropriate transform, load/download CIFAR10 train and test datasets with transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the training dataset
trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# Define trainloader and testloader
batch_size_train, batch_size_test = 64, 1000

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True,)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False)

# Code to display images
batch_idx, (images, targets) = next(enumerate(train_loader))
fig, ax = plt.subplots(3,3,figsize = (9,9))
for i in range(3):
    for j in range(3):
        image = images[i*3+j].permute(1,2,0)
        image = image/2 + 0.5
        ax[i,j].imshow(image)
        ax[i,j].set_axis_off()
        ax[i,j].set_title(f'{classes[targets[i*3+j]]}')
fig.show()\
  
class ourCIFAR10CNN(nn.Module):

  def __init__(self):
    super(ourCIFAR10CNN, self).__init__()
    self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(1024, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    # Convolutional layers with ReLU and max pooling
    x = F.relu(self.conv(x))
    x = F.max_pool2d(x, 2)

    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)

    x = F.relu(self.conv3(x))
    x = F.max_pool2d(x, 2)

    x = x.view(x.size(0), -1)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return F.log_softmax(x, dim=1)
  
# Train your neural network here.
start = time.time()
max_epoch = 4

model = ourCIFAR10CNN()
optimize = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

for epoch in range(1, max_epoch+1):
  train(model,F.nll_loss, optimize, train_loader, epoch)
  test(model, F.nll_loss, test_loader, epoch)

net = model
epoch = 5
criterion = nn.CrossEntropyLoss()
output = test(net, criterion, test_loader, epoch)
end = time.time()
print(f'Finished Training after {end-start} s ')

total_images = 3
predictions = output['prediction']
targets = torch.tensor(testset.targets)
misclassified_images = []
misclassified_labels = []
misclassified_predictions = []

for idx, prediction in enumerate(predictions):
    if prediction != targets[idx]:
        misclassified_images.append(testset[idx][0])  # Get image from testset
        misclassified_labels.append(targets[idx])
        misclassified_predictions.append(prediction)
        if len(misclassified_images) == total_images:
            break

fig, ax = plt.subplots(1, total_images, figsize=(total_images * 3, 9))

for i, img in enumerate(misclassified_images):
    img = img.detach().cpu() if img.is_cuda else img  # Move to CPU if on GPU
    img = img / 2 + 0.5  # Unnormalize if necessary

    ax[i].imshow(img.permute(1, 2, 0))  # Permute for display
    ax[i].set_title(f'Predicted: {classes[misclassified_predictions[i].item()]}, True: {classes[misclassified_labels[i].item()]}')
    ax[i].axis('off')

plt.tight_layout()
plt.show()