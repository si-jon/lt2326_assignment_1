import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn.modules.module import Module

# Input - bilder / tre dimensionell array
# Output - Koordinater till bounding box
# Dimension på bilderna: 2048 x 2048 x 3
# Dimension på bounding box: 2 x 4

# Två saker!
#   1. Hitta tecken
#   2. Avgör om det är kinesiskt eller inte

class charDetectionCNN(nn.Module):

    def __init__(self, height, width, channels, cords_dim, cords_count):
        super(charDetectionCNN, self).__init__()

        self.conv1 = nn.Conv2d(channels, )

    def forward(self, image):
        return self.softmax(self.linear(image))

transforms.ToTensor()

'''
class SimpleCNN(torch.nn.Module):
   def __init__(self):
      super(SimpleCNN, self).__init__()
      #Input channels = 3, output channels = 18
      self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
      self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
      #4608 input features, 64 output features (see sizing flow below)
      self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
      #64 input features, 10 output features for our 10 defined classes
      self.fc2 = torch.nn.Linear(64, 10)


'''