import torch
from torchvision import transforms
from PIL import Image
import imageio
import random
import time
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_tensors = []
label_tensors = []

train_pos = ["1x6", "2x3", "2x5", "2x6", "3x3", "3x5", "3x8", "4x3", "4x4", "4x5", "4x6", "4x7", "4x8", "5x2", "5x3", "5x5", "5x6", "6x1", "6x2", "6x3", "6x4", "6x5", "7x1", "7x2", "8x1", "8x2"]
valid_pos = ["1x5", "2x4", "3x4", "3x6", "5x7", "7x3"]
test_pos = ["4x24", "9x10", "12x21", "22x38", "32x32", "35x25"]

for i in range(0, 26):
    image = imageio.imread("data_trainset/fracture_"+str(train_pos[i])+".tif").astype('float32')
    label = imageio.imread("data_trainset/fracture_"+str(train_pos[i])+"_mask.tif").astype('float32')

    image_tensor = torch.from_numpy(image).view(1, 1000, 1000).float() / 65536.0
    label_tensor = torch.from_numpy(label).view(1, 1000, 1000).float() / 256.0

    print(torch.min(image_tensor), torch.max(image_tensor))
    print(torch.min(label_tensor), torch.max(label_tensor))
    print("")

    image_tensors.append(image_tensor)
    label_tensors.append(label_tensor)

data = torch.cat((torch.stack(image_tensors), torch.stack(label_tensors)), dim=1)

data = data.permute(1, 0, 2, 3)

print(data.size())

torch.save(data, "LDEO_TRAIN.pt")