import torch
import random

data = torch.load("LDEO_TRAIN.pt")

scaled_down = torch.nn.functional.interpolate(data, scale_factor = 0.1)

images = []

for i in range(1):
    image = torch.zeros(2, 1, 1000, 1000)
    for row in range(10):
        scaled_down = scaled_down[:, torch.randperm(scaled_down.size(1)), :, :]
        for column in range(10):
            image[:, :, row*100:(row+1)*100, column*100:(column+1)*100] = torch.rot90(scaled_down[:, column:column+1, :, :], int(torch.rand(1).item() * 4), [2, 3])
    images.append(image)

aug = torch.cat(images, dim=1)
print(aug.size())
torch.save(aug, "CUSTOM_AUG.pt")