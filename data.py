import os
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

train = datasets.MNIST("", train=True, download=True,
					transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
					transform = transforms.Compose([transforms.ToTensor()]))

#out of sample testing data

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

for data in trainset:
	print(data)
	break

x,y = data[0][0], data[0][1]

print (y)

print(data[0][0].shape)

plt.imshow(data[0][0].view(28,28))
plt.show()

total = 0 
counter_dict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
	Xs, ys = data
	for y in ys:
		counter_dict[int(y)] += 1
		total += 1

print(counter_dict)

for i in couner_dict:
	print (f"{i}: {counter_dict[i]/total*100}")