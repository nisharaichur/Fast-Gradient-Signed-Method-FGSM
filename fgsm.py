import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


alexnet = models.alexnet(pretrained=True)
alexnet.eval()

f = open("/imagenet_class_index.json")
id_classname = json.load(f)	

image = Image.open("/panda.jpg")
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
preprocess = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
                            ])

input_image = preprocess(image).unsqueeze(0)
input_image = Variable(input_image, requires_grad=True)

predictions = alexnet(input_image)
(target_class, target_dim) = return_class_name(predictions)
target_acc = return_class_accuracy(predictions, target_dim)

target = Variable(torch.LongTensor([target_dim]), requires_grad=False)
loss = torch.nn.CrossEntropyLoss()
loss_val = loss(predictions, target)
loss_val.backward(retain_graph = True)
grads = torch.sign(input_image.grad.data)
