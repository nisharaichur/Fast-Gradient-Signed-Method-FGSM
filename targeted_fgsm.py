import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from utils import return_class_name, return_class_accuracy, visualize

alexnet = models.resnet50(pretrained=True)
alexnet.eval()
f = open("/content/imagenet_class_index.json")
id_classname = json.load(f)

image = Image.open("/content/panda.jpg")
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
preprocess = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            ])

norm = transforms.Normalize(mean=mean, std=std)
input_image = preprocess(image).unsqueeze(0)
input_image = Variable(input_image)# we do not take the gradients wrt to this image, we take the gradients wrt the delta(perturbations on this image)
original_image = norm(input_image.squeeze(0)).unsqueeze(0)

predictions = alexnet(original_image)
(target_class, target_dim) = return_class_name(predictions)
target_acc = return_class_accuracy(predictions, target_dim)

actual_class = torch.LongTensor([target_dim])
required_class = torch.LongTensor([504])#this is the class whose accuracy we want to increase(coffee-mug in this case)
required_class_name = id_classname[str(required_class.item())][1]

delta = torch.zeros_like(input_image, requires_grad=True)# we are optimizing the perturbations on the image and not the image itself, so set the required_grad  for delta as true
optimizer = optim.SGD([delta], lr=0.005)

epsilon = 0.0081 #can try out lesser values as well
for i in range(100):
  data = input_image + delta
  data = norm(data.squeeze(0))
  predictions = alexnet(data.unsqueeze(0))
  
  loss = torch.nn.CrossEntropyLoss() 
  loss_maximize = loss(predictions, actual_class) #we try to maximize this loss
  loss_minimize = loss(predictions, required_class) #we try to minimize this loss
  total_loss = -loss_maximize + loss_minimize #total loss to be optimized

  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()
  adersarial_class = return_class_name(predictions)[0]
  adversarial_acc = return_class_accuracy(predictions, 504)
  acc_of_original = return_class_accuracy(predictions, target_dim)
  delta.data.clamp_(-epsilon, epsilon)
  if i % 10 == 0:
    visualize(original_image, data.unsqueeze(0), epsilon, delta, target_class, required_class_name, target_acc, adversarial_acc, acc_of_original)
