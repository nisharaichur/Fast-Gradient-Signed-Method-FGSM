import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from utils import return_class_name, return_class_accuracy, visualize

alexnet = models.alexnet(pretrained=True)
alexnet.eval() #keeping this in eval() mode will keep the requires_grad of the weights to False (We update the image instead)

f = open("/imagenet_class_index.json")#ImageNet class to index mappings
id_classname = json.load(f)	

image = Image.open("/panda.jpg")
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
preprocess = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std)
                            ])#always a better idea to normalize images before we give as an input to CNN models

input_image = preprocess(image).unsqueeze(0)
input_image = Variable(input_image, requires_grad=True)#we need to update the image, hence keep its requires_grad to True, backward() detects computes the gradients of the loss with respect to the input_image

predictions = alexnet(input_image)
(target_class, target_dim) = return_class_name(predictions)
target_acc = return_class_accuracy(predictions, target_dim)

target = Variable(torch.LongTensor([target_dim]), requires_grad=False)
loss = torch.nn.CrossEntropyLoss()
loss_val = loss(predictions, target)
loss_val.backward(retain_graph = True)#calculate the gradients of loss with respect to input_image
grads = torch.sign(input_image.grad.data)#gradients are stored in the input_image.grad variable

list_accu = []
epsilon = [0.0001, 0.001, 0.0013, 0.009, 0.015, 0.120, 0.25, 0.302] #we add noise based on the epsilon values
for i in epsilon:#FGSM method of adding the noise to an image
  adversarial_image = input_image.data + i * grads
  torch.clamp(adversarial_image, min=0, max=1)
  predictions_adv = alexnet.forward(Variable(adversarial_image))

  adversarial_class, class_index = return_class_name(predictions_adv)
  adversarial_acc = return_class_accuracy(predictions_adv, class_index)

  acc_of_original = return_class_accuracy(predictions_adv, target_dim)

  visualize(input_image, adversarial_image, i, grads, target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original)
  list_accu.append(acc_of_original)
