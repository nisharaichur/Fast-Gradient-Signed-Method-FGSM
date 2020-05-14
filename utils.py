import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import json
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def return_class_name(predictions):
  max_dim = predictions.argmax(dim=1).item()
  class_name = id_classname[str(max_dim)][1]
  return class_name , max_dim


def return_class_accuracy(predictions, class_id):
  prob = F.softmax(predictions, dim=1)
  accuracy = prob[0, class_id] * 100
  return torch.round(accuracy).item()


def visualize(image, adv_image, epsilon, gradients,  target_class, adversarial_class, target_acc, adversarial_acc, acc_of_original):
    image = image.squeeze(0) 
    image = image.detach() 
    image = image.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    image = np.transpose( image , (1,2,0))   
    image = np.clip(image, 0, 1)

    adv_image = adv_image.squeeze(0)
    adv_image = adv_image.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    adv_image = np.transpose( adv_image , (1,2,0))  
    adv_image = np.clip(adv_image, 0, 1)

    gradients = gradients.squeeze(0).numpy()
    gradients = np.transpose(gradients, (1,2,0))
    gradients = np.clip(gradients, 0, 1)

    figure, ax = plt.subplots(1,3, figsize=(18,8))
    ax[0].imshow(image)
    ax[0].set_title('Original Image', fontsize=20)
    ax[0].axis("off")

    ax[1].imshow(gradients)
    ax[1].set_title('Perturbation epsilon: {}'.format(epsilon), fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].imshow(adv_image)
    ax[2].set_title('Adversarial Example', fontsize=20)
    ax[2].axis("off")

    ax[0].text(0.5,-0.13, "Prediction: {}\n Accuracy: {}%".format(target_class, target_acc), size=15, ha="center", transform=ax[0].transAxes)
    ax[2].text(0.5,-0.13, "Prediction of {} is {}%\n Prediction of {} is {}%".format(adversarial_class, adversarial_acc, target_class, acc_of_original), size=15, ha="center", transform=ax[2].transAxes)
    plt.show()

