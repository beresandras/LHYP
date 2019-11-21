import PIL
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle

import data_reader
from data_reader import PatientData

mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
  plt.imshow(img)
  plt.axis('off')

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR)
])

dataset = []
data_folder = "../bindata"
data_paths = os.listdir(data_folder)
if len(data_paths) > 0:
    for data_path in data_paths:
        print(os.path.join(data_folder, data_path))
        input_file = open(os.path.join(data_folder, data_path), "rb")
        try:
            data = pickle.load(input_file)
            dataset.append(data)
        except Exception as e:
            print(e)
        input_file.close()

        print(data.patient_id)
        print(data.images_sa)
        print(data.images_sale)
        print(data.images_la)
        print(data.images_lale)