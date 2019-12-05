import os
from enum import Enum
import numpy as np
import PIL
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import data_reader
from data_reader import PatientData

class DatasetMode(Enum):
    SA = 1
    SALE = 2
    LA_2C = 3
    LA_3C = 4
    LA_4C = 5
    LALE = 6

class DatasetParameters:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.patient_ids = [pid for pid in sorted(os.listdir(data_dir)) if pid[0] is not '_']
        if len(self.patient_ids) == 0:
            print("Error: no data found in folder {}".format(data_dir))
            exit()

        self.patient_ids_sa = []
        self.patient_ids_sale = []
        self.patient_ids_la = []
        self.patient_ids_lale = []

        self.start_indices_sa = {}
        self.start_indices_sale = {}
        self.start_indices_la = {}
        self.start_indices_lale = {}

    def analyze_data(self):
        for pid in self.patient_ids:
            with open(os.path.join(self.data_dir, pid), "rb") as input_file:
                try:
                    loaded_data = pickle.load(input_file)

                    if len(loaded_data.images_sa) > 0:
                        self.start_indices_sa[pid] = len(self.patient_ids_sa)
                    for _ in range(len(loaded_data.images_sa)):
                        self.patient_ids_sa.append(pid)

                    if len(loaded_data.images_sale) > 0:
                        self.start_indices_sale[pid] = len(self.patient_ids_sale)
                    for _ in range(len(loaded_data.images_sale)):
                        self.patient_ids_sale.append(pid)

                    if loaded_data.images_la_2C == loaded_data.images_la_3C and loaded_data.images_la_3C == loaded_data.images_la_4C:
                        if len(loaded_data.images_la_2C) > 0:
                            self.start_indices_la[pid] = len(self.patient_ids_la)
                        for _ in range(len(loaded_data.images_la_2C)):
                            self.patient_ids_la.append(pid)
                    else:
                        print("Error: the number of 2-, 3-, and 4-chamber LA images is not equal.")
                        exit()

                    if len(loaded_data.images_lale) > 0:
                        self.start_indices_lale[pid] = len(self.patient_ids_lale)
                    for _ in range(len(loaded_data.images_lale)):
                        self.patient_ids_lale.append(pid)

                except Exception as e:
                    print(e)
                    exit()

    def save_object(self):
        with open(self.data_dir + "/_dataset_params", "wb") as outfile:
            pickle.dump(self, outfile)

class GenericHypertrophyDataset(Dataset):
        def __init__(self, mode, dataset_params, augmenter):
            super(GenericHypertrophyDataset).__init__()
            self.mode = mode
            self.data_dir = dataset_params.data_dir
            self.augmenter = augmenter

            if self.mode == DatasetMode["SA"]:
                self.patient_ids = dataset_params.patient_ids_sa
                self.start_indices = dataset_params.start_indices_sa
            elif self.mode == DatasetMode["SALE"]:
                self.patient_ids = dataset_params.patient_ids_sale
                self.start_indices = dataset_params.start_indices_sale
            elif self.mode == DatasetMode["LA_2C"] or self.mode == DatasetMode["LA_3C"] or self.mode == DatasetMode["LA_4C"]:
                self.patient_ids = dataset_params.patient_ids_la
                self.start_indices = dataset_params.start_indices_la
            elif self.mode == DatasetMode["LALE"]:
                self.patient_ids = dataset_params.patient_ids_lale
                self.start_indices = dataset_params.start_indices_lale

            self.length = len(self.patient_ids)

        def _get_image(self, loaded_data, index):
            real_index = index - self.start_indices[self.patient_ids[index]]
            if self.mode == DatasetMode["SA"]:
                return loaded_data.images_sa[real_index]
            elif self.mode == DatasetMode["SALE"]:
                return loaded_data.images_sale[real_index]
            elif self.mode == DatasetMode["LA_2C"]:
                return loaded_data.images_la_2C[real_index]
            elif self.mode == DatasetMode["LA_3C"]:
                return loaded_data.images_la_3C[real_index]
            elif self.mode == DatasetMode["LA_4C"]:
                return loaded_data.images_la_4C[real_index]
            elif self.mode == DatasetMode["LALE"]:
                return loaded_data.images_lale[real_index]

        def __getitem__(self, index):
            with open(os.path.join(self.data_dir, self.patient_ids[index]), "rb") as data_file:
                return self.augmenter((self._get_image(pickle.load(data_file), index)).astype(np.uint8)) #[..., np.newaxis]

        def __len__(self):
            return self.length


if __name__=='__main__':
    data_dir = "../bindata"
    force_analyzis = True

    if (force_analyzis != True and os.path.isfile(data_dir + "/_dataset_params")):
        with open(data_dir + "/_dataset_params", "rb") as param_file:
            dataset_params = pickle.load(param_file)
    else:
        dataset_params = DatasetParameters(data_dir)
        dataset_params.analyze_data()
        dataset_params.save_object()

    print("Number of all patients: {}".format(len(dataset_params.patient_ids)))
    print("Number of SA images: {}, patients: {}".format(len(dataset_params.patient_ids_sa), len(set(dataset_params.patient_ids_sa))))
    print("Number of SALE images: {}, patients: {}".format(len(dataset_params.patient_ids_sale), len(set(dataset_params.patient_ids_sale))))
    print("Number of LA images: {}, patients: {}".format(len(dataset_params.patient_ids_la), len(set(dataset_params.patient_ids_la))))
    print("Number of LALE images: {}, patients: {}".format(len(dataset_params.patient_ids_lale), len(set(dataset_params.patient_ids_lale))))

    augmenter = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = GenericHypertrophyDataset(DatasetMode["LALE"], dataset_params, augmenter)

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=1
        #shuffle=True
    )

    model = models.resnet.resnet18(num_classes=3)

    for batch_index, image_batch in enumerate(loader):
        print('batch {}, image shape {}'.format(batch_index, image_batch.shape))
        print(image_batch[0][0].numpy())
        plt.imshow(image_batch[0][0].numpy())
        plt.show()