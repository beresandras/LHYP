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
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

class MedicalMode(Enum):
    SA = 0
    SALE = 1
    LA_2C = 2
    LA_3C = 3
    LA_4C = 4
    LALE = 5

class Pathology(Enum):
    UNDEFINED     = -1
    NORMAL        = 0
    HCM           = 1
    AMY           = 2
    AMYLOIDOSIS   = 2
    EMF           = 3
    ADULT_M_SPORT = 4
    ADULT_F_SPORT = 5
    U18_F         = 6
    U18_M         = 7
    AORTASTENOSIS = 8
    FABRY         = 9

class DatasetParameters:
    def __init__(self, data_dir, expected_pathologies):
        self.data_dir = data_dir
        self.patient_ids = [pid for pid in sorted(os.listdir(data_dir)) if pid[0] is not '_']
        if len(self.patient_ids) == 0:
            print('Error: no data found in folder {}'.format(data_dir))
            exit()
        self.expected_pathologies = expected_pathologies

        # patient_id -> starting image index for that patient
        self.start_indices_sa = {}
        self.start_indices_sale = {}
        self.start_indices_la = {}
        self.start_indices_lale = {}

        # image_index -> patient_id
        self.patient_ids_sa = []
        self.patient_ids_sale = []
        self.patient_ids_la = []
        self.patient_ids_lale = []

        # shuffled_image_index -> image_index
        self.shuffled_indices_sa = []
        self.shuffled_indices_sale = []
        self.shuffled_indices_la = []
        self.shuffled_indices_lale = []

        # image_index -> pathology label
        self.pathology_labels_sa = []
        self.pathology_labels_sale = []
        self.pathology_labels_la = []
        self.pathology_labels_lale = []

    def pathology_from_meta_str(self, meta_str):
        for line in meta_str.splitlines():
            if line.startswith('Pathology'):
                pathology_str = line[11:].strip()
                if pathology_str == 'NORM':
                    pathology_str = 'NORMAL'
                return Pathology[pathology_str]
        return Pathology['UNDEFINED']

    def analyze_data(self):
        for pid in self.patient_ids:
            with open(os.path.join(self.data_dir, pid), 'rb') as input_file:
                try:
                    loaded_data = pickle.load(input_file)
                    
                    pathology = self.pathology_from_meta_str(loaded_data.meta_str)
                    if pathology in self.expected_pathologies:
                        if len(loaded_data.images_sa) > 0:
                            self.start_indices_sa[pid] = len(self.patient_ids_sa)
                        for _ in range(len(loaded_data.images_sa)):
                            self.patient_ids_sa.append(pid)
                            self.pathology_labels_sa.append(self.expected_pathologies[pathology])

                        if len(loaded_data.images_sale) > 0:
                            self.start_indices_sale[pid] = len(self.patient_ids_sale)
                        for _ in range(len(loaded_data.images_sale)):
                            self.patient_ids_sale.append(pid)
                            self.pathology_labels_sale.append(self.expected_pathologies[pathology])

                        if loaded_data.images_la_2C == loaded_data.images_la_3C and loaded_data.images_la_3C == loaded_data.images_la_4C:
                            if len(loaded_data.images_la_2C) > 0:
                                self.start_indices_la[pid] = len(self.patient_ids_la)
                            for _ in range(len(loaded_data.images_la_2C)):
                                self.patient_ids_la.append(pid)
                                self.pathology_labels_la.append(self.expected_pathologies[pathology])
                        else:
                            print('Error: the number of 2-, 3-, and 4-chamber LA images is not equal.')
                            exit()

                        if len(loaded_data.images_lale) > 0:
                            self.start_indices_lale[pid] = len(self.patient_ids_lale)
                        for _ in range(len(loaded_data.images_lale)):
                            self.patient_ids_lale.append(pid)
                            self.pathology_labels_lale.append(self.expected_pathologies[pathology])

                except Exception as e:
                    print(e)
                    exit()

        self.shuffled_indices_sa = np.arange(0, len(self.patient_ids_sa))
        self.shuffled_indices_sale = np.arange(0, len(self.patient_ids_sale))
        self.shuffled_indices_la = np.arange(0, len(self.patient_ids_la))
        self.shuffled_indices_lale = np.arange(0, len(self.patient_ids_lale))
        
        np.random.shuffle(self.shuffled_indices_sa)
        np.random.shuffle(self.shuffled_indices_sale)
        np.random.shuffle(self.shuffled_indices_la)
        np.random.shuffle(self.shuffled_indices_lale)

    def save_object(self):
        with open(self.data_dir + '/_dataset_params', 'wb') as outfile:
            pickle.dump(self, outfile)

class GenericHypertrophyDataset(Dataset):
    def __init__(self, dataset_mode, dataset_split, medical_mode, dataset_params, augmenter):
        super(GenericHypertrophyDataset).__init__()
        self.dataset_mode = dataset_mode
        self.medical_mode = medical_mode
        self.data_dir = dataset_params.data_dir
        self.augmenter = augmenter

        if self.medical_mode == MedicalMode['SA']:
            self.patient_ids = dataset_params.patient_ids_sa
            start_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][0])
            stop_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][1])
            self.shuffled_indices = dataset_params.shuffled_indices_sa[start_index:stop_index]
            self.start_indices = dataset_params.start_indices_sa
            self.pathology_labels = dataset_params.pathology_labels_sa
        elif self.medical_mode == MedicalMode['SALE']:
            self.patient_ids = dataset_params.patient_ids_sale
            start_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][0])
            stop_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][1])
            self.shuffled_indices = dataset_params.shuffled_indices_sale[start_index:stop_index]
            self.start_indices = dataset_params.start_indices_sale
            self.pathology_labels = dataset_params.pathology_labels_sale
        elif self.medical_mode == MedicalMode['LA_2C'] or self.medical_mode == MedicalMode['LA_3C'] or self.medical_mode == MedicalMode['LA_4C']:
            self.patient_ids = dataset_params.patient_ids_la
            start_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][0])
            stop_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][1])
            self.shuffled_indices = dataset_params.shuffled_indices_la[start_index:stop_index]
            self.start_indices = dataset_params.start_indices_la
            self.pathology_labels = dataset_params.pathology_labels_la
        elif self.medical_mode == MedicalMode['LALE']:
            self.patient_ids = dataset_params.patient_ids_lale
            start_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][0])
            stop_index = int(len(self.patient_ids) * dataset_split[self.dataset_mode][1])
            self.shuffled_indices = dataset_params.shuffled_indices_lale[start_index:stop_index]
            self.start_indices = dataset_params.start_indices_lale
            self.pathology_labels = dataset_params.pathology_labels_lale
        else:
            print('Unsupported dataset mode: {}'.format(self.medical_mode))
            exit()

        self.length = stop_index - start_index

    def _get_image(self, loaded_data, index):
        deshuffled_index = self.shuffled_indices[index]
        real_index = deshuffled_index - self.start_indices[self.patient_ids[deshuffled_index]]
        if self.medical_mode == MedicalMode['SA']:
            np_image = loaded_data.images_sa[real_index]
        elif self.medical_mode == MedicalMode['SALE']:
            np_image = loaded_data.images_sale[real_index]
        elif self.medical_mode == MedicalMode['LA_2C']:
            np_image = loaded_data.images_la_2C[real_index]
        elif self.medical_mode == MedicalMode['LA_3C']:
            np_image = loaded_data.images_la_3C[real_index]
        elif self.medical_mode == MedicalMode['LA_4C']:
            np_image = loaded_data.images_la_4C[real_index]
        elif self.medical_mode == MedicalMode['LALE']:
            np_image = loaded_data.images_lale[real_index]
        else:
            print('Unexpected unsupported dataset mode: {}'.format(self.medical_mode))
            exit()
        np_image = np_image.astype(np.float32)
        image_min = np.percentile(np_image, 1)
        image_max = np.percentile(np_image, 99)
        np_image = (np.clip(np_image, image_min, image_max) - image_min) / (image_max - image_min)
        np_image *= 255
        #np_image = np.array([np_image for _ in range(3)]).astype(np.uint8)
        return self.augmenter(np_image), self.pathology_labels[deshuffled_index]

    def __getitem__(self, index):
        with open(os.path.join(self.data_dir, self.patient_ids[self.shuffled_indices[index]]), 'rb') as data_file:
            return self._get_image(pickle.load(data_file), index)

    def __len__(self):
        return self.length

if __name__=='__main__':
    data_dir = './data/beres'
    force_analyzis = True

    if not torch.cuda.is_available():
        print('CUDA is not available')
        exit()
    device = torch.device("cuda")

    seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)

    pathology_to_label = {
        Pathology['NORMAL']: 0,
        Pathology['HCM']: 1,
        Pathology['AMY']: 2,
        Pathology['AMYLOIDOSIS']: 2,
        Pathology['EMF']: 2,
        Pathology['AORTASTENOSIS']: 2,
        Pathology['FABRY']: 2
    }

    if (force_analyzis != True and os.path.isfile(data_dir + '/_dataset_params')):
        with open(data_dir + '/_dataset_params', 'rb') as param_file:
            dataset_params = pickle.load(param_file)
    else:
        dataset_params = DatasetParameters(data_dir, pathology_to_label)
        dataset_params.analyze_data()
        dataset_params.save_object()

    print('Number of all patients: {}'.format(len(dataset_params.patient_ids)))
    print('Number of SA images: {}, patients: {}'.format(len(dataset_params.patient_ids_sa), len(set(dataset_params.patient_ids_sa))))
    print('Number of SALE images: {}, patients: {}'.format(len(dataset_params.patient_ids_sale), len(set(dataset_params.patient_ids_sale))))
    print('Number of LA images: {}, patients: {}'.format(len(dataset_params.patient_ids_la), len(set(dataset_params.patient_ids_la))))
    print('Number of LALE images: {}, patients: {}'.format(len(dataset_params.patient_ids_lale), len(set(dataset_params.patient_ids_lale))))

    dataset_split = {
        DatasetMode['TRAIN']: [0.0, 0.7],
        DatasetMode['VALIDATION']: [0.7, 0.85],
        DatasetMode['TEST']: [0.85, 1.0]
    }

    medical_mode_str = 'SALE'
    medical_mode = MedicalMode[medical_mode_str]
    epochs = 2
    batch_size = 16

    model = models.resnet.resnet18(num_classes=3) #models.vgg.vgg11_bn(num_classes=3)
    test_model = models.resnet.resnet18(num_classes=3)
    model_input_size = 224

    model.to(device)
    test_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_loader = DataLoader(
        GenericHypertrophyDataset(
            DatasetMode['TRAIN'],
            dataset_split,
            medical_mode,
            dataset_params,
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine((-20, 20), translate=(0.1, 0.1), scale=(1.8, 2.2)),
                transforms.CenterCrop((model_input_size, model_input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
        ),
        batch_size=batch_size
    )

    validation_loader = DataLoader(
        GenericHypertrophyDataset(
            DatasetMode['VALIDATION'],
            dataset_split,
            medical_mode,
            dataset_params,
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine((0, 0), translate=(0, 0), scale=(2, 2)),
                transforms.CenterCrop((model_input_size, model_input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
        ),
        batch_size=batch_size
    )

    test_loader = DataLoader(
        GenericHypertrophyDataset(
            DatasetMode['TEST'],
            dataset_split,
            medical_mode,
            dataset_params,
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine((0, 0), translate=(0, 0), scale=(2, 2)),
                transforms.CenterCrop((model_input_size, model_input_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
        ),
        batch_size=batch_size
    )

    model_save_frequency = 5
    model_path = '../models/resnet18_{}_{}.pth'
    show_image = False

    print('Training on {}'.format(medical_mode_str))
    for epoch in range(epochs):
        avg_train_loss = 0.0
        avg_train_acc = 0
        for batch_index, data_batch in enumerate(train_loader):
            image_batch, label_batch = data_batch
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            model.train()
            optimizer.zero_grad()

            outputs = model(image_batch)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()

            avg_train_loss += loss.item()
            avg_train_acc += torch.sum(predictions == label_batch)

            if ((batch_index + 1) % 10 == 0):
                print(batch_index, end=', ')
            if (show_image):
                plt.imshow(image_batch[0][0].numpy(), cmap='gray')
                plt.show()

        avg_validation_loss = 0.0
        avg_validation_acc = 0
        for batch_index, data_batch in enumerate(validation_loader):
            model.eval()
            image_batch, label_batch = data_batch
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            outputs = model(image_batch)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, label_batch)

            avg_validation_loss += loss.item()
            avg_validation_acc += torch.sum(predictions == label_batch)

        avg_train_loss /= len(train_loader)
        avg_validation_loss /= len(validation_loader)

        avg_train_acc = float(avg_train_acc) / (len(train_loader) * batch_size)
        avg_validation_acc = float(avg_validation_acc) / (len(validation_loader) * batch_size)

        print()
        print("epoch: {}, train_loss: {:.3f}, train_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}"\
                .format(epoch, avg_train_loss, avg_train_acc, avg_validation_loss, avg_validation_acc))

        if (epoch + 1) % model_save_frequency == 0:
            torch.save(model.state_dict(), model_path.format(medical_mode_str, epoch))
            print('Model saved at epoch {}'.format(epoch))
    if (epoch + 1) % model_save_frequency != 0:
        torch.save(model.state_dict(), model_path.format(medical_mode_str, epoch))
        print('Model saved at epoch {}'.format(epoch))
    
    load_epoch = input('Load model from epoch: ')
    test_model.load_state_dict(torch.load(model_path.format(medical_mode_str, load_epoch)))

    avg_test_loss = 0.0
    avg_test_acc = 0
    for batch_index, data_batch in enumerate(test_loader):
        model.eval()
        image_batch, label_batch = data_batch
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        outputs = model(image_batch)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, label_batch)

        avg_test_loss += loss.item()
        avg_test_acc += torch.sum(predictions == label_batch)
    avg_test_loss /= len(test_loader)
    avg_test_acc = float(avg_test_acc) / (len(test_loader) * batch_size)
    print("test_loss: {:.3f}, test_acc: {:.3f}".format(avg_test_loss, avg_test_acc))