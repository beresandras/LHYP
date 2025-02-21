import os
import time
import copy
from enum import Enum
import numpy as np
import PIL
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    def __init__(self, data_dir, dataset_path, expected_pathologies):
        self.data_dir = data_dir
        self.dataset_path = dataset_path
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
                pathology_str = line[11:].strip().upper()
                if pathology_str == 'NORM':
                    pathology_str = 'NORMAL'
                return Pathology[pathology_str]
        return Pathology['UNDEFINED']

    def analyze_data(self):
        for pid in tqdm(self.patient_ids):
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
        with open(self.dataset_path, 'wb') as outfile:
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

def train_model(model, model_info, epochs, batch_size, gpu_index, seed):
    start_time = time.time()
    force_analyzis = False
    show_image = False

    data_dir = '../data/beres'
    results_dir = '../results'
    models_dir = '../models'

    dataset_str = '{}Class'.format(model_info['num_classes'])
    model_str = '{}_{}_{}Class_{}Aug_{}'.format(model_info['name'], model_info['medical_mode'], model_info['num_classes'], \
                                                model_info['augmentation'], model_info['transfer'])

    dataset_path = results_dir + '/dataset_' + dataset_str + '.pickle'
    model_path = models_dir + '/model_' + model_str + '.pth'
    results_path = results_dir + '/result_' + model_str + '.pickle'

    if not torch.cuda.is_available():
        print('CUDA is not available')
        exit()
    #torch.cuda.empty_cache()
    #torch.cuda.ipc_collect()
    device = torch.device(gpu_index)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.seed_all()

    pathology_to_label_dict = {
        2: {
            Pathology['NORMAL']: 0,
            Pathology['HCM']: 1,
            Pathology['AMY']: 1,
            Pathology['AMYLOIDOSIS']: 1,
            Pathology['EMF']: 1,
            Pathology['AORTASTENOSIS']: 1,
            Pathology['FABRY']: 1
        },
        3: {
            Pathology['NORMAL']: 0,
            Pathology['HCM']: 1,
            Pathology['AMY']: 2,
            Pathology['AMYLOIDOSIS']: 2,
            Pathology['EMF']: 2,
            Pathology['AORTASTENOSIS']: 2,
            Pathology['FABRY']: 2
        },
        4: {
            Pathology['NORMAL']: 0,
            Pathology['HCM']: 1,
            Pathology['AMY']: 2,
            Pathology['AMYLOIDOSIS']: 2,
            Pathology['EMF']: 2,
            Pathology['AORTASTENOSIS']: 2,
            Pathology['FABRY']: 2,
            Pathology['ADULT_M_SPORT']: 3,
            Pathology['ADULT_F_SPORT']: 3,
            Pathology['U18_F']: 3,
            Pathology['U18_M']: 3,
        }
    }
    pathology_to_label = pathology_to_label_dict[model_info['num_classes']]

    if (force_analyzis != True and os.path.isfile(dataset_path)):
        with open(dataset_path, 'rb') as param_file:
            dataset_params = pickle.load(param_file)
    else:
        dataset_params = DatasetParameters(data_dir, dataset_path, pathology_to_label)
        dataset_params.analyze_data()
        dataset_params.save_object()
        tqdm.write('New dataset parameters have been created:')
        tqdm.write('Number of all patients: {}'.format(len(dataset_params.patient_ids)))
        tqdm.write('Number of SA images: {}, patients: {}'.format(len(dataset_params.patient_ids_sa), len(set(dataset_params.patient_ids_sa))))
        tqdm.write('Number of SALE images: {}, patients: {}'.format(len(dataset_params.patient_ids_sale), len(set(dataset_params.patient_ids_sale))))
        tqdm.write('Number of LA images: {}, patients: {}'.format(len(dataset_params.patient_ids_la), len(set(dataset_params.patient_ids_la))))
        tqdm.write('Number of LALE images: {}, patients: {}'.format(len(dataset_params.patient_ids_lale), len(set(dataset_params.patient_ids_lale))))

    dataset_split = {
        DatasetMode['TRAIN']: [0.0, 0.7],
        DatasetMode['VALIDATION']: [0.7, 0.85],
        DatasetMode['TEST']: [0.85, 1.0]
    }

    medical_mode = MedicalMode[model_info['medical_mode']]

    model_input_size = 224
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    augmentation_dict = {
        'min': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine((0, 0), translate=(0, 0), scale=(2, 2)),
            transforms.CenterCrop((model_input_size, model_input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]),
        'norm': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine((-20, 20), translate=(0.1, 0.1), scale=(1.8, 2.2)),
            transforms.CenterCrop((model_input_size, model_input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ]),
        'max': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine((-30, 30), translate=(0.15, 0.15), scale=(1.7, 2.3)),
            transforms.CenterCrop((model_input_size, model_input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor()
        ]),
        'min_transf': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine((0, 0), translate=(0, 0), scale=(2, 2)),
            transforms.CenterCrop((model_input_size, model_input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'norm_transf': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine((-20, 20), translate=(0.1, 0.1), scale=(1.8, 2.2)),
            transforms.CenterCrop((model_input_size, model_input_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    train_loader = DataLoader(
        GenericHypertrophyDataset(
            DatasetMode['TRAIN'],
            dataset_split,
            medical_mode,
            dataset_params,
            (augmentation_dict[model_info['augmentation']] if model_info['transfer'] == 'retrain' else augmentation_dict[model_info['augmentation'] + '_transf'])
        ),
        batch_size=batch_size,
        drop_last=True
    )

    validation_loader = DataLoader(
        GenericHypertrophyDataset(
            DatasetMode['VALIDATION'],
            dataset_split,
            medical_mode,
            dataset_params,
            (augmentation_dict['min'] if model_info['transfer'] == 'retrain' else augmentation_dict['min_transf'])
        ),
        batch_size=batch_size,
        drop_last=True
    )

    test_loader = DataLoader(
        GenericHypertrophyDataset(
            DatasetMode['TEST'],
            dataset_split,
            medical_mode,
            dataset_params,
            (augmentation_dict['min'] if model_info['transfer'] == 'retrain' else augmentation_dict['min_transf'])
        ),
        batch_size=batch_size,
        drop_last=True
    )

    tqdm.write('\nTraining {} for {} epochs'.format(model_str, epochs))
    model_info['num_trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    min_validation_loss = None
    for epoch in range(epochs):
        avg_train_loss = 0.0
        avg_train_acc = 0
        for _, data_batch in tqdm(enumerate(train_loader), total=len(train_loader)):
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

            if (show_image):
                plt.imshow(image_batch[0][0].numpy(), cmap='gray')
                plt.show()

        avg_validation_loss = 0.0
        avg_validation_acc = 0
        for ba_tch_index, data_batch in enumerate(validation_loader):
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

        tqdm.write('epoch: {}, train_loss: {:.3f}, train_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}'\
                .format(epoch, avg_train_loss, avg_train_acc, avg_validation_loss, avg_validation_acc))
        model_info['val_losses'].append(avg_validation_loss)
        model_info['val_accs'].append(avg_validation_acc)

        if min_validation_loss == None or min_validation_loss > avg_validation_loss:
            min_validation_loss = avg_validation_loss
            torch.save(model.state_dict(), model_path)
            tqdm.write('Model saved at epoch {}'.format(epoch))
    
    model.load_state_dict(torch.load(model_path))
    
    avg_test_loss = 0.0
    avg_test_acc = 0
    model_info['confusion_matrix'] = torch.zeros(model_info['num_classes'], model_info['num_classes'])
    for _, data_batch in enumerate(test_loader):
        model.eval()
        image_batch, label_batch = data_batch
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        outputs = model(image_batch)
        _, predictions = torch.max(outputs, 1)
        loss = criterion(outputs, label_batch)

        avg_test_loss += loss.item()
        avg_test_acc += torch.sum(predictions == label_batch)

        for truth, prediction in zip(label_batch.view(-1), predictions.view(-1)):
            model_info['confusion_matrix'][truth.long(), prediction.long()] += 1

    avg_test_loss /= len(test_loader)
    avg_test_acc = float(avg_test_acc) / (len(test_loader) * batch_size)
    tqdm.write('test_loss: {:.3f}, test_acc: {:.3f}'.format(avg_test_loss, avg_test_acc))
    model_info['test_loss'] = avg_test_loss
    model_info['test_acc'] = avg_test_acc
    model_info['train_time']= (time.time() - start_time) / (epochs * batch_size)

    with open(results_path, 'wb') as out_file:
        pickle.dump(model_info, out_file)

if __name__=='__main__':
    epochs = 2
    batch_size = 32

    gpu_index = 1
    seed = 10

    model_objects = [
        models.resnet.resnet18(num_classes=3),
        models.resnet.resnet18(num_classes=3),
        models.resnet.resnet18(num_classes=3),
        models.resnet.resnet18(num_classes=3),
        models.resnet.resnet18(num_classes=3),
        models.resnet.resnet18(num_classes=3),
        
        models.vgg.vgg11_bn(num_classes=3),
        models.squeezenet.squeezenet1_0(num_classes=3),
        models.mobilenet.mobilenet_v2(num_classes=3),
        
        models.resnet.resnet18(num_classes=2),
        models.resnet.resnet18(num_classes=4),
        
        models.resnet.resnet18(num_classes=3),
        models.resnet.resnet18(num_classes=3),
        
        models.resnet.resnet18(pretrained=True),
        models.resnet.resnet18(pretrained=True)
    ]

    # finetune: modify final layer
    model_objects[-2].fc = nn.Linear(model_objects[-2].fc.in_features, 3)

    # transfer-learn: modify final layer and freeze the others
    for param in model_objects[-1].parameters():
        param.requires_grad = False
    model_objects[-1].fc = nn.Linear(model_objects[-2].fc.in_features, 3)

    base_model_info = {
        'name': 'ResNet',
        'year': 2015,
        'medical_mode': 'SA',
        'num_classes': 3,
        'augmentation': 'norm',
        'transfer': 'retrain',

        'num_trainable_params': None,
        'train_time': None,
        'val_losses': [],
        'val_accs': [],
        'test_loss': None,
        'test_acc': None,
        'confusion_matrix': None
    }
    model_infos = [base_model_info]

    model_info = copy.deepcopy(base_model_info)
    model_info['medical_mode'] = 'SALE'
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['medical_mode'] = 'LA_2C'
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['medical_mode'] = 'LA_3C'
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['medical_mode'] = 'LA_4C'
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['medical_mode'] = 'LALE'
    model_infos.append(model_info)


    model_info = copy.deepcopy(base_model_info)
    model_info['name'] = 'VGG'
    model_info['year'] = 2014
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['name'] = 'SqueezeNet'
    model_info['year'] = 2016
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['name'] = 'MobileNet'
    model_info['year'] = 2018
    model_infos.append(model_info)
    

    model_info = copy.deepcopy(base_model_info)
    model_info['num_classes'] = 2
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['num_classes'] = 4
    model_infos.append(model_info)
    

    model_info = copy.deepcopy(base_model_info)
    model_info['augmentation'] = 'min'
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['augmentation'] = 'max'
    model_infos.append(model_info)
    

    model_info = copy.deepcopy(base_model_info)
    model_info['transfer'] = 'finetune'
    model_infos.append(model_info)

    model_info = copy.deepcopy(base_model_info)
    model_info['transfer'] = 'transfer'
    model_infos.append(model_info)

    for model, model_info in zip(model_objects, model_infos):
        train_model(
            model=model,
            model_info=model_info,
            epochs=epochs,
            batch_size=batch_size,
            gpu_index=gpu_index,
            seed=seed
        )
