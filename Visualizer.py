import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

def sorted_by(array_to_sort, array_by):
    return [element for _, element in sorted(zip(array_by, array_to_sort), key=lambda pair: pair[0])]

def matshow_with_values(matrix, xlabels, ylabels):
    _, ax = plt.subplots()
    ax.matshow(matrix)
    ax.set_xticklabels([''] + xlabels)
    ax.set_yticklabels([''] + ylabels)
    ax.xaxis.set_label_position('top')

    for (i, j), z in np.ndenumerate(matrix):
        ax.text(j, i, '{}'.format(int(z)), ha='center', va='center', color='white', fontweight='bold', fontsize=16)

results_dir = '../results/'
charts_dir = '../charts/'

model_infos = []
for result_file_path in os.listdir(results_dir):
    if result_file_path.startswith('result_'):
        with open(results_dir + result_file_path, 'rb') as infile:
            model_infos.append(pickle.load(infile))

chart_medical_modes = []
chart_test_losses = []
for model_info in model_infos:
    if model_info['name'] == 'ResNet' and model_info['num_classes'] == 3 and model_info['augmentation'] == 'norm' and model_info['transfer'] == 'retrain':
        chart_medical_modes.append(model_info['medical_mode'])
        chart_test_losses.append(model_info['test_loss'])

chart_medical_modes = sorted_by(chart_medical_modes, chart_test_losses)
chart_test_losses = sorted(chart_test_losses)

plt.bar(chart_medical_modes, chart_test_losses)
plt.xlabel('MRI mode')
plt.ylabel('test loss')
plt.savefig(charts_dir + 'MRI_modes.png')
plt.close()

chart_names = []
chart_num_trainable_params = []
chart_train_times = []
chart_test_losses = []
chart_test_accs = []
for model_info in model_infos:
    if model_info['medical_mode'] == 'SA' and model_info['num_classes'] == 3 and model_info['augmentation'] == 'norm' and model_info['transfer'] == 'retrain':
        chart_names.append(model_info['name'])
        chart_num_trainable_params.append(model_info['num_trainable_params'])
        chart_train_times.append(model_info['train_time'])
        chart_test_losses.append(model_info['test_loss'])
        chart_test_accs.append(model_info['test_acc'])

chart_names = sorted_by(chart_names, chart_test_losses)
chart_num_trainable_params = sorted_by(chart_num_trainable_params, chart_test_losses)
chart_train_times = sorted_by(chart_train_times, chart_test_losses)
chart_test_accs = sorted_by(chart_test_accs, chart_test_losses)
chart_test_losses = sorted(chart_test_losses)

fig = plt.figure(figsize=(6, 7))
ax = plt.subplot(3, 1, 1)
plt.bar(chart_names, chart_test_losses)
plt.ylabel('test loss')
plt.subplot(3, 1, 2, sharex=ax)
plt.bar(chart_names, chart_test_accs)
plt.ylabel('test accuracy')
plt.subplot(3, 1, 3, sharex=ax)
plt.bar(chart_names, chart_train_times)
plt.ylabel('train time per image (s)')
plt.xlabel('model name')
plt.tight_layout()
plt.savefig(charts_dir + 'Different_models.png')
plt.close()

chart_augmentations = []
chart_test_losses = []
for model_info in model_infos:
    if model_info['name'] == 'ResNet' and model_info['medical_mode'] == 'SA' and model_info['num_classes'] == 3 and model_info['transfer'] == 'retrain':
        chart_augmentations.append(model_info['augmentation'])
        chart_test_losses.append(model_info['test_loss'])

chart_augmentations = sorted_by(chart_augmentations, chart_test_losses)
chart_test_losses = sorted(chart_test_losses)

plt.bar(chart_augmentations, chart_test_losses)
plt.xlabel('Augmentation mode')
plt.ylabel('test loss')
plt.savefig(charts_dir + 'Augmentation_mode.png')
plt.close()

chart_transfers = []
chart_test_losses = []
for model_info in model_infos:
    if model_info['name'] == 'ResNet' and model_info['medical_mode'] == 'SA' and model_info['num_classes'] == 3 and model_info['augmentation'] == 'norm':
        chart_transfers.append(model_info['transfer'])
        chart_test_losses.append(model_info['test_loss'])

chart_transfers = sorted_by(chart_transfers, chart_test_losses)
chart_test_losses = sorted(chart_test_losses)

plt.bar(chart_transfers, chart_test_losses)
plt.xlabel('Transfer learning mode')
plt.ylabel('test loss')
plt.savefig(charts_dir + 'Transfer_learning.png')
plt.close()

chart_num_classes = []
chart_test_losses = []
chart_confusion_matrices = []
for model_info in model_infos:
    if model_info['name'] == 'ResNet' and model_info['medical_mode'] == 'SA' and model_info['augmentation'] == 'norm' and model_info['transfer'] == 'retrain':
        chart_num_classes.append(model_info['num_classes'])
        chart_test_losses.append(model_info['test_loss'])
        chart_confusion_matrices.append(model_info['confusion_matrix'])

chart_num_classes = [str(num_class) for num_class in sorted_by(chart_num_classes, chart_test_losses)]
chart_test_losses = sorted(chart_test_losses)

plt.bar(chart_num_classes, chart_test_losses)
plt.xlabel('Number of classes')
plt.ylabel('test loss')
plt.savefig(charts_dir + 'Number_of_classes.png')
plt.close()

labels_array = [
    ['Normal', 'LVH'],
    ['Normal', 'HCM', 'Other LVH'],
    ['Normal', 'HCM', 'Other LVH', 'Sports']
]
for labels, chart_confusion_matrix in zip(labels_array, chart_confusion_matrices):
    matshow_with_values(chart_confusion_matrix, labels, labels)
    plt.xlabel('Prediction')
    plt.ylabel('Reality')
    plt.savefig(charts_dir + 'Confusion_matrix_{}.png'.format(len(labels)))
    plt.close()
