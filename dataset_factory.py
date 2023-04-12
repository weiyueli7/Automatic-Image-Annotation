################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

import csv, os
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as transforms
from vocab import load_vocab
from coco_dataset import CocoDataset, collate_fn


# Builds your datasets here based on the configuration.
# You are not required to modify this code but you are allowed to.
def get_datasets(config_data):
    images_root_dir = config_data['dataset']['images_root_dir']
    root_train = os.path.join(images_root_dir, 'train')
    root_val = os.path.join(images_root_dir, 'val')
    root_test = os.path.join(images_root_dir, 'test')

    train_ids_file_path = config_data['dataset']['training_ids_file_path']
    val_ids_file_path = config_data['dataset']['validation_ids_file_path']
    test_ids_file_path = config_data['dataset']['test_ids_file_path']

    train_annotation_file = config_data['dataset']['training_annotation_file_path']
    test_annotation_file = config_data['dataset']['test_annotation_file_path']
    coco = COCO(train_annotation_file)
    coco_test = COCO(test_annotation_file)

    vocab_threshold = config_data['dataset']['vocabulary_threshold']
    vocabulary = load_vocab(train_annotation_file, vocab_threshold)

    train_data_loader = get_coco_dataloader(train_ids_file_path, root_train, train_annotation_file, coco, vocabulary,
                                            config_data)
    val_data_loader = get_coco_dataloader(val_ids_file_path, root_val, train_annotation_file, coco, vocabulary,
                                          config_data)
    test_data_loader = get_coco_dataloader(test_ids_file_path, root_test, test_annotation_file, coco_test, vocabulary,
                                           config_data)

    return coco_test, vocabulary, train_data_loader, val_data_loader, test_data_loader


def get_coco_dataloader(img_ids_file_path, imgs_root_dir, annotation_file_path, coco_obj, vocabulary, config_data):
    with open(img_ids_file_path, 'r') as f:
        reader = csv.reader(f)
        img_ids = list(reader)

    img_ids = [int(i) for i in img_ids[0]]

    ann_ids = [coco_obj.imgToAnns[img_ids[i]][j]['id'] for i in range(0, len(img_ids)) for j in
               range(0, len(coco_obj.imgToAnns[img_ids[i]]))]

    transform_list = []
    if config_data['transform']['center_crop']:
        transform_list.append(transforms.CenterCrop((config_data['transform']['center_crop'])))
    if config_data['transform']['horizontal_flip']:
        transform_list.append(transforms.RandomHorizontalFlip())
    if config_data['transform']['rotation']:
        transform_list.append(transforms.RandomRotation(config_data['transform']['rotation']))
    if config_data['transform']['normalize']:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform_list)

    dataset = CocoDataset(root=imgs_root_dir,
                          json=annotation_file_path,
                          ids=ann_ids,
                          vocab=vocabulary,
                          img_size=config_data['dataset']['img_size'],
                          transform=transform)
                          
    return DataLoader(dataset=dataset,
                      batch_size=config_data['dataset']['batch_size'],
                      shuffle=True,
                      num_workers=config_data['dataset']['num_workers'],
                      collate_fn=collate_fn,
                      pin_memory=True)
