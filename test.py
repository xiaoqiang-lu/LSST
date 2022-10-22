from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from dataset_name import *
from utils import count_params, meanIOU, color_map

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


NUM_CLASSES = {'GID-15': 15, 'iSAID': 15, 'DFC22': 12, 'MER': 9, 'MSL': 9, 'Vaihingen': 5}
DATASET = 'DFC22'     # ['GID-15', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'DFC22]
WEIGHTS = 'Your local path'

GID15_DATASET_PATH = 'Your local path'
iSAID_DATASET_PATH = 'Your local path'
DFC22_DATASET_PATH = 'Your local path'
MER_DATASET_PATH = 'Your local path'
MSL_DATASET_PATH = 'Your local path'
Vaihingen_DATASET_PATH = 'Your local path'

def parse_args():
    parser = argparse.ArgumentParser(description='LSST Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['GID-15', 'iSAID', 'DFC22', 'MER', 'MSL', 'Vaihingen'],
                        default=DATASET)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv2')
    parser.add_argument('--save-path', type=str, default='test_results/' + WEIGHTS.split('/')[-1].replace('.pth', ''))

    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)

    model = DeepLabV2(args.backbone, NUM_CLASSES[args.dataset])
    model.load_state_dict(torch.load(WEIGHTS))
    model = DataParallel(model).cuda()

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=8,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    eval(model, valloader, args)

def eval(model, valloader, args):

    model.eval()
    tbar = tqdm(valloader)

    data_list = []

    with torch.no_grad():
        for img, mask, _ in tbar:
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            data_list.append([mask.numpy().flatten(), pred.flatten()])
        filename = os.path.join(args.save_path, 'result.txt')
        get_iou(data_list, NUM_CLASSES[args.dataset], filename, DATASET)


def get_iou(data_list, class_num, save_path=None, dataset_name=None):
    from multiprocessing import Pool
    from metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    if dataset_name == 'MSL' or dataset_name == 'MER':
        classes, _ = MARS()
    elif dataset_name == 'iSAID':
        classes, _ = iSAID()
    elif dataset_name == 'GID-15':
        classes, _ = GID15()
    elif dataset_name == 'Vaihingen':
        classes, _ = Vaihingen()
    elif dataset_name == 'DFC22':
        classes, _ = DFC22()

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i] * 100))

    print('meanIOU {:.2f}'.format(aveJ * 100) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i] * 100) + '\n')
            f.write('meanIOU {:.2f}'.format(aveJ * 100) + '\n')


if __name__ == '__main__':
    args = parse_args()

    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    print(args)
    main(args)
