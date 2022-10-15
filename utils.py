import numpy as np
from PIL import Image


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == 'GID-15':
        cmap[0] = np.array([200, 0, 0])
        cmap[1] = np.array([250, 0, 150])
        cmap[2] = np.array([200, 150, 150])
        cmap[3] = np.array([250, 150, 150])
        cmap[4] = np.array([0, 200, 0])
        cmap[5] = np.array([150, 250, 0])
        cmap[6] = np.array([150, 200, 150])
        cmap[7] = np.array([200, 0, 200])
        cmap[8] = np.array([150, 0, 250])
        cmap[9] = np.array([150, 150, 250])
        cmap[10] = np.array([250, 200, 0])
        cmap[11] = np.array([200, 200, 0])
        cmap[12] = np.array([0, 0, 200])
        cmap[13] = np.array([0, 150, 200])
        cmap[14] = np.array([0, 200, 250])

    elif dataset == 'iSAID':
        cmap[0] = np.array([0, 0, 63])
        cmap[1] = np.array([0, 63, 63])
        cmap[2] = np.array([0, 63, 0])
        cmap[3] = np.array([0, 63, 127])
        cmap[4] = np.array([0, 63, 191])
        cmap[5] = np.array([0, 63, 255])
        cmap[6] = np.array([0, 127, 63])
        cmap[7] = np.array([0, 127, 127])
        cmap[8] = np.array([0, 0, 127])
        cmap[9] = np.array([0, 0, 191])
        cmap[10] = np.array([0, 0, 255])
        cmap[11] = np.array([0, 191, 127])
        cmap[12] = np.array([0, 127, 191])
        cmap[13] = np.array([0, 127, 255])
        cmap[14] = np.array([0, 100, 155])

    elif dataset == 'MSL' or dataset == 'MER':
        cmap[0] = np.array([128, 0, 0])
        cmap[1] = np.array([0, 128, 0])
        cmap[2] = np.array([128, 128, 0])
        cmap[3] = np.array([0, 0, 128])
        cmap[4] = np.array([128, 0, 128])
        cmap[5] = np.array([0, 128, 128])
        cmap[6] = np.array([128, 128, 128])
        cmap[7] = np.array([64, 0, 0])
        cmap[8] = np.array([192, 0, 0])

    elif dataset == 'Vaihingen' or dataset == 'Potsdam':
        cmap[0] = np.array([255, 255, 255])
        cmap[1] = np.array([0, 0, 255])
        cmap[2] = np.array([0, 255, 255])
        cmap[3] = np.array([0, 255, 0])
        cmap[4] = np.array([255, 255, 0])

    elif dataset == 'MF_DFC22':
        cmap[0] = np.array([219, 95, 87])
        cmap[1] = np.array([219, 151, 87])
        cmap[2] = np.array([219, 208, 87])
        cmap[3] = np.array([173, 219, 87])
        cmap[4] = np.array([117, 219, 87])
        cmap[5] = np.array([123, 196, 123])
        cmap[6] = np.array([88, 177, 88])
        cmap[7] = np.array([0, 128, 0])
        cmap[8] = np.array([88, 176, 167])
        cmap[9] = np.array([153, 93, 19])
        cmap[10] = np.array([87, 155, 219])
        cmap[11] = np.array([0, 98, 255])


    return cmap
