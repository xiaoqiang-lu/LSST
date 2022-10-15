import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms


def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask

def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def bflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def Rotate(img, mask, p=0.5):  # [-30, 30]
    if random.random() < p:
        v = random.randint(-30, 30)
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask

def Rotate_90(img, mask, p=0.5):
    if random.random() < p:
        v = 90
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask

def Rotate_180(img, mask, p=0.5):
    if random.random() < p:
        v = 180
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask

def Rotate_270(img, mask, p=0.5):
    if random.random() < p:
        v = 270
        return img.rotate(v), mask.rotate(v)
    else:
        return img, mask

def shearX(img, mask, p=0.5):
    if random.random() < p:
        v = random.randint(-30, 30) / 100
        return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0)), mask.transform(mask.size, Image.AFFINE, (1, v, 0, 0, 1, 0))
    else:
        return img, mask

def translateX(img, mask, p=0.5):
    if random.random() < p:
        v = random.randint(-45, 45) / 100
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0)), mask.transform(mask.size, Image.AFFINE, (1, 0, v, 0, 1, 0))
    else:
        return img, mask

def normalize(img, mask=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def resize(img, mask, base_size, ratio_range):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def edge_enhence(img, p=0.5):
    if random.random() < p:
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    return img

def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask


def color_transformation(img):
    if random.random() < 0.8:
        img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
    img = edge_enhence(img, p=0.5)
    img = transforms.RandomGrayscale(p=0.2)(img)
    img = blur(img, p=0.5)

    return img

def geometric_transformation(img, mask):
    img, mask = hflip(img, mask, p=0.5)
    # img, mask = bflip(img, mask, p=0.1)
    # img, mask = Rotate(img, mask, p=0.5)
    img, mask = Rotate_90(img, mask, p=0.5)
    img, mask = Rotate_180(img, mask, p=0.5)
    img, mask = Rotate_270(img, mask, p=0.5)
    # img, mask = shearX(img, mask, p=0.5)
    # img, mask = translateX(img, mask, p=0.5)

    return img, mask

