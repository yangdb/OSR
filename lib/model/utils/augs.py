import copy

import torch
import numpy as np
import random
import imgaug.augmenters as iaa
import imgaug as ia
from model.utils.config import cfg
import os
import cv2

ia.seed(cfg.RNG_SEED)
# iaa.reseed
np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)  ##
os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
torch.backends.cudnn.deterministic = True

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL
            # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                       # convert images into their superpixel representation
                       # iaa.OneOf([
                       #     iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                       #     iaa.AverageBlur(k=(2, 7)),
                       #     # blur image using local means with kernel sizes between 2 and 7
                       #     iaa.MedianBlur(k=(3, 11)),
                       #     # blur image using local medians with kernel sizes between 2 and 7
                       # ]),
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 2.0)),  # blur images with a sigma between 0 and 3.0
                           # iaa.AverageBlur(k=(2, 7)),
                           # # blur image using local means with kernel sizes between 2 and 7
                           # iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       # iaa.SimplexNoiseAlpha(iaa.OneOf([
                       #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                       #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                       # ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),
                           # randomly remove up to 10% of the pixels
                           iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               second=iaa.LinearContrast((0.5, 2.0))
                           )
                       ]),
                       iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                       # improve or worsen the contrast
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                       # move pixels locally around (with random strengths)
                       sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                       # sometimes move parts of the image around
                       sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True, random_state=cfg.RNG_SEED
                   )
    ],
    random_order=True, random_state=cfg.RNG_SEED
)
# Pipeline:
# (1) Crop images from each side by 1-16px, do not resize the results
#     images back to the input size. Keep them at the cropped size.
# (2) Horizontally flip 50% of the images.
# (3) Blur images using a gaussian kernel with sigma between 0.0 and 3.0.
seq = iaa.Sequential([
    iaa.Crop(px=(1, 10), keep_size=True, random_state=cfg.RNG_SEED),#16
    iaa.Fliplr(0.5, random_state=cfg.RNG_SEED),
    # iaa.GaussianBlur(sigma=(0, 3.0))
])

superpixel = iaa.Sequential([
    iaa.Superpixels(p_replace=1.0, n_segments=20, random_state=cfg.RNG_SEED),
    iaa.GaussianBlur(sigma=(3.0, 5.0), random_state=cfg.RNG_SEED)
])

def getFile(path, fileList, cls_list=None):
    fList = os.listdir(path) #将指定目录内的文件以列表格式输出
    # os.chdir(path)
    docuPath = path #获取根路径
    # fileList = []
    for f in fList: #对目录内的文件进行遍历
        if os.path.isdir(os.path.join(docuPath,f)): #判断文件类型是否为目录
            if cls_list is not None:
                cls_list.append(f)
            getFile(os.path.join(docuPath,f), fileList) #若是目录，递归运行此函数，继续进行遍历。
        else:
            fl = os.path.join(docuPath,f) #若不是目录，则结合文件名和根路径获得文件的绝对路径
            fileList.append(fl)
    #print(fileList)
    # return fileList

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

import torch.nn.functional as F
def mixup_criterion_kl(pred, y_a, y_b, lam):
    y_a_oh = F.one_hot(y_a, num_classes = pred.shape[1]).float()
    y_b_oh = F.one_hot(y_b, num_classes = pred.shape[1]).float()
    y_a_oh = y_a_oh * torch.Tensor([lam]).cuda()
    y_b_oh = y_b_oh * torch.Tensor([(1-lam)]).cuda()
    y = y_a_oh + y_b_oh
    loss = F.kl_div(F.log_softmax(pred,1), y)
    return loss

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data_old(x, y, x_old, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x_old[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_data_img(im_data, gt_boxes, croplist_mix, class_mix, alpha=1.0, use_cuda=True):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)  ##
    os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    '''Returns mixed inputs, pairs of targets, and lambda'''
    mix_im = copy.deepcopy(im_data)
    mix_gt_boxes = torch.zeros(gt_boxes.shape[0],6).cuda()#
    mix_gt_boxes[:,:5] = gt_boxes
    if alpha > 0:
        # lam = np.random.beta(alpha, alpha)0.5 #
        lam = np.random.randint(low=40,high=61)/100
    else:
        lam = 1

    for idx, gtb in enumerate(gt_boxes):
        gtb = gtb.cpu().long()
        cidx = random.randint(0, len(croplist_mix)-1)
        crop = cv2.imread(croplist_mix[cidx])
        crop_mix = cv2.resize(crop, ((gtb[2]-gtb[0]).item(), (gtb[3]-gtb[1]).item()))
        mix_im[gtb[1].item():gtb[3].item(), gtb[0].item():gtb[2].item()] = lam * mix_im[gtb[1].item():gtb[3].item(), gtb[0].item():gtb[2].item()] + (1-lam)*torch.from_numpy(crop_mix).float().cuda()
        mix_gt_boxes[idx][5] = class_mix[cidx]
    return mix_im, mix_gt_boxes, lam


def mixup_data_bg(im, crops, alpha=1.0):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)  ##
    os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    coords = []
    '''Returns mixed inputs, pairs of targets, and lambda'''
    for b in range(0,3):
        i1 = random.randint(0, len(crops)-1)
        i2 = random.randint(0, len(crops)-1)
        i3 = random.randint(0, len(crops)-1)
        crop1 = cv2.imread(crops[i1])[:, :, ::-1]
        crop2 = cv2.imread(crops[i2])[:, :, ::-1]
        crop3 = cv2.imread(crops[i3])[:, :, ::-1]
        crop1_re = cv2.resize(crop1, (min(crop1.shape[1],crop2.shape[1],crop3.shape[1], im.shape[1]),min(crop1.shape[0],crop2.shape[0],crop3.shape[0], im.shape[0])))
        crop2_re = cv2.resize(crop2, (min(crop1.shape[1],crop2.shape[1],crop3.shape[1], im.shape[1]),min(crop1.shape[0],crop2.shape[0],crop3.shape[0], im.shape[0])))
        crop3_re = cv2.resize(crop3, (min(crop1.shape[1],crop2.shape[1],crop3.shape[1], im.shape[1]),min(crop1.shape[0],crop2.shape[0],crop3.shape[0], im.shape[0])))
        crop = (1/3)*crop1_re+(1/3)*crop2_re+(1/3)*crop3_re#(1/3)*

        if (im.shape[0]-crop.shape[0])<0:
            print('<0', im.shape, crop.shape)
        x = random.randint(0, max(im.shape[0]-crop.shape[0],0))
        y = random.randint(0, max(im.shape[1]-crop.shape[1],0))
        # print(crops[i1], x, y)
        # im[x:x+crop.shape[0], y:y+crop.shape[1]] = 0.8*im[x:x+crop.shape[0], y:y+crop.shape[1]]+0.2*crop
        im[x:x+crop.shape[0], y:y+crop.shape[1]] = crop
        coords.append([ y, x, y+crop.shape[1], x+crop.shape[0],  -2])
    # for b in range(0,3):
    #     i1 = random.randint(0, len(crops)-1)
    #     i2 = random.randint(0, len(crops)-1)
    #     # i3 = random.randint(0, len(crops)-1)
    #     crop1 = cv2.imread(crops[i1])[:, :, ::-1]
    #     crop2 = cv2.imread(crops[i2])[:, :, ::-1]
    #     # crop3 = cv2.imread(crops[i3])
    #     crop1_re = cv2.resize(crop1, (min(crop1.shape[1],crop2.shape[1], im.shape[1]),min(crop1.shape[0],crop2.shape[0], im.shape[0])))
    #     crop2_re = cv2.resize(crop2, (min(crop1.shape[1],crop2.shape[1], im.shape[1]),min(crop1.shape[0],crop2.shape[0], im.shape[0])))
    #     # crop3_re = cv2.resize(crop3, (min(crop1.shape[1],crop2.shape[1],crop3.shape[1], im.shape[1]),min(crop1.shape[0],crop2.shape[0],crop3.shape[0], im.shape[0])))
    #     crop = 0.5*crop1_re+0.5*crop2_re#+crop3_re#(1/3)*
    #
    #     if (im.shape[0]-crop.shape[0])<0:
    #         print('<0', im.shape, crop.shape)
    #     x = random.randint(0, max(im.shape[0]-crop.shape[0],0))
    #     y = random.randint(0, max(im.shape[1]-crop.shape[1],0))
    #     im[x:x+crop.shape[0], y:y+crop.shape[1]] = crop
    return coords

croplist_mix = []
if isinstance(cfg.TRAIN.CROP_PATH, list):
    crop_path_train = copy.deepcopy(cfg.TRAIN.CROP_PATH)
    # crop_path_train.append(cfg.TRAIN.NEW_PATH)
else:
    crop_path_train = [cfg.TRAIN.CROP_PATH]#[cfg.TRAIN.CROP_PATH, cfg.TRAIN.NEW_PATH]
print('crops list: {}'.format(crop_path_train))
print('cfg.TRAIN.CROP_PATH', cfg.TRAIN.CROP_PATH, 'crop_path_train', crop_path_train)
if isinstance(crop_path_train, list):
    for crp in crop_path_train:
        getFile(crp, croplist_mix)
else:
    getFile(crop_path_train, croplist_mix)