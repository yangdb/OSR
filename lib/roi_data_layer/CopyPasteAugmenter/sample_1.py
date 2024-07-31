import cv2
# import dataset_reader as dr
import os
# from labels import *
from .Augmenter import utils
from .Augmenter import base_augmenter_det as ba
import random
import numpy as np
from model.utils.config import cfg

np.random.seed(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)  ##
os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)

voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor']
coco_classes = [None for i in range(80)]
if cfg.TRAIN.SET=='coco':
    from pycocotools.coco import COCO
    ann_filepath = cfg.TRAIN.ROOT_P+ 'GFR-IL/faster-rcnn/data/coco/annotations/instances_train2017.json'
    coco = COCO(annotation_file=ann_filepath)
    all_classes_ann = coco.cats  # ann['categories']#['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    # 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    #             'sheep', 'sofa', 'train', 'tvmonitor']
    # all_classes = [None for i in range(len(all_classes_ann))]
    for idx, cc in enumerate(sorted(all_classes_ann)):
        coco_classes[idx] = all_classes_ann[cc]['name']


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
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
if __name__=='__mian__':
    img_path = "/mnt/disk7/ydb/Object_train/GFR-IL/faster-rcnn/data/VOCdevkit2007-1/VOC2007/JPEGImages/"#"Path to images"
    lbl_path = "/mnt/disk7/ydb/Object_train/GFR-IL/faster-rcnn/data/VOCdevkit2007-1/VOC2007/Annotations/"#"Path to labels"
    aug_class_path=[]
    getFile("/mnt/disk7/ydb/Object_train/GFR-IL/faster-rcnn/data/2007_crops/19/",aug_class_path)#"Path to object images extracted by class_extractor.py"
    save_path = '/mnt/disk7/ydb/Object_train/GFR-IL/faster-rcnn/data/2007_crops/aug/'

    # data = dr.DatasetReader(image_path=img_path, label_path=lbl_path)

    # Modify the class_id and placement_id for other classes, refer labels.py
    # class_id = names2labels["person"].color
    # placement_id = (names2labels["sidewalk"].color, names2labels["terrain"].color,
    #                 names2labels["parking"].color, names2labels["road"].color,
    #                 names2labels["ground"].color)

    # Modify horizon_line and max_height as per your requirement
    # rows, cols, _ = data[0][0].shape
    # horizon_line = int(rows * 0.4)
    # max_height = int(rows * 0.8)

    with open(img_path+'../ImageSets/Main/trainval.txt') as f:
        ObjBndBoxSet = {}
        for line in f.readlines():

            tree = ET.ElementTree(file=lbl_path + '' + line.strip() + '.xml')
            root = tree.getroot()
            ObjectSet = root.findall('object')
            gt_boxes = []
            for Object in ObjectSet:
                ObjName = Object.find('name').text
                BndBox = Object.find('bndbox')
                x1 = int(BndBox.find('xmin').text)
                y1 = int(BndBox.find('ymin').text)
                x2 = int(BndBox.find('xmax').text)
                y2 = int(BndBox.find('ymax').text)
                gt_boxes.append([x1, y1, x2, y2, ObjName])
                # if x2-x1<80 or y2-y1<80:
                #     continue
                # if ObjName in ObjBndBoxSet:
                #     ObjBndBoxSet[ObjName].append([line.strip(),x1,y1,x2,y2])
                # else:
                #     ObjBndBoxSet[ObjName] = [[line.strip(),x1,y1,x2,y2]]

            image = np.array(Image.open(img_path + '' + line.strip() + '.jpg'))
            rows, cols, _ = image.shape
            horizon_line = int(rows * 0.4)
            max_height = int(rows * 0.8)
            aug = ba.BaseAugmenter_det(image, label=1, class_id=1, max_iou=0.1, min_px=50,
                                   horizon_line=horizon_line, max_height=max_height, gt_boxes=gt_boxes)

            # aug.set_limit((0.6, 0.8))
            img = aug.place_class_list(num_class=1, path_list=aug_class_path)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            Image.fromarray(img).save(save_path+line.strip()+'.jpg')



    # for i in range(len(data)):
    #     image, label = data[i]
    #
    #     aug = ba.BaseAugmenter(image, label, class_id, placement_id=placement_id,
    #                            horizon_line=horizon_line, max_height=max_height)
    #
    #     # aug.set_limit((0.6, 0.8))
    #     img, lbl = aug.place_class_list(2, aug_class_path)
    #
    #     # cv2.imshow("image", cv2.resize(img, (1024, 512)))
    #     # cv2.imshow("label", cv2.resize(lbl, (1024, 512)))
    #     # cv2.imshow("placement", cv2.resize(utils.viz_placement(aug), (1024, 512)))
    #     # cv2.imshow("scaling_triangle", cv2.resize(utils.viz_scaling_triangle(aug), (1024, 512)))
    #     # cv2.waitKey(0)

def copy_paste_aug(img_array, crop_path, gt_boxes, save=False, gray=False, gray_base_gt=0, mix=False):
    # print('ydb copy paste procedure......')
    rows, cols, _ = img_array.shape
    horizon_line = int(rows * 0.4)
    max_height = int(rows * 0.8)#0.8
    if gray:
        max_height = int(rows)
    min_px = min(int(rows/5),int(cols/5))#,60
    aug_class_path = []
    aug_class_path_1 = []
    cls_list=[]
    # print('ydb', crop_path)
    if isinstance(crop_path, list):
        for crp in crop_path:
            getFile(crp, aug_class_path, cls_list=cls_list)
    else:
        getFile(crop_path, aug_class_path, cls_list=cls_list)

    if isinstance(cfg.TRAIN.MIX_PATH_1, list):
        for crp_1 in cfg.TRAIN.MIX_PATH_1:
            getFile(crp_1, aug_class_path_1, cls_list=cls_list)
    else:
        getFile(cfg.TRAIN.MIX_PATH_1, aug_class_path_1, cls_list=cls_list)

    if cfg.TRAIN.SET=='voc':
        cls_list = voc_classes
        cls_list.sort()
    elif cfg.TRAIN.SET=='coco':
        cls_list = coco_classes
    else:
        print('ydb unknown datasets of crops !!!!!!')
    # print('ydb number of augmented samples:', len(aug_class_path))
    aug = ba.BaseAugmenter_det(img_array, max_iou=0.1, min_px=min_px,
                               horizon_line=None, max_height=max_height, gt_boxes=gt_boxes)
    if gray:
        img, cord_list = aug.place_class_list(num_class=cfg.TRAIN.GRAY_AUG_NUM_SAMPLES, path_list=aug_class_path,
                                              cls_list=cls_list)
    elif gray_base_gt>0:
        if mix:
            if cfg.TRAIN.kg:
                split_new = {}
                for pl1 in aug_class_path_1:
                    if cls_list.index(pl1.split('/')[-2]) in split_new:
                        split_new[cls_list.index(pl1.split('/')[-2])].append(pl1)
                    else:
                        split_new[cls_list.index(pl1.split('/')[-2])] = [pl1]
                if os.path.exists('kg.npy'):
                    kg = np.load('kg.npy')
                else:
                    kg = None
            else:
                kg = None
                split_new = None
            img, cord_list = aug.place_class_list(num_class=gray_base_gt, path_list=aug_class_path,
                                                  cls_list=cls_list, mix=True, path_list_1=aug_class_path_1, split_new=split_new, kg=kg)
        else:
            img, cord_list = aug.place_class_list(num_class=gray_base_gt, path_list=aug_class_path,
                                              cls_list=cls_list)
    else:
        img, cord_list = aug.place_class_list(num_class=cfg.TRAIN.NUM_SAMPLES, path_list=aug_class_path, cls_list=cls_list)

    # save=False
    if save:
        save_path = cfg.TRAIN.ROOT_P+'CPR-IOD/faster-rcnn/save_cp/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_name = save_path + str(rows)+str(random.randint(1,1000)) + '.jpg'
        draw = True#False
        if draw:
            img_c= img.copy()
            img_d = Image.fromarray(img_c)
            a = ImageDraw.ImageDraw(img_d)  # 用a来表示
            # 在边界框的两点（左上角、右下角）画矩形，无填充，边框红色，边框像素为5
            for box in gt_boxes:
                x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
                a.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red')
                a.text((x1, y1), str(box[4]), fill=(255, 255, 0))
            for box in cord_list:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                a.rectangle(((x1, y1), (x2, y2)), fill=None, outline='green')
                a.text((x1, y1), str(box[4]), fill=(255,255, 0))
            img_d.save(save_name)
        else:
            img = Image.fromarray(img)
            img.save(save_name)
        img = np.array(img)
        print('ydb save', save_name)
    return img, cord_list