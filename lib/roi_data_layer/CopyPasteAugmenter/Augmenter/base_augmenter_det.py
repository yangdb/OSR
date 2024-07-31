import copy
import os
import cv2
import random
import numpy as np
# from Augmenter \
from . import utils_aug
import time
from model.utils.config import cfg
import numpy as np
from model.utils.augs import seq

np.random.seed(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)  ##
os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)

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

class BaseAugmenter_det(object):
    """
    Parent class for all object types in the image that can be augmented
    """

    def __init__(self, image, label=None, class_id=None, placement_id=None, horizon_line=None,
                 max_height=None, max_iou=0.4, padding=10, min_px=10, sigma=0, gt_boxes=None):
        """
        Constructor

        image: image to be augmented
        label: semantic label to be modified
        class_id: BGR value of object to be copied into the image
        placement_id: possible locations for the object to be placed
        horizon_line: location of the horizon for scaling accurately
        max_height: size of the object if it were copied in an area closest to the camera
        max_iou: maximum overlap allowed between objects of same class
        padding: padding applied around roi for optimal blurring
        min_px: number of pixels tall the scaled object should be to consider it a valid copy paste
        sigma: increase/decrease the value to decrease/increase the scaling ratio
        """

        self.called = 0
        self.counter = 0
        self.limits = None

        self.sigma = sigma
        self.max_iou = max_iou
        self.padding = padding
        self.min_px = min_px

        self.rows, self.cols, _ = image.shape

        self.image = image.copy()
        # self.label = label.copy()

        # self.class_id = class_id
        # self.fake_class_id = [i if i == 255 else i + 1 for i in class_id]

        self.placement_id = placement_id
        self.horizon_line = horizon_line
        self.max_height = max_height

        if self.max_height is None:
            self.max_height = self.rows * 0.8

        if placement_id is not None and label is not None:
            self.row_value, self.col_value = utils_aug.threshold(image, label, placement_id)

        else:
            self.row_value, self.col_value = np.mgrid[0:len(range(self.rows)), 0:len(range(self.cols))]
            self.row_value, self.col_value = self.row_value.ravel(), self.col_value.ravel()

        if self.horizon_line is not None:
            self.col_value = self.col_value[self.row_value - self.horizon_line > 0]
            self.row_value = self.row_value[self.row_value - self.horizon_line > 0]

            # Initialize scaling triangle
            #           pt1
            #           .
            #     pt2 .   . pt3
            # pt1 = main_triangle_side = (horizon_line, cols / 2)
            # pt2 = (rows, 0)

            self.main_triangle_side = np.sqrt(np.power(self.horizon_line - self.rows, 2) + np.power(self.cols / 2, 2))
            self.slope = float(self.horizon_line - self.rows) / (self.cols / 2)
            self.y_intercept = self.rows

        self.copy_row_value = self.row_value
        self.copy_col_value = self.col_value

        # self.class_placement = utils.get_class_pos(self.label, self.class_id)
        self.class_placement = gt_boxes.tolist()

    def set_limit(self, limit):
        """
        Filters the placement array to constrain the number of
        augmented pixels per image.

        limit = (lower_percent, higher_percent)
                 percentage of the total image height requested
        """
        assert self.horizon_line is not None, "Can't call set_limit without setting a horizon line!"

        self.limits = limit

        self.col_value = self.copy_col_value
        self.row_value = self.copy_row_value

        min_scaled_class_height, max_scaled_class_height = np.array(limit) * self.rows

        min_ratio = float(min_scaled_class_height) / self.max_height
        max_ratio = float(max_scaled_class_height) / self.max_height

        min_cur_triangle_side = min_ratio * (self.main_triangle_side + self.sigma)
        max_cur_triangle_side = max_ratio * (self.main_triangle_side + self.sigma)

        y_min = (min_cur_triangle_side * (self.rows - self.horizon_line) /
                 self.main_triangle_side + self.horizon_line)

        y_max = (max_cur_triangle_side * (self.rows - self.horizon_line) /
                 self.main_triangle_side + self.horizon_line)

        self.col_value = self.col_value[np.logical_and(self.row_value > y_min, self.row_value < y_max)]
        self.row_value = self.row_value[np.logical_and(self.row_value > y_min, self.row_value < y_max)]

    def scale(self, x, y, class_img):
        """
        Scales the object according to user inputs for copying

        x: x co-ord of selected point to copy to (col)
        y: y co-ord of selected point to copy to (row)
        class_img: object image to copy
        """

        # Modify sigma if you want to further reduce or increase the ratio
        # Random scaling of object class if placement_id is None

        if self.horizon_line is not None:
            x_intersect = (y - self.y_intercept) / self.slope
            cur_triangle_side = np.sqrt(np.power(self.horizon_line - y, 2) + np.power(self.cols / 2 - x_intersect, 2))
            ratio = cur_triangle_side / (self.main_triangle_side + self.sigma)

        else:
            ratio = random.random()# ydb  random.uniform(0.5, 1.5)

        class_height, class_width, _ = class_img.shape

        init_scale = float(self.max_height) / class_height

        scaled_class_width = int(class_width * init_scale * ratio)#ydb int(class_width*ratio)
        scaled_class_height = int(self.max_height * ratio)#ydb int(class_height*ratio)

        return scaled_class_width, scaled_class_height

    def create_roi(self, x, y, class_img, extra_class_id=0, flag=1, cord_list=[], cls='', mask_gt=None, mix=False):
        """
        Creates the required roi for the object and copies it into the image

        x: x co-ord of selected point to copy to (col)
        y: y co-ord of selected point to copy to (row)
        class_img: object image to copy
        extra_class_id: BGR value if copying multiple objects in
        flag: enables poisson blending
        """

        height, width, _ = class_img.shape
        roi_x_start = x - width // 2
        roi_x_end = x + 1 + width // 2

        x1, y1, x2, y2 = roi_x_start, y - height, roi_x_end, y

        roi = self.image[y1:y2, x1:x2]
        # roi_label = self.label[y1:y2, x1:x2]

        # Padding around the roi for blurring the edges of the class image properly
        pad = self.padding

        pad_roi = self.image[y1 - pad:y2 + pad, x1 - pad:x2 + pad]
        pad_class_img = np.uint8(np.zeros((height + 2 * pad, width + 2 * pad, 3)))
        pad_class_img[pad:pad + height, pad:pad + width] = class_img

        if mask_gt is not None:
            mask_gt_pad= np.uint8(np.zeros((height + 2 * pad, width + 2 * pad)))
            mask_gt_pad[pad:pad + height, pad:pad + width] = mask_gt
            mask_gt = mask_gt_pad

        if roi.shape == (height, width, 3) and pad_class_img.shape == pad_roi.shape:
            for a1, b1, a2, b2, _ in self.class_placement: ##gt boxes
                iou, iot, iotgt = utils_aug.get_iou([x1, y1, x2, y2], [a1, b1, a2, b2])

                # Control the max amount of overlap allowed
                if iou > self.max_iou :
                    return 1
                if iou > 0 and ((x2-x1)*(y2-y1)>(a2-a1)*(b2-b1)):#iotgt>0.3 or
                    return 1
                # if iot >0.9:
                #     return 1


            self.class_placement.append([x1-5, y1-5, x2+5, y2+5, cls])#0
            # if extra_class_id == 0:
            #     roi_label[np.where(class_img[:, :, 0] != 0)] = self.fake_class_id
            # else:
            #     roi_label[np.where(class_img[:, :, 0] != 0)] = extra_class_id


            roi = utils_aug.blend(pad_roi, pad_class_img, flag=0, mask_gt=mask_gt)
            if mix:
                self.image[y1 - pad:y2 + pad, x1 - pad:x2 + pad] = roi #0.5*self.image[y1 - pad:y2 + pad, x1 - pad:x2 + pad] + 0.5*roi
            else:
                self.image[y1 - pad:y2 + pad, x1 - pad:x2 + pad] = roi

            utils_aug.smooth_edges(pad_roi, pad_class_img, mask_gt=mask_gt)
            cord_list.append([x1-5, y1-5, x2+5, y2+5, cls])
            return 0

        else:
            return 1


    def place_class_list(self, num_class=1, path_list=[], cls_list=None, mix=False, path_list_1=None, split_new=None, kg=None):
        # if path_list_1 is not None:
        #
        # else:
        #     split_new = None
        """
        Copy the required amount of objects into the image

        num_class: number of objects to be copied per image
        path: path to the folder containing object images extracted
        """

        self.called += 1

        updated_img = self.image.copy()
        # updated_lbl = self.label.copy()
        cord_list=[]
        s=time.time()
        while num_class != 0 and len(self.row_value):
            e=time.time()
            # print(e-s)
            if e-s>2:
                print('rest paste:', num_class)
                break
            all_class_imgs = path_list
            rand_copy_name = random.choice(all_class_imgs)
            # print(rand_copy_name)
            class_img = cv2.imread(rand_copy_name)
            class_img = cv2.cvtColor(class_img, cv2.COLOR_BGR2RGB) ###ydb
            # if path_list_1 is not None:
            #     all_class_imgs_1 = path_list_1
            #     rand_copy_name_1 = random.choice(all_class_imgs_1)
            #     # print(rand_copy_name)
            #     class_img_1 = cv2.imread(rand_copy_name_1)
            #     class_img_1 = cv2.cvtColor(class_img_1, cv2.COLOR_BGR2RGB)  ###ydb
            if mix:
                old_cls = cls_list.index(rand_copy_name.split('/')[-2])

                all_class_imgs_bak = copy.deepcopy(all_class_imgs)
                if path_list_1 is not None:
                    all_class_imgs_1 = path_list_1
                    if split_new is not None:
                        all_class_imgs_1 = split_new[np.argmin(kg[len(cls_list)-len(split_new):,old_cls])+(len(cls_list)-len(split_new))]
                    all_class_imgs_bak = copy.deepcopy(all_class_imgs_1)
                else:
                    all_class_imgs_bak.remove(rand_copy_name)
                rand_copy_name_mix = random.choice(all_class_imgs_bak)
                new_cls = float(cls_list.index(rand_copy_name.split('/')[-2]))
                class_img_mix = cv2.imread(rand_copy_name_mix)
                class_img_mix = cv2.cvtColor(class_img_mix, cv2.COLOR_BGR2RGB)  ###ydb
                cfg.TRAIN.mix_lam = np.random.beta(1.0, 1.0)#np.random.beta(20.0, 20.0)#
                class_img = cfg.TRAIN.mix_lam*cv2.resize(class_img,(max(class_img.shape[1], class_img_mix.shape[1]), max(class_img.shape[0], class_img_mix.shape[0]))) + (1-cfg.TRAIN.mix_lam)*cv2.resize(class_img_mix,(max(class_img.shape[1], class_img_mix.shape[1]), max(class_img.shape[0], class_img_mix.shape[0])))
            mask_gt_path = os.path.join('/'.join(rand_copy_name.split('/')[:-3])+'/mask_'+rand_copy_name.split('/')[-3]+'/'+rand_copy_name.split('/')[-2]+'/'+rand_copy_name.split('/')[-1].split('.')[0]+'.png')## mask ydb
            # print('mask gt', mask_gt_path)
            if os.path.exists(mask_gt_path) and cfg.TRAIN.MASK:
                # print('mask gt',mask_gt_path)
                mask_gt = cv2.imread(mask_gt_path,cv2.IMREAD_GRAYSCALE)
                _,mask_gt = cv2.threshold(mask_gt,0,255,cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                mask_gt = cv2.dilate(mask_gt, kernel)
                _, mask_gt = cv2.threshold(mask_gt, 0, 255, cv2.THRESH_BINARY)
            else:
                mask_gt = None
            if random.randint(0,1):  ### ydb
                class_img = cv2.flip(class_img, 1)  ### ydb
                if mask_gt is not None:
                    mask_gt = cv2.flip(mask_gt, 1)  ### ydb
                # class_img = seq(image=class_img)




            class_height, class_width, _ = class_img.shape

            index = random.randint(0, len(self.row_value) - 1)
            x, y = self.col_value[index], self.row_value[index]

            self.row_value = np.delete(self.row_value, index)
            self.col_value = np.delete(self.col_value, index)

            scaled_class_width, scaled_class_height = self.scale(x, y, class_img)

            # Should be at least min_px tall, change accordingly
            if scaled_class_height < self.min_px:
                continue

            # Width needs to be odd for equal splitting about mid point
            scaled_class_width -= 1 if scaled_class_width % 2 == 0 else 0

            scaled_class_img = cv2.resize(class_img, (scaled_class_width, scaled_class_height),
                                          interpolation=cv2.INTER_CUBIC)
            if mask_gt is not None:
                mask_gt = cv2.resize(mask_gt, (scaled_class_width, scaled_class_height),
                                              interpolation=cv2.INTER_CUBIC)

            class_err_code = self.place_extra_class(x, y, scaled_class_img, cord_list=cord_list, cls=float(cls_list.index(rand_copy_name.split('/')[-2])+1), mask_gt=mask_gt, mix=mix)

            if class_err_code == 1:
                self.image = updated_img.copy()
                # self.label = updated_lbl.copy()
                continue

            else:
                updated_img = self.image.copy()
                # updated_lbl = self.label.copy()
                num_class -= 1
                self.counter = 1
        # e=time.time()
        # print('ydb time',e-s)
        if self.limits is not None and len(self.copy_row_value) and num_class != 0:
            diff = self.limits[1] - self.limits[0]
            lower_limit = round(self.limits[0] - diff, 1)
            upper_limit = self.limits[0]

            if lower_limit < 0:
                lower_limit = 0.0

            if upper_limit != 0:
                self.set_limit((lower_limit, upper_limit))
                self.place_class_list(num_class, path_list)
        if mix and cord_list:
            cord_list = np.array(cord_list)
            cord_list[:,-1] = -2
            cord_list = cord_list.tolist()
        return self.image, cord_list#, self.label


    def place_extra_class(self, x, y, scaled_class_img, cord_list=[], cls='', mask_gt=None, mix=False):
        """
        Function to be overloaded
        """

        class_err_code = self.create_roi(x, y, scaled_class_img, flag=1, cord_list=cord_list, cls=cls, mask_gt=mask_gt, mix=mix) ### ydb flag=0
        if class_err_code:
            return 1

        return 0
