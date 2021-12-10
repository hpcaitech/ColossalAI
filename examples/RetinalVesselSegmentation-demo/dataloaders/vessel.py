import os
import random
import copy

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
from imageio import imread
from glob import glob


class RetinalVesselSegmentation(Dataset):
    """
    Retinal vessel segmentation dataset
    including 4 domain datasets
    one for test others for training
    """

    def __init__(self,
                 base_dir,
                 phase='train',
                 splitid=[1, 2, 3],
                 transform=None,
                 state='train',
                 normalize='whitening',
                 ):
        # super().__init__()
        self.state = state
        self._base_dir = base_dir
        self.normalize = normalize
        self.image_list = []
        self.phase = phase
        self.image_pool = {'CHASEDB1':[], 'DRIVE':[], 'HRF':[], 'STARE':[]}
        self.label_pool = {'CHASEDB1':[], 'DRIVE':[], 'HRF':[], 'STARE':[]}
        self.roi_pool = {'CHASEDB1':[], 'DRIVE':[], 'HRF':[], 'STARE':[]}
        self.img_name_pool = {'CHASEDB1':[], 'DRIVE':[], 'HRF':[], 'STARE':[]}
        self.postfix = [['jpg', 'png', 'png'], ['tif', 'tif', 'gif'], ['jpg', 'tif', 'tif'], ['ppm', 'ppm', 'png']]
        self.splitid = splitid
        SEED = 1023
        random.seed(SEED)
        domain_dirs = os.listdir(self._base_dir)
        domain_dirs.sort()
        for i in range(len(domain_dirs)):
            domain_dirs[i] = os.path.join(self._base_dir, domain_dirs[i])

        for id in splitid:
            if id == 3:
                self._image_dir = domain_dirs[id]
            else:
                self._image_dir = os.path.join(domain_dirs[id], phase)
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))
            
            postfix_im, postfix_gt, postfix_roi = self.postfix[id][0], self.postfix[id][1], self.postfix[id][2]
            imagePath = glob(os.path.join(self._image_dir, 'image', '*.{}'.format(postfix_im)))
            gtPath = glob(os.path.join(self._image_dir, 'mask', '*.{}'.format(postfix_gt)))
            roiPath = glob(os.path.join(self._image_dir, 'roi', '*.{}'.format(postfix_roi)))
            imagePath.sort()
            gtPath.sort()
            roiPath.sort()

            if id == 3 and phase != 'test':
                imagePath, gtPath, roiPath = imagePath[:10], gtPath[:10], roiPath[:10]
            elif id == 3 and phase == 'test':
                imagePath, gtPath, roiPath = imagePath[10:], gtPath[10:], roiPath[10:]

            for idx, image_path in enumerate(imagePath):
                self.image_list.append({'image': image_path, 'label': gtPath[idx], 'roi': roiPath[idx]})
            
        self.transforms = transform
        self._read_img_into_memory()
        iterFlag = True
        while iterFlag:
            iterFlag = False
            for key in self.image_pool:
                if len(self.image_pool[key]) < 1:
                    print('key ' + key + ' has no data')
                    del self.image_pool[key]
                    del self.label_pool[key]
                    del self.roi_pool[key]
                    del self.img_name_pool[key]
                    iterFlag = True
                    break
                
        # Display stats
        for key in self.image_pool:
            print('{} images in {}'.format(len(self.image_pool[key]), key))
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))
    def set_thread_id(self, thread_id):
        self.thread_id = thread_id
    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key]) > max:
                 max = len(self.image_pool[key])
        if self.phase != 'test':
            max *= 3
        return max

    def __getitem__(self, index):
        if self.phase != 'test' or True:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _roi = self.roi_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'roi': _roi, 'img_name': _img_name, 'dc': domain_code}
                if self.transforms is not None:
                    anco_sample = self.transforms(anco_sample)
                sample = anco_sample

        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = self.image_list[index]['image']
            if 'DRIVE' in basename:          
                Flag = 'DRIVE'
            elif 'CHASEDB1' in basename:
                Flag = 'CHASEDB1'
            elif 'STARE' in basename:
                Flag = 'STARE'
            elif 'HRF' in basename:
                Flag = 'HRF'

            if Flag == 'STARE':
                im = imread(self.image_list[index]['image'])
                gt = imread(self.image_list[index]['label'])
                self.image_pool[Flag].append(Image.fromarray(im).convert('RGB').resize((512, 512), Image.LANCZOS))
                self.label_pool[Flag].append(Image.fromarray(gt).convert('L').resize((512, 512)))
            else:
                self.image_pool[Flag].append(
                    Image.open(self.image_list[index]['image']).convert('RGB').resize((512, 512), Image.LANCZOS))
                self.label_pool[Flag].append(
                    Image.open(self.image_list[index]['label']).convert('L').resize((512, 512)))

            self.roi_pool[Flag].append(
                Image.open(self.image_list[index]['roi']).convert('L').resize((512, 512)))

            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name) 

        print('img_num: ' + str(img_num))


if __name__ == '__main__':
    RetinalVesselSegmentation('./dataset/RVS/')
