from skimage.io import imread,imsave
from skimage.transform import resize
from glob import glob
import numpy as np
from tqdm import tqdm

imagelist = glob("/home/lqyu/workspace/skin/data/ISIC-2017_Training_Data/*.jpg")
total_rgb = np.zeros(3)
for image_path in tqdm(imagelist):
    id = image_path.split('/')[-1].split('.')[0]
    gt_path = "/home/lqyu/workspace/skin/data/ISIC-2017_Training_Part1_GroundTruth/" + id + "_segmentation.png"
    image = imread(image_path)
    gt = imread(gt_path)

    image = resize(image, [513, 513], order=3)
    gt = resize(gt, [513, 513], order= 0)
    total_rgb += np.mean(image,axis=(0,1))

    imsave('/home/lqyu/workspace/skin/data/ISIC-2017_Training_Part1_resized/'+id+'_image.png', image)
    imsave('/home/lqyu/workspace/skin/data/ISIC-2017_Training_Part1_resized/'+id+'_segmentation.png', gt)

print(total_rgb/len(imagelist))