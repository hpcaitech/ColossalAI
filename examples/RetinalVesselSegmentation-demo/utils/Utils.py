import os.path as osp
import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from PIL import Image
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from skimage import measure, draw
import torch
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def postprocessing(prediction, threshold=0.75, dataset='G'):
    if dataset[0] == 'D':
        # prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        # for i in range(5):
        #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        prediction = torch.sigmoid(prediction).data.cpu().numpy()

        # disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        # cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.erosion(disc_mask, morphology.diamond(3))  # return 0,1
        # cup_mask = morphology.erosion(cup_mask, morphology.diamond(3))  # return 0,1

        prediction_copy = np.copy(prediction)
        prediction_copy = (prediction_copy > threshold)  # return binary mask
        prediction_copy = prediction_copy.astype(np.uint8)
        disc_mask = prediction_copy[1]
        cup_mask = prediction_copy[0]
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        # selem = disk(6)
        # disc_mask = morphology.closing(disc_mask, selem)
        # cup_mask = morphology.closing(cup_mask, selem)
        # print(sum(disc_mask))


        return prediction_copy


def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0))

    _pred_cup[:, :, 0] = prediction[0]
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:,:,0] = mask[0]
    _mask[:,:,1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1)
    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)):
        stack_image[i * img_shape[0] : (i + 1) * img_shape[0], :, : ] = img[i]
    imsave(name, stack_image)





def save_per_img(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp"):
    path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0]+'.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    disc_map = prob_map[0]
    cup_map = prob_map[1]
    size = disc_map.shape
    disc_map[:, 0] = np.zeros(size[0])
    disc_map[:, size[1] - 1] = np.zeros(size[0])
    disc_map[0, :] = np.zeros(size[1])
    disc_map[size[0] - 1, :] = np.zeros(size[1])
    size = cup_map.shape
    cup_map[:, 0] = np.zeros(size[0])
    cup_map[:, size[1] - 1] = np.zeros(size[0])
    cup_map[0, :] = np.zeros(size[1])
    cup_map[size[0] - 1, :] = np.zeros(size[1])

    # disc_mask = (disc_map > 0.75) # return binary mask
    # cup_mask = (cup_map > 0.75)
    # disc_mask = disc_mask.astype(np.uint8)
    # cup_mask = cup_mask.astype(np.uint8)


    contours_disc = measure.find_contours(disc_map, 0.5)
    contours_cup = measure.find_contours(cup_map, 0.5)


    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    disc_mask = get_largest_fillhole(gt[0].numpy()).astype(np.uint8)  # return 0,1
    cup_mask = get_largest_fillhole(gt[1].numpy()).astype(np.uint8)

    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)
    red = [255, 0, 0]
    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red


    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    patch_image.save(path1)