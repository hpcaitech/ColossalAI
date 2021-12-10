import torch
import numpy as np

bce = torch.nn.BCEWithLogitsLoss(reduction='none')

def _upscan(f):
    for i, fi in enumerate(f):
        if fi == np.inf: continue
        for j in range(1,i+1):
            x = fi+j*j
            if f[i-j] < x: break
            f[i-j] = x

def distance_transform(bitmap):
    f = np.where(bitmap, 0.0, np.inf)
    for ibatch in range(f.shape[0]):
        for i in range(f.shape[1]):
            _upscan(f[ibatch, i, :])
            _upscan(f[ibatch, i,::-1])
        for i in range(f.shape[2]):
            _upscan(f[ibatch, :,i])
            _upscan(f[ibatch, ::-1,i])
            np.sqrt(f[ibatch], f[ibatch])
    return f

def WatershedCrossEntropy(input, target):

    # Distance Transform
    discmap = target.data.cpu()[:, 0, :, :]
    cupmap = target.data.cpu()[:, 1, :, :]
    disc_DT = distance_transform(discmap)
    cup_DT = distance_transform(cupmap)
    disc_DT = torch.from_numpy(disc_DT).float()
    cup_DT = torch.from_numpy(cup_DT).float()

    disc_DT = discmap * (1.0 - disc_DT/torch.max(disc_DT)) + 1.0
    cup_DT = cupmap * (1.0 - cup_DT/torch.max(cup_DT)) + 1.0

    disc_DT = disc_DT.cuda()
    cup_DT = cup_DT.cuda()

    CEloss = bce(input, target)

    return torch.mean(disc_DT* CEloss[:, 0 , :, :]+
                      cup_DT*CEloss[:, 1 , :, :])

def cross_entropy2d(input, target, weight=None, size_average=False):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)

    log_p = torch.nn.functional.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = torch.nn.functional.nll_loss(log_p, target, weight=weight)
    loss = loss.float()
    if size_average:
        temp = mask.data.sum().float()
        loss = loss / temp
    return loss


def dice_coefficient_numpy(binary_segmentation, binary_gt_label):
    '''
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground truth
    '''

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

    # return it
    return dice_value


def dice_coeff(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    pred = torch.sigmoid(pred)
    pred = pred.data.cpu()
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    return dice_coefficient_numpy(pred, target)

def dice_coeff_2label(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    target = target.data.cpu()
    # pred = torch.sigmoid(pred)
    # pred = pred.data.cpu()
    # pred[pred > 0.75] = 1
    # pred[pred <= 0.75] = 0
    # print target.shape
    # print pred.shape
    if len(pred.shape) == 3:
        return dice_coefficient_numpy(pred[0, ...], target[0, ...]), dice_coefficient_numpy(pred[1, ...], target[1, ...])
    else:
        dice_cup = []
        dice_disc = []
        for i in range(pred.shape[0]):
            cup, disc = dice_coefficient_numpy(pred[i, 0, ...], target[i, 0, ...]), dice_coefficient_numpy(pred[i, 1, ...], target[i, 1, ...])
            dice_cup.append(cup)
            dice_disc.append(disc)
    return sum(dice_cup) / len(dice_cup), sum(dice_disc) / len(dice_disc)


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    return 1 - dice_coeff(pred, target)


def DiceLoss(input, target):
    '''
    in tensor fomate
    :param input:
    :param target:
    :return:
    '''
    smooth = 1.
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def Balanced_DiceLoss(input, target):
    '''

    :param input:
    :param target:
    :return:
    '''
    input = torch.sigmoid(input)

    return 0.5 * (DiceLoss(input[:, 0, ...], target[:, 0, ...]) + DiceLoss(input[:, 1, ...], target[:, 1, ...]))