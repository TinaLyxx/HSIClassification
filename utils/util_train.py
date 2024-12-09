import os
import math
import sys
import logging
import functools
import itertools

import torch
import numpy as np
import torch.distributed as dist
from collections import OrderedDict
from timm.utils import get_state_dict
from termcolor import colored
from sklearn.metrics import confusion_matrix
from PIL import Image



def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f'All checkpoints founded in {output_dir}: {checkpoints}')
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints],
            key=os.path.getmtime)
        print(f'The latest checkpoint founded: {latest_checkpoint}')
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def save_checkpoint(config,
                    epoch,
                    model,
                    max_accuracy,
                    optimizer,
                    lr_scheduler,
                    logger,
                    model_ema=None,
                    max_accuracy_ema=None,
                    ema_decay=None,
                    model_ems=None,
                    max_accuracy_ems=None,
                    ems_model_num=None,
                    best=None):

    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': max_accuracy,
        'epoch': epoch,
        'config': config
    }
    if model_ema is not None:
        save_state['model_ema'] = get_state_dict(model_ema)
    if max_accuracy_ema is not None:
        save_state['max_accuracy_ema'] = max_accuracy_ema
    if ema_decay is not None:
        save_state['ema_decay'] = ema_decay
    if model_ems is not None:
        save_state['model_ems'] = get_state_dict(model_ems)
    if max_accuracy_ems is not None:
        save_state['max_accuracy_ems'] = max_accuracy_ems
    if ems_model_num is not None:
        save_state['ems_model_num'] = ems_model_num
    if best is None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{best}.pth')
    logger.info(f'{save_path} saving......')
    torch.save(save_state, save_path)
    logger.info(f'{save_path} saved !!!')

    # if dist.get_rank() == 0 and isinstance(epoch, int):
    #     to_del = epoch - config.SAVE_CKPT_NUM * config.SAVE_FREQ
    #     old_ckpt = os.path.join(config.OUTPUT, f'ckpt_epoch_{to_del}.pth')
    #     if os.path.exists(old_ckpt):
    #         os.remove(old_ckpt)

    if isinstance(epoch, int):
        to_del = epoch - config.SAVE_CKPT_NUM * config.SAVE_FREQ
        old_ckpt = os.path.join(config.OUTPUT, f'ckpt_epoch_{to_del}.pth')
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)
    

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
        colored('(%(filename)s %(lineno)d)', 'yellow') + \
        ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(
        output_dir, f'log_rank{dist_rank}.txt'),
                                       mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


class MyAverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, max_len=-1):
        self.val_list = []
        self.count = []
        self.max_len = max_len
        self.val = 0
        self.avg = 0
        self.var = 0

    def update(self, val):
        self.val = val
        self.avg = 0
        self.var = 0
        if not math.isnan(val) and not math.isinf(val):
            self.val_list.append(val)
        if self.max_len > 0 and len(self.val_list) > self.max_len:
            self.val_list = self.val_list[-self.max_len:]
        if len(self.val_list) > 0:
            self.avg = np.mean(np.array(self.val_list))
            self.var = np.std(np.array(self.val_list))


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(
        f'==============> Resuming form {config.MODEL.RESUME}....................'
    )

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    print('resuming model')

    model_checkpoint = checkpoint['model']

    msg = model.load_state_dict(model_checkpoint, strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if optimizer is not None:
            print('resuming optimizer')
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('resume optimizer failed')
        if lr_scheduler is not None:
            print('resuming lr_scheduler')
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(
            f"=> loaded successfully {config.MODEL.RESUME} (epoch {checkpoint['epoch']})"
        )
    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()

    return max_accuracy



def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h



def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    print(f"target list: {target}")
    print(f"prediction list: {prediction}")
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results



def convert_to_color(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def save_predictions(pred, gt=None, caption=""):
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if gt is None:
        image = Image.fromarray(pred)
        path = os.path.join(results_dir, caption)
        image.save(path)
    else:
        pred = Image.fromarray(pred)
        pred_path = os.path.join(results_dir, caption)
        pred.save(pred_path)

        gt = Image.fromarray(gt)
        gt_path = os.path.join(results_dir, f"gt_{caption}")
        gt.save(gt_path)


def show_results(results, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Aggregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]

            
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    # Calculate and display average accuracy per class from the confusion matrix
    if label_values is not None:
        class_sums = np.sum(cm, axis=1)
        valid_classes = class_sums != 0  # Identify classes with at least one prediction
        class_accuracies = np.diag(cm) / np.where(class_sums > 0, class_sums, np.nan)
        average_accuracy = np.nanmean(class_accuracies)  # Safely compute mean, ignoring NaN values
        text += "Average Accuracy: {:.05f}%\n".format(average_accuracy * 100)
    text += "---\n"

    if agregated:
        text += ("Overall Accuracy: {:.05f} +- {:.05f}\n".format(np.mean(accuracies),
                                                                 np.std(accuracies)))
    else:
        text += "Overall Accuracy : {:.05f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.05f} +- {:.05f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.05f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.05f} +- {:.05f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.05f}\n".format(kappa)

    print(text)

