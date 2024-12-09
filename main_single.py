import os
import time
import random
import argparse
import datetime
import numpy as np
import subprocess
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import optim as optim
from timm.utils import ModelEma, ApexScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from datapipes import build_loader_single, build_model
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from utils.util_train import *
from datapipes.data_load import load_data
from config import get_config


def parse_option():
    parser = argparse.ArgumentParser(
        'Model training and evaluation script', add_help=False)
    parser.add_argument('--cfg',
                        type=str,
                        required=True,
                        metavar="FILE",
                        help='path to config file')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')

    # easy config modification
    parser.add_argument('--batch-size',
                        type=int,
                        help="batch size for single GPU")
    parser.add_argument('--dataset',
                        type=str,
                        help='dataset name',
                        default=None)
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--patch-size',
                        type=int,
                        help='patch size')
    parser.add_argument('--sample-mode',
                        type=str,
                        choices=['fixed','random','disjoint' ],
                        help='fixed: fixed number each class, '
                        'random: have the same ratio of each class'
                        )
    parser.add_argument('--model-type',
                        type=str,
                        choices=['v1', 'v2'],
                        help='different type of model')
    parser.add_argument('--head-type',
                        type=str,
                        choices=['AP','FC','AT'],
                        help='AP: Adaptive Avg Pooling,'
                        'FC: fully connect layer,'
                        'AT: attention head classification')
    # parser.add_argument('--zip',
    #                     action='store_true',
    #                     help='use zipped dataset instead of folder dataset')
    # parser.add_argument(
    #     '--cache-mode',
    #     type=str,
    #     default='part',
    #     choices=['no', 'full', 'part'],
    #     help='no: no cache, '
    #     'full: cache all data, '
    #     'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
    # )
    # parser.add_argument(
    #     '--pretrained',
    #     help=
    #     'pretrained weight from checkpoint, could be imagenet22k pretrained weight'
    # )
    parser.add_argument('--resume', help='resume from checkpoint')
    # parser.add_argument('--accumulation-steps',
    #                     type=int,
    #                     default=1,
    #                     help="gradient accumulation steps")
    parser.add_argument(
        '--use-checkpoint',
        action='store_true',
        help="whether to use gradient checkpointing to save memory")
    # parser.add_argument(
    #     '--amp-opt-level',
    #     type=str,
    #     default='O1',
    #     choices=['O0', 'O1', 'O2'],
    #     help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument(
        '--output',
        default='output',
        type=str,
        metavar='PATH',
        help=
        'root of output folder, the full path is <output>/<model_name>/<tag> (default: output)'
    )
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput',
                        action='store_true',
                        help='Test throughput only')
    parser.add_argument('--save-ckpt-num', default=1, type=int)
    parser.add_argument(
        '--use-zero',
        action='store_true',
        help="whether to use ZeroRedundancyOptimizer (ZeRO) to save memory")

    # distributed training
    parser.add_argument("--local-rank",
                        type=int,
                        # required=True,
                        default=0,
                        # required=False,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        '--train-size',
        type=int,
        help='the number of samples to train'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        help='the number of samples to test'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        help='the class number'
    )

    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config



@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return

    
def main(config):
    # prepare data loaders
    dataset_train, dataset_val, dataset_test, test_label, \
        data_loader_train, data_loader_val, data_loader_test = build_loader_single(config)
    
    # build runner
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model = model.to(device)

    # build optimizer
    optimizer = optim.AdamW(model.parameters(),
                            eps=config.TRAIN.OPTIMIZER.EPS,
                            betas=config.TRAIN.OPTIMIZER.BETAS,
                            lr=config.TRAIN.BASE_LR,
                            weight_decay=config.TRAIN.WEIGHT_DECAY)


    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")


    # build learning rate scheduler
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train)) \
        if not config.EVAL_MODE else None

    # build criterion
    criterion = torch.nn.CrossEntropyLoss()


    max_accuracy = 0.0
    # set auto resume
    if config.MODEL.RESUME == '' and config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume'
            )

     # set resume and pretrain
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer,
                                       lr_scheduler, logger=logger)
        if data_loader_val is not None:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(
                f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
            )
    
    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)

    if config.EVAL_MODE:
        return
    
    # train
    logger.info("Start training")
    logger.info(f"Train on {len(dataset_train)} train images")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        train_one_epoch(config,
                        model,
                        criterion,
                        data_loader_train,
                        optimizer,
                        epoch,
                        lr_scheduler)
        
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config,
                            epoch,
                            model,
                            max_accuracy,
                            optimizer,
                            lr_scheduler,
                            logger=logger)
        if data_loader_val is not None and epoch % config.EVAL_FREQ == 0:
            acc1, acc5, loss = validate(config, data_loader_val, model, epoch)
            logger.info(
                f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%"
            )
            if acc1 > max_accuracy:
                save_checkpoint(config,
                                epoch,
                                model,
                                acc1,
                                optimizer,
                                lr_scheduler,
                                logger=logger,
                                best='best')
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # Test
    logger.info("Start testing")
    best_path = os.path.join(config.OUTPUT, 'ckpt_epoch_best.pth')

    config.defrost()
    config.MODEL.RESUME = best_path
    config.EVAL_MODE = True
    config.freeze()

    max_accuracy = load_checkpoint(config, model, optimizer,
                                    lr_scheduler, logger=logger)
    logger.info(f'Load the best model, max accuracy is {max_accuracy:.2f}%')

    probs_map, gt, palette, label_values = test(model, config)
    print(f"Get test results!!")
    # print(f"probs_map shape {probs_map.shape}")
    num_cls = int(probs_map.shape[2])

    prob_map = np.argmax(probs_map, axis=-1)
    results = metrics(prob_map, test_label, ignored_labels=config.DATA.IGNOR_LABELS, n_classes=num_cls)

    mask = np.zeros(gt.shape, dtype='bool')
    for l in config.DATA.IGNOR_LABELS:
        mask[gt == l] = True
    
    color_pred_map = convert_to_color(prob_map, palette)
    prob_map[mask] = 0
    mask_color_pred_map = convert_to_color(prob_map, palette)

    file_name = config.DATA.DATASET + '.png'
    save_predictions(mask_color_pred_map, color_pred_map, caption=file_name)

    show_results(results, label_values=label_values, agregated=False)

        

def train_one_epoch(config,
                    model,
                    criterion,
                    data_loader,
                    optimizer,
                    epoch,
                    lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    model_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = MyAverageMeter(300)

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        iter_begin_time = time.time()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()

        loss.backward()
        grad_norm = get_grad_norm(model.parameters())
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(epoch * num_steps + idx)

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        model_time.update(time.time() - iter_begin_time)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'model_time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f}/{norm_meter.var:.4f})\t'
                f'mem {memory_used:.0f}MB')
            
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )


@torch.no_grad()
def validate(config, data_loader, model, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                        f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
    if epoch is not None:
        logger.info(
            f'[Epoch:{epoch}] * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}'
        )
    else:
        logger.info(
            f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def test(model, config):
    dataset = config.DATA.DATASET
    patch_size = config.DATA.PATCH_SIZE
    center_pixel = config.DATA.CENTER_PIXEL
    batch_size = config.DATA.BATCH_SIZE
    n_classes = config.MODEL.NUM_CLASSES

    model.eval()
    img, label, palette, label_values = load_data(dataset,config)

    probs_map = np.zeros(img.shape[:2] + (n_classes,))
    kwargs = {
        "step": config.TEST.STRIDE,
        "window_size": (patch_size, patch_size),
    }
    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
            grouper(batch_size, sliding_window(img, **kwargs)),
            total=(iterations),
            desc="Inference on the image",
    ):
        if patch_size == 1:
            data = [b[0][0, 0] for b in batch] 
            data = np.copy(data)
            data = torch.from_numpy(data)
        else:
            data = [b[0] for b in batch] 
            data = np.copy(data)
            data = data.transpose(0, 3, 1, 2)
            data = torch.from_numpy(data)
            data = data.unsqueeze(1)

        indices = [b[1:] for b in batch]
        data = data.cuda(non_blocking=True)
        output = model(data)

        if isinstance(output, tuple):
                output = output[0]

        output = output.to("cpu")
        if patch_size == 1 or center_pixel:
            output = output.numpy()
        else:
            output = np.transpose(output.numpy(), (0, 2, 3, 1))
        for (x, y, w, h), out in zip(indices, output):
            if center_pixel:
                probs_map[x + w // 2, y + h // 2] += out
            else:
                probs_map[x: x + w, y: y + h] += out
        
    return probs_map, label, palette, label_values




if __name__ == '__main__':
    _, config = parse_option()

    seed = config.SEED 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=0,
                           name=f"{config.MODEL.NAME}")
    
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    for run in range(config.N_RUNS):
        main(config)
    