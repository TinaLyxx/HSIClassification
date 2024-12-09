import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist

from datapipes import build_model
from utils.util_train import *
from datapipes.data_load import load_data, sample_gt
from config import get_config


def parse_option():
    parser = argparse.ArgumentParser(
        'Model training and evaluation script', add_help=False)
    parser.add_argument('--cfg',
                        type=str,
                        required=True,
                        metavar="FILE",
                        help='path to config file')

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
    parser.add_argument(
        '--output',
        default='output',
        type=str,
        metavar='PATH',
        help=
        'root of output folder, the full path is <output>/<model_name>/<tag> (default: output)'
    )
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=0,
                           name=f"{config.MODEL.NAME}")
    
    dataset = config.DATA.DATASET
    test_size = config.DATA.TEST_SIZE
    train_size = config.DATA.TRAIN_SIZE
    sample_mode = config.DATA.SAMPLE_MODE

    image, label,_,_= load_data(dataset, config)
    train_label, test_label = sample_gt(label, 1-test_size, sample_mode)

    model = build_model(config)
    model = model.to(device)

    best_path = os.path.join(config.OUTPUT, 'ckpt_epoch_best.pth')

    config.defrost()
    config.MODEL.RESUME = best_path
    config.EVAL_MODE = True
    config.freeze()

    max_accuracy = load_checkpoint(config, model, optimizer=None,
                                        lr_scheduler=None, logger=logger)
    logger.info(f'Load the best model, max accuracy is {max_accuracy:.2f}%')

    probs_map, gt, palette, label_values = test(model, config)
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
