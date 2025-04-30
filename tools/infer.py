import sys
import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from PIL import Image

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from stereo.utils.disp_color import disp_to_color
from stereo.datasets.dataset_template import build_transform_by_cfg
import time

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for eval')
    # data
    parser.add_argument('--left_img_path', type=str, default=None)
    parser.add_argument('--right_img_path', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--savename', type=str, default=None)

    args = parser.parse_args()
    args.cfg_file = "cfgs/efficientstereo/efficientstereo2_kitti.yaml"
    folder = "/home/rispro-sils/ADAS_Kedaireka/Dataset/Manually_gathered_17_07_24/data_img"
    args.left_img_path = os.path.join(folder, "20241210-182042-852_frame.png")
    args.right_img_path = os.path.join(folder, "20241210-182042-924_frame2.png")
    args.pretrained_model = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/output/KittiDataset/EfficientStereo2/efficientstereo2_kitti/default/ckpt/checkpoint_epoch_49.pth"
    args.savename = "output.png"
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    
    args.run_mode = 'infer'
    return args, cfgs

@torch.no_grad()
def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    seed = 0 if not args.dist_mode else dist.get_rank()
    common_utils.set_random_seed(seed=seed)

    # log
    logger = common_utils.create_logger(log_file=None, rank=local_rank)

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # model
    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model

    # data
    transform_config = cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING
    transform = build_transform_by_cfg(transform_config)
    left_img = np.array(Image.open(args.left_img_path).convert('RGB'), dtype=np.float32)
    right_img = np.array(Image.open(args.right_img_path).convert('RGB'), dtype=np.float32)
    sample = {
        'left': left_img,
        'right': right_img,
    }
    sample = transform(sample)
    sample['left'] = sample['left'].unsqueeze(0)
    sample['right'] = sample['right'].unsqueeze(0)

    model.eval()
    for k, v in sample.items():
        sample[k] = v.to(local_rank) if torch.is_tensor(v) else v

    with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
        # Warm-up
        model_pred = model(sample)

        # Measure inference time for 100 iterations
        start_time = time.time()
        for _ in range(100):
            model_pred = model(sample)
        end_time = time.time()

        # Calculate average inference time
        avg_inference_time = (end_time - start_time) / 100

        # Log model name and image size
        logger.info(f"Model Name: {cfgs.MODEL.NAME}")
        logger.info(f"Image Size: {left_img.shape[1]}x{left_img.shape[0]}")
        logger.info(f"Average Inference Time: {avg_inference_time:.6f} seconds")

    disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
    img_color = disp_to_color(disp_pred, max_disp=192)
    img_color = img_color.astype('uint8')
    img_color = Image.fromarray(img_color)
    img_color.save(args.savename)


if __name__ == '__main__':
    main()
