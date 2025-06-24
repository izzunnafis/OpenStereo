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
from thop import profile
import matplotlib.pyplot as plt

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
    args.cfg_file = "cfgs/efficientstereo/lse4.yaml"
    # args.cfg_file = "cfgs/efficientstereo/lightstereo_m_kitti.yaml"
    # folder = "/home/rispro-sils/ADAS_Kedaireka/Dataset/Manually_gathered_17_07_24/data_img"
    folder = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/data/KITTI15/kitti15/testing"
    # args.left_img_path = os.path.join(folder, "20241210-182042-852_frame.png")
    # args.right_img_path = os.path.join(folder, "20241210-182042-924_frame2.png")
    args.left_img_path = os.path.join(folder, "image_2/000010_11.png")
    args.right_img_path = os.path.join(folder, "image_3/000010_11.png")
    parent = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/output/KittiDataset/LightStereo4"
    you = "lse4/v15_1"
    child = "default/ckpt/checkpoint_epoch_499.pth"
    args.pretrained_model = os.path.join(parent, you, child)
    args.savename = "output.png"
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    
    args.run_mode = 'infer'
    return args, cfgs

def color_map_tensorboard(disp, max_disp=192):
    """
    Convert disparity map to color map for TensorBoard visualization.
    """
    cm = plt.get_cmap('plasma')
    disp_tmp = 255.0 * disp / max_disp
    disp_tmp = cm(disp_tmp.astype('uint8'))  # Apply colormap

    disp_tmp = disp_tmp[:, :, :3]*255  # Change to (C, H, W) format
    return disp_tmp


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

    # Check if the model is on GPU or CPU
    device = next(model.parameters()).device
    logger.info(f"Model is on device: {device}")

    # Check if the images are on GPU or CPU
    for key, value in sample.items():
        if torch.is_tensor(value):
            logger.info(f"Sample '{key}' is on device: {value.device}")

    with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
        # Warm-up
        for _ in range(100):
            model_pred = model(sample)

        # Measure inference time for 200 iterations
        start_time = time.time()
        for _ in range(200):
            # st_time = time.time()
            with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
                model_pred = model(sample)
            torch.cuda.synchronize()
            # print(f"Iteration time: {time.time() - st_time:.6f} seconds")
        end_time = time.time()

        # Calculate average inference time
        avg_inference_time = (end_time - start_time) / 200 * 1000

        # Log model name and image size
        logger.info(f"Model Name: {cfgs.MODEL.NAME}")
        logger.info(f"Image Size: {left_img.shape[1]}x{left_img.shape[0]}")
        logger.info(f"Average Inference Time: {avg_inference_time:.6f} ms")
        # Calculate MACs and FLOPs
        macs, params = profile(model, inputs=(sample,))
        logger.info(f"MACs: {macs / 1e9:.3f} G")
        logger.info(f"Parameters: {params / 1e6:.3f} M")

    disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
    img_color = color_map_tensorboard(disp_pred, max_disp=192)
    img_color = img_color.astype('uint8')
    # img_color.save(args.savename)

    filter = model_pred['filter'].squeeze().cpu().numpy()*255
    filter = filter.astype('uint8')
    filter = np.transpose(filter, (1, 2, 0))

    l_img = np.array(Image.open(args.left_img_path).convert('RGB').resize((1248, 384), Image.BILINEAR), dtype=np.uint8)
    l_img = l_img.astype('uint8')

    print("filter shape:", filter.shape)
    print("l_img shape:", l_img.shape)
    print("img_color shape:", img_color.shape)
    img_all = np.concatenate((l_img, img_color, filter), axis=0)

    l_img = Image.fromarray(l_img)
    img_color = Image.fromarray(img_color)
    filter = Image.fromarray(filter)
    img_all = Image.fromarray(img_all)
    filter.save("filter.png")
    img_all.save("output_concat.png")



    # Interpolate feat_l to match the original image size
    import torch.nn.functional as F
    feat_l = model_pred['freq_filter_low']
    original_size = (left_img.shape[0], left_img.shape[1])  # (height, width)
    feat_l = F.interpolate(feat_l, size=original_size, mode='nearest')
    feat_l = feat_l.squeeze().cpu().numpy()  # Remove batch and channel dimensions

    feat_avg = np.mean(feat_l, axis=0)
    feat_avg = (feat_avg - feat_avg.min()) / (feat_avg.max() - feat_avg.min()) * 255
    feat_avg = feat_avg.astype('uint8')
    feat_avg = Image.fromarray(feat_avg)
    feat_avg.save("feat_l_avg.png")
    logger.info("Average feature map saved as feat_l_avg.png")
    
    for i in range(feat_l.shape[0]):
        feat = feat_l[i]
        feat = (feat - feat.min()) / (feat.max() - feat.min()) * 255
        feat = feat.astype('uint8')
        feat = Image.fromarray(feat)
        feat.save(f"feat_l_{i}.png")
        logger.info(f"Feature map {i} saved as feat_l_{i}.png")

    feat_h = model_pred['freq_filter_high']
    original_size = (left_img.shape[0], left_img.shape[1])
    feat_h = F.interpolate(feat_h, size=original_size, mode='nearest')
    feat_h = feat_h.squeeze().cpu().numpy()
    feat_avg = np.mean(feat_h, axis=0)
    feat_avg = (feat_avg - feat_avg.min()) / (feat_avg.max() - feat_avg.min()) * 255
    feat_avg = feat_avg.astype('uint8')
    feat_avg = Image.fromarray(feat_avg)
    feat_avg.save("feat_h_avg.png")
    logger.info("Average feature map saved as feat_h_avg.png")

    for i in range(feat_h.shape[0]):
        feat = feat_h[i]
        feat = (feat - feat.min()) / (feat.max() - feat.min()) * 255
        feat = feat.astype('uint8')
        feat = Image.fromarray(feat)
        feat.save(f"feat_h_{i}.png")
        logger.info(f"Feature map {i} saved as feat_h_{i}.png")



if __name__ == '__main__':
    main()
