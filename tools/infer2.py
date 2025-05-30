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

import cv2

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
    args.cfg_file = "cfgs/efficientstereo/lse2_rev_test.yaml"
    # args.cfg_file = "cfgs/efficientstereo/lightstereo_m_kitti.yaml"
    # folder = "/home/rispro-sils/ADAS_Kedaireka/Dataset/Manually_gathered_17_07_24/data_img"
    folder = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/data/manual2"
    # args.left_img_path = os.path.join(folder, "20241210-182042-852_frame.png")
    # args.right_img_path = os.path.join(folder, "20241210-182042-924_frame2.png")
    args.left_img_path = os.path.join(folder, "output_images_4/left_image_20250521_111950_10_0156.png")
    args.right_img_path = os.path.join(folder, "output_images_4/right_image_20250521_111950_10_0156.png")
    args.disp_img_path = os.path.join(folder, "output_images_4/disparity_image_20250521_111950_10_0156.npy")
    parent = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/output/KittiDataset/LightStereo2"
    you = "lse2_rev"
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
    disp_img = np.load(args.disp_img_path)
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

    disp_img_color = color_map_tensorboard(disp_img, max_disp=192)
    disp_img_color = disp_img_color.astype('uint8')
    disp_img_color = Image.fromarray(disp_img_color).resize((672, 384), Image.BILINEAR)
    disp_img_color = np.array(disp_img_color, dtype=np.uint8)

    disp_pred = model_pred['disp_pred'].squeeze().cpu().numpy()
    img_color = color_map_tensorboard(disp_pred, max_disp=192)
    img_color = img_color.astype('uint8')
    # img_color.save(args.savename)

    filter = model_pred['filter'].squeeze().cpu().numpy()*255
    filter = filter.astype('uint8')
    filter = np.transpose(filter, (1, 2, 0))

    l_img = np.array(Image.open(args.left_img_path).convert('RGB').resize((672, 384), Image.BILINEAR), dtype=np.uint8)
    l_img = l_img.astype('uint8')

    coord_x = 100
    coord_y = 250    

    # Get depth and disparity at the specified coordinate
    disp_value = disp_img[coord_y, coord_x]
    depth_value = 0.12*335.95/disp_value if disp_value > 0 else 0  # Example depth calculation, adjust as needed

    disp_pred = disp_pred[coord_y, coord_x]
    depth_pred = 0.12*335.95/disp_pred if disp_pred > 0 else 0  # Example depth calculation, adjust as needed
    
    # Show the values on the image
    text1 = f"ZED Depth: {depth_value:.2f}m  ZED Disp: {disp_value:.0f}px"
    text2 = f"Pred Depth: {depth_pred:.2f}m  Pred Disp: {disp_pred:.0f}px"
    cv2.circle(l_img, (coord_x, coord_y), 5, (0, 0, 255), -1)
    cv2.circle(disp_img_color, (coord_x, coord_y), 5, (0, 0, 255), -1)
    cv2.putText(disp_img_color, text1, (coord_x + 10, coord_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.circle(img_color, (coord_x, coord_y), 5, (0, 0, 255), -1)
    cv2.putText(img_color, text2, (coord_x + 10, coord_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   

    print("filter shape:", filter.shape)
    print("l_img shape:", l_img.shape)
    print("img_color shape:", img_color.shape)
    print("disp_img_color shape:", disp_img_color.shape)
    img_all = np.concatenate((l_img, img_color, disp_img_color), axis=0)

    l_img = Image.fromarray(l_img)
    img_color = Image.fromarray(img_color)
    filter = Image.fromarray(filter)
    img_all = Image.fromarray(img_all)
    # filter.save("filter.png")
    img_all.save("output_concat3_2.png")




if __name__ == '__main__':
    main()
