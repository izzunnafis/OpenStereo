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
from stereo.datasets.dataset_utils.readpfm import readpfm
from stereo.datasets.dataset_template import build_transform_by_cfg
from thop import profile
import cv2
import matplotlib.pyplot as plt

def color_map_tensorboard(pred, disp_max=None):
    cm = plt.get_cmap('plasma')

    if disp_max is None:
        disp_max = np.max(pred)
    
    pred = pred/disp_max

    pred = cm(pred)

    pred = (pred[:, :, :3]*255.0).astype('uint8')

    return pred

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for eval')
    # data

    args = parser.parse_args()
    # args.cfg_file = "cfgs/efficientstereo/manual_efs_lite.yaml"
    args.cfg_file = "cfgs/efficientstereo/manual_lss.yaml"
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    args.pretrained_model = "/home/rispro-sils/ADAS_Kedaireka/Perception/OpenStereo/output/fix_ckpt/checkpoint_epoch_499_lss_manual_sf.pth"
    # args.pretrained_model = "/home/beliau/adas/OpenStereo/fix_ckpt/LightStereo-S-SceneFlow.ckpt"
    print("pretrained_model:", args.pretrained_model)
    args.model_name = "lss_manual_sf"

    cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model
    print(cfgs.MODEL.PRETRAINED_MODEL)
    
    args.run_mode = 'infer'
    return args, cfgs


def calculate_depth(disp, object_loc, logger, note=''):
    cam_fx = 349.5325  # Focal length in pixels
    cam_baseline = 0.12  # Baseline in meters
    # Calculate depth (distance)
    if disp > 0:
        distance = cam_fx * cam_baseline / disp
        # Calculate x and y distance from principal point (cx, cy)
        cx = 343.835
        cy = 194.70825
        x_distance = ((object_loc[1] - cx) * cam_baseline) / disp
        y_distance = ((object_loc[0] - cy) * cam_baseline) / disp
        # logger.info(f"[{note}] Distance to coordinate {object_loc}: {distance:.4f} meters")
        # logger.info(f"[{note}] X distance from principal point at {object_loc}: {x_distance:.4f} meters")
        # logger.info(f"[{note}] Y distance from principal point at {object_loc}: {y_distance:.4f} meters")
        
        zx_distance = np.sqrt(distance**2 + x_distance**2)
        # logger.info(f"[{note}] ZX distance (sqrt(depth^2 + x^2)): {zx_distance:.4f} meters")
        zxy_distance = np.sqrt(distance**2 + x_distance**2 + y_distance**2)
        logger.info(f"[{note}] ZXY distance (sqrt(depth^2 + x^2 + y^2)): {zxy_distance:.4f} meters")
    else:
        logger.info(f"[{note}] Disparity at {object_loc} is zero or invalid, cannot compute distance.")


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
    seed = 0
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

    samples = [
        {"name": '_20250709_165706_46_0000', "object_loc": (246, 315), "dist": 10.032},
        {"name": '_20250709_165802_95_0000', "object_loc": (249, 337), "dist": 6.915},
        {"name": '_20250709_165851_43_0000', "object_loc": (263, 330), "dist": 4.9213},
        {"name": '_20250709_170017_63_0000', "object_loc": (225, 304), "dist": 3.519},
        {"name": '_20250709_170055_74_0000', "object_loc": (224, 298), "dist": 4.939},
        {"name": '_20250709_170147_29_0000', "object_loc": (227, 307), "dist": 9.784},
        {"name": '_20250709_170416_63_0000', "object_loc": (235, 290), "dist": 6.304},
        {"name": '_20250709_170510_42_0000', "object_loc": (224, 326), "dist": 2.386},
        {"name": '_20250709_170617_31_0000', "object_loc": (223, 323), "dist": 12.869},
        {"name": '_20250709_170705_07_0000', "object_loc": (240, 309), "dist": 8.235},
        {"name": '_20250709_170731_53_0000', "object_loc": (244, 301), "dist": 5.881},
        {"name": '_20250709_170804_96_0000', "object_loc": (250, 307), "dist": 3.019},
        {"name": '_20250709_170940_27_0000', "object_loc": (259, 351), "dist": 3.452},
        {"name": '_20250709_171011_62_0000', "object_loc": (252, 315), "dist": 4.884},
        {"name": '_20250709_171040_24_0000', "object_loc": (252, 318), "dist": 6.471},
        {"name": '_20250709_171118_22_0000', "object_loc": (244, 330), "dist": 9.880},
        {"name": '_20250709_171332_34_0000', "object_loc": (239, 202), "dist": 8.819},
        {"name": '_20250709_171412_69_0000', "object_loc": (249, 40), "dist": 5.764},
    ]
    root_folder = "/home/rispro-sils/ADAS_Kedaireka/Dataset/manual_lab/data_izzun/take_car"

    # name = '_20250709_171146_31_0007'

    # [LEFT_CAM_VGA]
    # fx=349.5325
    # fy=349.5325
    # cx=343.835
    # cy=194.70825
    # k1=-0.173896
    # k2=0.0276094
    # p1=0.00122109
    # p2=8.15916e-05
    # k3=0


    
    for sample_info in samples:
        name = sample_info["name"]
        object_loc = sample_info["object_loc"]
        dist_gt = sample_info["dist"]

        left_path = os.path.join(root_folder, "left_image" + name + ".png")
        right_path = os.path.join(root_folder, "right_image" + name + ".png")
        disparity_path = os.path.join(root_folder, "disparity_image" + name + ".npy")

        file_name = "manual_" + name

        left_img = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32)
        # disp
        disp_img = np.load(disparity_path).astype(np.float32)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img
        }
        sample = transform(sample)
        sample['left'] = sample['left'].unsqueeze(0)
        sample['right'] = sample['right'].unsqueeze(0)

        model.eval()
        for k, v in sample.items():
            sample[k] = v.to(local_rank) if torch.is_tensor(v) else v


        with torch.amp.autocast('cuda', enabled=cfgs.OPTIMIZATION.AMP):
            model_pred = model(sample)

            # # Log model name and image size
            # logger.info(f"Model Name: {cfgs.MODEL.NAME}")
            # logger.info(f"Image Size: {left_img.shape[1]}x{left_img.shape[0]}")
            # # Calculate MACs and FLOPs
            # macs, params = profile(model, inputs=(sample,))
            # logger.info(f"MACs: {macs / 1e9:.3f} G")
            # logger.info(f"Parameters: {params / 1e6:.3f} M")

        result = model_pred['disp_pred'].squeeze().cpu().numpy().astype(np.float32)
        # Crop the result and ground truth disparity maps to 540x960
        img_height, img_width = left_img.shape[:2]
        trf_img_height, trf_img_width = result.shape[:2]
        result = result[trf_img_height-img_height:, :img_width]

        result_color = color_map_tensorboard(result, disp_max=25)
        result_color = result_color.astype('uint8')

        baseline_color = color_map_tensorboard(disp_img, disp_max=25)
        baseline_color = baseline_color.astype('uint8')

        patch_size = 2
        
        result_crop = result[object_loc[0]-patch_size:object_loc[0]+patch_size, object_loc[1]-patch_size:object_loc[1]+patch_size]
        baseline_crop = disp_img[object_loc[0]-patch_size:object_loc[0]+patch_size, object_loc[1]-patch_size:object_loc[1]+patch_size]
            
        mean_result_disp = np.mean(result_crop)
        mean_baseline_disp = np.mean(baseline_crop)
        logger.info(f"Sample Name: {name}")
        calculate_depth(mean_result_disp, object_loc, logger, note='Mean Result')
        # calculate_depth(mean_baseline_disp, object_loc, logger, note='Mean Baseline')    

        # Draw rectangle on left image
        left_img_rect = left_img.copy()
        cv2.rectangle(left_img_rect, 
                      (object_loc[1]-patch_size, object_loc[0]-patch_size), 
                      (object_loc[1]+patch_size, object_loc[0]+patch_size), 
                      (255, 0, 0), 2)

        # Draw rectangle on result color image
        result_color_rect = result_color.copy()
        cv2.rectangle(result_color_rect, 
                      (object_loc[1]-patch_size, object_loc[0]-patch_size), 
                      (object_loc[1]+patch_size, object_loc[0]+patch_size), 
                      (255, 0, 0), 2)

        # Draw rectangle on baseline (ground truth) color image
        baseline_color_rect = baseline_color.copy()
        cv2.rectangle(baseline_color_rect, 
                      (object_loc[1]-patch_size, object_loc[0]-patch_size), 
                      (object_loc[1]+patch_size, object_loc[0]+patch_size), 
                      (255, 0, 0), 2)
        
        # Save left image, result color image, and ground truth color image with rectangles
        os.makedirs('depth_test', exist_ok=True) 
        Image.fromarray(left_img_rect.astype('uint8')).save(f'depth_test/{file_name}_left_rect.png')
        Image.fromarray(result_color_rect.astype('uint8')).save(f'depth_test/{args.model_name}_{file_name}_disp_pred_color_rect.png')
        Image.fromarray(baseline_color_rect.astype('uint8')).save(f'depth_test/{file_name}_disp_gt_color_rect.png')
    

if __name__ == '__main__':
    main()
