# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import tqdm
from easydict import EasyDict

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT
from thop import profile


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # save path
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # dataloader
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    dataset_names = [x.DATASET for x in cfgs.DATA_CONFIG.DATA_INFOS]
    unique_dataset_names = list(set(dataset_names))
    if len(unique_dataset_names) == 1:
        exp_dataset_dir = unique_dataset_names[0]
    else:
        exp_dataset_dir = 'MultiDataset'
    args.exp_group_path = os.path.join(exp_dataset_dir, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'

    args.run_mode = 'eval'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        group_rank = int(os.environ["GROUP_RANK"])
    else:
        local_rank = 0
        global_rank = 0
        group_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    common_utils.set_random_seed(seed=42)

    # savedir
    args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))
    if os.path.exists(args.output_dir) and args.extra_tag != 'debug' and cfgs.MODEL.CKPT == -1:
        raise Exception('There is already an exp with this name')
    if args.dist_mode:
        dist.barrier()
    if global_rank == 0:
        common_utils.backup_source_code(os.path.join(args.output_dir, 'code'))
    if args.dist_mode:
        dist.barrier()



    # logger
    log_file = os.path.join(args.output_dir, 'eval_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), group_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_eval')) if global_rank == 0 else None
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # trainer
    model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer)

    model_trainer.evaluate(current_epoch=0)


if __name__ == '__main__':
    main()
