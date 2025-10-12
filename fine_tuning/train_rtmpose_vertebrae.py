"""
Training script for RTMPose with custom vertebrae keypoints.

This script fine-tunes a pretrained RTMPose model on your custom vertebrae dataset.

Usage:
    # Single GPU training
    python train_rtmpose_vertebrae.py

    # Multi-GPU training (4 GPUs)
    python -m torch.distributed.launch --nproc_per_node=4 train_rtmpose_vertebrae.py --launcher pytorch

    # Resume training from checkpoint
    python train_rtmpose_vertebrae.py --resume work_dirs/rtmpose_vertebrae/latest.pth
"""

import argparse
import os
import os.path as osp
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Train RTMPose with vertebrae keypoints')
    parser.add_argument('--config',
                       default='configs/rtmpose_vertebrae_config.py',
                       help='train config file path')
    parser.add_argument('--work-dir',
                       default='work_dirs/rtmpose_vertebrae',
                       help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='resume from the latest checkpoint in work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)

    # Merge CLI arguments into config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # Enable automatic mixed precision training
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # Resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # Set launcher
    if args.launcher != 'none':
        cfg.launcher = args.launcher

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    main()