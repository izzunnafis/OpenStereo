# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming

from .models.casnet.trainer import Trainer as CasStereoTrainer
from .models.cfnet.trainer import Trainer as CFNetTrainer
# from .models.aanet.trainer import Trainer as AANetTrainer
from .models.coex.trainer import Trainer as CoExTrainer
from .models.fadnet.trainer import Trainer as FADNetTrainer
from .models.gwcnet.trainer import Trainer as GwcNetTrainer
from .models.igev.trainer import Trainer as IGEVTrainer
from .models.msnet.trainer import Trainer as MSNetTrainer
from .models.psmnet.trainer import Trainer as PSMNetTrainer
from .models.sttr.trainer import Trainer as STTRTrainer
from .models.lightstereo.trainer import Trainer as LightStereoTrainer
from .models.stereobase.trainer import Trainer as StereoBaseGRUTrainer
from .models.efficient_stereo.trainer import Trainer as EfficientStereoTrainer
from .models.efficient_stereo2.trainer import Trainer as EfficientStereo2Trainer
from .models.efficient_stereo3.trainer import Trainer as EfficientStereo3Trainer
from .models.efficient_stereo4.trainer import Trainer as EfficientStereo4Trainer
from .models.efficient_stereo5.trainer import Trainer as EfficientStereo5Trainer
from .models.efficient_stereo6.trainer import Trainer as EfficientStereo6Trainer
from .models.efficient_stereo7.trainer import Trainer as EfficientStereo7Trainer
from .models.efficient_stereo8.trainer import Trainer as EfficientStereo8Trainer

# from .models.iinet.trainer import Trainer as IINetTrainer

# %If you want to train/eval NMRF-Stereo, you need to build deformable attention and superpixel-guided disparity downsample operator: 'cd stereo/modeling/models/nmrf/ops && sh make.sh && cd ..'
from .models.nmrf.trainer import Trainer as NMRFTrainer  


__all__ = {
    'STTR': STTRTrainer,
    'PSMNet': PSMNetTrainer,
    'MSNet2D': MSNetTrainer,
    'MSNet3D': MSNetTrainer,
    'IGEV': IGEVTrainer,
    'GwcNet': GwcNetTrainer,
    'FADNet': FADNetTrainer,
    'CoExNet': CoExTrainer,
    # 'AANet': AANetTrainer,
    'CFNet': CFNetTrainer,
    'CasGwcNet': CasStereoTrainer,
    'CasPSMNet': CasStereoTrainer,
    'LightStereo': LightStereoTrainer,
    'StereoBaseGRU': StereoBaseGRUTrainer,
    # 'IInet': IINetTrainer,
    'NMRF': NMRFTrainer,
    'EfficientStereo' : EfficientStereoTrainer,
    'EfficientStereo2' : EfficientStereo2Trainer,
    "EfficientStereo3" : EfficientStereo3Trainer,
    "EfficientStereo4" : EfficientStereo4Trainer,
    "EfficientStereo5" : EfficientStereo5Trainer,
    "EfficientStereo6" : EfficientStereo6Trainer,
    "EfficientStereo7" : EfficientStereo7Trainer,
    "EfficientStereo8" : EfficientStereo8Trainer,
}


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer)
    return trainer
