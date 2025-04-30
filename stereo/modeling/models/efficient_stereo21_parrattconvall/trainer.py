# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
from stereo.modeling.trainer_template import TrainerTemplate
from .efficient_stereo import EfficientStereo21

__all__ = {
    'EfficientStereo21': EfficientStereo21,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
