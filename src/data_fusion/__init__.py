from data_fusion.Intermediate_fusion import IntermediateFusion
from data_fusion.Intermediate_fusion_v2xvit import IntermediateFusion_v2xvit

__all__ = {
    'IntermediateFusion': IntermediateFusion,
    'IntermediateFusion_v2xvit': IntermediateFusion_v2xvit
}

def build_data(agent_name, data_cfg, visualize=False, train=True):
    dataset_name = data_cfg['fusion']['core_method']

    dataset = __all__[dataset_name](
        name=agent_name,
        params=data_cfg,
        visualize=visualize,
        train=train
    )

    return dataset