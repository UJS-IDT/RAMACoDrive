import logging
from util import Singleton
import yaml
import time
import os
from colorlog import ColoredFormatter
import numpy as np
import math


def init_logger(state="console"):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    log_file = f"../logs/{current_time}/log.txt"
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger()

    if state in ["file", "all"]:
        os.makedirs(f"../logs/{current_time}", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if state in ["console", "all"]:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if state != "none":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.CRITICAL)


def load_point_pillar_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['point_pillar_scatter']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


class Config(metaclass=Singleton):
    def __init__(self):
        with open('../config/base.yml', 'r') as base_config_file:
            self.config = yaml.safe_load(base_config_file)

    def update_config(self, updates, current_dict=None):
        """
        Recursively update the configuration dictionary with the provided updates.

        :param updates: A dictionary containing the updates.
        :param current_dict: The current level in the configuration dictionary.
        """
        if current_dict is None:
            current_dict = self.config

        for key, value in updates.items():
            if isinstance(value, dict) and key in current_dict:
                self.update_config(value, current_dict[key])
            else:
                current_dict[key] = value

    def update_dict_recursively(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self.update_dict_recursively(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def merge(self, args):
        update_params = {}
        if args.debug:
            with open("../config/"+args.debug+".yml", 'r') as test_config_file:
                update_params = yaml.safe_load(test_config_file)
        if args.town:
            update_params["map_name"] = args.town
        if args.show_sequence:
            update_params["show_sequence"] = True
        else:
            update_params["show_sequence"] = False
        if args.hypes:
            with open("../hypes_yaml/"+args.hypes+".yaml", 'r') as hypes_file:
                hypes_params = yaml.safe_load(hypes_file)
                if "yaml_parser" in hypes_params:
                    hypes_params = eval(hypes_params["yaml_parser"])(hypes_params)
                update_params["hypes_yaml"] = hypes_params
        if args.log:
            init_logger(args.log)
        else:
            init_logger("none")
        if args.epoches:
            update_params["rest_epoches"] = args.epoches
        else:
            if "hypes_yaml" in update_params:
                update_params["rest_epoches"] = update_params["hypes_yaml"]["train_params"]["epoches"]
            else:
                update_params["rest_epoches"] = 1
        update_params["model_dir"] = args.model_dir

        logging.info("Config init")

        self.config = self.update_dict_recursively(self.config, update_params)
        return self.config


config = Config()
