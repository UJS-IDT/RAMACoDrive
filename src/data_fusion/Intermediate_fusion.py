from data_fusion.process_function import *
import os
import random
import math
import warnings
from collections import OrderedDict
from pre_processor import build_preprocessor
import numpy as np
import torch


class IntermediateFusion():
    def __init__(self, name, params, visualize=False, train=True):
        self.name = name
        self.params = params
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)

    def get_single_lidar_feature(self, scenes_list):
        for scene in scenes_list:
            selected_cav_processed = self.get_processed_lidar(scene["lidar_data"])
            scene["processed_features"] = selected_cav_processed['processed_features']
            del scene["lidar_data"]
            del scene["adjacent_vehicles"]

        return scenes_list

    def get_processed_lidar(self, lidar_raw_data):
        selected_cav_processed = {}
        lidar_np = lidar_raw_data
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        processed_lidar = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update(
            {'projected_lidar': lidar_np,
             'processed_features': processed_lidar})

        return selected_cav_processed

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict
        
