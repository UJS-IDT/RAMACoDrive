# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import time
import torch
import multiprocessing
from multiprocessing import shared_memory
import pickle
from data.commuicate_manager import CommuniAgent
import numpy as np
import matplotlib.pyplot as plt
from util import create_model, create_loss

class Trainer(multiprocessing.Process):
    def __init__(self, ego_config, config, receive_mutex):
        super().__init__()
        self.ego_config = ego_config
        self.config = config
        self.receive_mutex = receive_mutex
        self.cav_config = self.generate_cav_config()
        self.running_event = multiprocessing.Event()
        self.mem_name = self.ego_config["train_share_mem"]
        self.mem_addr = None
        self.last_timestamp = -1.
        self.ego_data = None
        self.initial_flag = False
        self.receivers = {}
        self.model = create_model(self.config["hypes_yaml"])
        self.loss = create_loss(self.config["hypes_yaml"])


    def run(self):
        while True:
            if self.running_event.is_set():
                try:
                    self.ego_flag = False
                    if not self.mem_addr:
                        self.mem_addr = shared_memory.SharedMemory(name=self.mem_name)
                    self.ego_data, self.ego_flag = self.get_ego_lidar()
                    if self.ego_flag:
                        self.processed_features = []
                        self.merged_features = []
                        self.vehicle_speed = []
                        self.cav_data_list = []
                        self.label_list = []
                        for idx in range(len(self.ego_data)):
                            self.processed_features.append([])
                            self.merged_features.append([])
                            self.vehicle_speed.append([])
                            self.label_list.append(self.ego_data[idx]["label"])
                            self.processed_features[idx].append(self.ego_data[idx]["processed_features"])
                            self.vehicle_speed[idx].append(self.ego_data[idx]["speed"])
                        if not self.initial_flag:
                            self.initial_cav_comm()
                            self.initial_flag = True
                        self.get_cav_lidar()
                        cav_num = len(self.cav_data_list) + 1
                        if len(self.cav_data_list) > 0:
                            for cav_data in self.cav_data_list:
                                for idx in range(len(cav_data)):
                                    self.processed_features[idx].append(cav_data[idx]["processed_features"])
                                    self.vehicle_speed[idx].append(cav_data[idx]["speed"])
                                    self.label_list[idx].extend([item for item in cav_data[idx]["label"] if item not in self.label_list[idx]])
                        for idx in range(len(self.processed_features)):
                            self.merged_features[idx] = self.merge_features_to_dict(self.processed_features[idx])

                        
                        
                except Exception as e:
                    print("An error occurred:", e)
                    pass
                else:
                    pass
            else:
                time.sleep(0.05)
                self.last_timestamp = -1.
                if self.mem_addr:
                    self.mem_addr.close()
                self.mem_addr = None
                self.ego_data = None
                self.ego_flag = False
                self.receivers = {}
                self.initial_flag = False
                
    def get_ego_lidar(self):
        #if self.mem_addr.buf:
        if self.mem_addr is not None and self.mem_addr.buf is not None:
            self.receive_mutex.acquire()
            s = bytes(self.mem_addr.buf[:20])
            index = s.find(b"~")
            if index != -1:
                head = s[0:index]
                contentlength = int(head)
                content = bytes(self.mem_addr.buf[index + 1:index + 1 + contentlength])
                data = pickle.loads(content)
                self.receive_mutex.release()
                if data[-1]["timestamp"] != self.last_timestamp:
                    self.last_timestamp = data[-1]["timestamp"]
                    return data, True
                else:
                    return None, False
            else:
                self.receive_mutex.release()
                return None, False
        else:
            return None, False
        
    
    def get_cav_lidar(self):
        data_flag = False
        for key, value in self.receivers.items():
            try:
                cav_receiver = self.receivers[key]["receiver"]
                received_data = cav_receiver.receive_latest_message(self.receivers[key]["v2x_name"])
                if received_data is not None:
                    distance = self.calculate_distance(self.ego_data[-1]["lidar_pose"], received_data[-1]["lidar_pose"])
                    if distance <= self.config["communication_distance"]:
                        self.cav_data_list.append(received_data)
            except Exception as e:
                print("An error occurred:", e)
                pass
            else:
                pass
        
    def generate_cav_config(self):
        cav_config = []
        for agent in self.config["agents"]:
            if agent["name"] != self.ego_config["name"]:
                cav_config.append(agent)
        for agent in self.config["road_agents"]:
            if agent["name"] != self.ego_config["name"]:
                cav_config.append(agent)
        return cav_config
        
    def initial_cav_comm(self):
        for cav in self.cav_config:
            receiver = CommuniAgent("receiver" + cav["name"])
            receiver.init_subscriber(cav["v2x_name"], cav["inter_port"], self.config["comm_trans_addr"])
            self.receivers[cav['name']] = {}
            self.receivers[cav['name']]["receiver"] = receiver
            self.receivers[cav['name']]["v2x_name"] = cav["v2x_name"]
            
    def calculate_distance(self, ego_pose, cav_pose):
        ego_pose_x, ego_pose_y, ego_pose_z = ego_pose[:3]
        cav_pose_x, cav_pose_y, cav_pose_z = cav_pose[:3]
        distance = np.sqrt((ego_pose_x - cav_pose_x) ** 2 + (ego_pose_y - cav_pose_y) ** 2 + (ego_pose_z - cav_pose_z) ** 2)
        return distance
        
    def plot_voxel(self, voxel_features, voxel_coords, voxel_num_points):
        voxel_coords_gpu = torch.tensor(voxel_coords).cuda()
        voxel_colors_gpu = torch.tensor(voxel_features[:, 0, 0]).cuda()
    
        voxel_coords_cpu = voxel_coords_gpu.cpu().numpy()
        voxel_colors_cpu = voxel_colors_gpu.cpu().numpy()
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('auto')

        for i in range(len(voxel_coords_cpu)):
            color = voxel_colors_cpu[i]
            ax.scatter(voxel_coords_cpu[i, 0], voxel_coords_cpu[i, 1], voxel_coords_cpu[i, 2],
                       c=color, s=voxel_num_points[i])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        ax.view_init(elev=0, azim=0)

        plt.show()
        

    def merge_features_to_dict(self, processed_feature_list):
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

        merged_feature_dict = {}

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict
        
        
        
        
        
        
        
        
        
        
        
        
        
        