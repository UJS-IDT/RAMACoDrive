# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import sys
import time
import torch
import traceback
import multiprocessing
from multiprocessing import shared_memory
import os
import pickle
from data.commuicate_manager import CommuniAgent
from data_fusion.Async_data_receiver import Async_receiver
import numpy as np
import matplotlib.pyplot as plt
from util import create_model, create_loss, x_to_world, project_points_by_matrix_torch, x1_to_x2, \
    setup_train, load_saved_model, setup_optimizer, setup_lr_schedular, to_device, connect_to_server, world_to_x
from post_processor.box import box_utils
from tools.process_function import *
from pre_processor import build_preprocessor
from post_processor import build_postprocessor
from tensorboardX import SummaryWriter
import statistics


class Trainer(multiprocessing.Process):
    def __init__(self, ego_config, all_config, receive_mutex, cof_seg, model, init_epoch, saved_path):
        super().__init__()
        self.ego_config = ego_config
        self.config = all_config
        self.receive_mutex = receive_mutex
        self.cav_config = self.generate_cav_config()
        self.training_event = multiprocessing.Event()
        self.exit_event = multiprocessing.Event()
        self.save_event = multiprocessing.Event()
        self.epoch_event = multiprocessing.Event()
        self.eval_event = multiprocessing.Event()
        self.eval_mode_flag = False
        self.eval_calculate_flag = False
        self.best_epoch_num = 0
        self.valid_ave_loss = []
        self.save_flag = False
        self.scheduler_event = multiprocessing.Event()
        self.mem_name = self.ego_config["train_share_mem"]
        self.mem_addr = None
        self.last_timestamp = -1.
        self.ego_data = None
        self.this_ego_only = 0
        self.cof_seg = cof_seg
        self.best_loss = 1000000.
        self.this_loss = 0.
        self.async_data_mem_mutex = multiprocessing.Lock()
        self.max_cav_num = self.config["hypes_yaml"]['train_params']['max_cav']
        self.max_num = self.config["hypes_yaml"]['postprocess']['max_num']
        self.cav_lidar_range = self.config["hypes_yaml"]['postprocess']['anchor_args']['cav_lidar_range']
        self.initial_flag = False
        self.receivers = {}
        self.cav_data_dict = {}
        self.total_batches_processed = 0
        self.batches_processed = 0
        self.cav_data_change_flag = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.init_epoch = init_epoch
        self.saved_path = saved_path
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.model_without_ddp = self.model
        self.criterion = create_loss(self.config["hypes_yaml"])
        self.optimizer = setup_optimizer(self.config["hypes_yaml"], self.model_without_ddp)
        self.scheduler = setup_lr_schedular(self.config["hypes_yaml"], self.optimizer)
        self.writer = None
        self.order = self.config["hypes_yaml"]['postprocess']['order']
        
        self.pre_processor = build_preprocessor(self.config["hypes_yaml"]['preprocess'], train=True)
        self.post_processor = build_postprocessor(self.config["hypes_yaml"]['postprocess'], train=True)
        self.anchor_box = self.post_processor.generate_anchor_box()


    def run(self):
        self.async_data_receiver = Async_receiver(self.config, self.ego_config, self.async_data_mem_mutex, self.cof_seg)
        self.async_data_receiver.start()
        self.client, self.world = connect_to_server(1000, 2000)
        if not self.writer:
            self.writer = SummaryWriter(self.saved_path)
        for param_group in self.optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])
        while True:
            if self.eval_event.is_set():
                torch.cuda.empty_cache()
                self.eval_event.clear()
                self.batches_processed = 0
                self.this_ego_only = 0
                self.this_loss = 0.
                self.eval_mode_flag = True
            if self.scheduler_event.is_set():
                self.scheduler.step()
                torch.cuda.empty_cache()
                self.scheduler_event.clear()
                self.epoch_event.set()
                self.batches_processed = 0
                self.this_ego_only = 0
                self.this_loss = 0.
            if self.training_event.is_set():
                self.async_data_receiver.running_event.set()
                if self.ego_config["name"] not in self.cof_seg.keys():
                    continue
                try:
                    self.ego_flag = False
                    self.processed_lidar_list = []
                    if not self.mem_addr:
                        self.mem_addr = shared_memory.SharedMemory(name=self.mem_name)
                    self.ego_data, self.ego_flag = self.get_ego_lidar()
                    if self.ego_flag:
                        if self.eval_mode_flag:
                            self.model.eval()
                        else:
                            self.model.train()
                            self.model.zero_grad()
                            self.optimizer.zero_grad()
                        self.cav_data_dict[self.ego_config["name"]] = self.ego_data
                        if not self.initial_flag:
                            self.initial_cav_comm()
                            self.initial_flag = True
                        self.get_cav_lidar()
                        self.batch = []
                        self.intermediate_feature = []
                        self.feature_mask = []
                        for scene_id in range(len(self.ego_data)):
                            processed_features = []
                            object_id_stack = []
                            time_delay = []
                            transform_matrix_list = []
                            ego_pose = self.ego_data[scene_id]["lidar_pose"]
                            for cav_id, cav_data in self.cav_data_dict.items():
                                if len(time_delay) >= self.max_cav_num:
                                    continue
                                single_cav_processed = self.get_item_single_car(cav_data[scene_id], ego_pose)
                                if len(single_cav_processed) == 0:
                                    continue
                                else:
                                    delayed_time_value = float(self.ego_data[scene_id]["timestamp"] - cav_data[scene_id]["timestamp"])
                                    if delayed_time_value > 45.0 or delayed_time_value < -1.0:
                                        continue
                                    if scene_id == 0 and cav_id != self.ego_config["name"]:
                                        self.intermediate_feature.append(cav_data[0]["spatial_features_2d"])
                                        self.feature_mask.append(cav_data[0]["valid_mask"])
                                    if cav_id == self.ego_config["name"]:
                                        processed_features.append(cav_data[scene_id]["processed_lidar"])
                                    object_id_stack.extend([item for item in single_cav_processed["object_ids"] if item not in object_id_stack])
                                    transform_matrix_list.append(single_cav_processed['transform_matrix'])
                                    time_delay.append(delayed_time_value)
                            cav_num = len(time_delay)
                            merged_cavs_feature_dict = self.merge_features_to_dict(processed_features)
                            time_delay = time_delay + (self.max_cav_num - len(time_delay)) * [0.]
                            pairwise_t_matrix = self.get_pairwise_transformation(self.max_cav_num, transform_matrix_list)
                            self.batch.append(
                            {'object_ids': object_id_stack,
                             'processed_lidar': merged_cavs_feature_dict,
                             'cav_num': cav_num,
                             'pairwise_t_matrix': pairwise_t_matrix,
                             'time_delay': time_delay
                            })
                        if cav_num == -1:
                            self.this_ego_only += 1
                            if self.this_ego_only > 50:
                                continue
                        self.collate_batch = self.collate_batch_train(self.batch)
                        if len(self.intermediate_feature) > 0:
                            self.collate_batch.update({"intermediate_feature_v2x": self.intermediate_feature})
                        if len(self.feature_mask) > 0:
                            self.collate_batch.update({"feature_mask_v2x": self.feature_mask})
                        if self.eval_mode_flag:
                            with torch.no_grad():
                                self.input_data = to_device(self.collate_batch, self.device)
                                self.ouput_dict = self.model(self.input_data, True)
                                self.object_bbx_center = np.zeros((self.max_num, 7))
                                self.mask = np.zeros(self.max_num)
                                if self.ego_config["name"] not in self.cof_seg.keys():
                                    del self.input_data, self.ouput_dict
                                    torch.cuda.empty_cache()
                                    continue
                                self.label = self.generate_label()
                                if self.label is None:
                                    del self.input_data, self.ouput_dict
                                    torch.cuda.empty_cache()
                                    continue
                                self.label = to_device(self.label, self.device)
                                self.final_loss = self.criterion(self.ouput_dict, self.label)
                                self.valid_ave_loss.append(self.final_loss.item())
                                self.eval_calculate_flag = True
                                print("[epoch %d], || Eval_Loss: %.4f" % (self.init_epoch, self.final_loss.item()))
                        else:
                            self.input_data = to_device(self.collate_batch, self.device)
                            self.ouput_dict = self.model(self.input_data, True)
                            self.object_bbx_center = np.zeros((self.max_num, 7))
                            self.mask = np.zeros(self.max_num)
                            if self.ego_config["name"] not in self.cof_seg.keys():
                                del self.input_data, self.ouput_dict
                                torch.cuda.empty_cache()
                                continue
                            self.label = self.generate_label()
                            if self.label is None:
                                del self.input_data, self.ouput_dict
                                torch.cuda.empty_cache()
                                continue
                            self.label = to_device(self.label, self.device)
                            self.final_loss = self.criterion(self.ouput_dict, self.label)
                            self.criterion.logging(self.init_epoch, self.batches_processed, self.writer, self.total_batches_processed)
                            self.batches_processed += 1
                            self.total_batches_processed += 1
                            self.final_loss.backward()
                            self.optimizer.step()
                            self.save_flag = True
                            self.valid_ave_loss.append(self.final_loss.item())
                            this_loss = self.final_loss.item()
                            self.this_loss += this_loss
                        
                except Exception as e:
                    traceback.print_exc()
                    print("An error occurred:", e)
                    pass
                else:
                    pass
            else:
                time.sleep(0.05)
                if self.async_data_receiver.running_event.is_set():
                    self.async_data_receiver.running_event.clear()
                if self.save_event.is_set():
                    if self.save_flag:
                        torch.save(self.model_without_ddp.state_dict(), os.path.join(self.saved_path, 'net_epoch%d.pth' % (self.init_epoch)))
                        self.save_event.clear()
                        self.save_flag = False
                        self.valid_ave_loss_value = statistics.mean(self.valid_ave_loss)
                        print('At epoch %d, the validation loss is %f' % (self.init_epoch, self.valid_ave_loss_value))
                        self.writer.add_scalar('Train_Loss', self.valid_ave_loss_value, self.init_epoch)
                        self.valid_ave_loss = []
                if self.eval_mode_flag:
                    if self.eval_calculate_flag:
                        self.eval_mode_flag = False
                        self.eval_calculate_flag = False
                        self.valid_ave_loss_value = statistics.mean(self.valid_ave_loss)
                        print('At epoch %d, the validation loss is %f' % (self.init_epoch - 1, self.valid_ave_loss_value))
                        self.writer.add_scalar('Validate_Loss', self.valid_ave_loss_value, self.init_epoch - 1)
                        if self.valid_ave_loss_value < self.best_loss:
                            self.best_loss = self.valid_ave_loss_value
                            torch.save(self.model_without_ddp.state_dict(), os.path.join(self.saved_path, 'net_best.pth'))
                            self.best_epoch_num = self.init_epoch - 1
                        print("best epoch now is epoch", self.best_epoch_num)
                        self.valid_ave_loss = []
                if self.epoch_event.is_set():
                    self.init_epoch += 1
                    self.epoch_event.clear()
                self.last_timestamp = -1.
                if self.mem_addr:
                    self.mem_addr.buf[:] = b'\0' * self.mem_addr.size
                    self.mem_addr.close()
                self.mem_addr = None
                self.cav_data_dict = {}
                self.cav_data_change_flag = {}
                self.ego_data = None
                self.ego_flag = False
                self.receivers = {}
                self.initial_flag = False
                if self.exit_event.is_set():
                    self.async_data_receiver.exit_event.set()
                    sys.exit()
                
    def get_ego_lidar(self):
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
        if self.async_data_mem_addr is not None and self.async_data_mem_addr.buf is not None:
            self.async_data_mem_mutex.acquire()
            s = bytes(self.async_data_mem_addr.buf[:20])
            index = s.find(b"~")
            if index != -1:
                head = s[0:index]
                contentlength = int(head)
                content = bytes(self.async_data_mem_addr.buf[index + 1:index + 1 + contentlength])
                data = pickle.loads(content)
                self.async_data_mem_mutex.release()             
                for key, value in data.items():
                    try:
                        if "cav" in key:
                            if key not in self.cof_seg.keys():
                                if key in self.cav_data_dict.keys():
                                    del self.cav_data_dict[key]
                                    del self.cav_data_change_flag[key]
                                continue
                        distance = self.calculate_distance(self.ego_data[-1]["lidar_pose"], value[-1]["lidar_pose"])
                        if distance <= self.config["communication_distance"]:
                            self.cav_data_dict[key] = value
                            self.cav_data_change_flag[key] = True
                        else:
                            if key in self.cav_data_change_flag.keys():
                                if self.cav_data_change_flag[key]:
                                    self.cav_data_change_flag[key] = False
                                else:
                                    del self.cav_data_dict[key]
                                    del self.cav_data_change_flag[key]
                    except Exception as e:
                        print("An error occurred:", e)
                        pass
                    else:
                        pass
            else:
                self.async_data_mem_mutex.release()
    
    def get_cav_lidar_old(self):
        data_flag = False
        for key, value in self.receivers.items():
            try:
                if "cav" in key:
                    if key not in self.cof_seg.keys():
                        if key in self.cav_data_dict.keys():
                            del self.cav_data_dict[key]
                            del self.cav_data_change_flag[key]
                        continue
                cav_receiver = self.receivers[key]["receiver"]
                received_data = cav_receiver.receive_latest_message(self.receivers[key]["v2x_name"])
                if received_data is not None:
                    distance = self.calculate_distance(self.ego_data[-1]["lidar_pose"], received_data[-1]["lidar_pose"])
                    if distance <= self.config["communication_distance"]:
                        self.cav_data_dict[key] = received_data
                        self.cav_data_change_flag[key] = True
                    else:
                        if key in self.cav_data_change_flag.keys():
                            if self.cav_data_change_flag[key]:
                                self.cav_data_change_flag[key] = False
                            else:
                                del self.cav_data_dict[key]
                                del self.cav_data_change_flag[key]
            except Exception as e:
                print("An error occurred:", e)
                pass
            else:
                pass
        
    def generate_cav_config(self):
        cav_config = []
        for agent in self.config["road_agents"]:
            if agent["name"] != self.ego_config["name"]:
                cav_config.append(agent)
        for agent in self.config["agents"]:
            if agent["name"] != self.ego_config["name"]:
                cav_config.append(agent)
        return cav_config
        
    def initial_cav_comm(self):
        async_data_mem_name = "Async_receiver_to_" + self.ego_config["name"]
        self.async_data_mem_addr = shared_memory.SharedMemory(name=async_data_mem_name)
        
    def initial_cav_comm_old(self):
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
        
        
    def get_processed_lidar(self, lidar_raw_data, ego_pose, cav_pose):
        lidar_np = lidar_raw_data
        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        if self.config["hypes_yaml"]["proj_first"]:
            transform_matrix = x1_to_x2(cav_pose, ego_pose)
            lidar_np[:, :3] = project_points_by_matrix_torch(lidar_np[:, :3], transform_matrix)
            lidar_np = mask_points_by_range(lidar_np, self.config["hypes_yaml"]['preprocess']['cav_lidar_range'])
        if lidar_np.shape[0] == 0:
            return lidar_np, False
        else:
            processed_lidar = self.pre_processor.preprocess(lidar_np)
            return processed_lidar, True
        
    
    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        if True:
            transform_matrix = x1_to_x2(selected_cav_base["lidar_pose"], ego_pose)
            object_ids = selected_cav_base["label"]
            selected_cav_processed.update(
                {'transform_matrix': transform_matrix,
                 'object_ids': object_ids})
            return selected_cav_processed
    
    
    def get_item_single_car_early(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        cav_processed_feature, processed_flag = self.get_processed_lidar(selected_cav_base["lidar_data"], ego_pose, selected_cav_base["lidar_pose"])
        if processed_flag:
            delay_ego_lidar_pose = selected_cav_base["received_ego_pose"][self.ego_config["name"]]
            object_ids = selected_cav_base["label"]
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose, ego_pose)
            velocity = selected_cav_base["speed"]
            velocity = velocity / 30 # normalize veloccity by average speed 30 km/h
            selected_cav_processed.update(
                {'object_ids': object_ids,
                 'processed_features': cav_processed_feature,
                 'spatial_correction_matrix': spatial_correction_matrix,
                 'velocity': velocity})
            return selected_cav_processed
        else:
            return selected_cav_processed
        
        
    def collate_batch_train(self, batch):
        output_dict = {}
        object_ids = []
        processed_lidar_list = []
        record_len = []
        

        pairwise_t_matrix_list = []
        
        for i in range(len(batch)):
            input_dict = batch[i]
            object_ids.append(input_dict['object_ids'])
            
            processed_lidar_list.append(input_dict['processed_lidar'])
            record_len.append(input_dict['cav_num'])
            pairwise_t_matrix_list.append(input_dict['pairwise_t_matrix'])
        
        merged_batch_features_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_batch_features_dict)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))

        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        output_dict.update({
            'processed_lidar': processed_lidar_torch_dict,
            'record_len': record_len,
            'pairwise_t_matrix': pairwise_t_matrix,
            'object_ids': object_ids})
        return output_dict
        
        
    def get_object_info(self, obj, ego_pose):
        obj_transform = obj.get_transform()
        obj_center = obj.bounding_box.location
        obj_pose = [obj_transform.location.x + obj_center.x,
                    obj_transform.location.y + obj_center.y,
                    obj_transform.location.z + obj_center.z,
                    obj_transform.rotation.roll,
                    obj_transform.rotation.yaw,
                    obj_transform.rotation.pitch]
        object2lidar = x1_to_x2(obj_pose, ego_pose)
        obj_extent = obj.bounding_box.extent
        extent = [obj_extent.x, obj_extent.y, obj_extent.z]
        bbx = box_utils.create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = box_utils.corner_to_center(bbx_lidar, order=self.order)
        bbx_lidar = box_utils.mask_boxes_outside_range_numpy(bbx_lidar, self.cav_lidar_range, self.order)
        return bbx_lidar
        
        
    def find_ego_pose(self):
        ego_pose = None
        if "cav" in self.ego_config["name"]:
            world_temp_actors = self.world.get_actors().filter('*vehicle*')
        else:
            world_temp_actors = self.world.get_actors().filter('*streetsign*')
        for npc in world_temp_actors:
            if npc.attributes["role_name"] == self.ego_config["name"]:
                ego_transform = npc.get_transform()
                ego_pose = [ego_transform.location.x,
                            ego_transform.location.y,
                            ego_transform.location.z + 2.4,
                            ego_transform.rotation.roll,
                            ego_transform.rotation.yaw,
                            ego_transform.rotation.pitch]
        return ego_pose
        
        
    def generate_label(self):
        label_dict_list = []
        for batch_idx in range(len(self.batch)):
            output_dict = {}
            object_center = np.zeros((self.max_num, 7))
            mask = np.zeros(self.max_num)
            ego_pose_now = self.find_ego_pose()
            if ego_pose_now is None:
                return None
            world_actors = self.world.get_actors().filter('*vehicle*')
            for npc in world_actors:
                if npc.id in self.batch[batch_idx]["object_ids"]:
                    npc_bbx_lidar = self.get_object_info(npc, ego_pose_now)
                    if npc_bbx_lidar.shape[0] > 0:
                        output_dict.update({npc.id: npc_bbx_lidar})
            for i, (object_id, object_bbx) in enumerate(output_dict.items()):
                object_center[i] = object_bbx[0, :]
                mask[i] = 1
            object_center_stack = object_center[mask == 1]
            label_dict = self.post_processor.generate_label(
                            gt_box_center=object_center,
                            anchors=self.anchor_box,
                            mask=mask)
            label_dict_list.append(label_dict)
        label_torch_dict = self.post_processor.collate_batch(label_dict_list)
            
        return label_torch_dict
        
        
    def get_pairwise_transformation(self, max_cav, transform_matrix_list):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        pairwise_t_matrix[:, :] = np.identity(4)

        if self.config["hypes_yaml"]['proj_first']:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for t_m in transform_matrix_list:
                t_list.append(t_m)

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        t_matrix = np.eye(4)
                        pairwise_t_matrix[i, j] = t_matrix
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
        
