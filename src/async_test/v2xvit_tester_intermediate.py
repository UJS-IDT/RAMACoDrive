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
import open3d as o3d
import pickle
from data.commuicate_manager import CommuniAgent
from data_fusion.Async_data_receiver import Async_receiver
import numpy as np
import matplotlib.pyplot as plt
from util import create_model, create_loss, x_to_world, project_points_by_matrix_torch, x1_to_x2, compute_distance3D,\
    setup_train, load_saved_model, setup_optimizer, setup_lr_schedular, to_device, connect_to_server, world_to_x
from post_processor.box import box_utils
from post_processor.box import eval_utils
from visualize import vis_utils
from tools.process_function import *
from pre_processor import build_preprocessor
from post_processor import build_postprocessor
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.sub_modules.pillar_vfe import PillarVFE
from models.sub_modules.point_pillar_scatter import PointPillarScatter
from models.sub_modules.base_bev_backbone import BaseBEVBackbone
from models.sub_modules.downsample_conv import DownsampleConv
from models.sub_modules.naive_compress import NaiveCompressor


class Tester(multiprocessing.Process):
    def __init__(self, ego_config, all_config, receive_mutex, cof_seg, eval_all_frame, eval_remain, model, init_epoch, saved_path):
        super().__init__()
        self.ego_config = ego_config
        self.config = all_config
        self.receive_mutex = receive_mutex
        self.eval_all_frame = eval_all_frame
        self.eval_remain = eval_remain
        self.eval_remain[self.ego_config["name"]] = True
        self.eval_valid_frame = {}
        self.cav_config = self.generate_cav_config()
        self.test_event = multiprocessing.Event()
        self.exit_event = multiprocessing.Event()
        self.save_event = multiprocessing.Event()
        self.epoch_event = multiprocessing.Event()
        self.eval_event = multiprocessing.Event()
        self.scheduler_event = multiprocessing.Event()
        self.mem_name = self.ego_config["train_share_mem"]
        self.mem_addr = None
        self.pred_flag = False
        self.gt_flag = False
        self.last_timestamp = -1.
        self.ego_data = None
        self.ego_actor_id = None
        self.ego_data_num = 0
        self.cav_data_num = 0
        self.cof_seg = cof_seg
        self.async_data_mem_mutex = multiprocessing.Lock()
        self.max_cav_num = self.config["hypes_yaml"]['train_params']['max_cav']
        self.max_num = self.config["hypes_yaml"]['postprocess']['max_num']
        self.initial_flag = False
        self.receivers = {}
        self.cav_data_dict = {}
        self.total_batches_processed = 0
        self.batches_processed = 0
        self.eval_ready_flag = False
        self.vis = None
        self.cav_data_change_flag = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_cpu = torch.device('cpu')
        self.model = model
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.init_epoch = init_epoch
        self.saved_path = saved_path
        self.model.eval()
        self.model_without_ddp = self.model
        self.criterion = create_loss(self.config["hypes_yaml"])
        self.optimizer = setup_optimizer(self.config["hypes_yaml"], self.model_without_ddp)
        self.scheduler = setup_lr_schedular(self.config["hypes_yaml"], self.optimizer)
        self.writer = None
        self.order = self.config["hypes_yaml"]['postprocess']['order']
        
        self.pre_processor = build_preprocessor(self.config["hypes_yaml"]['preprocess'], train=True)
        self.post_processor = build_postprocessor(self.config["hypes_yaml"]['postprocess'], train=True)
        self.anchor_box = self.post_processor.generate_anchor_box()
        self.pillar_vfe = PillarVFE(self.config["hypes_yaml"]['model']['args']['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=self.config["hypes_yaml"]['model']['args']['voxel_size'],
                                    point_cloud_range=self.config["hypes_yaml"]['model']['args']['lidar_range'])
        self.scatter = PointPillarScatter(self.config["hypes_yaml"]['model']['args']['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(self.config["hypes_yaml"]['model']['args']['base_bev_backbone'], 64)
        
        self.shrink_flag = False
        if 'shrink_header' in self.config["hypes_yaml"]['model']['args']:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(self.config["hypes_yaml"]['model']['args']['shrink_header'])
            
        self.compression = False
        if self.config["hypes_yaml"]['model']['args']['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, self.config["hypes_yaml"]['model']['args']['compression'])


    def run(self):
        self.async_data_receiver = Async_receiver(self.config, self.ego_config, self.async_data_mem_mutex, self.cof_seg)
        self.async_data_receiver.start()
        self.client, self.world = connect_to_server(1000, 2000)
        if not self.writer:
            self.writer = SummaryWriter(self.saved_path)
        for param_group in self.optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])
        while True:
            if self.scheduler_event.is_set():
                self.scheduler.step()
                torch.cuda.empty_cache()
                self.scheduler_event.clear()
                self.epoch_event.set()
                self.batches_processed = 0
            if self.test_event.is_set():
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
                        self.cav_data_dict[self.ego_config["name"]] = self.ego_data
                        if not self.initial_flag:
                            self.initial_cav_comm()
                            self.initial_flag = True
                        self.get_cav_lidar()
                        self.batch = []
                        self.intermediate_feature = []
                        for scene_id in range(len(self.ego_data)):
                            processed_features = []
                            object_id_stack = []
                            velocity = []
                            time_delay = []
                            infra = []
                            spatial_correction_matrix = []
                            projected_lidar_stack = []
                            ego_pose = self.ego_data[scene_id]["lidar_pose"]
                            for cav_id, cav_data in self.cav_data_dict.items():
                                if len(velocity) >= self.max_cav_num:
                                    continue
                                single_cav_processed = self.get_item_single_car(cav_data[scene_id], ego_pose)
                                if len(single_cav_processed) == 0:
                                    continue
                                else:
                                    delayed_time_value = float(self.ego_data[scene_id]["timestamp"] - cav_data[scene_id]["timestamp"])
                                    if delayed_time_value > 45.0 or delayed_time_value < -1.0:
                                        continue
                                    if scene_id == 0 and cav_id != self.ego_config["name"]:
                                        self.intermediate_feature.append(cav_data[0]["spatial_features_2d"][-1].unsqueeze(0))
                                    if cav_id == self.ego_config["name"]:
                                        processed_features.append(cav_data[scene_id]["processed_lidar"])
                                    object_id_stack.extend([item for item in single_cav_processed["object_ids"] if item not in object_id_stack])
                                    velocity.append(single_cav_processed['velocity'])
                                    time_delay.append(delayed_time_value)
                                    spatial_correction_matrix.append(single_cav_processed["spatial_correction_matrix"])
                                    if "cav" in cav_id:
                                        infra.append(0)
                                    else:
                                        infra.append(1)
                                    if self.config["show_sequence"]:
                                        if cav_id == self.ego_config["name"]:
                                            projected_lidar_stack.append(cav_data[scene_id]['project_lidar'])
                            cav_num = len(velocity)
                            self.ego_data_num += 1
                            self.cav_data_num += (len(velocity) - 1)
                            merged_cavs_feature_dict = self.merge_features_to_dict(processed_features)
                            velocity = velocity + (self.max_cav_num - len(velocity)) * [0.]
                            time_delay = time_delay + (self.max_cav_num - len(time_delay)) * [0.]
                            infra = infra + (self.max_cav_num - len(infra)) * [0.]
                            spatial_correction_matrix = np.stack(spatial_correction_matrix)
                            padding_eye = np.tile(np.eye(4)[None],(self.max_cav_num - len(spatial_correction_matrix), 1, 1))
                            spatial_correction_matrix = np.concatenate([spatial_correction_matrix, padding_eye], axis=0)
                            if self.config["show_sequence"]:
                                origin_lidar_output = np.vstack(projected_lidar_stack)
                            else:
                                origin_lidar_output = []
                            self.batch.append(
                            {'object_ids': object_id_stack,
                             'processed_lidar': merged_cavs_feature_dict,
                             'origin_lidar': origin_lidar_output,
                             'cav_num': cav_num,
                             'velocity': velocity,
                             'time_delay': time_delay,
                             'infra': infra,
                             'spatial_correction_matrix': spatial_correction_matrix
                            })
                        self.batch = self.batch[-1:]
                        self.collate_batch = self.collate_batch_train(self.batch)
                        if len(self.intermediate_feature) > 0:
                            self.intermediate_feature_cat = torch.cat(self.intermediate_feature, dim=0)
                            self.collate_batch.update({"intermediate_feature_v2x": self.intermediate_feature_cat})
                        with torch.no_grad():
                            self.input_data = to_device(self.collate_batch, self.device)
                            self.ouput_dict = self.model(self.input_data, True)
                            ego_pose_now = self.find_ego_pose()
                            if ego_pose_now is None:
                                continue
                            if self.ego_actor_id is None:
                                if "cav" in self.ego_config["name"]:
                                    self.ego_actor_search = self.world.get_actors().filter('*vehicle*')
                                else:
                                    self.ego_actor_search = self.world.get_actors().filter('*streetsign*')
                                for npc in self.ego_actor_search:
                                    if npc.attributes["role_name"] == self.ego_config["name"]:
                                        self.ego_actor_id = npc.id
                            if len(self.eval_all_frame) == 0:
                                continue
                            self.latest_index = len(self.eval_all_frame) - 1
                            if self.ego_actor_id not in self.eval_all_frame[self.latest_index]["actor_pose"].keys():
                                continue
                            self.ouput_dict_cpu = to_device(self.ouput_dict, self.device_cpu)
                            self.input_data_cpu = to_device(self.input_data, self.device_cpu)
                            self.eval_valid_frame[self.latest_index] = {}
                            self.eval_valid_frame[self.latest_index]["output_dict"] = self.ouput_dict_cpu
                            del self.input_data_cpu["processed_lidar"]
                            del self.input_data_cpu["record_len"]
                            del self.input_data_cpu["prior_encoding"]
                            del self.input_data_cpu["spatial_correction_matrix"]
                            if "intermediate_feature_v2x" in self.input_data_cpu:
                                del self.input_data_cpu["intermediate_feature_v2x"]
                            self.eval_valid_frame[self.latest_index]["input_data"] = self.input_data_cpu
                            self.eval_ready_flag = True
                            print("[epoch %d][%d]" % (self.init_epoch, self.batches_processed + 1))
                            self.batches_processed += 1
                            self.total_batches_processed += 1
                        
                except Exception as e:
                    traceback.print_exc()
                    print("An error occurred:", e)
                    pass
                else:
                    pass
            else:
                time.sleep(0.5)
                if self.async_data_receiver.running_event.is_set():
                    self.async_data_receiver.running_event.clear()
                if self.eval_ready_flag:
                    if self.vis is None and self.config["show_sequence"]:
                        self.vis = o3d.visualization.Visualizer()
                        self.vis.create_window()
                        self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
                        self.vis.get_render_option().point_size = 1.0
                        self.vis.get_render_option().show_coordinate_frame = True
                        self.vis_pcd = o3d.geometry.PointCloud()
                        self.vis_aabbs_gt = []
                        self.vis_aabbs_pred = []
                        for _ in range(self.max_num):
                            self.vis_aabbs_gt.append(o3d.geometry.LineSet())
                            self.vis_aabbs_pred.append(o3d.geometry.LineSet())
                    relative_result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                    absolute_result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                    range_result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                         0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                         0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                    all_result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                       0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                                       0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                    short_result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                                         0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                                         0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                    middle_result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                                          0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                                          0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                    long_result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                                        0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                                        0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
                    # weird in V2X-VIT
                    transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
                    anchor_box = torch.from_numpy(np.array(self.anchor_box))
                    transformation_matrix_torch = to_device(transformation_matrix_torch, self.device)
                    anchor_box = to_device(anchor_box, self.device)
                    print("len_all=======", len(self.eval_all_frame))
                    print("len_valid=======", len(self.eval_valid_frame))
                    vis_idx = 0
                    for key, value_cpu in tqdm(self.eval_valid_frame.items()):
                        value = to_device(value_cpu, self.device)
                        all_actors_pose = self.eval_all_frame[key]["actor_pose"]
                        all_actors_extent = self.eval_all_frame[key]["actor_extent"]
                        all_actors_center = self.eval_all_frame[key]["actor_center"]
                        
                        relative_obj_id = value["input_data"]["object_ids"][0]
                        absolute_obj_id = self.get_absolute_obj(all_actors_pose)
                        range_obj_id = self.get_range_obj(all_actors_pose)
                        #relative_obj_id = self.get_relative_obj(range_obj_id, absolute_obj_id)
                        all_obj_id = self.get_all_obj(all_actors_pose)
                        
                        relative_obj_bbx_center, relative_obj_mask = self.generate_bbx_center(all_actors_pose, all_actors_extent, all_actors_center, relative_obj_id)
                        absolute_obj_bbx_center, absolute_obj_mask = self.generate_bbx_center(all_actors_pose, all_actors_extent, all_actors_center, absolute_obj_id)
                        range_obj_bbx_center, range_obj_mask = self.generate_bbx_center(all_actors_pose, all_actors_extent, all_actors_center, range_obj_id)
                        all_obj_bbx_center, all_obj_mask = self.generate_bbx_center(all_actors_pose, all_actors_extent, all_actors_center, all_obj_id)
                        
                        relative_obj_bbx_center = to_device(relative_obj_bbx_center, self.device)
                        relative_obj_mask = to_device(relative_obj_mask, self.device)
                        absolute_obj_bbx_center = to_device(absolute_obj_bbx_center, self.device)
                        absolute_obj_mask = to_device(absolute_obj_mask, self.device)
                        range_obj_bbx_center = to_device(range_obj_bbx_center, self.device)
                        range_obj_mask = to_device(range_obj_mask, self.device)
                        all_obj_bbx_center = to_device(all_obj_bbx_center, self.device)
                        all_obj_mask = to_device(all_obj_mask, self.device)

                        pred_box_tensor, pred_score = self.post_processor.post_process(value["output_dict"], transformation_matrix_torch, anchor_box)
                        relative_gt_box_tensor = self.post_processor.generate_gt_bbx(relative_obj_bbx_center, relative_obj_mask, transformation_matrix_torch)
                        absolute_gt_box_tensor = self.post_processor.generate_gt_bbx(absolute_obj_bbx_center, absolute_obj_mask, transformation_matrix_torch)
                        range_gt_box_tensor = self.post_processor.generate_gt_bbx(range_obj_bbx_center, range_obj_mask, transformation_matrix_torch)
                        all_gt_box_tensor = self.post_processor.generate_gt_bbx(all_obj_bbx_center, all_obj_mask, transformation_matrix_torch)
                        
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, relative_gt_box_tensor, relative_result_stat, 0.3)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, relative_gt_box_tensor, relative_result_stat, 0.5)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, relative_gt_box_tensor, relative_result_stat, 0.7)
                        
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, absolute_gt_box_tensor, absolute_result_stat, 0.3)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, absolute_gt_box_tensor, absolute_result_stat, 0.5)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, absolute_gt_box_tensor, absolute_result_stat, 0.7)
                        
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, range_gt_box_tensor, range_result_stat, 0.3)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, range_gt_box_tensor, range_result_stat, 0.5)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, range_gt_box_tensor, range_result_stat, 0.7)
                        
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, all_gt_box_tensor, all_result_stat, 0.3)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, all_gt_box_tensor, all_result_stat, 0.5)
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, all_gt_box_tensor, all_result_stat, 0.7)
                        
                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, short_result_stat, 0.3, left_range=0, right_range=30)
                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, short_result_stat, 0.5, left_range=0, right_range=30)
                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, short_result_stat, 0.7, left_range=0, right_range=30)

                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, middle_result_stat, 0.3, left_range=30, right_range=50)
                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, middle_result_stat, 0.5, left_range=30, right_range=50)
                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, middle_result_stat, 0.7, left_range=30, right_range=50)

                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, long_result_stat, 0.3, left_range=50, right_range=100)
                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, long_result_stat, 0.5, left_range=50, right_range=100)
                        eval_utils.caluclate_tp_fp_range(pred_box_tensor, pred_score, all_gt_box_tensor, long_result_stat, 0.7, left_range=50, right_range=100)
                        
                        if self.config["show_sequence"]:
                            vis_save_path = os.path.join(self.saved_path, 'vis')
                            if not os.path.exists(vis_save_path):
                                os.makedirs(vis_save_path)
                            vis_save_path = os.path.join(vis_save_path, '%05d.png' % key)
                            pcd, pred_o3d_box, gt_o3d_box = \
                                vis_utils.visualize_inference_sample_dataloader(
                                    pred_box_tensor,
                                    absolute_gt_box_tensor,
                                    value["input_data"]['origin_lidar'],
                                    self.vis_pcd,
                                    mode='constant')
                            if vis_idx == 0:
                                self.vis.add_geometry(pcd)
                            if not self.pred_flag:
                                if len(pred_o3d_box) > 0:
                                    vis_utils.linset_assign_list(
                                        self.vis,
                                        self.vis_aabbs_pred,
                                        pred_o3d_box,
                                        update_mode='add')
                                    self.pred_flag = True
                            if not self.gt_flag:
                                if len(gt_o3d_box) > 0:
                                    vis_utils.linset_assign_list(
                                        self.vis,
                                        self.vis_aabbs_gt,
                                        gt_o3d_box,
                                        update_mode='add')
                                    self.gt_flag = True
                            vis_idx += 1
                            if len(pred_o3d_box) > 0:
                                vis_utils.linset_assign_list(
                                    self.vis,
                                    self.vis_aabbs_pred,
                                    pred_o3d_box)
                                    
                            if len(gt_o3d_box) > 0:
                                vis_utils.linset_assign_list(
                                    self.vis,
                                    self.vis_aabbs_gt,
                                    gt_o3d_box)
                            self.vis.update_geometry(pcd)
                            self.vis.poll_events()
                            self.vis.update_renderer()
                            self.vis.capture_screen_image(vis_save_path)
                            time.sleep(0.001)
                        
                    eval_utils.eval_final_results(range_result_stat, self.saved_path, False, "range_result")
                    eval_utils.eval_final_results(relative_result_stat, self.saved_path, False, "relative_result")
                    eval_utils.eval_final_results(absolute_result_stat, self.saved_path, False, "absolute_result")
                    eval_utils.eval_final_results(all_result_stat, self.saved_path, False, "all_result")
                    eval_utils.eval_final_results(short_result_stat, self.saved_path, False, "short_range_result")
                    eval_utils.eval_final_results(middle_result_stat, self.saved_path, False, "middle_range_result")
                    eval_utils.eval_final_results(long_result_stat, self.saved_path, False, "long_range_result")
                    print("Ego data calculated times:", self.ego_data_num)
                    print("Cav data calculated times:", self.cav_data_num)
                    if self.config["show_sequence"]:
                        self.vis.destroy_window()
                    self.eval_ready_flag = False
                    del self.eval_remain[self.ego_config["name"]]
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
            lidar_np = mask_points_by_range(lidar_np, self.config["hypes_yaml"]['preprocess']['test_lidar_range'])
        if lidar_np.shape[0] == 0:
            return lidar_np, lidar_np, False
        else:
            processed_lidar = self.pre_processor.preprocess(lidar_np)
            return processed_lidar, lidar_np, True
        
        
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
            delay_ego_lidar_pose = selected_cav_base["received_ego_pose"][self.ego_config["name"]]
            object_ids = selected_cav_base["label"]
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose, ego_pose)
            velocity = selected_cav_base["speed"]
            velocity = velocity / 30 # normalize veloccity by average speed 30 km/h
            selected_cav_processed.update(
                {'object_ids': object_ids,
                 'spatial_correction_matrix': spatial_correction_matrix,
                 'velocity': velocity})
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
        cav_processed_feature, projected_lidar, processed_flag = self.get_processed_lidar(selected_cav_base["lidar_data"], ego_pose, selected_cav_base["lidar_pose"])
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
                 'projected_lidar': projected_lidar,
                 'velocity': velocity})
            return selected_cav_processed
        else:
            return selected_cav_processed
        
        
    def collate_batch_train(self, batch):
        output_dict = {}
        object_ids = []
        processed_lidar_list = []
        record_len = []
        
        velocity = []
        time_delay = []
        infra = []
        
        spatial_correction_matrix_list = []
        
        if self.config["show_sequence"]:
            origin_lidar = []
        
        for i in range(len(batch)):
            input_dict = batch[i]
            object_ids.append(input_dict['object_ids'])
            
            processed_lidar_list.append(input_dict['processed_lidar'])
            record_len.append(input_dict['cav_num'])
            velocity.append(input_dict['velocity'])
            time_delay.append(input_dict['time_delay'])
            infra.append(input_dict['infra'])
            spatial_correction_matrix_list.append(input_dict['spatial_correction_matrix'])
            if self.config["show_sequence"]:
                origin_lidar.append(input_dict['origin_lidar'])
        
        merged_batch_features_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_batch_features_dict)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = torch.from_numpy(np.array(spatial_correction_matrix_list))
        prior_encoding = torch.stack([velocity, time_delay, infra], dim=-1).float()
        output_dict.update({
            'processed_lidar': processed_lidar_torch_dict,
            'record_len': record_len,
            'object_ids': object_ids,
            'prior_encoding': prior_encoding,
            'spatial_correction_matrix': spatial_correction_matrix_list})
            
        if self.config["show_sequence"]:
            origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict.update({'origin_lidar': origin_lidar})
            
        return output_dict
        
        
    def get_object_info(self, obj_pose, ego_pose, extent):
        object2lidar = x1_to_x2(obj_pose, ego_pose)
        bbx = box_utils.create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]
        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = box_utils.corner_to_center(bbx_lidar, order=self.order)
        
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
                    npc_transform = npc.get_transform()
                    npc_extent = npc.bounding_box.extent
                    npc_center = npc.bounding_box.location
                    npc_pose = [npc_transform.location.x + npc_center.x,
                                npc_transform.location.y + npc_center.y,
                                npc_transform.location.z + npc_center.z,
                                npc_transform.rotation.roll,
                                npc_transform.rotation.yaw,
                                npc_transform.rotation.pitch]
                    extent = [npc_extent.x, npc_extent.y, npc_extent.z]
                    npc_bbx_lidar = self.get_object_info(npc_pose, ego_pose_now, extent)
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
        
        
    def get_absolute_obj(self, all_actors_pose):
        obj_id_list = []
        ego_pose = all_actors_pose[self.ego_actor_id].copy()
        ego_pose[2] += 2.4
        for key, value in all_actors_pose.items():
            if True:
                distance = compute_distance3D(value, ego_pose)
                if distance < self.config["hypes_yaml"]["absolute_range"]:
                    obj_id_list.append(key)
        return obj_id_list
        
    def get_range_obj(self, all_actors_pose):
        obj_id_list = []
        ego_pose = all_actors_pose[self.ego_actor_id].copy()
        ego_pose[2] += 2.4
        for key, value in all_actors_pose.items():
            if True:
                distance = compute_distance3D(value, ego_pose)
                if distance < int(self.ego_config["lidar_set"]["range"]):
                    obj_id_list.append(key)
        return obj_id_list
        
    def get_relative_obj(self, range_obj_list, absolute_obj_list):
        obj_id_list = []
        if len(absolute_obj_list) > 0:
            for obj in absolute_obj_list:
                if obj not in range_obj_list:
                    obj_id_list.append(obj)
        return obj_id_list
        
    def get_all_obj(self, all_actors_pose):
        obj_id_list = []
        for key, value in all_actors_pose.items():
            if True:
                obj_id_list.append(key)
        return obj_id_list
        
        
    def generate_bbx_center(self, all_actors_pose, all_actors_extent, all_actors_center, target_actors_id):
        object_bbx_center = []
        object_bbx_mask = []
        output_dict = {}
        object_center = np.zeros((self.max_num, 7))
        mask = np.zeros(self.max_num)
        ego_pose = all_actors_pose[self.ego_actor_id].copy()
        ego_pose[2] += 2.4
        for target_id in target_actors_id:
            if target_id in all_actors_pose.keys():
                target_initial_pose = all_actors_pose[target_id]
                target_extent = all_actors_extent[target_id]
                target_center = all_actors_center[target_id]
                target_pose = [target_initial_pose[0] + target_center[0],
                               target_initial_pose[1] + target_center[1],
                               target_initial_pose[2] + target_center[2],
                               target_initial_pose[3], target_initial_pose[4], target_initial_pose[5]]
                target_bbx_lidar = self.get_object_info(target_pose, ego_pose, target_extent)
                if target_bbx_lidar.shape[0] > 0:
                    output_dict.update({target_id: target_bbx_lidar})
        for i, (target_id, target_bbx) in enumerate(output_dict.items()):
            object_center[i] = target_bbx[0, :]
            mask[i] = 1
        object_center_stack = object_center[mask == 1]
        object_bbx_center.append(object_center)
        object_bbx_mask.append(mask)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        
        return object_bbx_center, object_bbx_mask
    
