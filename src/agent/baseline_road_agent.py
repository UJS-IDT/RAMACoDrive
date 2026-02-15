# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import sys
import json
import pickle
from agent.baseAgent import BaseAgent
from agent.lidarAgent import LidarAgent
#from module_util.baseline_v2x_agent import V2X_communicator
from data.commuicate_manager import CommuniAgent
from util import connect_to_server, spawn_road, time_const, is_within_distance, compute_distance, log_time_cost, txt_to_points, create_v2x
from view.debug_manager import draw_waypoints_arraw, draw_transforms, set_bird_view
import open3d as o3d
from perception.sensor_manager import SensorManager
from cythoncode.router_baseline import GlobalRoutePlanner
from cythoncode.controller_baseline import VehiclePIDController
from plan.planer_baseline import FrenetPlanner
import carla
import logging
import time
import math
import random
import cv2
import numpy as np
import copy
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing import Manager as manager

class BaselineRoadAgent(BaseAgent):
    def __init__(self, all_config, config, cof_dic, send_mutex, model):
        self.config = config
        self.all_config = all_config
        self.send_mutex = send_mutex
        self.model = model
        self.cof_dic = cof_dic
        self.stop_process = manager().dict()
        BaseAgent.__init__(
            self, self.config["name"], self.config["port"])
        self.count = 0
        self.vis = None
        self.road_dic = {}
        try:
            self.shared_mem = shared_memory.SharedMemory(name=self.config["share_mem"], create=True, size=1000000000)
        except FileExistsError:
            print("Shared memory exists.")
            self.shared_mem = shared_memory.SharedMemory(name=self.config["share_mem"])
        #self.data_process = build_data(self.config["name"], self.config["hypes_yaml"], visualize=False, train=True)
        self.data_buffer = []
        self.loc_buffer = []
        self.time_buffer = []
        self.scene_size = self.config["hypes_yaml"]["data_fusion"]["scene_size"]
        self.processed_feature = []
        
    def run(self):
        @time_const(fps=self.config["fps"])
        def run_step(world):
            if len(self.cof_dic) == 0 or self.communi_agent.rec_obj("main") == "end":
                self.stop_process[self.config["name"]] = True
                self.v2x_communicator.running_event.clear()
                while self.config["name"] in self.stop_process:
                    pass
                print(self.config["name"], " is closed")
                self.shared_mem.close()
                self.shared_mem.unlink()
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
                self.close_agent()
                self.communi_agent.close()
                time.sleep(2)
                sys.exit()

        client, world = connect_to_server(1000, 2000)
        map = world.get_map()
        self.start_agent()
        self.set_communi_agent()
        self.road = self.create_road(world, self.config["spawn_location"], self.config["type"])

        self.road_info = self.init_road_info()
        self.road_dic[self.road.id] = {"road": self.road, "info": self.road_info}
        self.mutex = multiprocessing.Lock()
        self.sensor_manager = SensorManager(
            world, self.road, self.road_info, self.config, self.shared_mem, self.mutex)
        #self.v2x_communicator = V2X_communicator(self.config, self.mutex, self.send_mutex)
        self.v2x_communicator = create_v2x(self.config["hypes_yaml"], self.all_config, self.config, self.mutex, self.send_mutex, self.model, self.stop_process)
        self.v2x_communicator.start()
        self.v2x_communicator.running_event.set()

        try:
            while True:
                run_step(world)
        except Exception as e:
            logging.error(f"ego vehicle agent error:{e}")
            print(e.__traceback__.tb_frame.f_globals["__file__"])
            print(e.__traceback__.tb_lineno)
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            self.close_agent()
            sys.exit()

    def update_lidar(self):
        if len(self.sensor_manager.raw_lidar_data_window) != 0:
            _, latest_data = self.sensor_manager.raw_lidar_data_window[-1]
            latest_pose = np.array([
                latest_data.transform.location.x,
                latest_data.transform.location.y,
                latest_data.transform.location.z,
                latest_data.transform.rotation.roll,
                latest_data.transform.rotation.yaw,
                latest_data.transform.rotation.pitch
            ])
            all_lidar_data = np.concatenate([
                np.frombuffer(_data.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
                for _, _data in list(self.sensor_manager.raw_lidar_data_window)], axis=0)
            
            # Memory Share
            flattened_lidar_data = all_lidar_data.flatten()
            flattened_pose = latest_pose.flatten()
            combined_data = np.concatenate((flattened_lidar_data, flattened_pose))
            metadata = np.array([latest_data.timestamp, len(flattened_lidar_data), len(flattened_pose)], dtype=combined_data.dtype)
            data_with_metadata = np.concatenate((metadata, combined_data))
            shared_array = np.ndarray(data_with_metadata.shape, dtype=data_with_metadata.dtype, buffer=self.shared_mem.buf)
            np.copyto(shared_array, data_with_metadata)

            # ZMQ Trans
            lidar_data_with_pose = (all_lidar_data, latest_pose)
            self.lidar_agent.lidar_agent.send_obj(lidar_data_with_pose)
    
    def update_intermediate_lidar(self):
        if len(self.sensor_manager.raw_lidar_data_window) != 0:
            _, latest_data = self.sensor_manager.raw_lidar_data_window[-1]
            latest_pose = np.array([
                latest_data.transform.location.x,
                latest_data.transform.location.y,
                latest_data.transform.location.z,
                latest_data.transform.rotation.roll,
                latest_data.transform.rotation.yaw,
                latest_data.transform.rotation.pitch
            ])
            all_lidar_data = np.concatenate([
                np.frombuffer(_data.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
                for _, _data in list(self.sensor_manager.raw_lidar_data_window)], axis=0)
            self.time_buffer.append(time.time())
            self.data_buffer.append(all_lidar_data)
            self.loc_buffer.append(latest_pose)
    
    def send_lidar_feature(self, lidar_feature, cav_location, timestamp):
        for idx in range(len(lidar_feature)):
            lidar_feature[idx]["cav_location"] = cav_location[idx]
            lidar_feature[idx]["timestamp"] = timestamp[idx]

        # Memory Share
        obj_data = pickle.dumps(lidar_feature)
        obj_len = len(obj_data)
        metadata = str(obj_len) + "~"
        byte_data = metadata.encode() + obj_data
        self.shared_mem.buf[:len(byte_data)] = byte_data
        
        # ZMQ Trans
        self.lidar_agent.lidar_agent.send_obj(lidar_feature)
    
    def update_visualization(self):
        if len(self.sensor_manager.raw_lidar_data_window) != 0:
            all_data = np.concatenate([np.frombuffer(_data.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
                                   for _, _data in self.sensor_manager.raw_lidar_data_window], axis=0)
            lidar_viz = self.visualize_data(all_data)
            cv2.imshow('vizs', lidar_viz)
            cv2.waitKey(1)

    # @log_time_cost(name="ego_vehicle_agent")
    def set_communi_agent(self):
        self.communi_agent.init_subscriber("router",
                                           self.config["traffic_agent_port"])


    def create_road(self, world, spawn_location, road_type):
        try:
            spawn_actor = spawn_road(
                world, road_type, spawn_location, hero=True, name=self.config["name"])
            while spawn_actor is None:
                logging.info(
                    f"spawn_actor{road_type} failed, trying another start point...")
            return spawn_actor
        except Exception as e:
            logging.error(f"create road device error:{e}")
            raise
            

    def init_road_info(self):
        r_length = self.road.bounding_box.extent.x
        r_widht = self.road.bounding_box.extent.y
        r_high = self.road.bounding_box.extent.z
        return {"width": r_widht, "length": r_length, "height": r_high}

    def get_trajection(self):
        return self.trajection

    def get_vehicle(self):
        return self.vehicle

    def visualize_data(self, lidar, text_args=(0.6)):
        lidar_viz = self.lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz, cv2.COLOR_GRAY2RGB)

        return lidar_viz

    def lidar_to_bev(self, lidar, min_x=-100, max_x=100, min_y=-100, max_y=100, pixels_per_meter=4,
                     hist_max_per_pixel=2):
        xbins = np.linspace(
            min_x, max_x + 1,
                   (max_x - min_x) * pixels_per_meter + 1,
        )
        ybins = np.linspace(
            min_y, max_y + 1,
                   (max_y - min_y) * pixels_per_meter + 1,
        )
        # Compute histogram of x and y coordinates of points.
        hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
        # Clip histogram
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        # Normalize histogram by the maximum number of points in a bin we care about.
        overhead_splat = hist / hist_max_per_pixel * 255.
        # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
        return overhead_splat[::-1, :]
