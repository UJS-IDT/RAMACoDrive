# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import time
import torch
import multiprocessing
from multiprocessing import shared_memory
import pickle
from data.commuicate_manager import CommuniAgent
from agent.lidarAgent import LidarAgent
import numpy as np
import matplotlib.pyplot as plt
from data_fusion import build_data
import sys
from util import spawn_vehicle, connect_to_server, time_const, thread_process_vehicles, get_ego_vehicle, get_speed, log_time_cost, get_vehicle_info, x_to_world, project_points_by_matrix_torch
from shapely.geometry import Polygon, Point

class V2X_communicator(multiprocessing.Process):
    def __init__(self, all_config, config, receive_mutex, send_mutex):
        super().__init__()
        self.config = config
        self.receive_mutex = receive_mutex
        self.send_mutex = send_mutex
        self.running_event = multiprocessing.Event()
        self.mem_addr = shared_memory.SharedMemory(name=self.config["share_mem"])
        self.send_mem_addr = shared_memory.SharedMemory(name=self.config["train_share_mem"])
        self.lidar_agent = LidarAgent(self.config["name"], self.config["inter_port"])
        self.last_timestamp = -1.
        self.ego_data = None
        self.ego_flag = False
        self.initial_flag = False
        self.receivers = {}
        self.data_buffer = []
        self.cav_actor = None
        self.lidar_range = int(self.config["lidar_set"]["range"])
        self.scene_size = self.config["hypes_yaml"]["data_fusion"]["scene_size"]
        self.data_process = build_data(self.config["name"], self.config["hypes_yaml"], visualize=False, train=True)

    def run(self):
        self.client, self.world = connect_to_server(1000, 2000)
        self.lidar_agent.start_agent()
        if "cav" in self.config["name"]:
            self.world_actors = self.world.get_actors().filter('*vehicle*')
            self.agent_type = "vehicle"
        else:
            self.world_actors = self.world.get_actors().filter('*streetsign*')
            self.agent_type = "road"
        for npc in self.world_actors:
            if npc.attributes["role_name"] == self.config["name"]:
                self.cav_actor = npc
        while True:
            if self.running_event.is_set():
                try:
                    if self.mem_addr is not None and self.mem_addr.buf is not None:
                        self.receive_mutex.acquire()
                        s = bytes(self.mem_addr.buf[:20])
                        index = s.find(b"~")
                        if index != -1:
                            head = s[0:index]
                            contentlength = int(head)
                            content = bytes(self.mem_addr.buf[index + 1:index + 1 + contentlength])
                            #self.receive_mutex.release()
                            data = pickle.loads(content)
                            self.receive_mutex.release()
                            if data["timestamp"] != self.last_timestamp:
                                temp_dict = {}
                                self.adjacent_vehicles = {}
                                self.label_vehicles = []
                                self.acc_track_dict = {}
                                self.last_timestamp = data["timestamp"]
                                temp_dict["lidar_data"] = data["lidar_data"]
                                temp_dict["timestamp"] = data["timestamp"]
                                temp_dict["lidar_pose"] = data["lidar_pose"]
                                temp_dict["agent_type"] = self.agent_type
                                #start_time = time.time()
                                if "cav" in self.config["name"]:
                                    self.speed = get_speed(self.cav_actor)
                                else:
                                    self.speed = 0.0
                                temp_dict["speed"] = self.speed
                                self.world_actors = self.world.get_actors().filter('*vehicle*')
                                for npc in self.world_actors:
                                    if npc.id != self.cav_actor.id:
                                        dist = npc.get_transform().location.distance(self.cav_actor.get_transform().location)
                                        if dist < self.lidar_range:
                                            bb = npc.bounding_box
                                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                            polygon = self.bounding_box_to_polygon(verts)
                                            self.adjacent_vehicles[npc.id] = polygon
                                temp_dict["adjacent_vehicles"] = self.adjacent_vehicles
                                if self.config["acc_track"]:
                                #if self.config["ego_flag"]:
                                    #self.world_lidar_data = self.transform_lidar_to_world(data["lidar_data"], data["lidar_pose"])
                                    self.transform_matrix = x_to_world(data["lidar_pose"])
                                    self.world_lidar_data = project_points_by_matrix_torch(data["lidar_data"][:, :3], self.transform_matrix)
                                    for key, value in self.adjacent_vehicles.items():
                                        if self.check_point_cloud_intersection(self.world_lidar_data, value):
                                            self.label_vehicles.append(key)
                                    temp_dict["label"] = self.label_vehicles
                                else:
                                    for key, value in self.adjacent_vehicles.items():
                                        self.label_vehicles.append(key)
                                    temp_dict["label"] = self.label_vehicles
                                self.data_buffer.append(temp_dict)
                        else:
                            self.receive_mutex.release()
                    else:
                        pass
                    if len(self.data_buffer) >= self.scene_size:
                        self.data_buffer = self.data_process.get_single_lidar_feature(self.data_buffer)
                        self.send_lidar_feature(self.data_buffer)
                        self.data_buffer = []
                except Exception as e:
                    print(e)
                    pass
                else:
                    pass
            else:
                time.sleep(0.05)
                self.last_timestamp = -1.
                if self.mem_addr:
                    self.mem_addr.close()
                if self.send_mem_addr:
                    self.send_mem_addr.close()
                self.lidar_agent.close_agent()
                self.mem_addr = None
                self.send_mem_addr = None
                self.ego_data = None
                self.ego_flag = False
                self.receivers = {}
                self.initial_flag = False
                sys.exit()
                
                
    def transform_lidar_to_world(self, lidar_points, lidar_transform):
        tx, ty, tz = lidar_transform[0], lidar_transform[1], lidar_transform[2]
        roll, yaw, pitch = np.deg2rad([lidar_transform[3], lidar_transform[4], lidar_transform[5]])

        lidar_points_3d = lidar_points[:, :3]
        R_pitch = np.array([[1, 0, 0],
                            [0, np.cos(pitch), -np.sin(pitch)],
                            [0, np.sin(pitch), np.cos(pitch)]])

        R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw), np.cos(yaw), 0],
                          [0, 0, 1]])

        R_roll = np.array([[np.cos(roll), 0, np.sin(roll)],
                           [0, 1, 0],
                           [-np.sin(roll), 0, np.cos(roll)]])

        R = R_yaw @ R_pitch @ R_roll

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = [tx, ty, tz]

        lidar_points_homogeneous = np.hstack((lidar_points_3d, np.ones((lidar_points_3d.shape[0], 1))))
    
        world_points_homogeneous = lidar_points_homogeneous @ transform_matrix.T

        world_points = world_points_homogeneous[:, :3]

        return world_points
        
    def bounding_box_to_polygon(self, verts):
        polygon_points = [(vert.x, vert.y) for vert in verts]
        polygon = Polygon(polygon_points)
        return polygon

    def check_point_cloud_intersection(self, point_cloud, polygon):
        for point in point_cloud:
            point_xy = Point(point[0], point[1])
            if polygon.contains(point_xy):
                return True
        return False

    
    def send_lidar_feature(self, lidar_feature_list):
        # Memory Share
        obj_data = pickle.dumps(lidar_feature_list)
        obj_len = len(obj_data)
        metadata = str(obj_len) + "~"
        byte_data = metadata.encode() + obj_data
        self.send_mutex.acquire()
        if self.send_mem_addr.buf is not None:
            self.send_mem_addr.buf[:len(byte_data)] = byte_data
        self.send_mutex.release()
        
        # ZMQ Trans
        self.lidar_agent.lidar_agent.send_obj(lidar_feature_list)






