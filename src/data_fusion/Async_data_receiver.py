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
import sys
from util import spawn_vehicle, connect_to_server, time_const, thread_process_vehicles, get_ego_vehicle, get_speed, log_time_cost, get_vehicle_info, x_to_world, project_points_by_matrix_torch, x1_to_x2, to_device
import traceback

class Async_receiver(multiprocessing.Process):
    def __init__(self, all_config, config, send_mutex, cof_seg):
        super().__init__()
        self.config = config
        self.all_config = all_config
        self.cof_seg = cof_seg
        self.cav_async_data_received = {}
        self.cav_config = self.generate_cav_config()
        self.send_mutex = send_mutex
        self.running_event = multiprocessing.Event()
        self.exit_event = multiprocessing.Event()        
        send_mem_addr_name = "Async_receiver_to_" + self.config["name"]
        try:
            self.send_mem_addr = shared_memory.SharedMemory(name=send_mem_addr_name, create=True, size=10000000000)
        except FileExistsError:
            print("Shared memory exists.")
            self.send_mem_addr = shared_memory.SharedMemory(name=send_mem_addr_name)
        self.last_timestamp = -1.
        self.initial_flag = False
        self.receivers = {}
        

    def run(self):
        while True:
            if self.running_event.is_set():
                try:
                    if not self.initial_flag:
                        self.initial_cav_comm()
                        self.initial_flag = True
                    #self.cav_async_data_received = {}
                    self.get_cav_lidar()
                    self.data_to_main_process()
                except Exception as e:
                    #print(e)
                    traceback.print_exc()                    
                    pass
                else:
                    pass
            else:
                time.sleep(0.05)
                self.last_timestamp = -1.
                if self.send_mem_addr:
                    self.send_mem_addr.buf[:] = b'\0' * self.send_mem_addr.size
                self.receivers = {}
                self.initial_flag = False
                self.cav_async_data_received = {}
                if self.exit_event.is_set():
                    self.send_mem_addr.close()
                    self.send_mem_addr.unlink()
                    self.send_mem_addr = None
                    sys.exit()
                
    
    def data_to_main_process(self):
        if len(self.cav_async_data_received) > 0:
            # Memory Share
            obj_data = pickle.dumps(self.cav_async_data_received)
            obj_len = len(obj_data)
            metadata = str(obj_len) + "~"
            byte_data = metadata.encode() + obj_data
            self.send_mutex.acquire()
            if self.send_mem_addr.buf is not None:
                self.send_mem_addr.buf[:len(byte_data)] = byte_data
            self.send_mutex.release()
        


    def generate_cav_config(self):
        cav_config = []
        for agent in self.all_config["agents"]:
            if agent["name"] != self.config["name"]:
                cav_config.append(agent)
        for agent in self.all_config["road_agents"]:
            if agent["name"] != self.config["name"]:
                cav_config.append(agent)
        return cav_config


    def initial_cav_comm(self):
        for cav in self.cav_config:
            receiver = CommuniAgent("receiver" + cav["name"])
            receiver.init_v2x_subscriber(cav["v2x_name"], cav["inter_port"], self.all_config["comm_trans_addr"])
            self.receivers[cav['name']] = {}
            self.receivers[cav['name']]["receiver"] = receiver
            self.receivers[cav['name']]["v2x_name"] = cav["v2x_name"]


    def get_cav_lidar(self):
        data_flag = False
        for key, value in self.receivers.items():
            try:
                if "cav" in key:
                    if key not in self.cof_seg.keys():
                        if key in self.cav_async_data_received.keys():
                            del self.cav_async_data_received[key]
                        continue
                cav_receiver = self.receivers[key]["receiver"]
                received_data = cav_receiver.receive_latest_message(self.receivers[key]["v2x_name"])
                if received_data is not None:
                    self.cav_async_data_received[key] = received_data
            except Exception as e:
                print("An error occurred:", e)
                pass
            else:
                pass


