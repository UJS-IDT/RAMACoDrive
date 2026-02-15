# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>

import os
from tools.world_manager import WorldManager
from tools.config_manager import config as cfg
from view.debug_manager import DebugManager, set_bird_view
#from tools.async_training import Trainer
from tools.input_manager import recieve_args
from agent.traffic_agent import TrafficFlowManager
from data.commuicate_manager import CommuniAgent
from data.recorder_manager import DataRecorder
from util import destroy_all_actors, time_const, log_time_cost, create_trainer, create_model, create_loss, setup_train, load_saved_model, to_device
import time
import torch
import multiprocessing
from multiprocessing import shared_memory
import logging
from multiprocessing import Manager as manager
from tools.loader import load_agents, load_batch_agents, load_conventional_agents, load_agents_shuffle, load_record_agents, load_road_agents
import sys
import csv
import carla


class ScenarioReplayer:
    def __init__(self, csv_file_path, world, record_remain, config, cof_seg, mutex_lockers, cof_load, record_frames, model, record_speed=1):
        self.csv_file_path = csv_file_path
        self.record_speed = record_speed
        self.world = world
        self.record_frames = record_frames
        self.model = model
        self.vehicles = {}
        self.vehicle_blueprints = {}
        self.active_vehicles = set()
        self.record_remain = record_remain
        self.config = config
        self.cof_seg = cof_seg
        self.mutex_lockers = mutex_lockers
        self.cof_load = cof_load
        self.all_frame = []
        self.read_frames()
        
    def read_frames(self):
        speed_idx = self.record_speed
        last_time = -1
        with open(self.csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            this_frame = []
            for row in reader:
                time_step = int(row['time'])
                if time_step != last_time:
                    last_time = time_step
                    if speed_idx == self.record_speed:
                        self.record_frames.append(time_step)
                        if len(this_frame) == 0:
                            this_frame.append(row)
                        else:
                            self.all_frame.append(this_frame)
                            this_frame = []
                            this_frame.append(row)
                    speed_idx += 1
                    if speed_idx > self.record_speed:
                        speed_idx = 1
                else:
                    if speed_idx == self.record_speed:
                        this_frame.append(row)
            self.all_frame.append(this_frame)

    def replay(self, frame_id):
        last_time = frame_id
        last_real_time = None
        if True:
            this_frame = self.all_frame[frame_id]
            for row in this_frame:
                time_step = int(row['time'])
                real_time = float(row['real_time'])

                vehicle_id = row['vehicle_id']
                if vehicle_id not in self.active_vehicles:
                    self.active_vehicles.add(vehicle_id)

                vehicle_type = row['vehicle_type']
                vehicle_pose = [float(row['location_x']),
                                float(row['location_y']),
                                float(row['location_z']),
                                float(row['rotation_roll']),
                                float(row['rotation_yaw']),
                                float(row['rotation_pitch'])]
                vehicle_extent = [float(row['extent_x']), float(row['extent_y']), float(row['extent_z'])]
                vehicle_center = [float(row['center_x']), float(row['center_y']), float(row['center_z'])]

                location = carla.Location(float(row['location_x']), float(row['location_y']),
                                          float(row['location_z']) + 1.0)  # Raise height to avoid collision
                rotation = carla.Rotation(float(row['rotation_pitch']), float(row['rotation_yaw']),
                                          float(row['rotation_roll']))
                transform = carla.Transform(location, rotation)

                if vehicle_id not in self.vehicles:
                    if vehicle_type not in self.vehicle_blueprints:
                        self.vehicle_blueprints[vehicle_type] = self.world.get_blueprint_library().find(vehicle_type)
                    blueprint = self.vehicle_blueprints[vehicle_type]
                    blueprint.set_attribute('role_name', vehicle_id)
                    vehicle = self.world.try_spawn_actor(blueprint, transform)
                    if vehicle:          
                        self.vehicles[vehicle_id] = vehicle
                        # Drop the vehicle to the ground
                        location.z -= 1.0
                        vehicle.set_transform(carla.Transform(location, rotation))
                    else:
                        print(f"Failed to spawn vehicle {vehicle_id} at {transform}")
                else:
                    vehicle = self.vehicles[vehicle_id]
                    location.z -= 1.0
                    vehicle.set_transform(carla.Transform(location, rotation))
                
                if vehicle_id in self.record_remain:
                    for i, agent_info in enumerate(self.config["agents"]):
                        if agent_info["name"] == vehicle_id:
                            print("loading.............." + vehicle_id)
                            load_record_agents(self.config, self.cof_seg, self.mutex_lockers, self.cof_load, agent_info, self.model)
                            if len(self.record_remain) > 1:
                                del self.record_remain[vehicle_id]
                            else:
                                del self.record_remain[vehicle_id]
                                while len(self.cof_load) > 0:
                                    pass
                if frame_id != 0:
                    self.apply_control(vehicle, row)
                actor_id = vehicle.id
            self.record_frames.pop(0)
            self.update_active_vehicles()

    def apply_control(self, vehicle, row):
        control = carla.VehicleControl()
        control.brake = float(row['control_brake']) * 0.8
        control.throttle = float(row['control_throttle']) * 0.8
        control.steer = float(row['control_steer']) * 0.8
        vehicle.apply_control(control)

    def update_active_vehicles(self):
        current_vehicle_ids = set(self.vehicles.keys())
        for vehicle_id in current_vehicle_ids:
            if vehicle_id not in self.active_vehicles:
                self.vehicles[vehicle_id].destroy()
                del self.vehicles[vehicle_id]
                if vehicle_id in self.cof_seg:
                    del self.cof_seg[vehicle_id]
        self.active_vehicles.clear()

    def destroy_all(self):
        for vehicle in self.vehicles.values():
            vehicle.destroy()


class UnifiedSignalSystem:
    def __init__(self, config, world, traffic_manager, trainers, cof_seg, main_com, save_flag=False, eval_flag=False,
                 replayer=None, record_remain=None, record_frames=None):
        self.config = config
        self.world = world
        self.TM = traffic_manager
        self.trainers = trainers
        self.save_flag = save_flag
        self.eval_flag = eval_flag
        self.cof_seg = cof_seg
        self.main_com = main_com
        self.replayer = replayer
        self.record_remain = record_remain
        self.record_frames = record_frames
        self.start_time = time.time()
        self.reset_time = int(self.config["hypes_yaml"]["reset_time"])
        self.frame_id = 0

    def start(self):
        @time_const(fps=self.config["fps"])
        def run_step():
            if self.replayer is not None:
                self.replayer.replay(self.frame_id)
            self.world.tick()
        try:
            for name, trainer in self.trainers.items():
                if self.eval_flag:
                    trainer.eval_event.set()
                else:
                    if self.save_flag:
                        trainer.save_event.set()
                    trainer.scheduler_event.set()
                trainer.training_event.set()
            if self.eval_flag:
                while len(self.record_frames) > 0:
                    run_step()
                    self.frame_id += 1
            else:
                while len(self.cof_seg) != 0:
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    if elapsed_time > self.reset_time:
                        print("Exit the loop")
                        break
                    run_step()
        except Exception as e:
            logging.error(f"main error:{e}")
        finally:
            self.main_com.send_obj("end")
            for name, trainer in self.trainers.items():
                trainer.training_event.clear()
            time.sleep(10)
            if self.eval_flag:
                self.record_remain.clear()
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.TM.set_synchronous_mode(False)
            return


def MainSystem():
    args = recieve_args()
    config = cfg.merge(args)
    config["hypes_yaml"]["model"]["args"]["training_flag"] = True
    rest_epoches = config['rest_epoches']
    traffic_light = TrafficFlowManager()
    traffic_light.start()
    main_com = MainCommuicator(config)
    trainers = {}
    mutex_lockers = {}
    shared_mems = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cof_seg = manager().dict()
    cof_load = manager().dict()
    record_remain = {}
    record_remain_copy = manager().dict()
    for agent_config in config["agents"]:
        agent_mutex = multiprocessing.Lock()
        mutex_lockers[agent_config["name"]] = agent_mutex
        record_remain[agent_config["name"]] = True
        try:
            shared_mem = shared_memory.SharedMemory(name=agent_config["train_share_mem"], create=True, size=100000000)
        except FileExistsError:
            print("Shared memory exists.")
            shared_mem = shared_memory.SharedMemory(name=agent_config["train_share_mem"])
        
        shared_mems[agent_config["name"]] = shared_mem
        if agent_config["ego_flag"]:
            max_cav_num = config["hypes_yaml"]['train_params']['max_cav']
            model = create_model(config["hypes_yaml"], max_cav_num)
            if config["model_dir"]:
                saved_path = config["model_dir"] + "/" + agent_config["name"] + "/"
                init_epoch, model = load_saved_model(saved_path, model)
            else:
                init_epoch = 0
                saved_path = setup_train(config["hypes_yaml"], agent_config["name"])
            trainer = create_trainer(agent_config, config, agent_mutex, cof_seg, model, init_epoch, saved_path)
            trainer.start()
            trainers[agent_config["name"]] = trainer
            
    for road_agent_config in config["road_agents"]:
        road_agent_mutex = multiprocessing.Lock()
        mutex_lockers[road_agent_config["name"]] = road_agent_mutex
        try:
            shared_mem = shared_memory.SharedMemory(name=road_agent_config["train_share_mem"], create=True, size=100000000)
        except FileExistsError:
            print("Shared memory exists.")
            shared_mem = shared_memory.SharedMemory(name=road_agent_config["train_share_mem"])
            
        shared_mems[road_agent_config["name"]] = shared_mem
        if road_agent_config["ego_flag"]:
            max_cav_num = config["hypes_yaml"]['train_params']['max_cav']
            model = create_model(config["hypes_yaml"], max_cav_num)
            if config["model_dir"]:
                saved_path = config["model_dir"] + "/" + agent_config["name"] + "/"
                init_epoch, model = load_saved_model(saved_path, model)
            else:
                init_epoch = 0
                saved_path = setup_train(config["hypes_yaml"], agent_config["name"])
            trainer = create_trainer(road_agent_config, config, road_agent_mutex, cof_seg, model, init_epoch, saved_path)
            trainer.start()
            trainers[road_agent_config["name"]] = trainer
            
    for i in range(0, rest_epoches):
        print("Now is the Loop", i)
        save_flag = False
        eval_flag = False
        if i % config['hypes_yaml']['train_params']['save_freq'] == 0:
            save_flag = True
        cof_seg.clear()
        cof_load.clear()
        if config['hypes_yaml']["fusion_mode"] == "intermediate" or config['hypes_yaml']["fusion_mode"] == "late":
            if i == 0:
                model = create_model(config["hypes_yaml"], max_cav_num)
            else:
                init_epoch, model = load_saved_model(saved_path, model)
                #model.to(device)
            body(config, main_com, trainers, mutex_lockers, cof_seg, cof_load, save_flag, model)
        else:
            body(config, main_com, trainers, mutex_lockers, cof_seg, cof_load, save_flag)
        
        
        if i % config['hypes_yaml']['train_params']['eval_freq'] == 0:
            eval_flag = True
            save_flag = False
            record_remain_copy.clear()
            for key, value in record_remain.items():
                record_remain_copy[key] = value
            if config['hypes_yaml']["fusion_mode"] == "intermediate" or config['hypes_yaml']["fusion_mode"] == "late":
                if i == 0:
                    model = create_model(config["hypes_yaml"], max_cav_num)
                else:
                    init_epoch, model = load_saved_model(saved_path, model)
                body_eval(config, main_com, trainers, mutex_lockers, cof_seg, cof_load, eval_flag, record_remain_copy, model)
            else:
                body_eval(config, main_com, trainers, mutex_lockers, cof_seg, cof_load, eval_flag, record_remain_copy)

    main_com.close()
    for name, trainer in trainers.items():
        trainer.terminate()
        trainer.join()
    for cav_name, cav_shared_mem in shared_mems.items():
        cav_shared_mem.close()
        cav_shared_mem.unlink()
    logging.info("Simulation ended\n")
    for name, trainer in trainers.items():
        trainer.exit_event.set()
    sys.exit()

def body(config, main_com, trainers, mutex_lockers, cof_seg, cof_load, save_flag, model=None):
    world_manager = WorldManager(config)
    world = world_manager.get_world()
    TM = world_manager.get_traffic_manager()
    destroy_all_actors(world)
    main_com.send_obj("start")
    load_agents_shuffle(config, cof_seg, mutex_lockers, cof_load, model)
    while len(cof_load) != 0:
        pass
    load_conventional_agents(world, TM, config)
    
    data_recorder = DataRecorder(config)

    uss = UnifiedSignalSystem(config, world, TM, trainers, cof_seg, main_com, save_flag=save_flag)
    uss.start()
    destroy_all_actors(world)
    logging.info("step ended\n")
    return

def body_eval(config, main_com, trainers, mutex_lockers, cof_seg, cof_load, eval_flag, record_remain, model=None):
    world_manager = WorldManager(config)
    world = world_manager.get_world()
    TM = world_manager.get_traffic_manager()
    destroy_all_actors(world)
    main_com.send_obj("start")
    load_road_agents(config, cof_seg, mutex_lockers, cof_load, model)

    record_frames = manager().list()
    record_speed = int(config["record_speed"])
    replayer = ScenarioReplayer('../carla_record/data_record_scene_new7.csv', world, record_remain, config,
                                cof_seg, mutex_lockers, cof_load, record_frames, model, record_speed)
    replayer.destroy_all()

    uss = UnifiedSignalSystem(config, world, TM, trainers, cof_seg, main_com, eval_flag=eval_flag, replayer=replayer,
                              record_remain=record_remain, record_frames=record_frames)
    uss.start()
    destroy_all_actors(world)
    logging.info("step ended\n")
    return

def MainCommuicator(config):
    world_control_center = CommuniAgent("World")
    world_control_center.init_publisher(config["main_port"])
    return world_control_center


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    MainSystem()
    
