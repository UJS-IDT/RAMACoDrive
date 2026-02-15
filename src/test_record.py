# -*- coding: utf-8 -*-
# Author: Shunyao Zhang <ca19p@163.com>

import os
from tools.world_manager import WorldManager
from tools.config_manager import config as cfg
from view.debug_manager import DebugManager, set_bird_view
from tools.input_manager import recieve_args
from agent.traffic_agent import TrafficFlowManager
from data.commuicate_manager import CommuniAgent
from data.recorder_manager import DataRecorder
from util import destroy_all_actors, time_const, log_time_cost, create_tester, create_model, load_saved_model, setup_train, to_device, load_best_model
import csv
import carla
import time
import torch
import multiprocessing
from multiprocessing import shared_memory
import logging
from multiprocessing import Manager as manager
from tools.loader import load_agents, load_batch_agents, load_conventional_agents, load_road_agents, load_record_agents
import sys
import copy


class ScenarioReplayer:
    def __init__(self, csv_file_path, world, record_remain, config, cof_seg, mutex_lockers, cof_load, record_frames, eval_all_frame, model, record_speed=1):
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
        self.eval_all_frame = eval_all_frame
        
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
            eval_actors = {}
            eval_actors["actor_pose"] = {}
            eval_actors["actor_extent"] = {}
            eval_actors["actor_center"] = {}
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
                eval_actors["actor_pose"][actor_id] = vehicle_pose
                eval_actors["actor_extent"][actor_id] = vehicle_extent
                eval_actors["actor_center"][actor_id] = vehicle_center
            self.eval_all_frame.append(eval_actors)
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
    def __init__(self, config, world, traffic_manager, testers, cof_seg, main_com, save_flag=False,
                 replayer=None, eval_remain=None, record_frames=None):
        self.config = config
        self.world = world
        self.TM = traffic_manager
        self.testers = testers
        self.save_flag = save_flag
        self.cof_seg = cof_seg
        self.main_com = main_com
        self.replayer = replayer
        self.eval_remain = eval_remain
        self.record_frames = record_frames
        self.start_time = time.time()
        self.reset_time = int(self.config["hypes_yaml"]["reset_time"])
        self.frame_id = 0

    def start(self):
        @time_const(fps=self.config["fps"])
        def run_step():
            self.replayer.replay(self.frame_id)
            self.world.tick()
        try:
            for name, tester in self.testers.items():
                if self.save_flag:
                    tester.save_event.set()
                tester.scheduler_event.set()
                tester.test_event.set()
            while len(self.record_frames) > 0:
                run_step()
                self.frame_id += 1
        except Exception as e:
            logging.error(f"main error:{e}")
        finally:
            self.replayer.destroy_all()
            self.main_com.send_obj("end")
            for name, tester in self.testers.items():
                tester.test_event.clear()
            while len(self.eval_remain) > 0:
                time.sleep(1)
            time.sleep(10)
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            self.TM.set_synchronous_mode(False)
            return


def MainSystem():
    args = recieve_args()
    assert os.path.exists(args.model_dir) and os.path.isdir(args.model_dir), "Model directory does not exist or is not a directory"
    config = cfg.merge(args)
    traffic_light = TrafficFlowManager()
    traffic_light.start()
    main_com = MainCommuicator(config)
    testers = {}
    mutex_lockers = {}
    shared_mems = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cof_seg = manager().dict()
    cof_load = manager().dict()
    eval_remain = manager().dict()
    eval_all_frame = manager().list()
    record_remain = manager().dict()
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
                init_epoch, model = load_best_model(saved_path, model)
            else:
                init_epoch = 0
                saved_path = setup_train(config["hypes_yaml"], agent_config["name"])

            model_to_agent = copy.deepcopy(model)
            tester = create_tester(agent_config, config, agent_mutex, cof_seg, eval_all_frame, eval_remain, model, init_epoch, saved_path)
            tester.start()
            testers[agent_config["name"]] = tester
            
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
                init_epoch, model = load_best_model(saved_path, model)
            else:
                init_epoch = 0
                saved_path = setup_train(config["hypes_yaml"], agent_config["name"])

            model_to_agent = copy.deepcopy(model)
            tester = create_tester(road_agent_config, config, road_agent_mutex, cof_seg, eval_all_frame, eval_remain, model, init_epoch, saved_path)
            tester.start()
            testers[road_agent_config["name"]] = tester
            
    save_flag = False
    cof_seg.clear()
    cof_load.clear()

    if config['hypes_yaml']["fusion_mode"] == "intermediate" or config['hypes_yaml']["fusion_mode"] == "late":
        body(config, main_com, testers, mutex_lockers, cof_seg, cof_load, save_flag, eval_all_frame, eval_remain, record_remain, model_to_agent)
    else:
        body(config, main_com, testers, mutex_lockers, cof_seg, cof_load, save_flag, eval_all_frame, eval_remain, record_remain)
    main_com.close()
    for name, tester in testers.items():
        tester.terminate()
        tester.join()
    for cav_name, cav_shared_mem in shared_mems.items():
        cav_shared_mem.close()
        cav_shared_mem.unlink()
    logging.info("Simulation ended\n")
    for name, tester in testers.items():
        tester.exit_event.set()
    sys.exit()


def body(config, main_com, testers, mutex_lockers, cof_seg, cof_load, save_flag, eval_all_frame, eval_remain, record_remain, model=None):
    world_manager = WorldManager(config)
    world = world_manager.get_world()
    TM = world_manager.get_traffic_manager()
    destroy_all_actors(world)
    main_com.send_obj("start")
    load_road_agents(config, cof_seg, mutex_lockers, cof_load, model)

    record_frames = manager().list()
    record_speed = int(config["record_speed"])
    replayer = ScenarioReplayer('../carla_record/data_record_scene_new2.csv', world, record_remain, config, cof_seg, mutex_lockers, cof_load, record_frames, eval_all_frame, model, record_speed)
    replayer.destroy_all()

    uss = UnifiedSignalSystem(config, world, TM, testers, cof_seg, main_com, save_flag=save_flag, replayer=replayer,
                              eval_remain=eval_remain, record_frames=record_frames)
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
    
