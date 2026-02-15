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
import time
import torch
import multiprocessing
from multiprocessing import shared_memory
import logging
from multiprocessing import Manager as manager
from tools.loader import load_agents, load_batch_agents, load_conventional_agents
import sys
import copy


class UnifiedSignalSystem:
    def __init__(self, config, world, traffic_manager, testers, cof_seg, main_com, save_flag=False,
                 eval_all_frame=None, eval_remain=None):
        self.config = config
        self.world = world
        self.TM = traffic_manager
        self.testers = testers
        self.save_flag = save_flag
        self.cof_seg = cof_seg
        self.main_com = main_com
        self.eval_all_frame = eval_all_frame
        self.eval_remain = eval_remain
        self.start_time = time.time()
        self.reset_time = int(self.config["hypes_yaml"]["reset_time"])
        self.timeout = 5 * 60

    def start(self):
        @time_const(fps=self.config["fps"])
        def run_step():
            self.world.tick()
        try:
            for name, tester in self.testers.items():
                if self.save_flag:
                    tester.save_event.set()
                tester.scheduler_event.set()
                tester.test_event.set()
            while len(self.cof_seg) > 0:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time > self.timeout:
                    print("Exit the loop")
                    break
                eval_actors = {}
                eval_actors["actor_pose"] = {}
                eval_actors["actor_extent"] = {}
                eval_actors["actor_center"] = {}
                world_actors = self.world.get_actors().filter('*vehicle*')
                for actor in world_actors:
                    actor_transform = actor.get_transform()
                    actor_pose = [actor_transform.location.x,
                                  actor_transform.location.y,
                                  actor_transform.location.z,
                                  actor_transform.rotation.roll,
                                  actor_transform.rotation.yaw,
                                  actor_transform.rotation.pitch]
                    actor_extent = actor.bounding_box.extent
                    actor_center_location = actor.bounding_box.location
                    extent = [actor_extent.x, actor_extent.y, actor_extent.z]
                    center = [actor_center_location.x, actor_center_location.y, actor_center_location.z]
                    eval_actors["actor_pose"][actor.id] = actor_pose
                    eval_actors["actor_extent"][actor.id] = extent
                    eval_actors["actor_center"][actor.id] = center
                self.eval_all_frame.append(eval_actors)
                run_step()
        except Exception as e:
            logging.error(f"main error:{e}")
        finally:
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
    for agent_config in config["agents"]:
        agent_mutex = multiprocessing.Lock()
        mutex_lockers[agent_config["name"]] = agent_mutex
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
        body(config, main_com, testers, mutex_lockers, cof_seg, cof_load, save_flag, eval_all_frame, eval_remain, model_to_agent)
    else:
        body(config, main_com, testers, mutex_lockers, cof_seg, cof_load, save_flag, eval_all_frame, eval_remain)
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

def body(config, main_com, testers, mutex_lockers, cof_seg, cof_load, save_flag, eval_all_frame, eval_remain, model=None):
    world_manager = WorldManager(config)
    world = world_manager.get_world()
    TM = world_manager.get_traffic_manager()
    destroy_all_actors(world)
    main_com.send_obj("start")
    load_agents(config, cof_seg, mutex_lockers, cof_load, model)
    while len(cof_load) != 0:
        pass
    load_conventional_agents(world, TM, config)
    
    uss = UnifiedSignalSystem(config, world, TM, testers, cof_seg, main_com, save_flag=save_flag,
                              eval_all_frame=eval_all_frame, eval_remain=eval_remain)
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
    
