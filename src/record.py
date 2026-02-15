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
from util import destroy_all_actors, time_const, log_time_cost, create_tester
import time
import multiprocessing
from multiprocessing import shared_memory
import logging
from multiprocessing import Manager as manager
from tools.loader import load_agents, load_batch_agents, load_conventional_agents
import sys


def test():
    args = recieve_args()
    assert os.path.exists(args.model_dir) and os.path.isdir(args.model_dir), "Model directory does not exist or is not a directory"
    config = cfg.merge(args)
    traffic_light = TrafficFlowManager()
    traffic_light.start()
    main_com = MainCommuicator(config)
    testers = {}
    mutex_lockers = {}
    shared_mems = {}
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
            tester = create_tester(agent_config, config, agent_mutex, cof_seg, eval_all_frame, eval_remain)
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
            tester = create_tester(road_agent_config, config, road_agent_mutex, cof_seg, eval_all_frame, eval_remain)
            tester.start()
            testers[road_agent_config["name"]] = tester
            
    save_flag = False
    cof_seg.clear()
    cof_load.clear()
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

def body(config, main_com, testers, mutex_lockers, cof_seg, cof_load, save_flag, eval_all_frame, eval_remain):
    start_time = time.time()
    timeout = 5 * 60
    world_manager = WorldManager(config)
    world = world_manager.get_world()
    TM = world_manager.get_traffic_manager()
    destroy_all_actors(world)
    main_com.send_obj("start")
    load_agents(config, cof_seg, mutex_lockers, cof_load)
    while len(cof_load) != 0:
        pass
    load_conventional_agents(world, TM, config)
    
    data_recorder = DataRecorder(config)
    #DataRecorder(config).start()

    #@log_time_cost
    @time_const(fps=config["fps"])
    def run_step(world):
        data_recorder.run_step(world)
        world.tick()
    try:
        for name, tester in testers.items():
            if save_flag:
                tester.save_event.set()
            tester.scheduler_event.set()
            tester.test_event.set()
        while len(cof_seg) > 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > timeout:
                print("Exit the loop")
                break
            eval_actors = {}
            eval_actors["actor_pose"] = {}
            eval_actors["actor_extent"] = {}
            world_actors = world.get_actors().filter('*vehicle*')
            for actor in world_actors:
                actor_transform = actor.get_transform()
                actor_pose = [actor_transform.location.x,
                              actor_transform.location.y,
                              actor_transform.location.z,
                              actor_transform.rotation.roll,
                              actor_transform.rotation.yaw,
                              actor_transform.rotation.pitch]
                actor_extent = actor.bounding_box.extent
                extent = [actor_extent.x, actor_extent.y, actor_extent.z]
                eval_actors["actor_pose"][actor.id] = actor_pose
                eval_actors["actor_extent"][actor.id] = extent
            eval_all_frame.append(eval_actors)
            run_step(world)
    except Exception as e:
        logging.error(f"main error:{e}")
    finally:
        main_com.send_obj("end")
        for name, tester in testers.items():
            tester.test_event.clear()
        while len(eval_remain) > 0:
            time.sleep(1)
        time.sleep(10)
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        TM.set_synchronous_mode(False)
        destroy_all_actors(world)
        logging.info("step ended\n")
        return

def MainCommuicator(config):
    world_control_center = CommuniAgent("World")
    world_control_center.init_publisher(config["main_port"])
    return world_control_center


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    test()
    
