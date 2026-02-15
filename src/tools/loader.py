from agent.baseline_vehicle_agent import BaselineVehicleAgent
from agent.baseline_road_agent import BaselineRoadAgent
from agent.baseline_record_agent import BaselineRecordAgent
import random
import carla
import copy


def load_agents(config, cof_dic, mutex_lockers, cof_load, model):
    for i, agent_info in enumerate(config["agents"]):
        cof_dic[agent_info["name"]] = True
        cof_load[agent_info["name"]] = True
    for i, agent_info in enumerate(config["agents"]):
        mutex_lock = mutex_lockers[agent_info["name"]]
        agent_info["ignore_traffic_light"] = False
        agent_info["fps"] = config["fps"]
        agent_info["hypes_yaml"] = config["hypes_yaml"]
        BaselineVehicleAgent(config, agent_info, cof_dic, mutex_lock, cof_load, model).start()
    for i, road_agent_info in enumerate(config["road_agents"]):
        mutex_lock = mutex_lockers[road_agent_info["name"]]
        road_agent_info["ignore_traffic_light"] = False
        road_agent_info["fps"] = config["fps"]
        road_agent_info["hypes_yaml"] = config["hypes_yaml"]
        BaselineRoadAgent(config, road_agent_info, cof_dic, mutex_lock, model).start()
        
        
def load_agents_shuffle(config, cof_dic, mutex_lockers, cof_load, model):
    shuffle_spawn_points = []
    for i, agent_info in enumerate(config["agents"]):
        cof_dic[agent_info["name"]] = True
        cof_load[agent_info["name"]] = True
        shuffle_spawn_points.append(agent_info["start_point"])
    random.shuffle(shuffle_spawn_points)
    for i, agent_info in enumerate(config["agents"]):
        mutex_lock = mutex_lockers[agent_info["name"]]
        agent_info["ignore_traffic_light"] = False
        agent_info["fps"] = config["fps"]
        agent_info["hypes_yaml"] = config["hypes_yaml"]
        agent_info["start_point"] = shuffle_spawn_points[i]
        BaselineVehicleAgent(config, agent_info, cof_dic, mutex_lock, cof_load, model).start()
    for i, road_agent_info in enumerate(config["road_agents"]):
        mutex_lock = mutex_lockers[road_agent_info["name"]]
        road_agent_info["ignore_traffic_light"] = False
        road_agent_info["fps"] = config["fps"]
        road_agent_info["hypes_yaml"] = config["hypes_yaml"]
        BaselineRoadAgent(config, road_agent_info, cof_dic, mutex_lock, model).start()


def load_batch_agents(config):

    config["spwan_list"] = random.sample(range(0, 50), 25)
    config["target_list"] = random.sample(range(50, 100), 25)
    agent_info = {}
    for i, spawn_target in enumerate(zip(config["spwan_list"], config["target_list"])):
        agent_info["name"] = f"agent_{i}"
        agent_info["fps"] = config["fps"]
        agent_info["port"] = int(9985+i)
        agent_info["start_point"] = spawn_target[0]
        agent_info["end_point"] = spawn_target[1]
        agent_info["type"] = "vehicle.tesla.model3"
        agent_info["traffic_agent_port"] = config["traffic_agent_port"]
        agent_info["main_port"] = config["main_port"]
        agent_info["ignore_traffic_light"] = False
        BaselineVehicleAgent(agent_info).start()


def load_road_agents(config, cof_dic, mutex_lockers, cof_load, model):
    for i, agent_info in enumerate(config["agents"]):
        cof_dic[agent_info["name"]] = True
        cof_load[agent_info["name"]] = True
    for i, road_agent_info in enumerate(config["road_agents"]):
        mutex_lock = mutex_lockers[road_agent_info["name"]]
        road_agent_info["ignore_traffic_light"] = False
        road_agent_info["fps"] = config["fps"]
        road_agent_info["hypes_yaml"] = config["hypes_yaml"]
        BaselineRoadAgent(config, road_agent_info, cof_dic, mutex_lock, model).start()


def load_record_agents(config, cof_dic, mutex_lockers, cof_load, agent_info, model):
    mutex_lock = mutex_lockers[agent_info["name"]]
    agent_info["ignore_traffic_light"] = False
    agent_info["fps"] = config["fps"]
    agent_info["hypes_yaml"] = config["hypes_yaml"]
    BaselineRecordAgent(config, agent_info, cof_dic, mutex_lock, cof_load, model).start()


def load_conventional_agents(world, tm, config):
    try:
        spawn_point = world.get_map().get_spawn_points()
        config["spwan_list"] = random.sample(range(1, 230), 80)
        config["target_list"] = random.sample(range(100, 300), 80)
        for i, spwan_target in enumerate(zip(config["spwan_list"], config["target_list"])):
            try:
                vehicle_bp = world.get_blueprint_library().filter(
                    "vehicle.tesla.model3*")[0]
                vehicle_bp.set_attribute('role_name', f"agent_{i}")
                vehicle = world.spawn_actor(
                    vehicle_bp, spawn_point[spwan_target[0]])
                tm.ignore_lights_percentage(vehicle, 0)
                vehicle.set_autopilot(True, tm.get_port())
                speed_diff = random.randint(-10, 50)
                tm.vehicle_percentage_speed_difference(vehicle, speed_diff)
            except Exception as e:
                print(f"load_conventional_agents error:{e}")
                pass
    except Exception as e:
        print(f"load_conventional_agents error:{e}")
        pass


def load_conventional_agents_backup(world, tm, config):
    try:
        spawn_point = world.get_map().get_spawn_points()
        config["spwan_list"] = [66] + random.sample(range(68, 120), 47)
        config["target_list"] = random.sample(range(100, 300), 70)
        for i, spwan_target in enumerate(zip(config["spwan_list"], config["target_list"])):
            vehicle_bp = world.get_blueprint_library().filter(
                "vehicle.tesla.model3*")[0]
            vehicle_bp.set_attribute('role_name', f"agent_{i}")
            vehicle = world.spawn_actor(
                vehicle_bp, spawn_point[spwan_target[0]])
            tm.ignore_lights_percentage(vehicle, 0)
            tm.global_percentage_speed_difference(50)
            vehicle.set_autopilot(True, tm.get_port())

    except Exception as e:
        print(f"load_conventional_agents error:{e}")
        pass
