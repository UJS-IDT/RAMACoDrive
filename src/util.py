# -*- coding: utf-8 -*-
# Author: Chengzhi Gao <Gaochengzhi1999@gmail.com>

import sys
from datetime import datetime
import curses
import time
import functools
import logging
from colorlog import ColoredFormatter
import numpy as np
import os
import re
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
# from tools.config_manager import config
import carla
import math
import csv
import glob
from concurrent.futures import ThreadPoolExecutor
import importlib
import yaml


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def list_to_points(line):
    sp = carla.Transform(carla.Location(float(line[0]), float(line[1]), float(
        line[2])), carla.Rotation(float(line[3]), float(line[4]), float(line[5])))
    return sp


def interpolate_points(start, end, spacing=3):
    distance = compute_distance2D(start, end)
    num_points = max(int(distance / spacing), 2)
    x_spacing = (end[0] - start[0]) / (num_points - 1)
    y_spacing = (end[1] - start[1]) / (num_points - 1)
    return [(start[0] + i * x_spacing, start[1] + i * y_spacing) for i in range(num_points)]


def compute_3D21d(vector):
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def txt_to_points(input_string):
    numbers_list = [float(item) for item in input_string.split(',')]
    return list_to_points(numbers_list)


def get_cache_file(map_name, sp_distance=20):
    map_name = map_name.split("/")[-1]
    cache_dir = "cache/sp_points"
    filename = f"{map_name}.csv"
    filepath = os.path.join(cache_dir, filename)

    return filepath


def load_points_from_csv(filepath):
    wp_list = []
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        for line in reader:
            wp_list.append(list_to_points(line))
    return wp_list


def connect_to_server(timeout, port, host="192.168.31.36"):
    carla_timeout = timeout
    carla_port = port
    client = carla.Client(host, carla_port)
    client.set_timeout(carla_timeout)
    world = client.get_world()
    return client, world


def destroy_all_actors(world):
    for actor in world.get_actors():
        if actor.type_id.startswith("vehicle") or actor.type_id.startswith("sensor") or actor.type_id.startswith("static.prop.streetsign"):
            actor.destroy()
    logging.debug("All actors destroyed")
    # world.tick()


def spawn_vehicle(world, vehicle_type, spawn_point, hero=False, name="hero"):
    lz = spawn_point.location.z + 0.5
    spawn_point = carla.Transform(carla.Location(
        spawn_point.location.x, spawn_point.location.y, lz), spawn_point.rotation)
    vehicle_bp = world.get_blueprint_library().filter(
        vehicle_type
    )[0]
    if hero:
        vehicle_bp.set_attribute('role_name', name)
    vehicle = world.try_spawn_actor(
        vehicle_bp, spawn_point)
    return vehicle


def spawn_road(world, road_type, spawn_location, hero=False, name="road_hero"):
    spawn_point = carla.Transform(carla.Location(spawn_location["x"], spawn_location["y"], spawn_location["z"]), 
                                  carla.Rotation(spawn_location["pitch"], spawn_location["yaw"], spawn_location["roll"]))
    road_bp = world.get_blueprint_library().filter(road_type)[0]
    if hero:
        road_bp.set_attribute('role_name', name)
    road = world.try_spawn_actor(
        road_bp, spawn_point)
    return road


def waypoints_center(waypoint_list):
    x = []
    y = []
    z = []
    for waypoint in waypoint_list:
        x.append(waypoint.transform.location.x)
        y.append(waypoint.transform.location.y)
        z.append(waypoint.transform.location.z)
    return carla.Location(
        np.mean(x), np.mean(y), np.mean(z)
    )


def get_ego_vehicle(world):
    actor = None
    while not actor:
        for actor in world.get_actors():
            if actor.type_id.startswith("vehicle"):
                if actor.attributes["role_name"] == "hero":
                    return actor
        time.sleep(0.2)
    raise RuntimeError("No ego vehicle found")


def log_time_cost(func=None, *, name=""):
    """
    Decorator to log the execution time of a function.


    """
    if func is None:
        return lambda func: log_time_cost(func, name=name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time of function execution
        result = func(*args, **kwargs)  # Execute the function
        elapsed_time = time.time() - start_time  # Calculate elapsed time

        # Log the time cost with debug level
        logging.debug(
            f"Function {name} {func.__name__} executed in {elapsed_time:.5f} seconds.")

        return result
    return wrapper


def time_const(fps):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            target_time_per_frame = 1.0 / fps
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            sleep_time = target_time_per_frame - elapsed_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            return result
        return wrapper
    return decorator

# turn carla way point into NetworkX graph point


def waypoint_to_graph_point(waypoint):
    return (waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z)


def get_vehicle_info(vehicle):
    location = vehicle.get_location()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    control = vehicle.get_control()
    transform = vehicle.get_transform()
    # vehicle_physics = vehicle.get_physics_control()
    vehicle_info = {
        'id': vehicle.attributes["role_name"],
        'location': location,
        'velocity': velocity,
        'acceleration': acceleration,
        'control': control,
        'transform': transform,
        # 'vehicle_pysics': vehicle_physics,
    }
    return vehicle_info


def thread_process_vehicles(world, func, *args, **kwargs):
    vehicles = []
    vehicle_actors = world.get_actors(
    ).filter("vehicle.*")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(func, world, vehicle, *args, **kwargs)
                   for vehicle in vehicle_actors]
        for future in futures:
            vehicles.append(future.result())
    return vehicles


def batch_process_surround_vehicles(world, ego,  max_distance, angle, func, *args, **kwargs):
    vehicles = []
    for actor in world.get_actors():
        if actor.type_id.startswith("vehicle") and actor.attributes["role_name"] != "hero":
            if is_within_distance(actor.get_transform(), ego.get_transform(), max_distance, angle_interval=angle):
                processed_vehicle = func(
                    world, actor, ego, *args, **kwargs)
                vehicles.append(processed_vehicle)
    return vehicles


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def get_trafficlight_trigger_location(traffic_light):
    """
    Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
    """
    def rotate_point(point, radians):
        """
        rotate a given point by a given angle
        """
        rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
        rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

        return carla.Vector3D(rotated_x, rotated_y, point.z)

    base_transform = traffic_light.get_transform()
    base_rot = base_transform.rotation.yaw
    area_loc = base_transform.transform(traffic_light.trigger_volume.location)
    area_ext = traffic_light.trigger_volume.extent

    point = rotate_point(carla.Vector3D(
        0, 0, area_ext.z), math.radians(base_rot))
    point_location = area_loc + carla.Location(x=point.x, y=point.y)

    return carla.Location(point_location.x, point_location.y, point_location.z)


class Location:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Velocity:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def compute_distance2D(location1, location2):
    dx = location1[0] - location2[0]
    dy = location1[1] - location2[1]
    return math.sqrt(dx ** 2 + dy ** 2)
    
    
def compute_distance3D(location1, location2):
    dx = location1[0] - location2[0]
    dy = location1[1] - location2[1]
    dz = location1[2] - location2[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def get_forward_vector(yaw):
    """
    Calculate the forward vector given a yaw angle.
    """
    rad = math.radians(yaw)
    return np.array([math.cos(rad), math.sin(rad)])


def is_within_distance_obs(ego_location, target_location, max_distance=60, ego_speed=0, target_velocity=0, ego_yaw=0, angle_interval=None):
    """
    Filters out the target object (B) within the angle_interval but with speed greater than ego (A).

    :param A: ego_transform, contains location and yaw of the ego vehicle
    :param B: future_state, contains location, velocity, and yaw of the target object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered.
    :return: boolean
    """
    # Calculate the vector from A to B
    target_vector = np.array([target_location[0]-ego_location[0],
                              target_location[1]-ego_location[1]])

    norm_target = math.sqrt(target_vector[0] ** 2 + target_vector[1] ** 2)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.01:
        return False  # Assuming we don't want to consider zero distance valid

    # Further than the max distance
    if norm_target > max_distance:
        return False

    # Calculate the angle between A's forward vector and the vector to B
    forward_vector = get_forward_vector(ego_yaw)
    angle = math.degrees(math.acos(
        np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    # Check if angle is within the specified interval
    if (angle_interval is None or angle_interval[0] <= angle <= angle_interval[1]) and target_velocity <= ego_speed:
        return True

    return False


def is_within_distance(target_transform, reference_transform, max_distance, angle_interval=None):
    """
    Check if a location is both within a certain distance from a reference object.
    By using 'angle_interval', the angle between the location and reference transform
    will also be tkaen into account, being 0 a location in front and 180, one behind.

    :param target_transform: location of the target object
    :param reference_transform: location of the reference object
    :param max_distance: maximum allowed distance
    :param angle_interval: only locations between [min, max] angles will be considered. This isn't checked by default.
    :return: boolean
    """
    target_vector = np.array([
        target_transform.location.x - reference_transform.location.x,
        target_transform.location.y - reference_transform.location.y
    ])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    # Further than the max distance
    if norm_target > max_distance:
        return False

    # We don't care about the angle, nothing else to check
    if not angle_interval:
        return True

    min_angle = angle_interval[0]
    max_angle = angle_interval[1]

    fwd = reference_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    angle = math.degrees(math.acos(
        np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return min_angle < angle < max_angle


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    :param target_location: location of the target object
    :param current_location: location of the reference object
    :param orientation: orientation of the reference object (in degrees)
    :return: a tuple composed by the distance to the object and the angle between both objects
    """
    dx = target_location.x - current_location.x
    dy = target_location.y - current_location.y
    norm_target = math.sqrt(dx ** 2 + dy ** 2)

    orientation_rad = math.radians(orientation)
    cos_ori = math.cos(orientation_rad)
    sin_ori = math.sin(orientation_rad)

    if norm_target == 0:
        return 0, 0

    d_angle = math.degrees(
        math.acos(max(-1.0, min(1.0, (dx * cos_ori + dy * sin_ori) / norm_target))))

    return norm_target, d_angle


def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = math.sqrt(x * x + y * y + z * z)

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    return math.sqrt(x * x + y * y + z * z)


def positive(num):
    """
    Return the given number if positive, else 0

        :param num: value to check
    """
    return num if num > 0.0 else 0.0


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def world_to_x(x1):
    """
    The transformation matrix from carla world system to x-coordinate system

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x1_to_world = x_to_world(x1)
    world_to_x1 = np.linalg.inv(x1_to_world)
    
    return world_to_x1


def x1_to_x2(x1, x2):
    """
    Transformation matrix from x1 to x2.

    Parameters
    ----------
    x1 : list
        The pose of x1 under world coordinates.
    x2 : list
        The pose of x2 under world coordinates.

    Returns
    -------
    transformation_matrix : np.ndarray
        The transformation matrix.

    """
    x1_to_world = x_to_world(x1)
    x2_to_world = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world)

    transformation_matrix = np.dot(world_to_x2, x1_to_world)
    return transformation_matrix


def project_points_by_matrix_torch(points, transformation_matrix):
    """
    Project the points to another coordinate system based on the
    transformation matrix.

    Parameters
    ----------
    points : torch.Tensor
        3D points, (N, 3)
    transformation_matrix : torch.Tensor
        Transformation matrix, (4, 4)
    Returns
    -------
    projected_points : torch.Tensor
        The projected points, (N, 3)
    """
    points, is_numpy = \
        check_numpy_to_torch(points)
    transformation_matrix, _ = \
        check_numpy_to_torch(transformation_matrix)

    # convert to homogeneous coordinates via padding 1 at the last dimension.
    # (N, 4)
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    # (N, 4)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)

    return projected_points[:, :3] if not is_numpy \
        else projected_points[:, :3].numpy()


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def create_model(hypes, max_cav_num):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config, max_cav_num)
    return instance


def torch_tensor_to_numpy(torch_tensor):
    """
    Convert a torch tensor to numpy.

    Parameters
    ----------
    torch_tensor : torch.Tensor

    Returns
    -------
    A numpy array.
    """
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    method_dict["args"]["eps"] = float(method_dict["args"]["eps"])
    method_dict["args"]["weight_decay"] = float(method_dict["args"]["weight_decay"])
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    print('optimizer method is: %s' % optimizer_method)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, n_iter_per_epoch=0.):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        sys.exit('not supported lr schedular')

    return scheduler


def create_v2x(hypes, all_config, config, mutex, send_mutex, model, stop_process, record_cof=None):
    """
    Create the v2x module.
    """
    v2x_module_name = hypes['v2x_module']

    v2x_filename = "v2x_processor." + v2x_module_name
    v2x_lib = importlib.import_module(v2x_filename)
    v2x_func = None
    target_v2x_name = "V2X_communicator"
    
    for name, vfunc in v2x_lib.__dict__.items():
        if name.lower() == target_v2x_name.lower():
            v2x_func = vfunc
    
    v2x_module = v2x_func(all_config, config, mutex, send_mutex, model, stop_process, record_cof)
    return v2x_module


def create_trainer(agent_config, config, agent_mutex, cof_seg, model, init_epoch, saved_path):
    """
    Create the trainer.
    """
    trainer_module_name = config['hypes_yaml']['trainer']

    trainer_filename = "async_training." + trainer_module_name
    trainer_lib = importlib.import_module(trainer_filename)
    trainer_func = None
    target_trainer_name = "Trainer"
    
    for name, tfunc in trainer_lib.__dict__.items():
        if name.lower() == target_trainer_name.lower():
            trainer_func = tfunc
    
    trainer_module = trainer_func(agent_config, config, agent_mutex, cof_seg, model, init_epoch, saved_path)
    return trainer_module
    
    
def create_tester(agent_config, config, agent_mutex, cof_seg, eval_all_frame, eval_remain, model, init_epoch, saved_path):
    """
    Create the trainer.
    """
    tester_module_name = config['hypes_yaml']['tester']

    tester_filename = "async_test." + tester_module_name
    tester_lib = importlib.import_module(tester_filename)
    tester_func = None
    target_tester_name = "Tester"
    
    for name, tfunc in tester_lib.__dict__.items():
        if name.lower() == target_tester_name.lower():
            tester_func = tfunc
    
    tester_module = tester_func(agent_config, config, agent_mutex, cof_seg, eval_all_frame, eval_remain, model, init_epoch, saved_path)
    return tester_module


def setup_train(hypes, name):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name + "/" + name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path


def load_best_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        if os.path.exists(os.path.join(saved_path, 'net_best.pth')):
            return 10000
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        model_file = os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch) \
            if initial_epoch != 10000 else os.path.join(saved_path,
                         'net_best.pth')
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = torch.load(
            model_file,
            map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

        del checkpoint

    return initial_epoch, model


def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        if os.path.exists(os.path.join(saved_path, 'latest.pth')):
            return 10000
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        model_file = os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch) \
            if initial_epoch != 10000 else os.path.join(saved_path,
                         'latest.pth')
        print('resuming by loading epoch %d' % initial_epoch)
        checkpoint = torch.load(
            model_file,
            map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

        del checkpoint

    return initial_epoch, model


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


import csv
def init_data_file(folder, file):
    if not os.path.exists(folder):
            os.makedirs(folder)
            print("ok")
    print("record_path===============", os.path.join(folder, file))
    fp = open(os.path.join(folder,file),"w")
    return  fp,csv.writer(fp)


def batch_process_vehicles(world, func, *args, **kwargs):
    vehicle_actors = world.get_actors().filter("vehicle.*")
    results = list(map(lambda vehicle: func(world, vehicle, *args, **kwargs), vehicle_actors))
    return results





