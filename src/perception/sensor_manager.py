import carla
import numpy as np
import time
import logging
import weakref
from util import compute_3D21d
from view.debug_manager import draw_transforms
from matplotlib import cm
import open3d as o3d
import math
import collections
import copy
import sys
import cv2
import pickle

class SensorManager:
    def __init__(self, world, vehicle, vehicle_info, config, shared_mem, mutex):
        self.world = world
        self.vehicle = vehicle
        self.config = config
        self.shared_mem = shared_mem
        self.mutex = mutex
        self.lidar_data_window = collections.deque()
        self.raw_lidar_data_window = collections.deque()
        self.VIDIDIS = np.array(cm.get_cmap("plasma").colors)
        self.VID_RANGE = np.linspace(0.0, 1.0, self.VIDIDIS.shape[0])
        self.point_cloud = o3d.geometry.PointCloud()
        self.vehicle_info = vehicle_info
        self.lidar_rotation_frequency = self.config["lidar_set"]["rotation_frequency"]
        self.lidar_time_window = 1 / self.lidar_rotation_frequency
        self.lidar_data_buffer = collections.deque(maxlen=int(self.lidar_rotation_frequency * 2))
        self.radar_list = []
        self.lidar_list = []
        self.radar_res = {
            radar_id: None for radar_id in ["front", "left", "right"]
        }
        #self.setup_radars()
        self.setup_lidars()
        #self.add_obstacle_sensor()
        # self.add_camera()
        self.camera_queue = []
        self.radar_queue = []
        self.obstacle = None
        self.rear_obstacle = None

    def add_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '192')
        camera_bp.set_attribute('image_size_y', '108')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self.camera_callback)

    def camera_callback(self, data):
        # data.save_to_disk('_out/%08d' % data.frame)
        pass

    def setup_lidars(self):
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(pitch=0, yaw=0, roll=0))
        self.lidar_sensor = self.add_lidar_sensor(lidar_transform)

    def add_lidar_sensor(self, lidar_transform):
        weak_self = weakref.ref(self)
        point_list = o3d.geometry.PointCloud()
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute("dropoff_general_rate", self.config["lidar_set"]["dropoff_general_rate"])
        # lidar_bp.set_attribute("dropoff_intensity_limit", "1.0")
        # lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
        lidar_bp.set_attribute('upper_fov', self.config["lidar_set"]["upper_fov"])
        lidar_bp.set_attribute('lower_fov', self.config["lidar_set"]["lower_fov"])
        lidar_bp.set_attribute('horizontal_fov', self.config["lidar_set"]["horizontal_fov"])
        lidar_bp.set_attribute('channels', self.config["lidar_set"]["channels"])
        lidar_bp.set_attribute('range', self.config["lidar_set"]["range"])
        lidar_bp.set_attribute('rotation_frequency', str(self.lidar_rotation_frequency))
        lidar_bp.set_attribute('points_per_second', self.config["lidar_set"]["points_per_second"])
        lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar_sensor.listen(lambda data: self.lidar_callback(weak_self, data, point_list))
        return lidar_sensor

    @staticmethod
    def lidar_callback(weak_self, lidar_data, point_list):
        self = weak_self()
        if not self:
            return

        current_time = lidar_data.timestamp

        self.raw_lidar_data_window.append((current_time, lidar_data))

        while self.raw_lidar_data_window and current_time - self.raw_lidar_data_window[0][0] > self.lidar_time_window:
            self.raw_lidar_data_window.popleft()
            self.v2x_dic = {}
            #if self.shared_mem is not None:
            if self.shared_mem.buf is not None:
                latest_pose = np.array([
                    lidar_data.transform.location.x,
                    lidar_data.transform.location.y,
                    lidar_data.transform.location.z,
                    lidar_data.transform.rotation.roll,
                    lidar_data.transform.rotation.yaw,
                    lidar_data.transform.rotation.pitch
                ])
                all_lidar_data = np.concatenate([
                    np.frombuffer(_data.raw_data, dtype=np.dtype("f4")).reshape(-1, 4)
                    for _, _data in list(self.raw_lidar_data_window)], axis=0)
                    
                # Memory Share
                self.v2x_dic["lidar_data"] = all_lidar_data
                self.v2x_dic["lidar_pose"] = latest_pose
                self.v2x_dic["timestamp"] = current_time
                v2x_obj_data = pickle.dumps(self.v2x_dic)
                v2x_obj_len = len(v2x_obj_data)
                v2x_metadata = str(v2x_obj_len) + "~"
                v2x_byte_data = v2x_metadata.encode() + v2x_obj_data
                self.mutex.acquire()
                if self.shared_mem.buf is not None:
                    self.shared_mem.buf[:len(v2x_byte_data)] = v2x_byte_data
                self.mutex.release()
            else:
                print("session is completed")
                sys.exit()

    def setup_radars(self):
        front_radar_transform = carla.Transform(carla.Location(
            x=self.vehicle_info["length"], z=self.vehicle_info["height"]), carla.Rotation(pitch=7))
        # rear_radar_transform = carla.Transform(carla.Location(
        #     x=-self.vehicle_info["length"], z=self.vehicle_info["height"]), carla.Rotation(yaw=180, pitch=7))
        left_radar_transform = carla.Transform(carla.Location(
            y=-self.vehicle_info["width"], z=self.vehicle_info["height"]), carla.Rotation(yaw=-80, pitch=7))
        right_radar_transform = carla.Transform(carla.Location(
            y=self.vehicle_info["width"], z=self.vehicle_info["height"]), carla.Rotation(yaw=80, pitch=7))

        self.radar_list.append(self.add_radar("front", h_fov=45, v_fov=20,
                                              radar_transform=front_radar_transform, range=25))
        # self.radar_list.append(self.add_radar("rear", h_fov=15, v_fov=15, radar_transform=rear_radar_transform,range=15))
        self.radar_list.append(self.add_radar("left", h_fov=120, v_fov=15, radar_transform=left_radar_transform,
                                              range=3))
        self.radar_list.append(self.add_radar("right", h_fov=120, v_fov=15, radar_transform=right_radar_transform,
                                              range=3))

    def add_radar(self, radar_id, h_fov, v_fov, radar_transform, points_per_second="100", range="50"):
        weak_self = weakref.ref(self)
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', str(h_fov))
        radar_bp.set_attribute('vertical_fov', str(v_fov))
        radar_bp.set_attribute('points_per_second', str(points_per_second))
        radar_bp.set_attribute('range', str(range))
        radar = self.world.spawn_actor(
            radar_bp, radar_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        radar.listen(lambda data: self.radar_callback(
            weak_self, radar_id, data))
        return radar

    def add_obstacle_sensor(self):
        weak_self = weakref.ref(self)
        obstacle_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
        obstacle_bp.set_attribute('distance', '50')
        obstacle_bp.set_attribute("only_dynamics", str(False))
        # obstacle_bp.debug_linetrace = True
        front_obs_transform = carla.Transform(
            carla.Location(x=self.vehicle_info["length"]+5, z=self.vehicle_info["height"]))
        rear_obs_transform = carla.Transform(
            carla.Location(x=-self.vehicle_info["length"], z=self.vehicle_info["height"]))
        self.obstacle_sensor = self.world.spawn_actor(
            obstacle_bp, front_obs_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.back_obstacle_sensor = self.world.spawn_actor(
            obstacle_bp, front_obs_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.obstacle_sensor.listen(
            lambda event: self.obstacle_callback(weak_self, event))

    @staticmethod
    def obstacle_callback(weak_self, data):
        self = weak_self()
        if not self:
            return
        obs_loc = data.other_actor.get_location()
        obs_speed = compute_3D21d(data.other_actor.get_velocity())
        self.obstacle = Obstacle(obs_loc,
                                 data.distance, obs_speed)
        # if data.other_actor.is_alive:
        #     data_transform = data.other_actor.get_transform()
        #     data_transform.location.z = 3
        #     draw_transforms(
        #         self.world, [data_transform], size=0.2, life_time=0.2)

    @staticmethod
    def radar_callback(weak_self, radar_id, radar_data):
        self = weak_self()
        if not self:
            return
        if not radar_data:
            self.radar_res[radar_id] = None
            return
        for detect in radar_data:
            distance = detect.depth
            velocity = detect.velocity
            self.radar_res[radar_id] = (distance, velocity)

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)  # azimuth
            alt = math.degrees(detect.altitude)
            velocity = detect.velocity  # velocity of the detected object
            fw_vec = carla.Vector3D(x=detect.depth - 0.15)
            transform = carla.Transform(carla.Location(), carla.Rotation(
                pitch=current_rot.pitch + alt, yaw=current_rot.yaw + azi, roll=current_rot.roll))
            fw_vec = transform.transform(fw_vec)
            color = carla.Color(255, 0, 0)
            # self.world.debug.draw_point(
            #     radar_data.transform.location + fw_vec,
            #     size=0.75,
            #     life_time=0.16,
            #     persistent_lines=False,
            #     color=color)

    def _parse_lidar_cb(self, lidar_data):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points

    def visualize_data(self, lidar, text_args=(0.6)):
        lidar_viz = self.lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz, cv2.COLOR_GRAY2RGB)

        return lidar_viz

    def lidar_to_bev(self, lidar, min_x=-100, max_x=100, min_y=-100, max_y=100, pixels_per_meter=4, hist_max_per_pixel=2):
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

class Obstacle:
    def __init__(self, location, distance, velocity):
        self.location = location
        self.distance = distance
        self.velocity = velocity
