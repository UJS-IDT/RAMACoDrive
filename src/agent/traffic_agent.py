import random
import math
import os
import logging
import carla
from agent.baseline_vehicle_agent import BaselineVehicleAgent
from prediction.predict_baseline import predict
from view.debug_manager import draw_future_locations
from util import connect_to_server, time_const, log_time_cost, thread_process_vehicles, get_speed
from cythoncode.cutil import is_within_distance_obs
from agent.baseAgent import BaseAgent
import time
from tools.config_manager import config as cfg
from pyinstrument import Profiler
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import numpy as np


def is_within_angle_range(ego_location, target_location, ego_yaw, angle_interval):
    dx = target_location.x - ego_location[0]
    dy = target_location.y - ego_location[1]
    angle = np.degrees(np.arctan2(dy, dx))
    angle_diff = (angle - ego_yaw + 180) % 360 - 180
    return angle_interval[0] <= angle_diff <= angle_interval[1]


class TrafficFlowManager(BaseAgent):
    def __init__(
        self,
    ) -> None:
        self.config = cfg.config
        self.fps = 20
        BaseAgent.__init__(self, "TrafficFlow",
                           self.config["traffic_agent_port"])
        self.ctr_sig = True
    
    def cleanup(self):
        self.close_agent()
        return
        
    def stop(self):
        self.ctr_sig = False       
    
    def run(self):
        @time_const(fps=self.config["fps"])
        def run_step(world):
            try:
                perception_res = thread_process_vehicles(
                    world, predict, self.fps)
                if self.config.get("debug_intersection", False):
                    for perception in perception_res:
                        if perception["id"] in ["agent4", "agent1", "agent2", "agent3"]:
                            perception["except_v"] = 21
                        else:
                            perception["except_v"] = 10
                if self.config.get("emergency", False):
                    for perception in perception_res:
                        if perception["id"] == "emergency":
                            for other_perception in perception_res:
                                if other_perception["id"] != "emergency":
                                    ego_location = np.array(
                                        [perception["location"].x, perception["location"].y], dtype=np.float32)
                                    target_location = np.array(
                                        [other_perception["location"].x, other_perception["location"].y], dtype=np.float32)
                                    ego_speed = perception["velocity"]
                                    target_velocity = other_perception["velocity"]
                                    ego_yaw = perception["yaw"]
                                    if is_within_distance_obs(ego_location, target_location, max_distance=50, ego_speed=ego_speed, target_velocity=target_velocity, ego_yaw=ego_yaw, angle_interval=(-80, 80)):
                                        other_perception["except_offset"] = 1
                if self.config.get("emergency", False):
                    for perception in perception_res:
                        if perception["id"] in ["agent"+str(i) for i in range(5)]:
                            pass

                self.communi_agent.send_obj(perception_res)
            except Exception as e:
                logging.error(e)
                logging.error(e.__traceback__.tb_lineno)
        client, world = connect_to_server(
            self.config["carla_timeout"], self.config["carla_port"])
        self.map = world.get_map()
        self.start_agent()
        time.sleep(1)
        while True:
            run_step(world)


def main():
    TrafficFlowManager().run()


if __name__ == "__main__":
    main()

    def obstacle_change_lane(self, world, obstacle, ego_v, tm, control, threshold_long_distance, ego_lane_index, current_location):

        return
        obstacle_location = obstacle.get_location()
        obstacle_yaw = obstacle.get_transform().rotation.yaw
        obs_speed = get_speed(obstacle)
        distance_obs = math.sqrt(
            (obstacle_location.x - current_location.x) ** 2
            + (obstacle_location.y - current_location.y) ** 2
        )
        obs_lane_index = self.get_lane_id(obstacle_location)
        lane_shit = -1.1 if abs(obs_lane_index) == 1 else 1.1
        if distance_obs < threshold_long_distance + 10 and distance_obs > 10:
            if obs_lane_index == ego_lane_index:
                if abs(obs_lane_index) == 1:
                    tm.force_lane_change(
                        obstacle, False
                    )  # -1 means left and 1 means right
                else:
                    tm.force_lane_change(
                        obstacle, True
                    )  # -1 means left and 1 means right
            # -1 means left and 1 means right
            self.block = False
            if (
                distance_obs < threshold_long_distance + 10
                # and obs_speed < 2
                # and distance_obs > 5
            ):
                control.steer = lane_shit * 0.55 * (1.9 - obs_speed)
                # control.throttle = 2
                control.steer = 2
                # control.brake = 2
                obstacle.apply_control(control)
            elif distance_obs < threshold_long_distance - 8:
                tm.vehicle_lane_offset(obstacle, lane_shit)

            if distance_obs < threshold_long_distance + 5 and (
                obs_lane_index == ego_lane_index
            ):
                self.block = True
                self.step = 90
                self.lane_shift = 1.7 if abs(ego_lane_index) == 1 else -1.7
        pass

    def get_sensor_manager(self):
        return self.ego_vehicle.sensor_manager
