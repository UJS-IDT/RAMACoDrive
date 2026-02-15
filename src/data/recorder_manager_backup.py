# -*- coding: utf-8 -*-
# Author: Chengzhi Gao <Gaochengzhi1999@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import random
import math
import os
import logging
import carla
from agent.baseline_vehicle_agent import BaselineVehicleAgent
from view.debug_manager import DebugManager as debug
from util import spawn_vehicle, connect_to_server, time_const, thread_process_vehicles, get_ego_vehicle, get_speed, log_time_cost, get_vehicle_info
from agent.baseAgent import BaseAgent
import time
import csv
from threading import Lock


class DataRecorder(BaseAgent):
    def __init__(
        self,
        config,
    ) -> None:
        self.config = config
        self.step = 0
        BaseAgent.__init__(self, "DataRecorder",
                           config["data_port"])

    def run(self):
        # @log_time_cost
        @time_const(fps=self.config["fps"])
        def run_step(world, writer):
            lock = Lock()
            current_time = self.step
            try:
                v_list = thread_process_vehicles(
                    world, self.write_vehicle_info, writer, lock, time_now=current_time)
                self.step += 1
            except Exception as e:
                logging.error(e)
        if not self.config["record"]:
            return
        client, world = connect_to_server(
            self.config["carla_timeout"], self.config["carla_port"])
        self.start_agent()


    def write_vehicle_info(self, world, vehicle, writer, lock, time_now):
        # Acquire the lock before writing to the file
        with lock:
            # Basic info
            location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            acceleration = vehicle.get_acceleration()
            angular_velocity = vehicle.get_angular_velocity()
            transform = vehicle.get_transform()
            control = vehicle.get_control()

            # Writing data to CSV within the locked context to ensure thread safety
            writer.writerow([
                time_now,
                vehicle.id,
                location.x, location.y, location.z,
                velocity.x, velocity.y, velocity.z,
                acceleration.x, acceleration.y, acceleration.z,
                angular_velocity.x, angular_velocity.y, angular_velocity.z,
                transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw,
                control.brake, control.throttle, control.steer,
            ])

    def init_data_file(self, folder_path):
        current_time = time.time()
        logging.info(f"creat_data_file: {current_time}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.fp = open(folder_path+"/data.csv", "w")
        writer = csv.writer(self.fp)
        return writer
        # write csv file header

    def close(self) -> None:
        self.fp.fflush()
        self.fp.close()
        time.sleep(2)
        return super().close()
