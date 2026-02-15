import multiprocessing
import threading
from data.commuicate_manager import CommuniAgent
import logging
from util import time_const


class LidarAgent(multiprocessing.Process):
    def __init__(self, agent_name, agent_port):
        super(LidarAgent, self).__init__()
        self.name = agent_name
        self.agent_port = agent_port

    def start_agent(self):
        self.lidar_agent = self.init_lidar_agent(
            self.name, self.agent_port)

    def close_agent(self):
        self.lidar_agent.close()

    def init_lidar_agent(self, agent_name, agent_port):
        communi_agent = CommuniAgent(agent_name)
        communi_agent.init_v2x_publisher(
            agent_port, "192.168.31.79")
        communi_agent.send_obj(f"{agent_name} lidar started")
        return communi_agent
