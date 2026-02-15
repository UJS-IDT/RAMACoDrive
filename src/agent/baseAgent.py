import multiprocessing
import threading
from data.commuicate_manager import CommuniAgent
import logging
from util import time_const


class BaseAgent(multiprocessing.Process):
    def __init__(self, agent_name, agent_port):
        super(BaseAgent, self).__init__()
        self.name = agent_name
        self.agent_port = agent_port

    def start_agent(self):
        self.communi_agent = self.init_communi_agent(
            self.name, self.agent_port)

    def close_agent(self):
        self.communi_agent.close()

    def init_communi_agent(self, agent_name, agent_port):
        communi_agent = CommuniAgent(agent_name)
        communi_agent.init_publisher(
            agent_port)
        communi_agent.send_obj(f"{agent_name} started")
        communi_agent.init_subscriber("main",
                                      self.config["main_port"])
        return communi_agent
