# -*- coding: utf-8 -*-
# Author: Chengzhi Gao <Gaochengzhi1999@gmail.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import zmq
import logging
from uuid import uuid1


class CommuniAgent:
    def __init__(self, name: str):
        self.context = zmq.Context()
        self.pub_socket = None
        self.sub_sockets = {}
        self.push_socket = None
        self.pull_socket = None
        self.type = name
        self.id = uuid1()

    def init_publisher(self, pub_port):
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{pub_port}")
        
    def init_v2x_publisher(self, pub_port, pub_ip="*"):
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.CONFLATE, 1)
        self.pub_socket.bind(f"tcp://{pub_ip}:{pub_port}")

    def init_subscriber(self, name: str, sub_port, sub_ip="localhost"):
        sub_socket = self.context.socket(zmq.SUB)
        sub_socket.connect(f"tcp://{sub_ip}:{sub_port}")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_sockets[name] = sub_socket
        
    def init_v2x_subscriber(self, name: str, sub_port, sub_ip="localhost"):
        sub_socket = self.context.socket(zmq.SUB)
        sub_socket.connect(f"tcp://{sub_ip}:{sub_port}")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        sub_socket.setsockopt(zmq.RCVHWM, 10)
        sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_sockets[name] = sub_socket

    def send_obj(self, data):
        self.pub_socket.send_pyobj(data)

    def send_int(self, data):
        self.pub_socket.send(data)

    def rec_obj(self, sub_name: str):
        try:
            if sub_name in self.sub_sockets:
                msg = self.sub_sockets[sub_name].recv_pyobj(flags=zmq.NOBLOCK)
                return msg
            else:
                raise ValueError(
                    f"No subscriber with name '{sub_name}' initialized.")
        except zmq.Again:
            return None

    def rec_obj_block(self, sub_name: str):
        if sub_name in self.sub_sockets:
            msg = self.sub_sockets[sub_name].recv_pyobj()
            return msg
        else:
            raise ValueError(
                f"No subscriber with name '{sub_name}' initialized.")

    def receive_latest_message(self, sub_name: str):
        """Receive the latest message available in the queue."""
        latest_message = None
        if sub_name in self.sub_sockets:
            sub_socket = self.sub_sockets[sub_name]
            while True:
                try:
                    message = sub_socket.recv_pyobj(zmq.NOBLOCK)
                    latest_message = message  # Keep updating the message until the queue is empty
                except zmq.Again:
                    break  # No more messages
            return latest_message
        else:
            raise ValueError(f"No subscriber with name '{sub_name}' initialized.")

    def close(self):
        self.pub_socket.close()
        if self.push_socket:
            self.push_socket.close()
        if self.pull_socket:
            self.pull_socket.close()
        for sub_socket in self.sub_sockets.values():
            sub_socket.close()
        self.context.term()
        return
        #exit(0)
