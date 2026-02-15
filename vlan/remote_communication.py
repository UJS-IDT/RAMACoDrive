import multiprocessing

import yaml
import zmq

def comm(port):
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://192.168.31.79:{port}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')

    pub_socket = context.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.CONFLATE, 1)
    pub_socket.bind(f"tcp://192.168.31.95:{port}")

    try:
        while True:
            try:
                message = sub_socket.recv_pyobj(zmq.NOBLOCK)
                # message = sub_socket.recv_pyobj()
                pub_socket.send_pyobj(message)
                # print(f"Forwarded: {message}")
            except KeyboardInterrupt:
                print("Relay terminated by user.")
            except zmq.Again:
                pass
    except KeyboardInterrupt:
        print("Relay terminated by user.")
    finally:
        sub_socket.close()
        pub_socket.close()
        context.term()

def create_processes(ports):
    processes = []
    for port in ports:
        process = multiprocessing.Process(target=comm, args=(port,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    with open("../config/test.yml", 'r') as test_config_file:
        update_params = yaml.safe_load(test_config_file)
    ports = []
    for agent_info in update_params["agents"]:
        sub_port = agent_info["inter_port"]
        ports.append(sub_port)
    for agent_info in update_params["road_agents"]:
        sub_port = agent_info["inter_port"]
        ports.append(sub_port)
    create_processes(ports)