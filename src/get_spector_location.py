import carla
import time
from util import spawn_vehicle, connect_to_server, time_const, thread_process_vehicles, get_ego_vehicle, get_speed, log_time_cost, get_vehicle_info, x_to_world, project_points_by_matrix_torch

def main():
    client, world = connect_to_server(1000, 2000)

    try:
        while True:
            spectator = world.get_spectator()
            transform = spectator.get_transform()
            location = transform.location
            rotation = transform.rotation

            print(f"Spectator location: (x: {location.x:.2f}, y: {location.y:.2f}, z: {location.z:.2f})")
            print(f"Spectator rotation: (pitch: {rotation.pitch:.2f}, yaw: {rotation.yaw:.2f}, roll: {rotation.roll:.2f})")

            time.sleep(1)

    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
        
if __name__ == '__main__':
    main()
