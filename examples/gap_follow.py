import csv
import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from pyglet.gl import GL_POINTS
from f110_gym.envs.base_classes import Integrator


def find_largest_gap(ranges, min_dist=2.0):
    """
    Find the largest contiguous set of indices in 'ranges'
    whose values are >= min_dist. Returns a tuple:
        (start_index, end_index)
    of the largest gap.
    """

    # Mark which indices are above threshold
    above_threshold = ranges >= min_dist

    max_len = 0
    max_start = 0
    max_end = 0

    curr_start = None
    for i, val in enumerate(above_threshold):
        if val:
            if curr_start is None:
                curr_start = i
        else:
            if curr_start is not None:
                length = i - curr_start
                if length > max_len:
                    max_len = length
                    max_start = curr_start
                    max_end = i - 1
                curr_start = None
    # Check at end in case the last segment extends to the final index
    if curr_start is not None:
        length = len(above_threshold) - curr_start
        if length > max_len:
            max_len = length
            max_start = curr_start
            max_end = len(above_threshold) - 1

    return max_start, max_end


def compute_steer_angle_for_gap(start_idx, end_idx):
    """
    Given the largest gap, compute the midpoint index and convert
    that index into an angle (in radians). Then map to steering.
    - Index 0 is angle 0°, index 1440 => 360° (each index ~ 0.25°).
    - Convert the index to a heading relative to forward = 0°.
    - Return the desired steering angle (radians) scaled by a gain.
    """

    # Midpoint index
    mid_idx = (start_idx + end_idx) // 2

    # Each index is 0.25 degrees
    resolution_deg = 0.25
    angle_deg = resolution_deg * mid_idx - 90.0

    # Convert degrees to radians
    angle_rad = np.deg2rad(angle_deg)

    # A typical approach is to scale the heading angle to a steering angle.
    STEERING_GAIN = 1.0  # Adjust or tune this gain as needed
    desired_steer = STEERING_GAIN * angle_rad

    # Saturate at the car's steering limits
    max_steer = 0.52
    if desired_steer > max_steer:
        desired_steer = max_steer
    elif desired_steer < -max_steer:
        desired_steer = -max_steer

    return desired_steer


def main():
    # Load config and create environment
    with open("examples/config_example_map.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    env = gym.make(
        "f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )

    visited_points = []
    drawn_points = []
    last_recorded_point = None  # Track the last recorded waypoint

    def render_callback(env_renderer):
        # Follow the car
        e = env_renderer
        x_coords = e.cars[0].vertices[::2]
        y_coords = e.cars[0].vertices[1::2]
        top, bottom = max(y_coords), min(y_coords)
        left, right = min(x_coords), max(x_coords)

        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        # Draw trajectory points
        for i, pt in enumerate(visited_points):
            scaled_x = 50.0 * pt[0]
            scaled_y = 50.0 * pt[1]
            if i < len(drawn_points):
                drawn_points[i].vertices = [scaled_x, scaled_y, 0.0]
            else:
                b = e.batch.add(
                    1,
                    GL_POINTS,
                    None,
                    ("v3f/stream", [scaled_x, scaled_y, 0.0]),
                    ("c3B/stream", [255, 0, 0]),  # Red
                )
                drawn_points.append(b)

    env.add_render_callback(render_callback)
    obs, step_reward, done, info = env.reset(
        poses=np.array([[conf.sx, conf.sy, conf.stheta]])
    )
    env.render()

    print("Gap-Follow Enabled with Constant Speed.")
    print("Press CTRL+C in the console to terminate early.")

    constant_speed = 5.0  # m/s

    # In-memory storage for data
    collected_data = []

    try:
        while not done:
            # --- GAP FOLLOW ALGORITHM ---
            # 1) Extract LiDAR scan
            lidar_ranges = np.array(obs["scans"]).flatten()
            lidar_ranges = lidar_ranges[360:1080]

            # 2) Find the largest gap above some threshold
            MIN_RANGE_THRESHOLD = 2.0
            start_idx, end_idx = find_largest_gap(lidar_ranges, MIN_RANGE_THRESHOLD)

            # 3) Compute steering angle toward the middle of that gap
            steer = compute_steer_angle_for_gap(start_idx, end_idx)

            # Current car position
            current_point = [obs["poses_x"][0], obs["poses_y"][0]]

            # Check if the current point is at least 0.1 meters from the last recorded point
            if (
                last_recorded_point is None
                or np.linalg.norm(
                    np.array(current_point) - np.array(last_recorded_point)
                )
                >= 0.1
            ):
                # Update the last recorded point
                last_recorded_point = current_point

                # Store steer and lidar data in-memory
                collected_data.append([steer] + lidar_ranges.tolist())

            # Record visited points for rendering
            visited_points.append(current_point)

            # Step the environment
            action = np.array([[steer, constant_speed]])
            obs, step_reward, done, info = env.step(action)

            # Render the environment
            env.render(mode="human")

            # Small delay for stability
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    finally:
        # Save all collected data to a CSV file
        with open("steering_and_lidar_data.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["steer"] + [f"lidar_{i}" for i in range(720)])
            # Write the data
            writer.writerows(collected_data)

        env.close()
        print("Data saved to 'steering_and_lidar_data.csv'.")


if __name__ == "__main__":
    main()
