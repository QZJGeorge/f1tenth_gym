import curses
import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from pyglet.gl import GL_POINTS
from f110_gym.envs.base_classes import Integrator


def curses_main(stdscr):
    """
    Main function to be called by curses.wrapper(), so that
    we can capture single keystrokes without requiring Enter.
    """

    # 1) Make getch() non-blocking
    curses.cbreak()
    stdscr.nodelay(True)

    # Optional: Hide cursor
    curses.curs_set(0)

    # Load config and environment
    with open("examples/config_example_map.yaml") as file:
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

    # Car control parameters
    speed = 0.0
    steer = 0.0
    speed_increment = 0.5
    steer_increment = 0.05
    max_steer = 0.52
    min_steer = -0.52

    # Print instructions once in the curses terminal
    stdscr.addstr(
        0,
        0,
        "Controls:\n"
        "  Arrow UP    -> speed += 0.5\n"
        "  Arrow DOWN  -> speed -= 0.5 (min 0)\n"
        "  Arrow LEFT  -> steer left by 0.05 (bounded)\n"
        "  Arrow RIGHT -> steer right by 0.05 (bounded)\n"
        "  q           -> quit\n",
    )
    stdscr.refresh()

    while not done:
        # 2) Check for keypress
        ch = stdscr.getch()
        if ch != -1:
            # A key was pressed
            if ch == ord("q"):
                break

            elif ch == curses.KEY_UP:
                speed += speed_increment
            elif ch == curses.KEY_DOWN:
                speed -= speed_increment
                if speed < 0.0:
                    speed = 0.0
            elif ch == curses.KEY_LEFT:
                steer += steer_increment
            elif ch == curses.KEY_RIGHT:
                steer -= steer_increment

            # Bound steering
            steer = max(min_steer, min(max_steer, steer))

        # Step environment
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        visited_points.append([obs["poses_x"][0], obs["poses_y"][0]])

        # Render the environment (Pyglet window)
        env.render(mode="human")

        # Sleep for stability
        time.sleep(0.01)

    env.close()


def main():
    curses.wrapper(curses_main)


if __name__ == "__main__":
    main()
