import gym
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from copy import deepcopy
from PIL import Image
import torch

# Robomimic / stable-baselines3 / other imports as needed
# e.g., from stable_baselines3 import PPO

class StackEnv(gym.Env):
    """
    A custom Gym environment for controlling a robot environment
    and training with stable-baselines3.
    """
    def __init__(self, env, policy, rollout_horizon, video_record=False, collect_data=False, save_path="", device="cpu"):
        super(StackEnv, self).__init__()

        self.env = env
        self.policy = policy
        self.rollout_horizon = rollout_horizon
        self.video_record = video_record
        self.collect_data = collect_data
        self.save_path = save_path
        self.device = device

        # Action space: discrete with 19 possible actions
        self.action_space = gym.spaces.Discrete(19)

        # Observation space: simple 3 x 84 x 84 image
        # (You can adjust this to match your actual environment)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )

        # Internal trackers
        self.steps = 0
        self.obs = None
        self.video_writer = None
        self.is_sequence = False  # If obs has extra dimension

        # Initial environment reset
        self.obs = self.env.reset()
        if len(self.obs["agentview_image"].shape) == 4: 
            # If there's a batch dimension
            self.is_sequence = True

    def reset(self):
        print("Resetting the environment...")
        st = self.env.get_state()

        # Parse and modify the XML as needed
        xml_str = st["model"]
        root = ET.fromstring(xml_str)

        for geom in root.findall(".//geom"):
            if geom.attrib['name'] == 'table_visual':
                geom.attrib['size'] = "0.4 0.4 0.025"  #0.4 0.4 0.025

        #cube visual size
        for geom in root.findall(".//geom"):
            if geom.attrib['name'] == 'cubeA_g0_vis':
                geom.attrib['size'] = "0.0213203 0.0206657 0.020327"  #0.4 0.4 0.025

        #cube color
        for geom in root.findall(".//geom"):
            if geom.attrib['name'] == 'cubeA_g0_vis':
                geom.attrib['rgba'] = "1 0 0 1"  

        for geom in root.findall(".//geom"):
            if "robot0_g" in geom.attrib['name']:  
                if 'rgba' in geom.attrib:
                    del geom.attrib['rgba']

        #table color
        for geom in root.findall(".//geom"):
            if geom.attrib['name'] == 'table_visual':
                geom.attrib['rgba'] = "0.5 0.5 0.5 1" 
        
        for light in root.findall(".//light"):
            # Example of restoring some typical defaults:
            light.set("diffuse", "1 1 1")
            light.set("specular", "0.1 0.1 0.1")
            light.set("pos", "1 1 1.5")
            light.set("dir", "-0.19245 -0.19245 -0.96225")

        new_xml_str = ET.tostring(root, encoding="unicode")
        st["model"] = new_xml_str

        # Actually reset
        self.obs = self.env.reset_to(st)
        self.env.reset()

        # If recording video, initialize the VideoWriter
        if self.video_record:
            self.steps += 1
            rendered_img = self.env.render(mode="rgb_array", width=300, height=300)
            video_filename = f"episode_{self.steps}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            (h, w) = rendered_img.shape[:2]
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, 30, (w, h))
            self._save_rendered_frame(rendered_img)

        if self.is_sequence:
            return self.obs["agentview_image"][0]

        return self.obs["agentview_image"]

    def step(self, action):
        """
        Apply one discrete action to the environment,
        change the XML accordingly, then run a short rollout.
        """
        self.steps += 1
        robot_state = self.env.get_state()

        # Parse XML
        xml_str = robot_state["model"]
        root = ET.fromstring(xml_str)

        for geom in root.findall(".//geom"): 
        # Cube Color
            if action == 0:
                if geom.attrib['name'] == 'cubeA_g0_vis': # Cube red color
                    geom.attrib['rgba'] = "1 0 0 1"  
            elif action == 1:
                if geom.attrib['name'] == 'cubeA_g0_vis': # Cube green color
                    geom.attrib['rgba'] = "0 1 0 1"
            elif action == 2:
                if geom.attrib['name'] == 'cubeA_g0_vis': # Cube blue color
                    geom.attrib['rgba'] = "0 0 1 1"
            elif action == 3:
                if geom.attrib['name'] == 'cubeA_g0_vis': 
                    geom.attrib['rgba'] = "0.5 0.5 0.5 1"
        
        # Table Color
            elif action == 4:
                if geom.attrib['name'] == 'table_visual': # Table green color
                    geom.attrib['rgba'] = "0 1 0 1"
            elif action == 5:
                if geom.attrib['name'] == 'table_visual': # Table blue color
                    geom.attrib['rgba'] = "0 0 1 1"
            elif action == 6:
                if geom.attrib['name'] == 'table_visual': # Table red color
                    geom.attrib['rgba'] = "1 0 0 1"
            elif action == 7:
                if geom.attrib['name'] == 'table_visual': # Table default color
                    geom.attrib['rgba'] = "0.7 0.7 0.7 1"
        
        # Table Size
            elif action == 8:
                if geom.attrib['name'] == 'table_visual': 
                    geom.attrib['size'] = "0.8 0.2 0.025"
            elif action == 9:
                if geom.attrib['name'] == 'table_visual': 
                    geom.attrib['size'] = "0.2 0.8 0.025"
        
        # Cube Size
            elif action == 10:
                if geom.attrib['name'] == 'cubeA_g0_vis':
                    geom.attrib['size'] = "0.04 0.04 0.04"  # enlarge the cube
            elif action == 11:
                if geom.attrib['name'] == 'cubeA_g0_vis':
                    geom.attrib['size'] = "0.01 0.01 0.01"  # shrink the cube
            elif action == 12:
                if geom.attrib['name'] == 'cubeA_g0_vis':
                    geom.attrib['size'] = "0.04 0.01 0.01"  
        
        # Robot Color
            elif action == 13:
                if "robot0_g" in geom.attrib['name']:  # or pick specific ones if needed
                    geom.attrib['rgba'] = "1 0 0 1"  # Make them yellow
            
            elif action == 14:
                if "robot0_g" in geom.attrib['name']:  # or pick specific ones if needed
                    geom.attrib['rgba'] = "0 1 0 1"  # Make them yellow
            
            elif action == 15:
                if "robot0_g" in geom.attrib['name']:  # or pick specific ones if needed
                    geom.attrib['rgba'] = "0 1 1 1"  # Make them yellow
            elif action == 16:
                if "robot0_g" in geom.attrib['name']:  # or pick specific ones if needed
                    geom.attrib['rgba'] = "0.5 0.5 0.5 1"  # Make them yellow
        
        # Lighting
        lights = root.findall(".//light")

        if action == 17:
            for light in lights:
                r, g, b = 1, 0, 0
                light.set("diffuse", f"{r} {g} {b}")
        
        elif action == 18:
            for light in lights:
                r, g, b = 0, 1, 0
                light.set("diffuse", f"{r} {g} {b}")

        elif action == 19:
            for light in lights:
                r, g, b = 0, 0, 1
                light.set("diffuse", f"{r} {g} {b}")
        
        elif action == 20:
            for light in lights:
                r, g, b = 0.5, 0.5, 0.5
                light.set("diffuse", f"{r} {g} {b}")

        # Update model
        new_xml_str = ET.tostring(root, encoding="unicode")
        robot_state["model"] = new_xml_str

        # Reset the environment to the new XML
        self.obs = self.env.reset_to(robot_state)

        # If we're recording video, save the frame
        if self.video_record:
            rendered_img = self.env.render(mode="rgb_array", width=300, height=300)
            self._save_rendered_frame(rendered_img)

        # Now do a short rollout for `rollout_horizon` steps
        total_reward = 0.0
        success = False

        for step_i in range(self.rollout_horizon):
            # Query policy
            with torch.no_grad():
                act = self.policy(ob=self.obs)

            # Step the environment
            next_obs, r, done, _ = self.env.step(act)
            total_reward += r
            success = self.env.is_success()["task"]

            # Collect data if requested
            if self.collect_data:
                # Save frames to a designated folder
                self._save_demo_frames(step_i, action)

            # If also recording video, save each new frame
            if self.video_record:
                rendered_img = self.env.render(mode="rgb_array", width=300, height=300)
                self._save_rendered_frame(rendered_img)

            # Break if done or success
            if done or success:
                break

            self.obs = deepcopy(next_obs)

        # Write stats
        stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
        print(stats)
        self._log_episode_stats(stats, action)

        # Terminal or not
        done = True
        if success:
            # If success, we might do some custom logic
            reward = -1
            done = False
        else:
            reward = 1000
            print("Episode Completed")

        # If done, stop video recording
        if done and self.video_record and self.video_writer is not None:
            self.video_writer.release()
            print(f"Episode {self.steps} video saved.")

        if self.is_sequence:
            return self.obs["agentview_image"][0], reward, done, {}
        return self.obs["agentview_image"], reward, done, {}


    def _save_rendered_frame(self, img_array):
        """
        Converts an RGB array to BGR for OpenCV and writes it to the video.
        """
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        if self.video_writer is not None:
            self.video_writer.write(img_bgr)

    def _save_demo_frames(self, step_i, action):
        """
        Save frames to disk for data collection.
        """
        from PIL import Image
        import os

        img_save_dir = os.path.join("stack_rl_data", self.save_path, "demo")
        step_save_dir = os.path.join(img_save_dir, f"demo{self.steps}_action_{action}")
        os.makedirs(step_save_dir, exist_ok=True)

        rendered_img = self.env.render(mode="rgb_array", width=300, height=300)
        image_path = os.path.join(step_save_dir, f"frame_{step_i:04d}.png")
        Image.fromarray(rendered_img).save(image_path)

    def _log_episode_stats(self, stats, action):
        """
        Write stats and actions to log files.
        """
        import os

        # Always log stats
        with open(f"stack_logs/{self.save_path}/episode_stats.txt", "a") as file:
            file.write(str(stats) + "\n")

        # If collecting data, also log success/action
        if self.collect_data:
            with open(f"stack_rl_data/{self.save_path}/success_rate.txt", "a") as file:
                file.write(str(stats["Success_Rate"]) + "\n")
            with open(f"stack_rl_data/{self.save_path}/actions.txt", "a") as file:
                file.write(str(action) + "\n")

