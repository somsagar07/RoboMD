import os
import gym
import xml.etree.ElementTree as ET
from copy import deepcopy
import numpy as np
import cv2
import torch

from scipy.spatial.distance import cdist
from utils.losses import find_closest_value

class LatentActionEnv(gym.Env):
    """
    An environment that:
      1) Receives a 512-d float action (latent vector).
      2) Finds the nearest known embedding from 'embeddings_array'.
      3) Applies that action's XML modifications.
      4) Rolls out the robomimic policy for 'horizon' steps.
      5) Rewards are based on success/failure / distance / repeated actions, etc.

    Observations: 3x84x84 images from 'agentview_image'.
    """

    def __init__(self, env, policy, horizon, embeddings_array, values_array,
                 log_dir, task_name, video_record=False):
        super().__init__()

        # RL action is 512-d latent
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(512,), dtype=np.float32)
        # Observation is the image
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)

        self.env = env
        self.policy = policy
        self.horizon = horizon
        self.embeddings_array = embeddings_array
        self.values_array = values_array
        self.log_dir = log_dir
        self.task_name = task_name

        self.video_record = video_record
        self.video_writer = None
        self.steps = 0
        self.prev_action = -1

        # initial reset
        self.obs = self.env.reset()
        self.is_sequence = (len(self.obs["agentview_image"].shape) == 4)

    def reset(self):
        print('Resetting env')
        
        st = self.env.get_state()

        # Parse the XML model
        xml_str = st['model']
        root = ET.fromstring(xml_str)

        if self.task_name == 'lift' or self.task_name == 'stack':
        
            ##table visual size
            for geom in root.findall(".//geom"):
                if geom.attrib['name'] == 'table_visual':
                    geom.attrib['size'] = "0.4 0.4 0.025"  #0.4 0.4 0.025

            #cube visual size
            for geom in root.findall(".//geom"):
                if geom.attrib['name'] == 'cube_g0_vis':
                    geom.attrib['size'] = "0.0213203 0.0206657 0.020327"  #0.4 0.4 0.025

            #cube color
            for geom in root.findall(".//geom"):
                if geom.attrib['name'] == 'cube_g0_vis':
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
            
        elif self.task_name == 'square':
                        ##table visual size
            for geom in root.findall(".//geom"):
                if "name" in geom.attrib and geom.attrib["name"] == "table_visual":
                    geom.attrib['size'] = "0.4 0.4 0.025"  #0.4 0.4 0.025


            #cube color
            for body in root.findall(".//body"):
                if body.attrib.get("name") == "peg2":
                    for geom in body.findall(".//geom"):
                        # Maybe match both type=cylinder AND a certain size
                        if geom.attrib.get("type") == "cylinder" and geom.attrib.get("size") == "0.02 0.1":
                
                            if "material" in geom.attrib:
                                del geom.attrib["material"]
                            #geom.set("rgba", "0.5 0.5 0 1") 


            
            for light in root.findall(".//light"):
                # Example of restoring some typical defaults:
                light.set("diffuse", "1 1 1")
                light.set("specular", "0.1 0.1 0.1")
                light.set("pos", "1 1 1.5")
                light.set("dir", "-0.19245 -0.19245 -0.96225")

        elif self.task_name == 'can':
            for geom in root.findall(".//geom"):
                if "robot" in geom.attrib.get("name", "") or "robot" in geom.attrib.get("material", ""): 
                    if 'rgba' in geom.attrib:
                        del geom.attrib['rgba']

            
            for light in root.findall(".//light"):
                # Example of restoring some typical defaults:
                light.set("diffuse", "1 1 1")
                light.set("specular", "0.1 0.1 0.1")
                light.set("pos", "1 1 1.5")
                light.set("dir", "-0.19245 -0.19245 -0.96225")

        
        new_xml_str = ET.tostring(root, encoding='unicode')

        # Update the model in the state
        st['model'] = new_xml_str

        self.obs = self.env.reset_to(st)

        self.env.reset() 

        if self.video_record:
            rendered_img = self.env.render(mode="rgb_array", width=300, height=300)
            self.save_rendered_frame(rendered_img)

            rendered_img = self.env.render(mode="rgb_array", width=300, height=300)  # Set width and height
            original_height, original_width = rendered_img.shape[:2]

            #Initialize the VideoWriter with the rendered frame dimensions
            video_filename = f'episode_{self.steps}.avi'
            self.video_writer = cv2.VideoWriter(
                video_filename, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'), 30, (original_width, original_height)
            )

        self.policy.start_episode()
        if self.is_sequence:
            return self.obs["agentview_image"][0]
        
        return self.obs["agentview_image"]
    
    def step(self, action):
        self.steps+=1

        # Find the index of the closest action
        action, penalty = find_closest_value(action, self.embeddings_array, self.values_array)

        print(f'Closest known action : {action}')
        with open(f"{self.log_dir}/actions.txt", "a") as file:
            file.write(str(action) + '\n')

        same_action_penalty = 0
        if self.prev_action == action:
            same_action_penalty = 100
            self.prev_action = action
        
        if self.steps == 1:
            self.prev_action = action

        robot_state = self.env.get_state()

        # Parse the XML model
        xml_str = robot_state['model']
        root = ET.fromstring(xml_str)


        if self.task_name == 'lift' or self.task_name == 'stack':
        
            for geom in root.findall(".//geom"): 
                # Cube Color
                if action == 0:
                    if geom.attrib['name'] == 'cube_g0_vis': # Cube red color
                        geom.attrib['rgba'] = "1 0 0 1"  
                elif action == 1:
                    if geom.attrib['name'] == 'cube_g0_vis': # Cube green color
                        geom.attrib['rgba'] = "0 1 0 1"
                elif action == 2:
                    if geom.attrib['name'] == 'cube_g0_vis': # Cube blue color
                        geom.attrib['rgba'] = "0 0 1 1"
                elif action == 3:
                    if geom.attrib['name'] == 'cube_g0_vis': 
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
                    if geom.attrib['name'] == 'cube_g0_vis':
                        geom.attrib['size'] = "0.04 0.04 0.04"  # enlarge the cube
                elif action == 11:
                    if geom.attrib['name'] == 'cube_g0_vis':
                        geom.attrib['size'] = "0.01 0.01 0.01"  # shrink the cube
                elif action == 12:
                    if geom.attrib['name'] == 'cube_g0_vis':
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
        
        elif self.task_name == 'square':

            for body in root.findall(".//body"):
                if body.attrib.get("name") == "peg2":
                    for geom in body.findall(".//geom"):
                        # Maybe match both type=cylinder AND a certain size
                        if geom.attrib.get("type") == "cylinder" and geom.attrib.get("size") == "0.02 0.1":
                
                            if "material" in geom.attrib:
                                del geom.attrib["material"]
                            if action == 0:
                                geom.set("rgba", "1 0 0 1") 
                            elif action == 1:
                                geom.set("rgba", "0 1 0 1")
                            elif action == 2:
                                geom.set("rgba", "0 0 1 1")
                            elif action == 3:
                                geom.set("rgba", "0.5 0.5 0.5 1")
                
            
            for body in root.findall(".//body"):
                if body.attrib.get("name") == "peg2":
                    for geom in body.findall(".//geom"):
                        # Maybe match both type=cylinder AND a certain size
                        if geom.attrib.get("type") == "cylinder" and geom.attrib.get("size") == "0.02 0.1":
                
                            if "material" in geom.attrib:
                                del geom.attrib["material"]
                            if action == 4:
                                geom.set("size", "0.03 0.1") 
                            elif action == 5:
                                geom.set("size", "0.02 0.15")
                            elif action == 6:
                                geom.set("size", "0.02 0.13")
                            elif action == 7:
                                geom.set("size", "0.03 0.08")
                        
                
            for geom in root.findall(".//geom"): 
            
            # Table Size
                if action == 8:
                    if "name" in geom.attrib and geom.attrib["name"] == "table_visual": 
                        geom.attrib['size'] = "0.8 0.2 0.025"
                elif action == 9:
                    if "name" in geom.attrib and geom.attrib["name"] == "table_visual":  
                        geom.attrib['size'] = "0.2 0.8 0.025"
            
            # Lighting
            lights = root.findall(".//light")

            if action == 10:
                for light in lights:
                    r, g, b = 1, 0, 0
                    light.set("diffuse", f"{r} {g} {b}")
            
            elif action == 11:
                for light in lights:
                    r, g, b = 0, 1, 0
                    light.set("diffuse", f"{r} {g} {b}")

            elif action == 12:
                for light in lights:
                    r, g, b = 0, 0, 1
                    light.set("diffuse", f"{r} {g} {b}")
            
            elif action == 13:
                for light in lights:
                    r, g, b = 0.5, 0.5, 0.5
                    light.set("diffuse", f"{r} {g} {b}")
        
        elif self.task_name == 'can':
            for geom in root.findall(".//geom"): 
                # cylinder Color
                if action == 0:
                    if "name" in geom.attrib and geom.attrib["name"] == "Can_g0_visual":
                        geom.attrib["rgba"] = "1 0 0 1" 
                elif action == 1:
                    if "name" in geom.attrib and geom.attrib["name"] == "Can_g0_visual": # Cube green color
                        geom.attrib['rgba'] = "0 1 0 1"
                elif action == 2:
                    if "name" in geom.attrib and geom.attrib["name"] == "Can_g0_visual": # Cube blue color
                        geom.attrib['rgba'] = "0 0 1 1"
                elif action == 3:
                    if "name" in geom.attrib and geom.attrib["name"] == "Can_g0_visual": 
                        geom.attrib['rgba'] = "0.5 0.5 0.5 1"
            
            # Table Color
                elif action == 4:
                    if geom.attrib.get("type") == "box" and geom.attrib.get("material") == "light-wood":
                        geom.attrib['rgba'] = "0 1 0 1"
                elif action == 5:
                    if geom.attrib.get("type") == "box" and geom.attrib.get("material") == "light-wood":
                        geom.attrib['rgba'] = "0 0 1 1"
                elif action == 6:
                    if geom.attrib.get("type") == "box" and geom.attrib.get("material") == "light-wood":
                        geom.attrib['rgba'] = "1 0 0 1"
                elif action == 7:
                    if geom.attrib.get("type") == "box" and geom.attrib.get("material") == "light-wood":
                        geom.attrib['rgba'] = "0.7 0.7 0.7 1"
            
            # Table Size
                elif action == 8:
                    if geom.attrib.get("type") == "box" and geom.attrib.get("material") == "light-wood":
                        geom.attrib["size"] = "0.3 0.3 0.02"  # Modify size
                elif action == 9:
                    if geom.attrib.get("type") == "box" and geom.attrib.get("material") == "light-wood":
                        geom.attrib["size"] = "0.2 0.2 0.02"  # Modify size
                elif action == 10:
                    if geom.attrib.get("type") == "box" and geom.attrib.get("material") == "light-wood":
                        geom.attrib["size"] = "0.1 0.1 0.02"
            
            # Robot Color
                elif action == 11:
                    if "robot" in geom.attrib.get("name", "") or "robot" in geom.attrib.get("material", ""):
                        geom.attrib["rgba"] = "0 0 1 1"  

                
                elif action == 12:
                    if "robot" in geom.attrib.get("name", "") or "robot" in geom.attrib.get("material", ""):
                        geom.attrib['rgba'] = "0 1 0 1"  # Make them yellow
                
                elif action == 13:
                    if "robot" in geom.attrib.get("name", "") or "robot" in geom.attrib.get("material", ""):
                        geom.attrib['rgba'] = "0 1 1 1"  # Make them yellow
                elif action == 14:
                    if "robot" in geom.attrib.get("name", "") or "robot" in geom.attrib.get("material", ""):
                        geom.attrib['rgba'] = "0.5 0.5 0.5 1"  # Make them yellow
                
                # Lighting
            lights = root.findall(".//light")

            if action == 15:
                for light in lights:
                    r, g, b = 1, 0, 0
                    light.set("diffuse", f"{r} {g} {b}")
            
            elif action == 16:
                for light in lights:
                    r, g, b = 0, 1, 0
                    light.set("diffuse", f"{r} {g} {b}")

            elif action == 17:
                for light in lights:
                    r, g, b = 0, 0, 1
                    light.set("diffuse", f"{r} {g} {b}")
            
            elif action == 18:
                for light in lights:
                    r, g, b = 0.5, 0.5, 0.5
                    light.set("diffuse", f"{r} {g} {b}")


        new_xml_str = ET.tostring(root, encoding='unicode')

        # Update the model in the state
        robot_state['model'] = new_xml_str

        self.obs = self.env.reset_to(robot_state)
        # self.env.reset()


        if self.video_record:
            rendered_img = self.env.render(mode="rgb_array", width=300, height=300)
            self.save_rendered_frame(rendered_img)

        total_reward = 0.
        traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=robot_state)

        for step_i in range(self.horizon):

            # get action from policy
            act = self.policy(ob=self.obs)
            
            # play action
            next_obs, r, done, _ = self.env.step(act)

            # compute reward
            total_reward += r
            success = self.env.is_success()["task"]

            # visualization
            #if render:
                #   self.env.render(mode="human", camera_name=camera_names[0])

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(robot_state["states"])


            # Record each frame
            if self.video_record:
                rendered_img = self.env.render(mode="rgb_array", width=300, height=300)
                self.save_rendered_frame(rendered_img)


            # break if done or if success
            if done or success:
                break

            # update for next iter
            self.obs = deepcopy(next_obs)
            # st = self.env.get_state()
            
        stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))
    
        
        print(stats)
        with open(f"{self.log_dir}/stats.txt", "a") as file:
            file.write(str(stats) + '\n')
            
        
        done = True

        if stats['Success_Rate'] == 1.0:
            reward = -1 * (500/stats['Horizon']) * (1000/penalty + 1) - same_action_penalty
            done = False
            self.env.reset_to(robot_state)
        else:
            reward = 10000 / (penalty + 1) - same_action_penalty
            print('Failure Found, episode Completed\n')

        if done and self.video_record:
            self.video_writer.release()
            print(f"Episode {self.steps - 1} video saved successfully.")
        
        print(f"Steps : {self.steps}")

        if self.is_sequence:
            return self.obs["agentview_image"][0], reward, done, {}
            
        return self.obs["agentview_image"], reward, done, {}
        
    def save_rendered_frame(self, img_array):
        # Convert to uint8 if necessary and ensure BGR format for OpenCV
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        # Convert RGB to BGR (OpenCV expects BGR format)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Write the rendered frame to the video
        self.video_writer.write(img_bgr)
