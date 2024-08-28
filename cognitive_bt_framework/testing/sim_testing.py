from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from scipy.spatial import distance
from ai2thor.controller import Controller
from typing import Tuple
import numpy as np
import re
import time
import math
import threading
import cv2

import os
from glob import glob
import shutil

'''
trooms = [4, 7, 9, 11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28]
'''

robots = [{'name': 'agent', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']},
          {'name': 'robot2', 'skills': ['GoToObject', 'OpenObject', 'CloseObject', 'BreakObject', 'SliceObject', 'SwitchOn', 'SwitchOff', 'PickupObject', 'PutObject', 'DropHandObject', 'ThrowObject', 'PushObject', 'PullObject']}]


# sim = AI2ThorSimEnv()
# tables = [obj for obj in sim.get_graph()['objects'] if 'table' in obj['name'].lower()]
# for table in tables:
#     print(table)

floor_no = 4

c = Controller( height=1000, width=1000)
c.reset("FloorPlan" + str(floor_no))
no_robot = len(robots)
# get reachabel positions
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = positions_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]

print(c.last_event.metadata['objects'])
print(c.last_event.metadata['agent'])
event = c.step(action="GetMapViewCameraProperties")
event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
def distance_pts(p1: Tuple[float, float, float], p2: Tuple[float, float, float]):
    return ((p1[0] - p2[0]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

def closest_node(node, nodes, no_robot, clost_node_location):
    crps = []
    distances = distance.cdist([node], nodes)[0]
    dist_indices = np.argsort(np.array(distances))
    for i in range(no_robot):
        pos_index = dist_indices[(i * 5) + clost_node_location[i]]
        crps.append (nodes[pos_index])
    return crps
action_queue = []

task_over = False
def GoToObject(robots, dest_obj):
    print("Going to ", dest_obj)
    # check if robots is a list

    if not isinstance(robots, list):
        # convert robot to a list
        robots = [robots]
    no_agents = len(robots)
    # robots distance to the goal
    dist_goals = [10.0] * len(robots)
    prev_dist_goals = [10.0] * len(robots)
    count_since_update = [0] * len(robots)
    clost_node_location = [0] * len(robots)

    # list of objects in the scene and their centers
    objs = list([obj["objectId"] for obj in c.last_event.metadata["objects"]])
    objs_center = list([obj["axisAlignedBoundingBox"]["center"] for obj in c.last_event.metadata["objects"]])

    # look for the location and id of the destination object
    for idx, obj in enumerate(objs):
        match = re.match(dest_obj, obj)
        if match is not None:
            dest_obj_id = obj
            dest_obj_center = objs_center[idx]
            break  # find the first instance

    dest_obj_pos = [dest_obj_center['x'], dest_obj_center['y'], dest_obj_center['z']]

    # closest reachable position for each robot
    # all robots cannot reach the same spot
    # differt close points needs to be found for each robot
    crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

    goal_thresh = 0.3
    # at least one robot is far away from the goal

    while all(d > goal_thresh for d in dist_goals):
        for ia, robot in enumerate(robots):
            robot_name = robot['name']
            agent_id = 0

            # get the pose of robot
            metadata = c.last_event.events[agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"],
                "horizon": metadata["agent"]["cameraHorizon"]}

            prev_dist_goals[ia] = dist_goals[ia]  # store the previous distance to goal
            dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])

            dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            print(ia, "Dist to Goal: ", dist_goals[ia], dist_del, clost_node_location[ia])
            if dist_del < 0.2:
                # robot did not move
                count_since_update[ia] += 1
            else:
                # robot moving
                count_since_update[ia] = 0

            if count_since_update[ia] < 15:
                action_queue.append(
                    {'action': 'ObjectNavExpertAction', 'position': dict(x=crp[ia][0], y=crp[ia][1], z=crp[ia][2]),
                     'agent_id': agent_id})
            else:
                # updating goal
                clost_node_location[ia] += 1
                count_since_update[ia] = 0
                crp = closest_node(dest_obj_pos, reachable_positions, no_agents, clost_node_location)

            time.sleep(0.5)

    # align the robot once goal is reached
    # compute angle between robot heading and object
    metadata = c.last_event.events[agent_id].metadata
    robot_location = {
        "x": metadata["agent"]["position"]["x"],
        "y": metadata["agent"]["position"]["y"],
        "z": metadata["agent"]["position"]["z"],
        "rotation": metadata["agent"]["rotation"]["y"],
        "horizon": metadata["agent"]["cameraHorizon"]}

    robot_object_vec = [dest_obj_pos[0] - robot_location['x'], dest_obj_pos[2] - robot_location['z']]
    y_axis = [0, 1]
    unit_y = y_axis / np.linalg.norm(y_axis)
    unit_vector = robot_object_vec / np.linalg.norm(robot_object_vec)

    angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
    angle = 360 * angle / (2 * np.pi)
    angle = (angle + 360) % 360
    rot_angle = angle - robot_location['rotation']

    if rot_angle > 0:
        action_queue.append({'action': 'RotateRight', 'degrees': abs(rot_angle), 'agent_id': agent_id})
    else:
        action_queue.append({'action': 'RotateLeft', 'degrees': abs(rot_angle), 'agent_id': agent_id})

    print("Reached: ", dest_obj)


def exec_actions():
    # delete if current output already exist
    cur_path = os.path.dirname(__file__) + "/*/"
    for x in glob(cur_path, recursive=True):
        shutil.rmtree(x)

    # create new folders to save the images from the agents
    for i in range(no_robot):
        folder_name = "agent_" + str(i + 1)
        folder_path = os.path.dirname(__file__) + "/" + folder_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # create folder to store the top view images
    folder_name = "top_view"
    folder_path = os.path.dirname(__file__) + "/" + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    img_counter = 0

    while not task_over:
        if len(action_queue) > 0:
            try:
                act = action_queue[0]
                if act['action'] == 'ObjectNavExpertAction':
                    multi_agent_event = c.step(
                        dict(action=act['action'], position=act['position'], agentId=act['agent_id']))
                    next_action = multi_agent_event.metadata['actionReturn']

                    if next_action != None:
                        multi_agent_event = c.step(action=next_action, agentId=act['agent_id'], forceAction=True)

                elif act['action'] == 'MoveAhead':
                    multi_agent_event = c.step(action="MoveAhead", agentId=act['agent_id'])

                elif act['action'] == 'MoveBack':
                    multi_agent_event = c.step(action="MoveBack", agentId=act['agent_id'])

                elif act['action'] == 'RotateLeft':
                    multi_agent_event = c.step(action="RotateLeft", degrees=act['degrees'], agentId=act['agent_id'])

                elif act['action'] == 'RotateRight':
                    multi_agent_event = c.step(action="RotateRight", degrees=act['degrees'], agentId=act['agent_id'])

                elif act['action'] == 'PickupObject':
                    multi_agent_event = c.step(action="PickupObject", objectId=act['objectId'], agentId=act['agent_id'],
                                               forceAction=True)

                elif act['action'] == 'PutObject':
                    multi_agent_event = c.step(action="PutObject", objectId=act['objectId'], agentId=act['agent_id'],
                                               forceAction=True)

                elif act['action'] == 'ToggleObjectOn':
                    multi_agent_event = c.step(action="ToggleObjectOn", objectId=act['objectId'],
                                               agentId=act['agent_id'], forceAction=True)

                elif act['action'] == 'ToggleObjectOff':
                    multi_agent_event = c.step(action="ToggleObjectOff", objectId=act['objectId'],
                                               agentId=act['agent_id'], forceAction=True)

                elif act['action'] == 'Done':
                    multi_agent_event = c.step(action="Done")

            except Exception as e:
                print(e)

            for i, e in enumerate(multi_agent_event.events):
                cv2.imshow('agent%s' % i, e.cv2img)
                f_name = os.path.dirname(__file__) + "/agent_" + str(i + 1) + "/img_" + str(img_counter).zfill(
                    5) + ".png"
                cv2.imwrite(f_name, e.cv2img)
            top_view_rgb = cv2.cvtColor(c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
            cv2.imshow('Top View', top_view_rgb)
            f_name = os.path.dirname(__file__) + "/top_view/img_" + str(img_counter).zfill(5) + ".png"
            cv2.imwrite(f_name, e.cv2img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            img_counter += 1
            action_queue.pop(0)


actions_thread = threading.Thread(target=exec_actions)
actions_thread.start()


GoToObject(robots[0], 'Fridge')