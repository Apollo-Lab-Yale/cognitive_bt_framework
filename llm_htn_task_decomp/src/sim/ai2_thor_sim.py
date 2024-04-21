import cv2
from ai2thor.controller import Controller
import numpy as np
import time
import prior
from PIL import Image
import random
from llm_htn_task_decomp.src.sim.utils import get_visible_objects, get_predicates, CLOSE_DISTANCE, find_closest_position, is_in_room, get_yaw_angle, get_vhome_to_thor_dict, get_inf_floor_polygon, \
    NO_VALID_PUT, Event, PUT_COLLISION
from llm_htn_task_decomp.src.sim.image_saver import SaveImagesThread

from llm_htn_task_decomp.utils.logic_utils import parse_instantiated_predicate
def get_ithor_scene_single_room(room, index = -1):
    broken_rooms = [2, 5, 17, 22]
    if index > 0:
        return f"FloorPlan{index}"
    kitchens = [f"FloorPlan{i}" for i in range(1, 31) if i not in broken_rooms]
    living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 31)]
    bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 31)]
    bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 31)]
    rooms = []
    if "kitchen" in room.lower():
        rooms = kitchens
    if "livingroom" in room.lower().replace("_", ""):
        rooms = living_rooms
    if "bedroom" in room.lower():
        rooms = bedrooms
    if "bathroom" in room.lower():
        rooms = bathrooms
    assert len(rooms) > 0
    return np.random.choice(rooms)

class AI2ThorSimEnv:
    def __init__(self, scene_index=-1, width=600, height=600, gridSize=0.25, visibilityDistance=20, single_room='kitchen', save_video=False, use_find = False):
        self.single_room = single_room
        self.scene = None
        self.controller = None
        self.comm = self
        self.image_saver = None
        self.save_video = save_video
        self.use_find = use_find
        self.reset(scene_index, width, height, gridSize, visibilityDistance, single_room, save_video)
        self.valid_positions = self.get_valid_positions()
        self.object_waypoints = {}
        self.action_fn_from_str = {
            "walk_to_room": self.navigate_to_room,
            "walk_to_object": self.navigate_to_object,
            "grab": self.grab_object,
            "turnleft": self.turn_left,
            "turnright": self.turn_right,
            "put": self.place_object,
            "putin": self.place_object,
            "open": self.open_object,
            "close": self.close_object,
            "cook": self.cook_object,
            "clean": self.clean_object,
            "switchon": self.turn_object_on,
            "switchoff": self.turn_object_off,
            "slice": self.slice_object,
            "find": self.cheat_and_find_object,
            "give-up": self.give_up,
            "cut": self.slice_object,
            "moveforward": self.move_forward,
            "movebackward": self.move_back,
            "turnaround": self.turn_around,
            "lookup": self.look_up,
            "lookdown": self.look_down
        }

    def reset(self, scene_index=-1, width=600, height=600, gridSize=0.25, visibilityDistance=10, single_room='kitchen', save_video=False):
        scene = None
        self.save_video = save_video
        if single_room is not None and single_room != '':
            scene = get_ithor_scene_single_room(single_room, index=scene_index)
        else:
            dataset = prior.load_dataset("procthor-10k")
            scene = dataset["train"][scene_index]
        self.scene = scene
        if self.controller is None:
            self.controller = Controller(scene=scene, agentMode="default",
                                         visibilityDistance=visibilityDistance, width=width,
                                         height=height, allowAutoSimulation=True, gridSize=gridSize)
        self.controller.reset(scene)
        self.scene = {"rooms": [{"roomType": single_room, "floorPolygon": get_inf_floor_polygon(), "name":scene}]}
        print(self.scene)
        if self.save_video:
            if self.image_saver is not None:
                self.image_saver.keep_running = False
            self.image_saver = SaveImagesThread(controller=self.controller)
            self.image_saver.start()

    def end_sim(self):
        self.controller.stop()
        if self.image_saver is not None:
            self.image_saver.keep_running = False

    def get_object_location(self, object_type):
        objects = self.controller.last_event.metadata['objects']
        for obj in objects:
            if obj['objectType'] == object_type:
                return obj['position']
        return None

    def turn_around(self):
        act = np.random.choice([self.turn_left, self.turn_right])
        return act(180)

    def get_state(self, goal_objs=None):
        event = self.controller.last_event.metadata
        state = {}
        state["objects"] = get_visible_objects(event["objects"])
        for object in state["objects"]:
            if 'surface' in object['name']:
                object['onTop'] = object['parentReceptacles']
        state["predicates"] = get_predicates(state["objects"])
        state["room_names"] = [room['roomType'] for room in self.scene['rooms']]
        state["robot_state"] = event['agent']
        state["memory_dict"] = {}
        return state

    def get_robot_state(self, state, robot = "character", relations=None):
        state = self.get_state()
        nl_state = ''
        agent_pose = state["robot_state"]["position"]
        robot_room = None
        for room in state["room_names"]:
            polygon = [room_obj for room_obj in self.scene["rooms"] if room_obj["roomType"]==room][0]["floorPolygon"]
            if is_in_room(agent_pose, polygon):
                robot_room = room

        holding = [obj['objectId'] for obj in state["objects"] if obj["isPickedUp"]]
        # nl_state = f"I am in the {robot_room}."
        if len(holding) == 0:
            nl_state+= " I am not holding anything."
        else:
            holding_str = ", ".join(holding)
            nl_state+=f" I am holding the {holding_str}."
        return nl_state, robot_room

    def cheat_and_find_object(self, object):
        print(object)
        try:
            obj = [obj for obj in self.get_graph()['objects'] if object.lower() in obj['objectId'].lower()][0]
            if self.use_find:
                self.navigate_to_object(obj)
            self.handle_scan_room(obj['objectId'], memory=None, pause_time=0)
        except Exception as e:
            print(f"Failed to find object {object}")

        return self.controller.last_event

    def done(self):
        pass
        # return self.controller.step(action="Done")

    def turn_left(self, degrees=90):
        return self.controller.step("RotateLeft", degrees=degrees)

    def turn_right(self, degrees=90):
        return self.controller.step("RotateRight", degrees=degrees)

    def look_up(self, degrees=10):
        return self.controller.step("LookUp", degrees=degrees)

    def look_down(self, degrees=10):
        return self.controller.step("LookDown", degrees=degrees)

    def move_forward(self, meters=1):
        return self.controller.step("MoveAhead", moveMagnitude=meters)

    def move_back(self, meters=0.25):
        return self.controller.step("MoveBack", moveMagnitude=meters)

    def give_up(self):
        print("womp womp")
        return self.navigate_to_room(target_room_str=self.single_room)

    def navigate_to_room(self, target_room_str="bedroom"):
        target_room = None
        for room in self.scene["rooms"]:
            if room["roomType"].lower() == target_room_str.lower():
                target_room = room
        assert target_room is not None
        positions = [pose for pose in self.valid_positions if is_in_room(pose, target_room["floorPolygon"])]
        teleport_pose = np.random.choice(positions)
        return self.controller.step(
            action="Teleport",
            position=teleport_pose,
            rotation={'x': 0.0, "y": np.random.uniform(0, 360), "z": 0.0},
            forceAction=False
        )

    def grab_object(self, object):
        assert object["distance"] <= CLOSE_DISTANCE
        ret =  self.controller.step("PickupObject",
                             objectId=object["objectId"],
                             forceAction=False,
                             manualInteract=True)
        self.controller.step("MoveHeldObjectUp", moveMagnitude=0.1, forceVisible=True)
        for i in range(20):
            self.controller.step("MoveHeldObjectUp", moveMagnitude=0.01, forceVisible=True)
            ret = self.controller.step("MoveHeldObjectBack", moveMagnitude=0.01, forceVisible=True)
        return ret

    def place_object(self, target):
        if target['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {target['objectId']} to place object."
            return fail_event
        return self.controller.step(
            action="PutObject",
            objectId=target["objectId"],
            forceAction=False,
            placeStationary=False
        )

    def get_interactable_poses(self, object):
        positions = self.controller.step(
                action="GetInteractablePoses",
                objectId=object['objectId']
            ).metadata['actionReturn']
        return positions

    def navigate_to_object(self, object):
        positions = self.get_valid_positions()
        teleport_pose = find_closest_position(object["position"], object["rotation"], positions, 1.2, facing=False)
        # teleport_pose = np.random.choice(positions)
        assert teleport_pose is not None
        event = self.controller.step(
            action="Teleport",
            position=teleport_pose,
            rotation={'x': 0.0, 'y': get_yaw_angle(teleport_pose,{'y':0.0}, object['position']), 'z': 0.0},
            forceAction=True
        )
        self.handle_scan_room(object['objectId'], None)
        return event

    def open_object(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['objectId']} to place object."
            return fail_event
        event = self.controller.step(
            action="OpenObject",
            objectId=object['objectId'],
            openness=1,
            forceAction=False
        )

        # Image.fromarray(event.frame).show("str")
        return event

    def close_object(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['objectId']} to place object."
            return fail_event
        return self.controller.step(
            action="CloseObject",
            objectId=object['objectId'],
            forceAction=False
        )

    def cook_object(self, object):
        return self.controller.step(
            action="CookObject",
            objectId=object['objectid'],
            forceAction=False
        )

    def slice_object(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['objectId']} to place object."
            return fail_event
        return self.controller.step(
            action="SliceObject",
            objectId=object["objectId"],
            forceAction=False
        )

    def turn_object_on(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['objectId']} to place object."
            return fail_event
        return self.controller.step(
            action="ToggleObjectOn",
            objectId=object["objectId"],
            forceAction=False
        )

    def turn_object_off(self, object):
        if object['distance'] > CLOSE_DISTANCE:
            fail_event = Event()
            fail_event.metadata['lastActionSuccess'] = False
            fail_event.metadata['errorMessage'] = f"Not close enough to {object['objectId']} to place object."
            return fail_event
        return self.controller.step(
            action="ToggleObjectOff",
            objectId=object["objectId"],
            forceAction=False
        )

    def get_valid_positions(self):
        return self.controller.step(
                    action="GetReachablePositions").metadata["actionReturn"]

    def make_object_dirty(self, object):
        return  self.controller.step(
            action="DirtyObject",
            objectId=object["objectId"],
            forceAction=True
        )

    def clean_object(self, object):

        return self.controller.step(
            action="CleanObject",
            objectId=object["objectId"],
            forceAction=True
        )

    def fill_with_liquid(self, object, liquid="water"):
        return self.controller.step(
            action="FillObjectWithLiquid",
            objectId=object["objectId"],
            fillLiquid=liquid,
            forceAction=False
        )

    def empty_liquid(self, object):
        return self.controller.step(
            action="EmptyLiquidFromObject",
            objectId=object["objectId"],
            forceAction=False
        )

    def handle_fry_in_pan(self, object):
        pass

    def drop_object(self):
        return self.controller.step(action="DropHandObject",
                             forceAction=False)

    def handle_scan_room(self, goal_obj, memory, pause_time = 0.5):
        print(f"scanning for object {goal_obj}")
        action = random.choice([self.turn_left, self.turn_right])
        for i in range(12):
            evt = action(degrees=30)
            time.sleep(pause_time)
            self.done()
            state = self.get_state()
            if not self.controller.last_event.metadata["lastActionSuccess"]:
                return self.controller.last_event.metadata["lastActionSuccess"], self.controller.last_event.metadata[
                    "errorMessage"]
            if any([goal_obj == object["objectId"] for object in state['objects']]):
                return self.controller.last_event.metadata["lastActionSuccess"], self.controller.last_event.metadata["errorMessage"]
        return False, self.controller.last_event.metadata["errorMessage"]

    def translate_action_for_sim(self, action, state):
        return [action]

    def add_object_waypoint(self, x, y):
        pass

    def get_graph(self):
        return self.controller.last_event.metadata

    def environment_graph(self):
        graph = self.get_graph()
        graph['nodes'] = graph['objects']
        for node in graph['nodes']:
            node['class_name'] = node['objectType']
            node['id'] = '|'.join(node['objectId'].split('|')[1:])
        return True, graph

    def check_cooked(self):
        pass

    def execute_actions(self, actions, state):
        event = self.controller.last_event
        for action in actions:
            state = self.get_state()
            act, params = parse_instantiated_predicate(action)
            objects = []
            print(f"Action: {action}")
            if "find" in action:
                event = self.cheat_and_find_object(params[0])
                print(event.metadata["lastActionSuccess"], event.metadata['errorMessage'])
                continue
            new_action = f"{action}"
            if act == 'walk':
                new_action = f"walk_to_object"
                if any(param in state['room_names'] for param in params):
                    new_action = f"walk_to_room"
                act = new_action
            for param in params:
                if "character" in param:
                    continue
                object = [obj for obj in state["objects"] if obj["objectId"] == param]
                object = None if len(object) == 0 else object[0]
                if object is None and param not in state["room_names"]:
                    return False, f"I cannot currently see a {param} to complete action {action}."
                elif param in state["room_names"]:
                    object = param
                objects.append(object)
            event = self.controller.last_event
            try:
                if len(objects) == 0:
                    event = self.action_fn_from_str[act]()
                else:
                    target = objects[0] if len(objects) == 1 else objects[1]
                    event = self.action_fn_from_str[act](target)
                    print(event.metadata["lastActionSuccess"], event.metadata['errorMessage'])
                time.sleep(0.5)
                self.move_back(0.01)
            except Exception as e:
                print(f"Failed to edecute action {act} due to: {e}")
        return event.metadata["lastActionSuccess"], event.metadata['errorMessage']

    def get_world_predicate_set(self, graph, custom_preds=()):
        return set(get_predicates(graph['objects']))

    def check_obj_contained(self, obj, receptacle):
        state = self.get_graph()
        receptacle = [object for object in state["objects"] if object["objectId"] == receptacle][0]
        print(obj)
        print(receptacle)
        print(obj in receptacle['receptacleObjectIds'])
        return obj in receptacle['receptacleObjectIds']

    def failure_msg_to_blocking_mode(self, msg, action):
        act, params = parse_instantiated_predicate(action)
        if msg == NO_VALID_PUT or msg == PUT_COLLISION:
            return f"no-placement {params[1]}"
        return None


    def check_satisfied(self, predicates, sub_goal):
        state = self.get_graph()
        to_remove = []
        success = False
        pred, params = parse_instantiated_predicate(sub_goal)
        if "WASHED" in pred:
            if "SINK" in pred:
                faucet_str = params[2]
                sink_str = params[1]
                item_str = params[0]
                faucet = [obj for obj in state['objects'] if obj['objectId'] == faucet_str]
                sink = [obj for obj in state['objects'] if obj['objectId'] == sink_str]

                if len(faucet) > 0 and len(sink)>0:
                    faucet = faucet[0]
                    sink = sink[0]

                    if faucet['isToggled'] and item_str in sink['receptacleObjectIds']:
                        self.clean_object({"objectId": item_str})
            param = params[0] if "character" not in params[0] else params[1]
            obj = [object for object in state["objects"] if object["objectId"] == param][0]
            success = not obj["isDirty"]
            print(f"check sat: {obj}")
        elif "INSIDE" in sub_goal or "IN" in sub_goal or ("ON" in sub_goal and len(params) == 2):
            success = self.check_obj_contained(params[0], params[1])
        elif "ACTIVE" in sub_goal:
            param = sub_goal.split()[1]
            obj= [object for object in state["objects"] if object["objectId"] == param][0]
            success  = obj['isToggled']
        # elif "FILL" in sub_goal:
        #     objs = [obj for obj in state["objects"] if obj["fillLiquid"] in params]
        #     success = len(objs) > 0
        else:
            vhome_to_aithor = get_vhome_to_thor_dict()

            key = vhome_to_aithor.get(pred, None)
            if key is None:
                print(f"failed to find predicate: {pred} ")
                return False, to_remove
            param = params[0] if "character" not in params[0] else params[1]
            obj = [object for object in state["objects"] if object["objectId"] ==param]
            if len(obj) == 0:
                return False, to_remove
            obj = obj[0]
            success = obj[key]
        if success:
            to_remove.append(sub_goal)
        return success, to_remove
