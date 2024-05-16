import copy
from PIL import Image
import math
import numpy as np

NO_VALID_PUT = "No valid positions to place object found"
PUT_COLLISION = "another object's collision is blocking held object from being placed"

AI2THOR_ACTIONS = [
    'walk_to_room',
    'walk_to_object',
    'grab',
    'turnleft',
    'turnright',
    'put',
    'open',
    'close',
    'cook',
    'switchon',
    'switchoff',
    'slice',
    'cut',
    'moveforward',
    'movebackward',
    'turnaround',
    'lookup',
    'lookdown',
    'scanroom',
    'wash'
]

AI2THOR_NO_TARGET = [
    'moveforward',
    'movebackward',
    'turnaround',
    'lookup',
    'lookdown',
    'turnleft',
    'turnright',
]

AI2THOR_PREDICATES = [
    'visible',
    'isInteractable',
    'receptacle',
    'toggleable',
    'isToggled',
    'breakable',
    'isBroken',
    'canFillWithLiquid',
    'isFilledWithLiquid',
    'dirtyable',
    'isDirty',
    'canBeUsedUp',
    'isUsedUp',
    'cookable',
    'isCooked',
    'isHeatSource',
    'isColdSource',
    'sliceable',
    'isSliced',
    'openable',
    'isOpen',
    'pickupable',
    'isPickedUp',
    "inRoom",
    'moveable'
]

AI2THOR_TO_VHOME = {
    'receptacle': 'SURFACES',
    'toggleable': 'HAS_SWITCH',
    'isToggled': 'ON',
    'isDirty': 'DIRTY',
    'isCooked': 'COOKED',
    'openable': 'CAN_OPEN',
    'pickupable': 'GRABBABLE',
    'isPickedUp': 'HOLDS',
    # 'onTop': 'ON_TOP',
    'moveable': "MOVEABLE",
    "isOpen": 'OPEN',
    "isSliced": "SLICED",
    "fillLiquid": "FILLED"
}

CLOSE_DISTANCE = 1.75

class Event:
    def __init__(self):
        self.metadata = {
            "lastActionSuccess": "",
            "errorMessage": ""
        }

def get_vhome_to_thor_dict():
    vhome_to_thor = {}
    for key, val in AI2THOR_TO_VHOME.items():
        vhome_to_thor[val] = key
    vhome_to_thor["CONTAINER"] = 'receptacle'
    return vhome_to_thor

def get_visible_objects(object_meta):
    visible = []
    for object in object_meta:
        if object["visible"]:
            visible.append(object)
    return visible

def get_predicates(objects):
    predicates = []
    for object in objects:
        for pred in AI2THOR_TO_VHOME.keys():
            if object[pred]:
                predicates.append(f"{AI2THOR_TO_VHOME[pred]} {object['objectId']}")
        if object['distance'] < CLOSE_DISTANCE:
            predicates.append(f"close {object['objectId']} character_1")
        if object['parentReceptacles'] is not None:
            for container in object["parentReceptacles"]:
                predicates += [f"IN {object['objectId']} {container}", f"ON {object['objectId']} {container}"]
    return predicates

def is_in_room(point, polygon):
    """
    Determine if the point is in the polygon.

    Args:
    point -- A tuple (x, y) representing the point to check.
    polygon -- A list of tuples [(x, y), (x, y), ...] representing the vertices of the polygon.

    Returns:
    True if the point is in the polygon; False otherwise.
    """
    if polygon == -1:
        return True
    x, z, y = point['x'], point['y'], point['z']
    inside = False

    for i in range(len(polygon)):
        point = polygon[i]
        x1, z1, y1 = point['x'], point['y'], point['z']
        point = polygon[(i + 1) % len(polygon)]
        x2, Z2, y2 = point['x'], point['y'], point['z']
        xinters = 0
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or x <= xinters:
                inside = not inside

    return inside

def is_facing(reference_position, reference_rotation, target_position):
    """
    Check if the reference position with given rotation is facing the target position.

    Args:
    reference_position -- A dict {'x': x_val, 'y': y_val, 'z': z_val} for the reference position.
    reference_rotation -- A dict {'x': x_val, 'y': y_val, 'z': z_val} for the reference rotation in degrees.
    target_position -- A dict {'x': x_val, 'y': y_val, 'z': z_val} for the target position.

    Returns:
    True if the reference position is facing the target position; False otherwise.
    """

    # Calculate forward vector from rotation
    yaw = math.radians(reference_rotation['y'])
    forward_vector = {
        'x': math.cos(yaw),
        'z': math.sin(yaw)
    }

    # Calculate the vector from reference to target position
    direction_to_target = {
        'x': target_position['x'] - reference_position['x'],
        'z': target_position['z'] - reference_position['z']
    }

    # Normalize direction_to_target vector
    mag = math.sqrt(direction_to_target['x'] ** 2 + direction_to_target['z'] ** 2)
    direction_to_target = {k: v / mag for k, v in direction_to_target.items()}

    # Dot product of forward_vector and direction_to_target
    dot_product = forward_vector['x'] * direction_to_target['x'] + forward_vector['z'] * direction_to_target['z']

    # Check if the reference is facing the target (dot product close to 1)
    return dot_product > .25  # Adjust this threshold as needed

def find_closest_position(point, orientation, positions, radius = .5, facing=False):
    """
    Find the closest position to a reference point.

    Args:
    point -- A dict {'x': x_value, 'y': y_value, 'z': z_value} representing the reference point.
    positions -- A list of dicts [{'x': x_value, 'y': y_value, 'z': z_value}, ...] representing the valid positions.

    Returns:
    A dict representing the closest position.
    """

    def distance(p1, p2):
        """Calculate Euclidean distance ignoring y coordinate."""
        return ((p1['x'] - p2['x']) ** 2 + (p1['z'] - p2['z']) ** 2) ** 0.5

    min_distance = float('inf')
    closest_position = None

    for pos in positions:
        dist = distance(point, pos)
        if min_distance > dist >= radius:
            if facing and not is_facing(point, orientation, pos):
                continue
            min_distance = dist
            closest_position = pos

    return closest_position


def get_top_down_frame(sim):
    # Setup the top-down camera
    event = sim.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = sim.controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def check_close(object):
    return object["distance"] <= CLOSE_DISTANCE
def get_object_properties_and_states(state):
    global PREDICATES
    object_properties_states = {}
    object_properties_states["HOLDS"] = {}
    object_properties_states["CLOSE"] = {}
    object_properties_states["FAR"] = {}
    object_properties_states["IN"] = {}
    object_properties_states["INSIDE"] = {}

    object_properties_states["ON_TOP"] = {}
    object_properties_states["CLOSED"] = {}
    object_properties_states["CONTAINERS"] = {}

    vhome_to_thor = get_vhome_to_thor_dict()
    for obj in state["objects"]:
        is_close = check_close(obj)
        if is_close:
            object_properties_states["CLOSE"][obj["objectId"]] = obj
        else:
            object_properties_states["FAR"][obj["objectId"]] = obj
        if obj["openable"] and not obj["isOpen"]:
            object_properties_states["CLOSED"][obj["objectId"]] = obj
        for pred in vhome_to_thor:
            if pred not in object_properties_states:
                object_properties_states[pred] = {}
            elif obj.get(vhome_to_thor[pred], False):
                object_properties_states[pred][obj["objectId"]] = obj
        if obj['fillLiquid'] not in [None, ""]:
            object_properties_states["FILLED"][(obj["objectId"], obj['fillLiquid'])] = obj
        if obj["receptacleObjectIds"] is not None:
            for cont_obj in obj['receptacleObjectIds']:
                object_properties_states["ON_TOP"][(cont_obj, obj["objectId"])] = obj
                object_properties_states["IN"][(cont_obj, obj['objectId'])] = obj
                object_properties_states["INSIDE"][(cont_obj, obj['objectId'])] = obj


    return object_properties_states

def preds_dict_to_set(object_properties_states):
    new_dict = {}
    for key in object_properties_states:
        new_dict[key] = set(object_properties_states[key].keys())
    return new_dict


def get_world_predicate_set(graph, custom_preds=()):
    return set(get_predicates(graph['objects']))


# def get_yaw_angle(pose1, pose2):
#     direction_vector = {
#         'x': pose1['x'] - pose2['x'],
#         'y': pose1['y'] - pose2['y'],
#         'z': pose1['z'] - pose2['z'],
#     }
#
#     angle = math.degrees(math.atan2(direction_vector['z'], direction_vector['x']))
#
#     # Normalize angle to be in the range [0, 360)
#     angle = angle % 360
#
#     return angle

def get_yaw_angle(pose1, orientation, pose2):
    x1, y1 = pose1['x'], pose1['z']
    x2, y2 = pose2['x'], pose2['z']
    # Calculate the angle from the first point to the second point
    angle_to_second_point = math.degrees(math.atan2( x2 - x1, y2 - y1))

    # Adjust for the orientation of the first point
    relative_angle = angle_to_second_point - orientation['y']

    # Normalize the angle to be between 0 and 360 degrees
    relative_angle = relative_angle % 360

    return relative_angle

def get_room_polygon(scene, room):
    try:
        room = [room for room in scene['rooms'] if room["roomType"].lower() == room][0]
        return room["floorPolygon"]
    except:
        raise Exception(f"Failed to find room {room}.")

def get_inf_floor_polygon():
    return -1