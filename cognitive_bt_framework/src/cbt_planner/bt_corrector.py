from cognitive_bt_framework.utils.bt_utils import Node, Sequence, Selector, Action, Condition

class BTCorrector(object):
    def __init__(self):
        pass

OBJ_PLACEHOLDER = '<OBJ>'
ACT_PLACEHOLDER = '<ACT>'



action_effects = {
    "walk_to_object": ["isClose <OBJ>", "visible <OBJ>"],
    "scanroom": ["visible <OBJ>"],

}


LOCATE_OBJ_SUBTREE = '''
        <Sequence>
            <Selector>
                <Condition name="visible" target="<OBJ>" val=1/>
                <Action name="scanroom" target="<OBJ>" />
                <Sequence name="locate stored object"/>
                    <Action name="getPossibleContainers" target="<OBJ>"/>
                    <Action name="searchPossibleContainers" target="<OBJ>"/>
                <Sequence/>
            <Selector/>
            <Selector>
                <Condition name="isClose" target="<OBJ>" val=1/>
                <Action name="walk_to_object" target="<OBJ>"/>
            <Selector/>
            <Action name="queryObjAffordance" target="<OBJ>" recipient="<ACT>" val=1/>
            <Action name="<ACT>" target="<OBJ>"/>
        <Sequence/>
'''

def generate_enable_interact_subtree(obj, action):
    xml_tree = LOCATE_OBJ_SUBTREE
    xml_tree = xml_tree.replace(OBJ_PLACEHOLDER, obj).replace(ACT_PLACEHOLDER, action)
    return xml_tree


def navigate_tree(root, interface, conditions, actions):
    isSeq = isinstance(root, Sequence)
    isSelector = isinstance(root, Selector)
    if isSeq or isSelector:
        for child in root.children:
            if isinstance(child, Action):
                action = child.name
                target = child.target
                recipient = None
                isVisible = any("visible" in cond and target in cond for cond in conditions)
                isClose = any("isClose" in cond and target in cond for cond in conditions)