import xml.etree.ElementTree as ET

BASE_EXAMPLE = '''
<?xml version="1.0"?>
<root>
  <selector>
    <sequence name="Navigate to Sink">
      <action name="Navigate_to_Room" target="kitchen"/>
      <action name="Navigate_to_Object" target="sink"/>
    </sequence>
    <action name="AlertHuman"/>
  </selector>
</root>
'''


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def execute(self, state, interface):
        raise NotImplementedError("This method should be overridden by subclasses")

class Action(Node):
    def __init__(self, name, target):
        self.name = name
        self.target = target

    def execute(self, state, interface):
        # Logic to execute the action
        print(f"Executing action: {self.name}")
        return interface.execute_actions(state, f"{self.name} {self.target}")# Modify and return the new state

class Condition(Node):
    def __init__(self, name, condition):
        super().__init__(name)
        self.condition = condition

    def execute(self, state, interface):
        # Evaluate the condition against the state
        return self.condition(state)

class Selector(Node):
    def execute(self, state, interface):
        for child in self.children:
            success, msg = child.execute(state, interface)
            if child.execute(state, interface):
                return True
        return False

class Sequence(Node):
    def execute(self, state, interface):
        for child in self.children:
            if not child.execute(state, interface):
                return False
        return True



def parse_bt_xml(xml_content):
    print(xml_content)
    root = ET.fromstring(xml_content)
    return parse_node(root)

def parse_node(element):
    node_type = element.tag.capitalize()
    if node_type == "Action":
        node = Action(element.get('name'), element.get('target'))
        return node
    elif node_type == "Condition":
        condition = lambda state: eval(element.get('condition'))
        return Condition(element.get('name'), condition)
    elif node_type in ["Selector", "Sequence"]:
        node_class = globals()[node_type]
        node = node_class(element.get('name'))
        for child_elem in element:
            child = parse_node(child_elem)
            node.children.append(child)
        return node
    elif node_type == "Root":
        node = Sequence('root')
        for child_elem in element:
            child = parse_node(child_elem)
            node.children.append(child)
        return node
    raise ValueError(f"Unsupported node type: {node_type}")
