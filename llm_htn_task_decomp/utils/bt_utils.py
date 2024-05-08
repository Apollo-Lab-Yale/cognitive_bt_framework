import xml.etree.ElementTree as ET

BASE_EXAMPLE = '''
<?xml version="1.0"?>
<root>
  <selector>
    <sequence name="Tea Making Task">
      <!-- Navigate to the kitchen -->
      <action name="walk_to_room" target="kitchen"/>
      <!-- Locate the kettle -->
      <selector name="Locate Kettle">
        <!-- Check if the kettle is visible -->
        <condition name="visible" target="kettle"/>
        <!-- If not visible, scan the room -->
        <sequence name="Adjust and Scan Kettle">
          <action name="scanroom" target="kettle"/>
          <condition name="visible" target="kettle"/>
        </sequence>
      </selector>
      <!-- Navigate to the kettle -->
      <action name="walk_to_object" target="kettle"/>
      <!-- Fill kettle if needed -->
      <selector name="Check and Fill Kettle">
        <!-- Check if kettle is filled with liquid -->
        <condition name="isFilledWithLiquid" target="kettle"/>
        <!-- If kettle is not filled, fill it -->
        <sequence name="Fill Kettle">
          <action name="open" target="kettle"/>
          <action name="putin" target="water_source"/>  <!-- Assume there is a target 'water_source' -->
          <action name="close" target="kettle"/>
        </sequence>
      </selector>
      <!-- Turn on the kettle -->
      <action name="switchon" target="kettle"/>
      <!-- Wait for water to boil -->
      <condition name="isBoiled" target="kettle"/>  <!-- Assuming 'isBoiled' is a valid condition -->
      <!-- Locate the mug -->
      <selector name="Locate Mug">
        <!-- Check if the mug is visible -->
        <condition name="visible" target="mug"/>
        <!-- If not visible, scan the room -->
        <sequence name="Adjust and Scan Mug">
          <action name="scanroom" target="mug"/>
          <condition name="visible" target="mug"/>
        </sequence>
      </selector>
      <!-- Navigate to the mug -->
      <action name="walk_to_object" target="mug"/>
      <!-- Pick up the mug -->
      <action name="grab" target="mug"/>
      <!-- Locate the tea bag -->
      <selector name="Locate Tea Bag">
        <!-- Check if the tea bag is visible -->
        <condition name="visible" target="tea_bag"/>
        <!-- If not visible, scan the room -->
        <sequence name="Adjust and Scan Tea Bag">
          <action name="scanroom" target="tea_bag"/>
          <condition name="visible" target="tea_bag"/>
        </sequence>
      </selector>
      <!-- Put the tea bag in the mug -->
      <action name="putin" target="mug"/>
      <!-- Pour hot water into the mug -->
      <action name="putin" target="mug"/>
      <!-- Let the tea steep -->
      <condition name="isSteeped" target="mug"/>  <!-- Assuming 'isSteeped' is a valid condition -->
      <!-- Remove the tea bag -->
      <action name="put" target="trash"/>
      <!-- Put the mug aside -->
      <action name="put" target="drying_rack"/>
    </sequence>
  </selector>
</root>

'''


class Node:
    def __init__(self, name):
        self.name = name
        self.children = []

    def execute(self, state, interface, memory):
        raise NotImplementedError("This method should be overridden by subclasses")

    def to_xml(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class Action(Node):
    def __init__(self, name, target):
        self.name = name
        self.target = target

    def execute(self, state, interface, memory):

        # Logic to execute the action
        print(f"Executing action: {self.name}")
        return *interface.execute_actions([f"{self.name} {self.target}"], memory), self.to_xml()# Modify and return the new state

    def to_xml(self):
        return f'<action name="{self.name}" target="{self.target}"/>'

class Condition(Node):
    def __init__(self, name, target):
        super().__init__(name)
        self.target = target

    def execute(self, state, interface, memory):
        # Evaluate the condition against the state
        success, msg = interface.check_condition(self.name, self.target, memory)
        if 'These objects satisfy the condition' in msg:
            msg = f"{self.name} is FALSE for object {self.target}. " + msg
        return success, msg, self.to_xml()

    def to_xml(self):
        return f'<condition name="{self.name}" target="{self.target}"/>'

class Selector(Node):
    def __init__(self, name):
        super().__init__(name)
        self.failures = []

    def execute(self, state, interface, memory):
        for child in self.children:
            success, msg, _ = child.execute(state, interface, memory)
            if success:
                return success, msg, ""
            self.failures.append(msg)
        print(f"{self.failures}")
        return False, f"{self.failures[-1]}", self.to_xml()

    def to_xml(self):
        children_xml = "\n".join(child.to_xml() for child in self.children)
        return f'<selector name="{self.name}">{children_xml}</selector>'


class Sequence(Node):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, state, interface, memory):
        for child in self.children:
            success, msg, child_xml = child.execute(state, interface, memory)
            if not success:
                print(f"{msg}")
                ret_xml = child_xml
                if type(child) not in [Sequence, Condition]:
                    ret_xml = self.to_xml()
                return False, f"{msg}.", self.to_xml()
        return True, "", ""

    def to_xml(self):
        children_xml = "\n".join(child.to_xml() for child in self.children)
        return f'<sequence name="{self.name}">{children_xml}</sequence>'



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
        target = element.get('target')
        return Condition(element.get('name'), target)
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
