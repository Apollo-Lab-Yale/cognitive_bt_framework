import xml.etree.ElementTree as ET

BASE_EXAMPLE = '''
<?xml version="1.0"?>
<root>
  <selector>
    <sequence name="Mug Washing Task">
      <!-- Navigate to the kitchen -->
      <action name="walk_to_room" target="kitchen"/>
      <!-- Attempt to locate the sink -->
      <selector name="Locate Sink">
        <!-- Check if the sink is visible -->
        <condition name="visible" target="sink"/>
        <!-- If not visible, scan the room or adjust position -->
        <sequence name="Adjust and Scan">
          <action name="scan_room" target="sink"/>
          <condition name="visible" target="sink"/>
        </sequence>
      </selector>
      <!-- Navigate to the sink -->
      <action name="walk_to_object" target="sink"/>
      <!-- Ensure the mug is present and interactable -->
      <condition name="isInteractable" target="mug"/>
      <!-- Check if the mug is dirty -->
      <condition name="isDirty" target="mug"/>
      <!-- Pick up the mug -->
      <action name="grab" target="mug"/>
      <!-- Sequence to wash the mug -->
      <sequence name="Washing Sequence">
        <!-- Check if the sink is a receptacle -->
        <condition name="receptacle" target="sink"/>
        <!-- Check if the sink is toggleable -->
        <condition name="toggleable" target="sink"/>
        <!-- Check if the sink is filled with liquid -->
        <condition name="isFilledWithLiquid" target="sink"/>
        <!-- Open the sink -->
        <action name="open" target="sink"/>
        <!-- Wash the mug -->
        <action name="putin" target="sink"/>
        <!-- Close the sink -->
        <action name="close" target="sink"/>
      </sequence>
      <!-- Put the clean mug aside -->
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

class Action(Node):
    def __init__(self, name, target):
        self.name = name
        self.target = target

    def execute(self, state, interface, memory):
        # Logic to execute the action
        print(f"Executing action: {self.name}")
        return interface.execute_actions([f"{self.name} {self.target}"], memory)# Modify and return the new state

class Condition(Node):
    def __init__(self, name, target):
        super().__init__(name)
        self.target = target

    def execute(self, state, interface, memory):
        # Evaluate the condition against the state
        return interface.check_condition(self.name, self.target, memory)

class Selector(Node):
    def __init__(self, name):
        super().__init__(name)
        self.failures = []

    def execute(self, state, interface, memory):
        for child in self.children:
            success, msg = child.execute(state, interface, memory)
            if success:
                return success, msg
            self.failures.append(msg)
        print(f"Selector {self.name} has {self.failures} failures")
        return False, f"{self.name} failed due to the failures: {self.failures}"

class Sequence(Node):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, state, interface, memory):
        for child in self.children:
            success, msg = child.execute(state, interface, memory)
            if not success:
                print(f"Sequence {self.name} failed to execute any children due to the failures: {msg}")
                return False, f"Sequence {self.name} failed to execute {child.name}: {msg}."
        return True, ""



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
