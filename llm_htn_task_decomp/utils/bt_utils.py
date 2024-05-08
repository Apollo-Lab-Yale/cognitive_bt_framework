import xml.etree.ElementTree as ET
from llm_htn_task_decomp.src.sim.ai2_thor.utils import AI2THOR_NO_TARGET

BASE_EXAMPLE = '''
<?xml version="1.0"?>
<root>
    <selector>
        <sequence name="Water Filling Task">
            <!-- Navigate to the room with the glass -->
            <action name="walk_to_room" target="kitchen"/>
            <!-- Locate the glass -->
            <selector name="Locate Glass">
                <!-- Check if the glass is visible -->
                <condition name="visible" target="glass"/>
                <!-- If not visible, scan the room -->
                <sequence name="Adjust and Scan Glass">
                    <action name="scanroom" target="glass"/>
                    <condition name="visible" target="glass"/>
                </sequence>
            </selector>
            <!-- Navigate to the glass -->
            <action name="walk_to_object" target="glass"/>
            <!-- Pick up the glass -->
            <action name="grab" target="glass"/>
            <!-- Locate the faucet -->
            <selector name="Locate Faucet After Grabbing Glass">
                <!-- Check if the faucet is visible -->
                <condition name="visible" target="faucet"/>
                <!-- If not visible, scan the room -->
                <sequence name="Adjust and Scan Faucet">
                    <action name="scanroom" target="faucet"/>
                    <condition name="visible" target="faucet"/>
                </sequence>
            </selector>
            <!-- Navigate to the faucet -->
            <action name="walk_to_object" target="faucet"/>
            <!-- Place the glass in the sink basin -->
            <action name="put" target="sinkBasin"/>
            <!-- Check if faucet is off before turning it on -->
            <selector name="Ensure Faucet Off">
                <condition name="isToggled" target="faucet"/>
                <!-- Turn on the faucet if it is not toggled-->
                <action name="switchon" target="faucet"/>
            </selector>
            <!-- Fill the glass with water -->
            <sequence name="Fill Glass">
                <!-- Check if the glass can be filled -->
                <condition name="canFillWithLiquid" target="glass"/>
                <!-- Check if the glass is filled with liquid -->
                <condition name="isFilledWithLiquid" target="glass"/>
            </sequence>
            <!-- Turn off the faucet -->
            <action name="switchoff" target="faucet"/>
            <!-- Pick up the filled glass from the sink basin -->
            <action name="grab" target="glass"/>
            <!-- Place the filled glass on the countertop -->
            <action name="put" target="CounterTop"/>
        </sequence>
    </selector>
</root>
'''

FORMAT_EXAMPLE = '''
<?xml version="1.0"?>
<root>
  <selector>
    <!-- Subtask 1: Define the first sequence and the conditions/actions with fallback options described in comments -->
    <sequence name="Subtask1">
      <selector>
        <!-- Primary conditions and action for the first action -->
        <sequence>
          <condition name="condition1_for_action1" target="target1_for_action1"/>
          <condition name="condition2_for_action1" target="target2_for_action1"/>
          <action name="action1"/>
        </sequence>
        <!-- Fallback options if the conditions above fail (described in comments) -->
        <!-- Fallback conditions and action for the first action -->
        <sequence>
          <condition name="alternative_condition1_for_action1" target="alternative_target1_for_action1"/>
          <action name="action2"/>
        </sequence>
      </selector>
      
      <!-- Repeat the pattern for the second action with fallback described in comments -->
      <selector>
        <sequence>
          <condition name="condition1_for_action2" target="target1_for_action2"/>
          <condition name="condition2_for_action2" target="target2_for_action2"/>
          <action name="action3"/>
        </sequence>
        <!-- Fallback conditions and action for the second action -->
        <sequence>
          <condition name="alternative_condition1_for_action2" target="alternative_target1_for_action2"/>
          <action name="action4"/>
        </sequence>
      </selector>
    </sequence>
    
    <!-- Subtask 2: Define the second sequence and the conditions/actions with fallback options described in comments -->
    <sequence name="Subtask2">
      <selector>
        <sequence>
          <condition name="condition1_for_action3" target="target1_for_action3"/>
          <condition name="condition2_for_action3" target="target2_for_action3"/>
          <action name="action5"/>
        </sequence>
        <!-- Fallback conditions and action for the third action -->
        <sequence>
          <condition name="alternative_condition1_for_action3" target="alternative_target1_for_action3"/>
          <action name="action6"/>
        </sequence>
      </selector>
      
      <!-- Repeat the pattern for the fourth action with fallback described in comments -->
      <selector>
        <sequence>
          <condition name="condition1_for_action4" target="target1_for_action4"/>
          <condition name="condition2_for_action4" target="target2_for_action4"/>
          <action name="action7"/>
        </sequence>
        <!-- Fallback conditions and action for the fourth action -->
        <sequence>
          <condition name="alternative_condition1_for_action4" target="alternative_target1_for_action4"/>
          <action name="action8"/>
        </sequence>
      </selector>
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
        if target is not None:
            self.target = self.target.replace('_', '')

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
        if target is not None:
            self.target = self.target.replace('_', '')

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



def parse_bt_xml(xml_content, actions, conditions):
    print(xml_content)
    root = ET.fromstring(xml_content)
    return parse_node(root, actions, conditions)




def parse_node(element, actions, conditions):
    node_type = element.tag.capitalize()
    if node_type == "Action":
        target = element.get('target', None)
        if target is None and element.get('name') not in AI2THOR_NO_TARGET:
            target = "TARGET MISSING"
            raise Exception(f"ERROR: Failed to parse <{node_type.lower()} name={element.get('name')} target={target}> DUE TO MISSING TARGET")
        if element.get('name') not in actions:
            raise Exception(f"ERROR: {element.get('name')} is not a valid action.")
        node = Action(element.get('name'), element.get('target'))
        return node
    elif node_type == "Condition":
        target = element.get('target', None)
        if target is None:
            target = "TARGET MISSING"
            raise Exception(f"ERROR: Failed to parse <{node_type.lower()} name={element.get('name')} target={target}> DUE TO MISSING TARGET")
        if element.get('name') not in conditions:
            raise Exception(f"ERROR: {element.get('name')} is not a valid condition.")
        return Condition(element.get('name'), target)
    elif node_type in ["Selector", "Sequence"]:
        node_class = globals()[node_type]
        node = node_class(element.get('name'))
        for child_elem in element:
            child = parse_node(child_elem, actions, conditions)
            node.children.append(child)
        return node
    elif node_type == "Root":
        node = Sequence('root')
        for child_elem in element:
            child = parse_node(child_elem, actions, conditions)
            node.children.append(child)
        return node
    raise ValueError(f"Unsupported node type: {node_type}")
