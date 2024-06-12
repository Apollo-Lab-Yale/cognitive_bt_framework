import openai
from openai import OpenAI
from cognitive_bt_framework.utils import setup_openai, get_openai_key
from ratelimit import limits, sleep_and_retry


class LLMInterface:
    def __init__(self, model_name="gpt-4o"):
        """
        Initialize the interface with your OpenAI API key and model choice.
        :param api_key: Your OpenAI API key.
        :param model_name: The model identifier for GPT-3 (e.g., "text-davinci-003").
        """
        self.client = OpenAI(api_key=get_openai_key(), timeout=10)
        self.model_name = model_name
        self.conversation_history = []
        self.token_limit = 4000

    def add_to_history(self, role, content):
        """
        Add a new message to the conversation history, managing token count.
        """
        new_message = {"role": role, "content": content}
        self.conversation_history.append(new_message)
        # Keep the last 4096 tokens of the conversation history to stay within token limits
        tokens = sum(len(m['content']) for m in self.conversation_history)
        while tokens > self.token_limit:
            tokens -= len(self.conversation_history[0]['content'])
            self.conversation_history.pop(0)

    def generate_prompt_htn(self, task):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :return: A string prompt for the GPT model.
        """
        instruction = {"role": 'system', 'content': 'You are assisting in decomposing high level goal for a robot. '
                                                    'each subgoal should be its own line with NO ADDITIONAL CHARACTERS, '
                                                    'in the format should be short camel case similar to a '
                                                    'c++ class name with no spaces. Please provide the most complete'
                                                    'decomposition possible, include a name for the high level task'
                                                    'in the same format as the subtasks as the first line of your response.'}
        message = {"role": "user", "content": f"Decompose the following task into detailed subtask steps: {task}"}
        return [instruction, message]

    def generate_prompt_task_id(self, task):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :return: A string prompt for the GPT model.
        """
        instruction = {"role": 'system', 'content': 'You are assisting in generating a task id for a user requested task'
                                                    ' the format should be short camel case similar to a '
                                                    'c++ class name with no spaces.'}
        message = {"role": "user", "content": f"Generate an id in the format described for this task: {task}"}
        return [instruction, message]

    def generate_behavior_tree_prompt(self, task, actions, conditions, example, relevant_objects):
        """
        Generate a prompt specifically for creating a behavior tree in XML format.
        """
        system_message = f'''
                    You are going to be tasked with creating a behavior tree in XML format for a robot to execute a specific task.
                    The <Action> tags must come from this list: {actions}. Other actions are not allowed.
                    The conditions that may apply to the state must come from this list: {conditions}. Other conditions are not allowed.
                    Here is a description of tags you are allowed to use:
                    <Sequence> tags execute all children until a child action fails or a <condition> check returns False then exits.
                    <Selector> tags execute each child until an action child succeeds or a <condition> check returns True then exits.
                    Both <Selector> and <Sequence> may exit before all children are executed.
                    <Action> describes an action for the robot to take.
                    Ensure each relevant object is properly located before attempting to interact with it.
                    These are the object classes I can detect, all targets must come from this list:  {relevant_objects}. You may not use any other objects besides the ones listed.
                    The one exception to the above requirement is: all food objects can be acted on by slice and become <item>sliced. For example, slicing "apple" results in "applesliced"
                    Here is an example of a properly formed XML behavior tree: 
                    {example}
                    Your response should contain only the BT and no additional text.
                    Now, please create a behavior tree in XML format for a robot to execute this task: 
                    "{task}"
                '''
        instruction = {"role": 'system', 'content': system_message}
        message = {"role": "user", "content": f"Behavior Tree for {task}:"}
        print([instruction, message])

        return [instruction, message]

    def generate_behavior_tree_refinement_prompt(self, task, actions, conditions, original_bt_xml, feedback,
                                                 known_objects, example, error_category=None):
        """
        Generate a prompt to refine an existing behavior tree based on user feedback, in XML format.
        """
        error_info = f"Error Category: {error_category}" if error_category else ""
        system_message = f'''
            This is the task that was attempted: "{task}"

            The behavior tree provided for this task resulted in an error. 
            This is the sub-tree where the error occurred: {original_bt_xml}
            This is the associated feedback and error info describing the error: {feedback}. {error_info}
            Please modify the behavior tree to account for this failure, with the following requirements:
            1. This is the list of valid actions for an <Action> tag: {actions}. Other actions are not allowed.
            2. The <condition>s that may apply to the actions come from this list: {conditions}. Other conditions are not allowed.
            3. These are the object classes I can detect, all targets must come from this list:  {known_objects}. You may not use any other objects besides the ones listed.
            The one exception to requirement 3 is: all food objects can be acted on by the slice action and become <item>sliced. For example, "apple" becomes "applesliced".
            Your response should contain only the BT and no additional text.
        '''
        prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Corrected Behavior Tree in XML:"}
        ]
        print(prompt)
        return prompt


    @sleep_and_retry
    @limits(calls=10, period=60)  # Example: Max 10 calls per minute
    def query_llm(self, prompt):
        """
        Query the GPT model with the generated prompt.
        :param prompt: The prompt for task decomposition.
        :return: The model's response as a task decomposition.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=1000,
                temperature=0.5
            )
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None  # Or handle appropriately
        return response.choices[0].message.content

    def query_llm_for_refinement(self, task, original_decomposition, user_feedback):
        """
        Query the LLM with the original decomposition and user feedback to generate a refined decomposition.
        """
        prompt = self.generate_refined_prompt(task, original_decomposition, user_feedback)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=1000,
                temperature=0.5
            )
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None  # Or handle appropriately
        return response.choices[0].message.content

    def get_task_decomposition(self, task):
        """
        Get the task decomposition from the GPT model.
        :param task: The task description.
        :return: A list or string of decomposed tasks.
        """
        prompt = self.generate_prompt_htn(task)
        decomposition = self.query_llm(prompt)
        return decomposition

    def get_task_id(self, task):
        prompt = self.generate_prompt_task_id(task)
        task_id = self.query_llm(prompt)
        return task_id


    def get_behavior_tree(self, task, actions, conditions, example, known_objects):
        """
        Get the behavior tree from the GPT model for a given task.
        """
        prompt = self.generate_behavior_tree_prompt(task, actions, conditions, example, known_objects)

        behavior_tree_xml = self.query_llm(prompt)
        for i in range(len(behavior_tree_xml)):
            if behavior_tree_xml[i] == '<':
                behavior_tree_xml = behavior_tree_xml[i:]
                break
        for i in range(len(behavior_tree_xml)):
            if behavior_tree_xml[len(behavior_tree_xml) - i - 1] == '>':
                behavior_tree_xml = behavior_tree_xml[:len(behavior_tree_xml) - i ]
                break
        behavior_tree_xml = behavior_tree_xml.replace('```', '')
        return behavior_tree_xml

    def refine_behavior_tree(self, task, actions, conditions, original_bt_xml, user_feedback, known_objects, example):
        """
        Refine a behavior tree based on user feedback.
        """
        prompt = self.generate_behavior_tree_refinement_prompt(task, actions, conditions, original_bt_xml, user_feedback, known_objects, example)
        refined_behavior_tree_xml = self.query_llm(prompt)
        print(f"Refined Behavior Tree: {refined_behavior_tree_xml}")
        for i in range(len(refined_behavior_tree_xml)):
            if refined_behavior_tree_xml[i] == '<':
                refined_behavior_tree_xml = refined_behavior_tree_xml[i:]
                break
        refined_behavior_tree_xml = refined_behavior_tree_xml.replace('```', '')
        return refined_behavior_tree_xml

if __name__ == "__main__":
    from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_ACTIONS
    # Example usage
    llm_interface = LLMInterface()
    task = "Clean the kitchen"
    bt_xml = llm_interface.get_behavior_tree(task, AI2THOR_ACTIONS)
    print(bt_xml)
    # Assuming some feedback was received
    feedback = "The robot fails to find the sink."
    refined_bt_xml = llm_interface.refine_behavior_tree(task, bt_xml, feedback)
    print(refined_bt_xml)
