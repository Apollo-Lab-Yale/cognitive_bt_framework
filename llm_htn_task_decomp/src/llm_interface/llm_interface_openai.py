import openai
from openai import OpenAI
from llm_htn_task_decomp.utils import setup_openai, get_openai_key
from ratelimit import limits, sleep_and_retry


class LLMInterface:
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initialize the interface with your OpenAI API key and model choice.
        :param api_key: Your OpenAI API key.
        :param model_name: The model identifier for GPT-3 (e.g., "text-davinci-003").
        """
        self.client = OpenAI(api_key=get_openai_key(), timeout=10)
        self.model_name = model_name

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
                                                    'decomposition possible.'}
        message = {"role": "user", "content": f"Decompose the following task into detailed subtask steps: {task}"}
        return [instruction, message]

    def generate_behavior_tree_prompt(self, task, conditions, actions, example):
        """
        Generate a prompt specifically for creating a behavior tree in XML format.
        """
        system_message = f'''
                    Create a behavior tree in XML format for a robot to execute a task. 
                    Each action should be clearly defined in XML tags, and should only come from this list: {actions}.
                    These are the possible predicates in the state that may apply to actions: {conditions}
                    this is an example of a properly formed xmlBT: {example}
                    Create a behavior tree for the task: {task}
                '''
        instruction = {"role": 'system', 'content': system_message}
        message = {"role": "user", "content": "Behavior Tree for {task}:"}
        print([instruction, message])

        return [instruction, message]

    def generate_behavior_tree_refinement_prompt(self, task, conditions, actions, original_bt_xml, feedback):
        """
        Generate a prompt to refine an existing behavior tree based on user feedback, in XML format.
        """
        system_message = f'''
            Task: {task}"
            Original Behavior Tree: {original_bt_xml}.
            The above behavior tree failed due to {feedback}. Update the behavior tree to account for this failure.
            Each action should be clearly defined in XML tags, and should only come from this list: {actions}
            These are the possible predicates in the state that may apply to actions: {conditions}
            Please suggest modifications to including robust more robust object detection and handling steps before attempting to interact with them.
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


    def get_behavior_tree(self, task, actions, conditions, example):
        """
        Get the behavior tree from the GPT model for a given task.
        """
        prompt = self.generate_behavior_tree_prompt(task, actions, conditions, example)

        behavior_tree_xml = self.query_llm(prompt)
        for i in range(len(behavior_tree_xml)):
            if behavior_tree_xml[i] == '<':
                behavior_tree_xml = behavior_tree_xml[i:]
                break
        behavior_tree_xml = behavior_tree_xml.replace('```', '')
        return behavior_tree_xml

    def refine_behavior_tree(self, task, actions, conditions, original_bt_xml, user_feedback):
        """
        Refine a behavior tree based on user feedback.
        """
        prompt = self.generate_behavior_tree_refinement_prompt(task, actions, conditions, original_bt_xml, user_feedback)
        refined_behavior_tree_xml = self.query_llm(prompt)
        return refined_behavior_tree_xml

if __name__ == "__main__":
    from llm_htn_task_decomp.src.sim.ai2_thor.utils import AI2THOR_ACTIONS
    # Example usage
    llm_interface = LLMInterface()
    task = "Clean the kitchen"
    bt_xml = llm_interface.get_behavior_tree(task, AI2THOR_ACTIONS)
    print(bt_xml)
    # Assuming some feedback was received
    feedback = "The robot fails to find the sink."
    refined_bt_xml = llm_interface.refine_behavior_tree(task, bt_xml, feedback)
    print(refined_bt_xml)
