import openai
from openai import OpenAI
import anthropic
from cognitive_bt_framework.utils import setup_openai, get_openai_key, parse_llm_response, get_claude_key
from cognitive_bt_framework.utils.bt_utils import DOMAIN_DEF
from ratelimit import limits, sleep_and_retry
from cognitive_bt_framework.src.sim.ai2_thor.utils import AI2THOR_PREDICATES_ANNOTATED

class LLMInterfaceClaude:
    def __init__(self, model_name="claude-3-5-sonnet-20240620"):
        """
        Initialize the interface with your OpenAI API key and model choice.
        :param api_key: Your OpenAI API key.
        :param model_name: The model identifier for GPT-3 (e.g., "text-davinci-003").
        """
        self.client = anthropic.Anthropic(api_key=get_claude_key())
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

    # def generate_prompt_htn(self, task, known_objects, context):
    #     """
    #     Generate a prompt for the task decomposition.
    #     :param task: The task description.
    #     :return: A string prompt for the GPT model.
    #     """
    #     msg = f"""
    #             You are assisting in decomposing a high-level goal for a robot.
    #             Each subgoal should be its own line with NO ADDITIONAL CHARACTERS.
    #             The format should be short and underscore separated similar to a pythonic class name
    #             with no spaces. Please provide the most complete decomposition
    #             possible, all subtasks should be as basic as possible
    #             is its own atomic step moving closer to the high-level goal. Include a
    #             name for the high-level task in the same format as the subtasks as
    #             the first line of your response. If the task is sufficiently small,
    #              then you do not need to generate subtasks. Sufficiently reduced
    #             subtasks include tasks like empty_trash, clear_counters,
    #             empty_dishwasher, etc.
    #             Additionally, for each subtask, provide a SINGLE object state condition
    #             that defines when the subtask is considered complete. Format these conditions
    #             as a bulleted list directly under the subtask, each condition on a new line
    #             with a dash "-" at the beginning. All Conditions should come from
    #             the following list without exception: {AI2THOR_PREDICATES_ANNOTATED} followed
    #             by a space and the object that the condition applies to and a 1
    #             or 0 indicating if the predicate should be true of false. All conditions
    #             must target a specific object and cannot contain placeholders.
    #              All objects should come from this list without exception: {known_objects}.
    #             Here is a brief description of the environment: {context}
    #             Decompose the following task into detailed subtask steps: {task} """
    #     instruction = {"role": 'user', 'content': [{'type': 'text', 'text': msg}]}
    #     message = {"role": "user", "content": instruction}
    #     return [instruction]

    def generate_prompt_htn(self, task, known_objects, context):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :param known_objects: List of known objects in the environment.
        :param context: Brief description of the environment.
        :return: A string prompt for the GPT model.
        """
        ai2thor_predicates = [
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
            'inRoom',
            'moveable',
            'isOnTop',
            'isInside'
        ]
        predicates_str = ", ".join(ai2thor_predicates)

        msg = f"""
        You are assisting in decomposing a high-level goal for a robot. Each subgoal should be its own line with 
        NO ADDITIONAL CHARACTERS. The format should be short and underscore separated, similar to a Pythonic class name 
        with no spaces. Provide the most complete decomposition possible, ensuring all subtasks are as basic as possible
         and represent an atomic step towards achieving the high-level goal. Include a name for the high-level task in 
         the same format as the subtasks as the first line of your response. If the task is sufficiently small, you do 
         not need to generate subtasks. Examples of sufficiently reduced subtasks include: empty_trash, clear_counters, empty_dishwasher.
         Keep in Mind only one item can be held at a time. Please emphasize only visible items relevant to the task.

        For each subtask, provide a SINGLE object state condition that defines when the subtask is considered complete. 
        Format these conditions as a bulleted list directly under the subtask, each condition on a new line with a dash ("-") at the beginning. All conditions should come from the following list without exception: {predicates_str}, followed by a space, the object that the condition applies to, and a 1 or 0 indicating if the predicate should be true or false. All conditions must target a specific object and cannot contain placeholders. All objects should come from this list without exception: {known_objects}.

        Here is a brief description of the environment: {context}

        Decompose the following task into detailed subtask steps: {task}
        """
        message = {"role": "user", "content": [{'type': 'text', 'text': f"{msg.strip()}"}]}
        return [message]

    def generate_prompt_task_id(self, task, context, states):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :return: A string prompt for the GPT model.
        """
        content = []
        instruction = f"""
        'You are assisting in generating a task id for a user requested task'
        ' The format should be short and underscore separated similar to a '
        'pythonic class name  with no spaces. Additionally, using the images provided'
        ' as task context, generate all relevant environmental'
        ' information that may assist an llm in generating subtasks '
        'and action sequences for the task'
        f' {task} and behavior trees to complete those sub tasks be as detailed as '
        'possible including detailed spacial relations and object descriptions'
        '. The task ID and context should be separated by a new line '
        'character. Additionally give hints on potential actions or insight into what '
        'might be needed to complete the task. The description should be '
        'long and easily readable. Include no other text!' 
        """
        content.append({"type": 'text', 'text': instruction})
        context_data = [{
                        "type": "text",
                        "text": "The following images and states are the context for the task to be completed"
                    }]
        for i in range(len(context)):
            image = context[i]
            state = states[i]
            img_msg = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": 'image/png',
                    "data": image,
                }
            }
            state_msg = {
                "type": "text",
                "text": f"The environmental state associated with image {i} is: {state}"
            }
            context_data.append(img_msg)
            context_data.append(state_msg)
        context_data.append({'type': 'text', 'text': f"Generate an id and context, in the format described for this task: {task}"})
        context_msg = {
            "role": "user",
            "content": content + context_data
        }
        message = {"role": "user", "content": context_msg}

        return [context_msg]

    def generate_behavior_tree_prompt(self, big_task, task, actions, conditions, example, relevant_objects,
                                      completed_subtasks, context, complete_condition):
        """
        Generate a prompt specifically for creating a behavior tree in XML format.
        """
        completed_goals_str = f'''
        **Completed Sub Goals**
        The following subgoals have already been completed:
        {completed_subtasks}
        ''' if len(completed_subtasks) > 0 else ""
        # user_message = f'''
        #             You are going to be tasked with creating a behavior tree in XML format for a robot to execute a specific task. Follow these rules carefully:
        #             **Actions**: Use only the actions from this list:
        #              {actions}.
        #              Other actions are not allowed.
        #              each action tag should contain name and target members where name is the action being performed and target is the target of the action
        #             **Conditions**: Use only the conditions from this list:
        #             {conditions}.
        #             No other conditions are allowed.
        #             each condition tag should contain name and target members. where name is the condition being checked and target is the target of the condition
        #             {completed_goals_str}
        #             **Domain**
        #             the domain is defined in pddl as follows:
        #             {DOMAIN_DEF}
        #             **Behavior Tree Structure**:
        #             - Use <Sequence> tags to execute all child actions or conditions in order until one fails or returns False.
        #     -       - Use <Selector> tags to execute each child action or condition in order until one succeeds or returns True.
        #
        #            **Object Classes**: You can only interact with objects from this list:
        #            {relevant_objects}.
        #            No other objects are allowed. Exception: Slicing a food object results in '<item>sliced', e.g., 'apple' becomes 'applesliced'.
        #             The one exception to the above requirement is: all food objects can be acted on by slice and become <item>sliced. For example, slicing "apple" results in "applesliced"
        #             **Example Behavior Tree**:
        #             Here is an example of a properly formed XML behavior tree for the task "FillGlassWater":
        #             {example}
        #             ** Environmental Description**
        #             {context}
        #             **Task**:
        #             Now, please create a behavior tree in XML format for the robot to execute the task: {task}'.
        #             This task is a sub task of {big_task}
        #             The following subtasks have already been completed: {completed_subtasks}
        #             Make sure that you consider the cases where objects are contained in other objects or the task is already complete in your answer!
        #             The completion condition for this task is: {complete_condition}
        #             Ensure your response contains only the XML behavior tree and no additional text.
        #             "{task}"
        #             Behavior Tree for {task}:
        #         '''
        user_message = f'''
                            You are going to be tasked with creating a behavior tree in XML format for a robot to execute a specific task. Follow these rules carefully:
                            **Actions**: Use only the actions from this list:
                             {actions}. 
                             Other actions are not allowed.
                             each action tag should contain name and target members where name is the action being performed and target is the target of the action
                            **Conditions**: Use only the conditions from this list:
                            {conditions}.
                            No other conditions are allowed.
                            each condition tag should contain name and target members. where name is the condition being checked and target is the target of the condition
                            {completed_goals_str}
                            **Domain**
                            the domain is defined in pddl as follows:
                            {DOMAIN_DEF}
                            **Behavior Tree Structure**:
                            - Use <Sequence> tags to execute all child actions or conditions in order until one fails or returns False.
                    -       - Use <Selector> tags to execute each child action or condition in order until one succeeds or returns True.

                           **Object Classes**: You can only interact with objects from this list:
                           {relevant_objects}.
                           No other objects are allowed. Exception: Slicing a food object results in '<item>sliced', e.g., 'apple' becomes 'applesliced'.
                            The one exception to the above requirement is: all food objects can be acted on by slice and become <item>sliced. For example, slicing "apple" results in "applesliced"
                            ** Environmental Description**
                            {context}
                            **Task**:
                            Now, please create a behavior tree in XML format for the robot to execute the task: {task}'.
                            This task is a sub task of {big_task}
                            The following subtasks have already been completed: {completed_subtasks}
                            Make sure that you consider the cases where objects are contained in other objects or the task is already complete in your answer!
                            The completion condition for this task is: {complete_condition}
                            Ensure your response contains only the XML behavior tree and no additional text. 
                            "{task}"
                            Behavior Tree for {task}:
                        '''
        instruction = {"role": 'user', 'content': [{'type': "text", "text": user_message}]}

        return [instruction]

    def generate_behavior_tree_refinement_prompt(self, big_task, task, actions, conditions, original_bt_xml, feedback,
                                                 known_objects, completed_subtasks, example, context, complete_condition,
                                                 image_context, error_category=None, state=None):
        """
        Generate a prompt to refine an existing behavior tree based on user feedback, in XML format.
        """
        img_msg = []
        content = []
        if image_context is not None:
            img_msg = [
                {
                    'type': 'text',
                    'text': "This is my current view of my environment"
                },
                {
                    'type': 'image_url',
                    "image_url": {
                        "url": f"data:image/png;base64,{image_context}",
                    }
                }
            ]
        error_info = f"Error Category: {error_category}" if error_category else ""
        # user_message = f'''
        #     Task attempted: "{task}"
        #     This is a subtask of: {big_task}
        #     The following subtasks have already been completed: {completed_subtasks}
        #     The behavior tree provided for this task resulted in an error.
        #     Sub-tree where the error occurred: {original_bt_xml}
        #     Associated feedback and error information: {feedback}. {error_info}
        #     **Domain**
        #     the domain is defined in pddl as follows:
        #     {DOMAIN_DEF}
        #     Please modify the behavior tree to address this failure, adhering to the following requirements:
        #     1. Valid actions for an <Action> tag: {actions}. No other actions are allowed, and each action should contain name and target members.
        #     where name is the action being performed and target is the target of the action.
        #     2. Applicable <condition>s for the actions: {conditions}. No other conditions are allowed, and each condition should contain name and target members.
        #     where name is the condition being checked and target is the target of the condition. an additional 'recipient'
        #     member can also be provided for <condition>s for example <Condition name='isOnTop' target='plate' recipient='diningtable'>.
        #     3. Detectable object classes: {known_objects}. Only these objects may be used.
        #        Exception: All food objects can be acted on by the slice action, becoming <item>sliced. For example, "apple" becomes "applesliced".
        #     4. The only valid tags are <Action>, <Condition>, <Sequence>, <Selector>, <root>, <?xml version="1.0"?>.
        #     - Use <Sequence> tags to execute all child actions or conditions in order until one fails or returns False.
        #     - Use <Selector> tags to execute each child action or condition in order until one succeeds or returns True.
        #     The following is an example of a complete behavior tree:
        #     {example}.
        #     Here is a brief description of the environment: {context}
        #     Your response should contain only the corrected behavior tree in XML format and no additional text.
        #     the completion criteria for this task is: {complete_condition}
        #     Make sure that you consider the cases where objects are contained in other objects or the task is already complete in your answer!
        #     Corrected Behavior Tree in XML:
        # '''
        user_message = f'''
                   Task attempted: "{task}"
                   This is a subtask of: {big_task}
                   The following subtasks have already been completed: {completed_subtasks}
                   The behavior tree provided for this task resulted in an error. 
                   Sub-tree where the error occurred: {original_bt_xml}
                   Associated feedback and error information: {feedback}. {error_info}
                   **Domain**
                   the domain is defined in pddl as follows:
                   {DOMAIN_DEF}
                   Please modify the behavior tree to address this failure, adhering to the following requirements:
                   1. Valid actions for an <Action> tag: {actions}. No other actions are allowed, and each action should contain name and target members.
                   where name is the action being performed and target is the target of the action.
                   2. Applicable <condition>s for the actions: {conditions}. No other conditions are allowed, and each condition should contain value, name and target members.
                   where value is the boolean value of the condition either 0 or 1, name is the condition being checked and target is the target of the condition. an additional 'recipient' 
                   member can also be provided for <condition>s for example <Condition name='isOnTop' target='plate' recipient='diningtable'>.
                   3. Detectable object classes: {known_objects}. Only these objects may be used.
                      Exception: All food objects can be acted on by the slice action, becoming <item>sliced. For example, "apple" becomes "applesliced".
                   4. The only valid tags are <Action>, <Condition>, <Sequence>, <Selector>, <root>, <?xml version="1.0"?>.
                   - Use <Sequence> tags to execute all child actions or conditions in order until one fails or returns False.
                   - Use <Selector> tags to execute each child action or condition in order until one succeeds or returns True.
                   
                   Here is a brief description of the environment: {context}
                   Your response should contain only the corrected behavior tree in XML format and no additional text.
                   the completion criteria for this task is: {complete_condition}
                   Make sure that you consider the cases where objects are contained in other objects or the task is already complete in your answer!
                   Corrected Behavior Tree in XML:
               '''
        content.append({
            'type': 'text',
            'text': user_message
        })
        # print(f"{image_context} *************************************")
        if image_context is not None:
            img_msg = [
                {
                    'type': 'text',
                    'text': "This is my current view of my environment"
                },
                {
                    'type': 'image_url',
                    "image_url": {
                        "url": f"data:image/png;base64,{image_context}",
                    }
                }
            ]
            content = content + img_msg
        prompt = [{
            "role": 'user',
            'content': content
        }]
        return prompt

    @sleep_and_retry
    @limits(calls=100, period=60)  # Example: Max 10 calls per minute
    def query_llm(self, prompt):
        """
        Query the GPT model with the generated prompt.
        :param prompt: The prompt for task decomposition.
        :return: The model's response as a task decomposition.
        """
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=4096,
                temperature=1
            )
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None  # Or handle appropriately
        print(response)
        return response.content[0].text

    def query_llm_for_refinement(self, big_task, task, original_decomposition, user_feedback):
        """
        Query the LLM with the original decomposition and user feedback to generate a refined decomposition.
        """
        prompt = self.generate_refined_prompt(big_task, task, original_decomposition, user_feedback)
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

    def get_task_decomposition(self, task, known_objects, context):
        """
        Get the task decomposition from the GPT model.
        :param task: The task description.
        :return: A list or string of decomposed tasks.
        """
        prompt = self.generate_prompt_htn(task, known_objects, context)
        decomposition = self.query_llm(prompt)
        return parse_llm_response(decomposition)

    def get_task_id(self, task, context, states):
        prompt = self.generate_prompt_task_id(task, context, states)
        ret = self.query_llm(prompt)
        print(ret)
        context_object = ret.split('\n')[1]
        task_id = ret.split('\n')[0]
        return task_id, context_object

    def get_behavior_tree(self, big_task, task, actions, conditions, example, known_objects, completed_subtasks,
                          context, complete_condition):
        """
        Get the behavior tree from the GPT model for a given task.
        """
        prompt = self.generate_behavior_tree_prompt(big_task, task, actions, conditions, example, known_objects,
                                                    completed_subtasks, context, complete_condition)

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

    def refine_behavior_tree(self, big_task, task, actions, conditions, original_bt_xml, user_feedback, known_objects,
                             completed_subtasks, example, context, complete_condition, image_context):
        """
        Refine a behavior tree based on user feedback.
        """
        prompt = self.generate_behavior_tree_refinement_prompt(big_task, task, actions, conditions, original_bt_xml,
                                                               user_feedback, known_objects, completed_subtasks,
                                                               example, context, complete_condition, image_context)
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
    task = "get a glass of water"
    bt_xml = llm_interface.get_behavior_tree(task, AI2THOR_ACTIONS)
    print(bt_xml)
    # Assuming some feedback was received
    feedback = "The robot fails to find the sink."
    refined_bt_xml = llm_interface.refine_behavior_tree(task, bt_xml, feedback)
    print(refined_bt_xml)
