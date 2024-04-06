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
        instruction = {"role": 'system', 'content': 'You are assisting in decomposing high level tasks for a robot. '
                                                    'each subtask should be its own line with NO ADDITIONAL CHARACTERS, '
                                                    'in the format should be short camel case similar to a '
                                                    'c++ class name with no spaces. Please provide the most complete'
                                                    'decomposition possible.'}
        message = {"role": "user", "content": f"Decompose the following task into detailed subtask steps: {task}"}
        return [instruction, message]

    def generate_refined_prompt(self, task, original_decomposition, user_feedback):
        """
        Generate a secondary prompt that includes the task, the original decomposition,
        and the user feedback to guide the LLM towards generating a refined task decomposition.
        """
        prompt = [
            {"role": "system",
             "content": "Based on the user feedback, refine the following task decomposition to improve clarity and completeness."},
            {"role": "user", "content": f"Task: {task}"},
            {"role": "system", "content": f"Original Decomposition: {original_decomposition}"},
            {"role": "user", "content": f"User Feedback: {user_feedback}"},
            {"role": "system", "content": "Refined Decomposition:"}
        ]
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

if __name__ == "__main__":
    # Example usage
    llm_interface = LLMInterface()
    task = "Clean the kitchen"
    decomposition = llm_interface.get_task_decomposition(task)
    print(decomposition)
