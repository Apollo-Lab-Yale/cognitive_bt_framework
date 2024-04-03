import openai
from openai import OpenAI
from llm_htn_task_decomp.utils import setup_openai, get_openai_key

class LLMInterface:
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initialize the interface with your OpenAI API key and model choice.
        :param api_key: Your OpenAI API key.
        :param model_name: The model identifier for GPT-3 (e.g., "text-davinci-003").
        """
        self.client = OpenAI(api_key=get_openai_key(), timeout=10)
        self.model_name = model_name

    def generate_prompt(self, task):
        """
        Generate a prompt for the task decomposition.
        :param task: The task description.
        :return: A string prompt for the GPT model.
        """
        message = {"role": "user", "content": f"Decompose the following task into detailed steps: {task}"}
        return [message]

    def query_llm(self, prompt):
        """
        Query the GPT model with the generated prompt.
        :param prompt: The prompt for task decomposition.
        :return: The model's response as a task decomposition.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=1000,
            temperature=0.5
        )
        print(response)
        return response.choices[0].message.content

    def get_task_decomposition(self, task):
        """
        Get the task decomposition from the GPT model.
        :param task: The task description.
        :return: A list or string of decomposed tasks.
        """
        prompt = self.generate_prompt(task)
        decomposition = self.query_llm(prompt)
        return decomposition

if __name__ == "__main__":
    # Example usage
    llm_interface = LLMInterface()
    task = "Plan a small wedding"
    decomposition = llm_interface.get_task_decomposition(task)
    print(decomposition)
