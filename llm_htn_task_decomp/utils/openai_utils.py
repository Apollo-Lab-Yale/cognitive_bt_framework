import openai

def set_api_key(api_key):
    """
        Initializes the OpenAI API with the provided API key.
        :param api_key: The API key for OpenAI.
    """
    openai.api_key = api_key

def get_openai_key():
    file_path = "/usr/config/llm_task_planning.txt"
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            return first_line.strip()
    except:
        raise "failed to find api key."

def setup_openai():
    file_path = "/usr/config/llm_task_planning.txt"
    try:
        with open(file_path, 'r') as file:
            first_line = file.readline()
            set_api_key(first_line.strip())
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
    except IOError as e:
        print(f"An error occurred while reading the file: {str(e)}")
