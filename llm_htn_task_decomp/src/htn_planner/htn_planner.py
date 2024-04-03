from llm_htn_task_decomp.src.llm_interface.llm_interface_openai import LLMInterface


class HTNPlanner:
    def __init__(self, llm_interface=None):
        self.llm_interface = llm_interface
        self.tasks = {}  # Dictionary to store task decompositions

    def register_task(self, task_name, decomposition):
        """
        Register a static decomposition for a task.
        :param task_name: The name of the task.
        :param decomposition: A function that takes no arguments and returns the decomposition of the task.
        """
        self.tasks[task_name] = decomposition

    def decompose_task(self, task_name):
        """
        Decompose a task into subtasks.
        :param task_name: The name of the task to decompose.
        :return: A list of subtasks or steps required to complete the task.
        """
        if task_name in self.tasks:
            # If there is a predefined decomposition, use it
            return self.tasks[task_name]()
        elif self.llm_interface:
            # If no predefined decomposition, try to use the LLM interface, if available
            return self.llm_interface.get_task_decomposition(task_name)
        else:
            raise ValueError(f"No decomposition found for task '{task_name}', and no LLM interface provided.")

    # Example static decomposition method
    def organize_conference_decomposition(self):
        return [
            "Select a date and time",
            "Choose a location",
            "Prepare a guest list",
            "Send out invitations",
            "Plan the menu",
            "Decorate the venue",
            "Arrange entertainment"
        ]

# Example usage
if __name__ == "__main__":
    # Assuming `llm_interface` is an instance of `LLMInterface` set up as previously described
    llm_interface = LLMInterface()
    planner = HTNPlanner(llm_interface)

    # Register a static task decomposition
    planner.register_task("organize conference", planner.organize_conference_decomposition)

    # Decompose a registered task
    print(planner.decompose_task("organize conference"))

    # Decompose a task using the LLM (assuming the LLM interface can handle this task)
    print(planner.decompose_task("plan a team building event"))
