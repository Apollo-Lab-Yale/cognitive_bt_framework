import csv
from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from cognitive_bt_framework.src.cbt_planner.cbtf import CognitiveBehaviorTreeFramework

goals = ['set_place', 'wash_mug', 'apple', 'coffee']

goals_nl = {
    'coffee': 'bring a mug of coffee to the table',
    'set_place': 'set a place at the table',
    'wash_mug': 'wash the dirty mug',
    'apple': 'put the apple in the fridge'
}

data_dir = '/home/liam/dev/cognitive_bt_framework/cognitive_bt_framework/data'
output_csv = f'{data_dir}/run_data.csv'


def main():
    sim = AI2ThorSimEnv(scene_index=17)
    cbtf = CognitiveBehaviorTreeFramework(sim)
    trials = 4
    scenes = [28, 27]

    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Goal', 'Scene', 'Trial', 'Success'])

        # Run simulations and log data
        for goal in goals:
            cbtf.set_goal(goal)
            for trial in range(trials):
                for scene in scenes:
                    sim.reset(scene_index=scene)
                    cbtf.set_goal(goal)
                    success = cbtf.manage_task_ordered(goals_nl[goal])
                    # Write data to CSV
                    writer.writerow([goal, scene, trial, success])


if __name__ == "__main__":
    main()
