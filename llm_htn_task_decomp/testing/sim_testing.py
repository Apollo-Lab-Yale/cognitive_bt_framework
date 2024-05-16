from llm_htn_task_decomp.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv

sim = AI2ThorSimEnv()
print(sim.get_state())
print(sim.get_state()['objects'][0])
print(sim.handle_scan_room("sink"))
input()