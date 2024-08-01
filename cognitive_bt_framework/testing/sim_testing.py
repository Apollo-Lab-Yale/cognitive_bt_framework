from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv

trooms = [4, 7, 9, 11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28]


sim = AI2ThorSimEnv()

print(sim.get_graph()['agent'])

sim.look_down()
sim.move_back()
sim.move_back()
print(sim.get_graph()['agent'])
sim.turn_left()
sim.turn_left()
print(sim.get_graph()['agent'])
input()