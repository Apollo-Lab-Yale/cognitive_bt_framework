from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv

trooms = [4, 7, 9, 11, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28]


sim = AI2ThorSimEnv()
state = sim.get_graph()
apple = [obj for obj in state['objects'] if 'apple' in obj['name'].lower()]
print(apple)
print(state['predicates'])
