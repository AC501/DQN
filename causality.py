from environment.sumotl_causality import SUMOTrafficLights
from learner.q_learning import QLearner
from exploration.epsilon_greedy import EpsilonGreedy
import datetime
import warnings
warnings.filterwarnings(action='ignore')

# a SUMO environment
env = SUMOTrafficLights('nets/3x3grid/3x3grid.sumocfg', 8813, True)
# env = SUMOTrafficLights('nets/MianYang/yuanyishan.sumocfg', 8813, True)

# an exploration strategy
exp = EpsilonGreedy(epsilon=1, min_epsilon=0.0, decay_rate= 0.95, manual_decay=True)

# for each traffic light in the net file
for tlID in env.get_trafficlights_ID_list():
    # create a learner
    # print("=======================")
    # print tlID
    _ = QLearner(tlID, env, 0, 0, 0.1, 0.8, exp)

# number of episodes
n_episodes = 1

# for each episode
for i in range(n_episodes):


    file_name = "queue.txt"
    env.run_episode(5000, file_name, exp)


# f.close()
