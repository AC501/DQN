from sumotl_action_causality_yuanshibanben import SUMOTrafficLights
from learner.q_learning import QLearner
from exploration.epsilon_greedy import EpsilonGreedy
import datetime
import warnings
warnings.filterwarnings(action='ignore')

# a SUMO environment
#env = SUMOTrafficLights('nets/3x3grid/3x3grid.sumocfg', 8813, True)
env = SUMOTrafficLights('nets/MianYangTl/yuanyishan.sumocfg', 8813, True)

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


    # print queue length
    print("episode=",i,"====================")
    arq_avg_nome = 'tl_%d.txt' % (i)
    arq_tl = open(arq_avg_nome, 'w')  # para salvar saida em um arquivo
    arq_tl.writelines('##%s## \n' % (datetime.datetime.now().time()))

    env.run_episode(1000, arq_tl, exp)

arq_tl.close()
# f.close()
