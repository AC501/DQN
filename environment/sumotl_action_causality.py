'''
Created on 12/12/2017

@author: Liza L. Lemos <lllemos@inf.ufrgs.br>
'''
import pandas as pd

from environment import Environment
import traci
#找一个二进制文件，即寻找可执行文件
import sumolib
from xml.dom import minidom
import sys, os
import subprocess
import atexit
from contextlib import contextmanager
import time
from array import array
import numpy as np
import datetime
import math


class SUMOTrafficLights(Environment):

    def __init__(self, cfg_file, port=8813, use_gui=False):

        super(SUMOTrafficLights, self).__init__()

        self.__create_env(cfg_file, port, use_gui)

    '''
    Create the environment as a MDP. The MDP is modeled as follows:
    * for each traffic light:
    * the STATE is defined as a vector [current phase, elapsed time of current phase, queue length for each phase]
    * for simplicity, the elapsed time is discretized in intervals of 5s
    * and, the queue length is calculated according to the occupation of the link. 
    * The occupation is discretized in 4 intervals (equally distributed)
    * The number of ACTIONS is equal to the number of phases
    * Currentlly, there are only two phases thus the actions are either keep green time at the current phase or 
    * allow green to another phase. As usual, we call these actions 'keep' and 'change'
    * At each junction, REWARD is defined as the difference between the current and the previous average queue length (AQL)
    * at the approaching lanes, i.e., for each traffic light the reward  is defined as $R(s,a,s')= AQL_{s} - AQL_{s'}$.
    * the transitions between states are deterministic
    '''

    def __create_env(self, cfg_file, port, use_gui):

        # check for SUMO's binaries
        if use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        # register SUMO/TraCI parameters
        self.__cfg_file = cfg_file
        self.__net_file = self.__cfg_file[:self.__cfg_file.rfind("/") + 1] + \
                          minidom.parse(self.__cfg_file).getElementsByTagName('net-file')[0].attributes['value'].value

        # read the network file
        self.__net = sumolib.net.readNet(self.__net_file)

        self.__env = {}

        d = ['keep', 'change']
        # d[0] = 'keep'
        # d[1] = 'change'

        # to each state the actions are the same
        # self.__env[state] has 160 possible variations
        # [idPhase, elapsed time, queue NS, queue EW] = [2, 5, 4, 4]
        # 2 * 5 * 4 * 4 = 160
        # idPhase: 2 phases - NS, EW
        # elapsed time: 30s that are discretize in 5 intervals
        # queue: 0 to 100% discretize in 4 intervals
        # Note: to change the number of phases, it is necessary to change the number of states, e.g. 3 phases: [3, 5, 4, 4, 4]
        # it is also necessary to change the method change_trafficlight
        for x in range(0, 160):
            self.__env[x] = d

        # create the set of traffic ligths
        self.__create_trafficlights()

        self.__create_edges()

    def __create_trafficlights(self):
        # set of all traffic lights in the simulation
        # each element in __trafficlights correspond to another in __learners
        self.__trafficlights = {}

        # process all trafficlights entries
        #获取交通信号灯，并获取其他相关信息
        junctions_parse = minidom.parse(self.__net_file).getElementsByTagName('junction')
        for element in junctions_parse:
            if element.getAttribute('type') == "traffic_light":

                tlID = element.getAttribute('id').encode('utf-8')
                # print((str(tlID))[2:3])
                tlID = (str(tlID))[2:3]
                #print(tlID)
                # create the entry in the dictionary
                self.__trafficlights[tlID] = {
                    'greenTime': 0,
                    'nextGreen': -1,
                    'yellowTime': -1,
                    'redTime': -1
                }

    def reset_episode(self):

        super(SUMOTrafficLights, self).reset_episode()

        # initialise TraCI
        traci.start([self._sumo_binary, "-c", self.__cfg_file])

        # reset traffic lights attributes
        for tlID in self.get_trafficlights_ID_list():
            self.__trafficlights[tlID]['greenTime'] = 0
            self.__trafficlights[tlID]['nextGreen'] = -1
            self.__trafficlights[tlID]['yellowTime'] = -1
            self.__trafficlights[tlID]['redTime'] = -1

    # define the edges/lanes that are controled for each traffic light
    # the function getControlledLanes() from TRACI, returned the names of lanes doubled
    # that's way is listed here
    def __create_edges(self):
        self._edgesNS = {}
        self._edgesEW = {}

        self._edgesNS[0] = ['0Ni_0', '0Ni_1', '0Si_0', '0Si_1']
        self._edgesEW[0] = ['0Wi_0', '0Wi_1', '0Ei_0', '0Ei_1']
        self._edgesNS[1] = ['1Ni_0', '1Ni_1', '1Si_0', '1Si_1']
        self._edgesEW[1] = ['1Wi_0', '1Wi_1', '1Ei_0', '1Ei_1']
        self._edgesNS[2] = ['2Ni_0', '2Ni_1', '2Si_0', '2Si_1']
        self._edgesEW[2] = ['2Wi_0', '2Wi_1', '2Ei_0', '2Ei_1']
        self._edgesNS[3] = ['3Ni_0', '3Ni_1', '3Si_0', '3Si_1']
        self._edgesEW[3] = ['3Wi_0', '3Wi_1', '3Ei_0', '3Ei_1']
        self._edgesNS[4] = ['4Ni_0', '4Ni_1', '4Si_0', '4Si_1']
        self._edgesEW[4] = ['4Wi_0', '4Wi_1', '4Ei_0', '4Ei_1']
        self._edgesNS[5] = ['5Ni_0', '5Ni_1', '5Si_0', '5Si_1']
        self._edgesEW[5] = ['5Wi_0', '5Wi_1', '5Ei_0', '5Ei_1']
        self._edgesNS[6] = ['6Ni_0', '6Ni_1', '6Si_0', '6Si_1']
        self._edgesEW[6] = ['6Wi_0', '6Wi_1', '6Ei_0', '6Ei_1']
        self._edgesNS[7] = ['7Ni_0', '7Ni_1', '7Si_0', '7Si_1']
        self._edgesEW[7] = ['7Wi_0', '7Wi_1', '7Ei_0', '7Ei_1']
        self._edgesNS[8] = ['8Ni_0', '8Ni_1', '8Si_0', '8Si_1']
        self._edgesEW[8] = ['8Wi_0', '8Wi_1', '8Ei_0', '8Ei_1']

    # calculates the capacity for each queue of each traffic light
    def __init_edges_capacity(self):
        #字典心事{‘tlid’:lengthNS}
        self._edgesNScapacity = {}
        self._edgesEWcapacity = {}

        for tlID in self.get_trafficlights_ID_list():
            # ~ print '----'
            # ~ print 'tlID', tlID
            lengthNS = 0
            lengthWE = 0
            #获取交通环境中的信息
            #存储道路长度总和
            for lane in self._edgesNS[int(tlID)]:
                lengthNS += traci.lane.getLength(lane)
            for lane in self._edgesEW[int(tlID)]:
                lengthWE += traci.lane.getLength(lane)
            #道路上的车辆数
            lengthNS = lengthNS / 7.5  # vehicle length 5m + 2.5m (minGap)
            lengthWE = lengthWE / 7.5
            self._edgesNScapacity[int(tlID)] = lengthNS
            #print('车道上车辆数',self._edgesNScapacity[int(tlID)])
            self._edgesEWcapacity[int(tlID)] = lengthWE



    def get_trafficlights_ID_list(self):
        # return a list with the traffic lights' IDs
        return self.__trafficlights.keys()

    # commands to be performed upon normal termination
    def __close_connection(self):
        traci.close()  # stop TraCI
        sys.stdout.flush()  # clear standard output

    def get_state_actions(self, state):
        self.__check_env()
        # print state
        # print self.__env[state]
        return self.__env[state]

    # check whether the environment is ready to run
    def __check_env(self):
        # check whether the environment data structure was defined
        if not self.__env:
            raise Exception("The traffic lights must be set before running!")

        # discretize the queue occupation in 4 classes equally distributed

    def discretize_queue(self, queue):
        q_class = math.ceil((queue) / 25)
        if queue >= 75:
            q_class = 3

        # percentage
        # ~ if queue < 25:
        # ~ q_class = 0 # 0 - 25%
        # ~ if queue >= 25 and queue < 50:
        # ~ q_class = 1 # 25 - 50%
        # ~ if queue >= 50 and queue < 75:
        # ~ q_class = 2 # 50 - 75%
        # ~ if queue >= 75:
        # ~ q_class = 3 # 75 - 100%

        return int(q_class)

    # http://stackoverflow.com/questions/759296/converting-a-decimal-to-a-mixed-radix-base-number
    #混合基数编码
    def mixed_radix_encode(self, idPhase, duration, queueNS, queueEW):
        factors = [2, 5, 4, 4]

        queueNS = self.discretize_queue(queueNS)
        queueEW = self.discretize_queue(queueEW)

        # the total elapsed time is 30s that are discretize in intervals
        # discretize the duration time (elapsed time) in intervals of 5s (interv_action_selection), except the first interval
        # the fisrt interval is 0 - minGreenTime
        #对最小绿灯时间进行离散化处理
        if duration > 0 and duration <= 10:  # minGreenTime
            duration = 0
        if duration > 10 and duration <= 15:
            duration = 1
        if duration > 15 and duration <= 20:
            duration = 2
        if duration > 20 and duration <= 25:
            duration = 3
        if duration > 25:
            duration = 4

        # idPhase = 0 (NS green), idPhase = 3 (EW green),
        # but for the mixed radix conversion idPhase can only assume 0 or 1
        if idPhase == 3:
            idPhase = 1

        # mixed radix conversion
        values = [idPhase, duration, queueNS, queueEW]
        res = 0
        for i in range(4):
            res = res * factors[i] + values[i]
        # print(values,res)

        return res

    # decode a mixed radix conversion
    #混合基数解码
    def mixed_radix_decode(self, value):
        print('value', value)
        #2544表示有几种状态
        factors = [2, 5, 4, 4]
        res = [0, 0, 0, 0]
        for i in reversed(range(4)):
            res[i] = value % factors[i]
            value = value / factors[i]

        print('reverse %s' % (res))

    # change the traffic light phase
    # set yellow phase and save the next green
    #切换南北东西向的绿灯相位
    def change_trafficlight(self, tlID):
        if traci.trafficlight.getPhase(tlID) == 0:  # NS phase
            traci.trafficlight.setPhase(tlID, 1)
            self.__trafficlights[tlID]["nextGreen"] = 3
        elif traci.trafficlight.getPhase(tlID) == 3:  # EW phase
            traci.trafficlight.setPhase(tlID, 4)
            self.__trafficlights[tlID]["nextGreen"] = 0

    # obs: traci.trafficlights.getPhaseDuration(tlID)
    # it is the time defined in .net file, not the current elapsed time
    def update_phaseTime(self, string, tlID):
        #满足条件时，对绿灯时间+1
        self.__trafficlights[tlID][string] += 1

    # for states
    def calculate_queue_size(self, tlID):
        minSpeed = 2.8  # 10km/h - 2.78m/s
        # 获取车辆ID
        allVehicles = traci.vehicle.getIDList()
        #获取所有车辆ID、速度等信息
        for vehID in allVehicles:
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        info_veh = traci.vehicle.getSubscriptionResults(None)

        # VAR_LANE_ID = 81
        # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001

        qNS = []
        qEW = []
        if len(info_veh) > 0:
            for x in info_veh.keys():
                if info_veh[x][81] in self._edgesNS[int(tlID)]:
                    qNS.append(x)
                if info_veh[x][81] in self._edgesEW[int(tlID)]:
                    qEW.append(x)
        #返回的是与交通信号灯相关联的车辆ID
        return [qNS, qEW]

    # for the reward
    def calculate_stopped_queue_length(self, tlID):
        minSpeed = 2.8  # 10km/h - 2.78m/s
        allVehicles = traci.vehicle.getIDList()

        for vehID in allVehicles:
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        info_veh = traci.vehicle.getSubscriptionResults(None)

        # VAR_LANE_ID = 81
        # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001

        qNS = []
        qEW = []
        if len(info_veh) > 0:
            for x in info_veh.keys():
                if info_veh[x][64] <= minSpeed:
                    if info_veh[x][81] in self._edgesNS[int(tlID)]:
                        qNS.append(x)
                    if info_veh[x][81] in self._edgesEW[int(tlID)]:
                        qEW.append(x)

        return [len(qNS), len(qEW)]

    def calculate_new_state(self, tlID):

        # 1) index of the current phase
        idPhase = traci.trafficlight.getPhase(tlID)

        # 2) the elapsed time in the current phase
        # obs: duration = traci.trafficlights.getPhaseDuration(tlID)
        # its the time defined in .net file, not the current elapsed time
        #获取的是预先定义的绿灯时间
        duration = self.__trafficlights[tlID]["greenTime"]

        # 3) queue size
        #计算南北东西车辆排队长度
        qNS_list, qEW_list = self.calculate_queue_size(tlID)

        qNS = len(qNS_list)
        qEW = len(qEW_list)

        # vehicle / capacity
        #计算车辆队列占用率
        qNS_occupation = 0
        qEW_occupation = 0
        if qNS > 0:
            qNS_occupation = (qNS * 100) / self._edgesNScapacity[int(tlID)]
        if qEW > 0:
            qEW_occupation = (qEW * 100) / self._edgesEWcapacity[int(tlID)]

        new_state = self.mixed_radix_encode(idPhase, duration, qNS_occupation, qEW_occupation)
        #将状态写入state.txt
        s = str(idPhase) + "," + str(duration) + "," + str(round(qNS_occupation,1)) + "," + str(round(qEW_occupation,1)) + "," + str(new_state) + "\n"
        file = 'state.txt'
        arq_tl = open(file, 'a')  # para salvar saida em um arquivo
        arq_tl.writelines(s)
        arq_tl.close()

        return new_state

    def run_episode(self, max_steps=-1, arq_tl='saida_tl.txt', exp=None):

        self.__check_env()

        start = time.time()

        #max_steps *= 1000  # traci returns steps in ms, not s
        self._has_episode_ended = False
        self._episodes += 1
        self.reset_episode()

        self.__init_edges_capacity()  # initialize the queue capacity of each traffic light
        # self.__create_tlogic()

        # ----------------------------------------------------------------------------------

        current_time = 0
        previousNSqueue = [0] * len(self.get_trafficlights_ID_list())
        previousEWqueue = [0] * len(self.get_trafficlights_ID_list())
        currentNSqueue = [0] * len(self.get_trafficlights_ID_list())
        currentEWqueue = [0] * len(self.get_trafficlights_ID_list())
        new_state = [0] * len(self.get_trafficlights_ID_list())
        state = [0] * len(self.get_trafficlights_ID_list())
        choose = [0] * len(self.get_trafficlights_ID_list())  # flag: if choose an action
        maxGreenTime = 180  # maximum green time, to prevent starvation
        minGreenTime = 10
        interv_action_selection = 5  # interval for action selection#选择动作的间隔时间
        update_epsilon = maxGreenTime * 2  # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase

        # main loop
        #主循环，确定训练是在仿真中运行有效
        while ((max_steps > -1 and traci.simulation.getCurrentTime() < max_steps) or max_steps <= -1) and (
                traci.simulation.getMinExpectedNumber() > 0 or traci.simulation.getArrivedNumber() > 0):
            #定义南北东西以及选择动作的列表
            queueNS = [0] * len(self.get_trafficlights_ID_list())

            queueEW = [0] * len(self.get_trafficlights_ID_list())

            actlist = [0] * len(self.get_trafficlights_ID_list())

            #初始化智能体状态和行为的字典
            learner_state_action = {}
            #对每一个交通信号灯进行循环处理
            for tlID in self.get_trafficlights_ID_list():
                # print(f'交通信号灯ID',tlID)
                # print(self.__trafficlights[tlID]["greenTime"])
            # for tlID in (str(3)):
                # A) LEARNER ACTION
                # each traffic light makes a decision at each interv_action_selection (5s)
                #绿灯的时间是动态的
                #__trafficlights[tlID]["greenTime"]最开始都是0
                if self.__trafficlights[tlID]["greenTime"] > 9 and \
                    (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 \
                        and (traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 3):
                    # print("*",tlID,self.__trafficlights[tlID]["greenTime"] , interv_action_selection)
                    new_state[int(tlID)] = self.calculate_new_state(tlID)
                    # print(new_state[int(tlID)])
                    state[int(tlID)], action = self._learners[tlID].act_last(new_state[int(tlID)],tlID)
                    # if int(tlID) == 3:
                    #     print(self._learners[tlID]._QTable)
                    #     print(traci.simulation.getCurrentTime() / 1000, tlID, state[int(tlID)], action, round((previousNSqueue[int(tlID)] + previousEWqueue[int(tlID)]),1))
                    learner_state_action[tlID] = [state[int(tlID)], action]
                    # if green time is equal or more than maxGreenTime, change phase
                    if self.__trafficlights[tlID]["greenTime"] >= maxGreenTime:
                        learner_state_action[tlID] = [state[int(tlID)], 'change']
                    choose[int(tlID)] = True  # flag: if choose an action
                else:
                    choose[int(tlID)] = False
            #print(self.get_trafficlights_ID_list())

            # run a single simulation step
            #仿真向前推进一步
            traci.simulationStep()
            #将仿真时间转化为秒，方便与其他对应
            current_time = traci.simulation.getCurrentTime() / 1000

            # update epsilon manually - traffic lights are not a episodic task
            # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase
            if update_epsilon == current_time:
                update_epsilon = update_epsilon + (maxGreenTime * 2)
                exp.update_epsilon_manually()

            # before start needs 'change' or 'keep' the phase according to the selected action
            #reward_t = 0
            #reward_tlID = 0
            for tlID in self.get_trafficlights_ID_list():
                #print(tlID)

            # for tlID in (str(3)):

                # green phase: idPhase = 0 or 3 (when have two phases)
                # if yellow or all red phase - do nothing

                if traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 3:
                    self.update_phaseTime('greenTime', tlID)

                    # if choose == True: run the action (change, keep)
                    # else: just calculate the queue length (reward will be the average queue length)
                    if choose[int(tlID)] == True:
                        # B) RUN ACTION

                        if learner_state_action[tlID][1] == 'change':  # TODO: more phases
                            self.__trafficlights[tlID]["greenTime"] = 0

                            # this method must set yellow phase and save the next green phase
                            #切换相位
                            self.change_trafficlight(tlID)
                            actlist[int(tlID)] = 1
                        else:  # if action = 'keep' just calculte queue size
                            # calculate queue size
                            queueNS[int(tlID)], queueEW[int(tlID)] = self.calculate_stopped_queue_length(tlID)
                            currentNSqueue[int(tlID)] += queueNS[int(tlID)]
                            currentEWqueue[int(tlID)] += queueEW[int(tlID)]


                    else:
                        # calculate queue size
                        queueNS[int(tlID)], queueEW[int(tlID)] = self.calculate_stopped_queue_length(tlID)
                        currentNSqueue[int(tlID)] += queueNS[int(tlID)]
                        currentEWqueue[int(tlID)] += queueEW[int(tlID)]

                    # if it will select action in the next step,
                    # in the previous you need to calculate the feedback and update Q-table
                    if self.__trafficlights[tlID]["greenTime"] > (minGreenTime - 1) and \
                            (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 and \
                            current_time > 13:
                        #  if current_time: it can enter in the beggining -  13 = 10 (minGreenTime) + 2 (yellow) + 1 (allRed)
                        # calculate the average queue length
                        if self.__trafficlights[tlID][
                            "greenTime"] == minGreenTime:  # action 'change': stay minGreenTime before select new action
                            aver_currentNSqueue = currentNSqueue[int(tlID)] / float(minGreenTime)
                            aver_currentEWqueue = currentEWqueue[int(tlID)] / float(minGreenTime)
                        else:  # action 'keep': stay interv_action_selection before select new action
                            aver_currentNSqueue = currentNSqueue[int(tlID)] / float(interv_action_selection)
                            aver_currentEWqueue = currentEWqueue[int(tlID)] / float(interv_action_selection)

                        # C) CALCULATE REWARD
                        trafficlight_to_proces_feedback = {}

                        # we define the reward as the difference between the previous and current average queue length (AQL)
                        # at the junction $R(s,a,s')= AQL_{s} - AQL_{s'}
                        df = pd.read_excel('D:/tsc_ddqn_prb_1Con_new/nets/3x3grid/e2_result/queue.xlsx')
                        # 初始化一个空的DataFrame来存储结果
                        result_df = pd.DataFrame(columns=['tlID', 'TotalQueue'])
                        # 循环9次，对应于tlID的值从0到8
                        for i in range(9):
                            # 使用布尔索引筛选出tlID等于i的行
                            row = df[df['tlID'] == i]
                            # 检查是否找到了对应的行
                            if not row.empty:
                                # 计算NSqueue和EWqueue的和
                                total_queue = row['NSqueue'].iloc[0]/4 + row['EWqueue'].iloc[0]/4

                                # 将结果添加到结果DataFrame中
                                # 将结果添加到结果DataFrame中
                                result_df = pd.concat(
                                    [result_df, pd.DataFrame({'tlID': [i],
                                                              'NSqueue': row['NSqueue'].iloc[0],
                                                              'EWqueue': row['EWqueue'].iloc[0],
                                                              'TotalQueue': [total_queue]})],

                                    ignore_index=True)

                                #print(f"tlID: {i}, Total_Queue: {total_queue}")
                            else:
                                print(f"No data found for tlID: {i}")
                            #print(result_df)
                            for i in range(9):
                                # 使用布尔索引筛选出tlID等于i的行
                                row = result_df[result_df['tlID'] == i]

                                # 检查是否找到了对应的行
                                if not row.empty:
                                    NSqueue_value = row['NSqueue'].iloc[0]/4
                                    EWqueue_value = row['EWqueue'].iloc[0]/4
                                    # 计算奖励
                                    reward = (row['TotalQueue'].iloc[0] - (
                                                previousEWqueue[int(tlID)] + previousNSqueue[int(tlID)]))
                                    reward *= -1
                                    trafficlight_to_proces_feedback[tlID] = [
                                        reward,
                                        new_state[int(tlID)],
                                        state[int(tlID)]
                                    ]
                                    # update previous queue
                                    previousNSqueue[int(tlID)] = NSqueue_value
                                    previousEWqueue[int(tlID)] = EWqueue_value
                                    # clean current queue
                                    # currentNSqueue[int(tlID)] = 0
                                    # currentEWqueue[int(tlID)] = 0

                        # reward = ((aver_currentEWqueue + aver_currentNSqueue) - (
                        #             previousEWqueue[int(tlID)] + previousNSqueue[int(tlID)]))
                        # reward *= -1

                        ########################################################################
                        # if(tlID == 3 or tlID == 1) :
                        # alpha = ((aver_currentEWqueue + aver_currentNSqueue) - (
                        #         previousEWqueue[int(tlID)] + previousNSqueue[int(tlID)]))
                        # reward = -1 * (aver_currentEWqueue + aver_currentNSqueue)
                        ########################################################################

                        # print(traci.simulation.getCurrentTime() / 1000, "", tlID, " updata queue", round(reward,1),
                        #       round((aver_currentEWqueue + aver_currentNSqueue),1),
                        #       round((previousEWqueue[int(tlID)] + previousNSqueue[int(tlID)]),1)
                        #       )


                        # D) PROCESS FEEDBACK
                        # trafficlight_to_proces_feedback[tlID] = [
                        #     reward,
                        #     new_state[int(tlID)],
                        #     state[int(tlID)]
                        # ]
                        # if int(tlID) == 3:
                        #     print("====================",traci.simulation.getCurrentTime() / 1000, trafficlight_to_proces_feedback[tlID])
                            # print(traci.simulation.getCurrentTime() / 1000, tlID, state[int(tlID)], action)

                        self.__process_trafficlights_feedback(trafficlight_to_proces_feedback)

                        # # update previous queue
                        # previousNSqueue[int(tlID)] = aver_currentNSqueue
                        # previousEWqueue[int(tlID)] = aver_currentEWqueue
                        # # clean current queue
                        # currentNSqueue[int(tlID)] = 0
                        # currentEWqueue[int(tlID)] = 0



            self.metrics(arq_tl, current_time)
            queue_list = ""
            for tlID in self.get_trafficlights_ID_list():
                queue_len = round((previousNSqueue[int(tlID)] + previousEWqueue[int(tlID)]),1)
                queue_list = queue_list + str(queue_len) + ","
                # queue_list = queue_list + "|" +str(queue_len) + "," + str(traci.trafficlight.getPhase(tlID)) + "| "
                # queue_list = queue_list + str(traci.trafficlight.getPhase(tlID)) + "," +str(traci.trafficlight.getPhaseDuration(tlID))

            # print(traci.simulation.getCurrentTime()/1000, ",", queue_list)
            # print(traci.simulation.getCurrentTime() / 1000, actlist)


        self.__close_connection()
        self._has_episode_ended = True

    def __process_trafficlights_feedback(self, traffic_lights):
        # feedback_last
        for tlID in traffic_lights.keys():
            self._learners[str(tlID)].feedback_last(traffic_lights[tlID][0], traffic_lights[tlID][1],
                                                    traffic_lights[tlID][2])

    def metrics(self, arquivo, current_time):
        minSpeed = 2.8  # 10km/h - 2.78m/s

        # using subcriptions
        allVehicles = traci.vehicle.getIDList()
        for vehID in allVehicles:
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        lanes = traci.vehicle.getSubscriptionResults(None)

        # VAR_LANE_ID = 81
        # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001
        # VAR_WAITING_TIME = 122 	Returns the waiting time [s]

        cont_veh_per_tl = [0] * len(self.get_trafficlights_ID_list())
        if len(lanes) > 0:
            for x in lanes.keys():
                for tlID in self.get_trafficlights_ID_list():
                    if lanes[x][64] <= minSpeed:
                        if (lanes[x][81] in self._edgesNS[int(tlID)]) or (lanes[x][81] in self._edgesEW[int(tlID)]):
                            cont_veh_per_tl[int(tlID)] += 1

        # save in a file
        # how many vehicles were in queue in each timestep
        average_queue = 0
        for tlID in self.get_trafficlights_ID_list():
            average_queue = average_queue + cont_veh_per_tl[int(tlID)]
        average_queue = average_queue / float(len(self.__trafficlights))
        arquivo.writelines(
            '%d,%s,%.1f,%d\n' % (current_time, str(cont_veh_per_tl)[1:-1], average_queue, len(allVehicles)))

    def run_step(self):
        raise Exception('run_step is not available in %s class' % self)
        return

    def has_episode_ended(self):
        return self._has_episode_ended

    def __calc_reward(self, state, action, new_state):
        raise Exception('__calc_reward is not available in %s class' % self)
        return

