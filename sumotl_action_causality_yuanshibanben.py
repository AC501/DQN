'''
Created on 12/12/2017

@author: Liza L. Lemos <lllemos@inf.ufrgs.br>
'''

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
import re

# #只提取数字
# def extract_number(string):
#     match = re.search(r'\b\d+\b', string)
#     if match:
#         return int(match.group())
#     else:
#         return None

#定义sumo仿真环境的相关配置信息
class SUMOTrafficLights(Environment):

    #继承SUMOTrafficLights类，并创建自己的仿真环境
    def __init__(self, cfg_file, port=8813, use_gui=False):

        super(SUMOTrafficLights, self).__init__()

        self.__create_env(cfg_file, port, use_gui)



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

        #根据策略选择状态
        self.__env = {}

        d = ['keep', 'change']

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
        junctions_parse = minidom.parse(self.__net_file).getElementsByTagName('junction')
        #提取交通信号灯的id
        for element in junctions_parse:
            if element.getAttribute('type') == "traffic_light":

                tlID = element.getAttribute('id').encode('utf-8')

                # print(f"Value of tlID: {tlID}")
                #提取交通信号灯的ID
                tlID = (str(tlID))[2:3]
                #print(tlID)


                # print('只提取数字：',tlID)
                # create the entry in the dictionary
                #存储交通信号灯的相关信息
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
        #重置仿真环境中交通信号灯的状态
        # reset traffic lights attributes
        for tlID in self.get_trafficlights_ID_list():
            # print('******')
            # print(tlID)
            self.__trafficlights[tlID]['greenTime'] = 0
            self.__trafficlights[tlID]['nextGreen'] = -1
            self.__trafficlights[tlID]['yellowTime'] = -1
            self.__trafficlights[tlID]['redTime'] = -1


    def __create_edges(self):
        self._edgesNS = {}
        self._edgesEW = {}
        #交通信号灯控制车道的方向
        self._edgesNS[0] = ['-2642_0', '-2642_1', '2636_0', '2636_1']
        self._edgesEW[0] = ['-3420_0', '-3420_1', '-3420_2', '-3414_0', '-3414_1', '-3414_2']
        self._edgesNS[1] = ['-3382_0', '-3382_1', '-3382_2', '3381_0', '3381_1']
        self._edgesEW[1] = ['4513_0', '4513_1', '4513_2', '-3380_0', '-3380_1', '-3380_2']
        self._edgesNS[2] = ['-3729_0', '-3729_1', '3738_0', '3738_1']
        self._edgesEW[2] = ['3403_0', '3403_1', '3403_2', '-3398_0', '-3398_1', '-3398_2']
        self._edgesNS[3] = ['-3401_0', '-3401_1', '3405_0', '3405_1']
        self._edgesEW[3] = ['3415_0', '3415_1', '3415_2', '-3403_0', '-3403_1', '-3403_2']
        self._edgesNS[4] = ['-3405_0', '-3405_1', '3404_0', '3404_1']
        self._edgesEW[4] = ['3411_0', '3411_1', '-3739_0', '-3739_1']
        self._edgesNS[5] = ['-3416_0', '-3416_1', '3412_0', '3412_1']
        self._edgesEW[5] = ['3414_0', '3414_1', '3414_2', '-3415_0', '-3415_1', '-3415_2']
        self._edgesNS[6] = ['-5550_0', '-5550_1', '-4424_0']
        self._edgesEW[6] = ['5548_0', '5548_1', '5548_2', '-5554_0', '-5554_1', '-5554_2', '-5554_3']
        self._edgesNS[7] = ['-4515_0', '-4515_1', '-4515_2', '4514_0', '4514_1', '4514_2']
        self._edgesEW[7] = ['4512_0', '4512_1', '4512_2', '-4513_0', '-4513_1', '-4513_2']


    # calculates the capacity for each queue of each traffic light


    def __init_edges_capacity(self):
        #创建交通信号灯控制下道路边缘的容量
        self._edgesNScapacity = {}
        self._edgesEWcapacity = {}

        #获取交通信号灯的id
        for tlID in self.get_trafficlights_ID_list():
            # print('----')
            # print('tlID：', tlID)

            lengthNS = 0
            lengthWE = 0
            #获取交通环境中的信息
            # 先将 tlID 转换为整数
            #tlID = int(tlID)
            for lane in self._edgesNS[int(tlID)]:
                lengthNS += traci.lane.getLength(lane)
            for lane in self._edgesEW[int(tlID)]:
                lengthWE += traci.lane.getLength(lane)
            lengthNS = lengthNS / 7.5  # vehicle length 5m + 2.5m (minGap)
            lengthWE = lengthWE / 7.5
            self._edgesNScapacity[int(tlID)] = lengthNS
            self._edgesEWcapacity[int(tlID)] = lengthWE
            #print(self._edgesNScapacity)




    def get_trafficlights_ID_list(self):
        # return a list with the traffic lights' IDs
        return self.__trafficlights.keys()


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

    #将车量排队长度进行离散化处理
    def discretize_queue(self, queue):
        q_class = math.ceil((queue) / 25)
        if queue >= 75:
            q_class = 3



        return int(q_class)

    # http://stackoverflow.com/questions/759296/converting-a-decimal-to-a-mixed-radix-base-number
    def mixed_radix_encode(self, idPhase, duration, queueNS, queueEW):
        factors = [2, 5, 4, 4]

        queueNS = self.discretize_queue(queueNS)
        queueEW = self.discretize_queue(queueEW)

        # the total elapsed time is 30s that are discretize in intervals
        # discretize the duration time (elapsed time) in intervals of 5s (interv_action_selection), except the first interval
        # the fisrt interval is 0 - minGreenTime
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
    def mixed_radix_decode(self, value):
        print
        'value', value
        factors = [2, 5, 4, 4]
        res = [0, 0, 0, 0]
        for i in reversed(range(4)):
            res[i] = value % factors[i]
            value = value / factors[i]

        print
        'reverse %s' % (res)

    # change the traffic light phase
    # set yellow phase and save the next green
    def change_trafficlight(self, tlID):

        #改变东西南北相位红绿灯的状态
        if traci.trafficlight.getPhase(tlID) == 0:  # NS phase
            traci.trafficlight.setPhase(tlID, 1)
            self.__trafficlights[tlID]["nextGreen"] = 3
        elif traci.trafficlight.getPhase(tlID) == 3:  # EW phase
            traci.trafficlight.setPhase(tlID, 4)
            self.__trafficlights[tlID]["nextGreen"] = 0

    # obs: traci.trafficlights.getPhaseDuration(tlID)
    # it is the time defined in .net file, not the current elapsed time
    def update_phaseTime(self, string, tlID):
        self.__trafficlights[tlID][string] += 1

    # for states，状态信息
    #计算交通信号灯南北和东西向的车队长度
    def calculate_queue_size(self, tlID):

        minSpeed = 2.8  # 10km/h - 2.78m/s
        #获取所有车辆的ID
        allVehicles = traci.vehicle.getIDList()

        for vehID in allVehicles:
            #循环遍历订阅车辆车辆的信息
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        #获取订阅车辆的信息
        info_veh = traci.vehicle.getSubscriptionResults(None)

        # VAR_LANE_ID = 81
        # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001

        qNS = []
        qEW = []
        if len(info_veh) > 0:
            for x in info_veh.keys():
                #判断车辆所在的车道是否是交通信号灯所控制的道路
                if info_veh[x][81] in self._edgesNS[int(tlID)]:
                    qNS.append(x)
                if info_veh[x][81] in self._edgesEW[int(tlID)]:
                    qEW.append(x)

        return [qNS, qEW]

    # for the reward
    #计算交通信号灯的南北和东北方向停止的车队长度
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
                #判断速度和最小阈值的关系以及行车道和所控制道路的关系
                if info_veh[x][64] <= minSpeed:
                    if info_veh[x][81] in self._edgesNS[int(tlID)]:
                        qNS.append(x)
                    if info_veh[x][81] in self._edgesEW[int(tlID)]:
                        qEW.append(x)

        return [len(qNS), len(qEW)]

    def calculate_new_state(self, tlID):


        # 1) index of the current phase
        #获取交通信号灯的相位索引
        idPhase = traci.trafficlight.getPhase(tlID)
        # print(idPhase)

        # 2) the elapsed time in the current phase
        # obs: duration = traci.trafficlights.getPhaseDuration(tlID)
        # its the time defined in .net file, not the current elapsed time
        duration = self.__trafficlights[tlID]["greenTime"]

        # 3) queue size
        qNS_list, qEW_list = self.calculate_queue_size(tlID)

        qNS = len(qNS_list)
        qEW = len(qEW_list)

        # vehicle / capacity
        qNS_occupation = 0
        qEW_occupation = 0
        if qNS > 0:
            qNS_occupation = (qNS * 100) / self._edgesNScapacity[int(tlID)]
        if qEW > 0:
            qEW_occupation = (qEW * 100) / self._edgesEWcapacity[int(tlID)]

        new_state = self.mixed_radix_encode(idPhase, duration, qNS_occupation, qEW_occupation)
        s = str(idPhase) + "," + str(duration) + "," + str(round(qNS_occupation,1)) + "," + str(round(qEW_occupation,1)) + "," + str(new_state) + "\n"
        file = 'state.txt'
        arq_tl = open(file, 'a')  # para salvar saida em um arquivo
        arq_tl.writelines(s)
        arq_tl.close()

        return new_state

    def run_episode(self, max_steps=-1, arq_tl='saida_tl.txt', exp=None):

        self.__check_env()

        start = time.time()

        max_steps *= 1000  # traci returns steps in ms, not s
        self._has_episode_ended = False
        self._episodes += 1
        self.reset_episode()

        self.__init_edges_capacity()  # initialize the queue capacity of each traffic light
        # self.__create_tlogic()

        # ----------------------------------------------------------------------------------

        current_time = 0
        #初始化先前和当前各个方向交通信号灯id的列表
        previousNSqueue = [0] * len(self.get_trafficlights_ID_list())

        previousEWqueue = [0] * len(self.get_trafficlights_ID_list())
        currentNSqueue = [0] * len(self.get_trafficlights_ID_list())
        currentEWqueue = [0] * len(self.get_trafficlights_ID_list())
        new_state = [0] * len(self.get_trafficlights_ID_list())
        state = [0] * len(self.get_trafficlights_ID_list())
        traffic_lights_ID_list = self.get_trafficlights_ID_list()
        #将所有交通信号灯初始化为0
        choose = [0] * len(traffic_lights_ID_list)
        #print(choose)
        #print(f"Length of choose list: {len(choose)}, Length of traffic lights ID list: {len(traffic_lights_ID_list)}")
        maxGreenTime = 180  # maximum green time, to prevent starvation
        minGreenTime = 10
        interv_action_selection = 5  # interval for action selection。策略在选择新的行动之前需要保持的绿灯时间间隔
        update_epsilon = maxGreenTime * 2  # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase

        # main loop
        #判断仿真是否继续
        while ((max_steps > -1 and traci.simulation.getCurrentTime() < max_steps) or max_steps <= -1) and (
                traci.simulation.getMinExpectedNumber() > 0 or traci.simulation.getArrivedNumber() > 0):

            queueNS = [0] * len(self.get_trafficlights_ID_list())
            queueEW = [0] * len(self.get_trafficlights_ID_list())

            actlist = [0] * len(self.get_trafficlights_ID_list())


            learner_state_action = {}
            # valid_tl_ids = [15, 306, 309, 310, 311, 313, 605, 626]
            # for tlID in valid_tl_ids:
            #     if tlID in self.__trafficlights:

            for tlID in self.get_trafficlights_ID_list():
                # print(f'红绿灯：',tlID)
                # print(f'先前东西方向排队车辆:',previousEWqueue[int(tlID)])
                # print(tlID)
                #tlID_str = str(tlID)
                #tlID = extract_number(tlID_str)
                # print(tlID)
            # for tlID in (str(3)):
                # A) LEARNER ACTION
                # each traffic light makes a decision at each interv_action_selection (5s)
                if self.__trafficlights[tlID]["greenTime"] > 9 and \
                    (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 \
                        and (traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 3):
                    # print("*",tlID,self.__trafficlights[tlID]["greenTime"] , interv_action_selection)
                    #满足条件更新状态，新的状态[idPhase, duration, qNS_occupation, qEW_occupation]
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
                    tlID_str = str(tlID)
                    # tlID = extract_number(tlID_str)

                    if 0 <= int(tlID) < len(choose):
                        choose[int(tlID)] = False
                    # else:
                        # print("Invalid traffic lights ID:", tlID)
                    # choose[int(tlID)] = False

            # run a single simulation step
            traci.simulationStep()
            #获取当前仿真时间以S为单位
            current_time = traci.simulation.getCurrentTime() / 1000

            # update epsilon manually - traffic lights are not a episodic task
            # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase
            if update_epsilon == current_time:
                update_epsilon = update_epsilon + (maxGreenTime * 2)
                exp.update_epsilon_manually()

            # before start needs 'change' or 'keep' the phase according to the selected action
            reward_t = 0
            reward_tlID = 0
            #优化交通流量

            for tlID in self.get_trafficlights_ID_list():
            # for tlID in (str(3)):

                # green phase: idPhase = 0 or 3 (when have two phases)
                # if yellow or all red phase - do nothing

                if traci.trafficlight.getPhase(str(tlID)) == 0 or traci.trafficlight.getPhase(str(tlID)) == 2:
                    #print(traci.trafficlight.getPhase(str(tlID)))
                    self.update_phaseTime('greenTime', tlID)

                    # if choose == True: run the action (change, keep)
                    # else: just calculate the queue length (reward will be the average queue length)

                    if choose[int(tlID)] == True:
                        # B) RUN ACTION
                        if learner_state_action[tlID][1] == 'change':  # TODO: more phases
                            self.__trafficlights[tlID]["greenTime"] = 0

                            # this method must set yellow phase and save the next green phase
                            self.change_trafficlight(tlID)
                            actlist[int(tlID)] = 1
                        else:  # if action = 'keep' just calculte queue size
                            # calculate queue size
                            queueNS[int(tlID)], queueEW[int(tlID)] = self.calculate_stopped_queue_length(tlID)
                            currentNSqueue[int(tlID)] += queueNS[int(tlID)]
                            currentEWqueue[int(tlID)] += queueEW[int(tlID)]
                            #print(currentNSqueue[int(tlID)])

                    else:
                        # calculate queue size
                        queueNS[int(tlID)], queueEW[int(tlID)] = self.calculate_stopped_queue_length(tlID)
                        currentNSqueue[int(tlID)] += queueNS[int(tlID)]
                        currentEWqueue[int(tlID)] += queueEW[int(tlID)]
                        #print(currentNSqueue[int(tlID)])

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
                        # at the junction $R(s,a,s')= AQL_{s} - AQL_{s'}$
                        #当前东西方向加上南北方向的队列长度后减去以前东西南北方向车辆和
                        reward = ((aver_currentEWqueue + aver_currentNSqueue) - (
                                    previousEWqueue[int(tlID)] + previousNSqueue[int(tlID)]))
                        reward *= -1




                        # D) PROCESS FEEDBACK
                        trafficlight_to_proces_feedback[tlID] = [
                            reward,
                            new_state[int(tlID)],
                            state[int(tlID)]
                        ]

                        self.__process_trafficlights_feedback(trafficlight_to_proces_feedback)

                        # update previous queue
                        previousNSqueue[int(tlID)] = aver_currentNSqueue
                        previousEWqueue[int(tlID)] = aver_currentEWqueue
                        # clean current queue
                        currentNSqueue[int(tlID)] = 0
                        currentEWqueue[int(tlID)] = 0


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

