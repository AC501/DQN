# coding=utf-8
'''
Created on 12/12/2017

@author: Liza L. Lemos <lllemos@inf.ufrgs.br>
'''

from environment import Environment
import traci
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
import pandas as pd
from pandas import DataFrame
import pyEDM


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
        for x in range(0, 320):
            self.__env[x] = d

        # create the set of traffic ligths
        self.__create_trafficlights()

    def __create_trafficlights(self):
        # set of all traffic lights in the simulation
        # each element in __trafficlights correspond to another in __learners
        self.__trafficlights = {}

        # process all trafficlights entries
        junctions_parse = minidom.parse(self.__net_file).getElementsByTagName('junction')
        i = 0
        for element in junctions_parse:
            if element.getAttribute('type') == "traffic_light":
                tlID = element.getAttribute('id')
                incline = element.getAttribute('incLanes').split(' ')
                # len_incline = len(str(incline))
                # incline = str(incline)[2:len_incline-1]
                # incline = list(incline.split(' '))

                # print("================")
                # print(type(incline))
                # print(incline)
                # # print(incline[0])

                # create the entry in the dictionary
                if i < 10:
                    self.__trafficlights[tlID] = {
                        'greenTime': 0,
                        'nextGreen': -1,
                        'yellowTime': -1,
                        'redTime': -1,
                        'incline': incline
                    }
                    i += 1

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

    # define the edges/lanes that are controlled for each traffic light
    # the function getControlledLanes() from TRACI, returned the names of lanes doubled
    # that's way is listed here

    # calculates the capacity for each queue of each traffic light
    def __init_edges_capacity(self):
        self._edgesCapacity = {}

        for tlID in self.get_trafficlights_ID_list():
            # 获取探测器长度
            length = 0
            for lane in self.__trafficlights[tlID]['incline']:
                # print("=============")
                # print(lane)
                length += traci.lane.getLength(lane)
            length = length / 7.5  # vehicle length 5m + 2.5m (minGap)
            self._edgesCapacity[int(tlID)] = length

    # https://sourceforge.net/p/sumo/mailman/message/35824947/
    # It's necessary set a new logic, because we need more duration time.
    # in SUMO the duration of the phases are set in .net file.
    # but if in .net the phase duration is set to 30s and if we want 40s, the simulator will change phase in 30s
    # thus, we set the duration with a high value
    # also, the yellow and all red phase duration can be set here
    # if prefer, this can be changed in .net file 'tllogic' tag
    # obs: the duration is set in ms

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

        if queue < 25:
            q_class = 0
        elif 25 <= queue < 50:
            q_class = 1
        elif 50 <= queue < 75:
            q_class = 2
        else:
            q_class = 3

        return int(q_class)

    # http://stackoverflow.com/questions/759296/converting-a-decimal-to-a-mixed-radix-base-number
    def mixed_radix_encode(self, idPhase, duration, queue):
        factors = [2, 5, 8]

        queue = self.discretize_queue(queue)

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
        if idPhase == 4:
            idPhase = 1

        # mixed radix conversion
        values = [idPhase, duration, queue]
        res = 0
        for i in range(3):
            res = res * factors[i] + values[i]

        return res

    # decode a mixed radix conversion
    def mixed_radix_decode(self, value):
        factors = [2, 5, 4, 4]
        res = [0, 0, 0, 0]
        for i in reversed(range(4)):
            res[i] = value % factors[i]
            value = value / factors[i]

    # change the traffic light phase
    # set yellow phase and save the next green
    def change_trafficlight(self, tlID):
        if traci.trafficlight.getPhase(tlID) == 0:  # NS phase
            traci.trafficlight.setPhase(tlID, 1)
            self.__trafficlights[tlID]["nextGreen"] = 4
        elif traci.trafficlight.getPhase(tlID) == 4:  # EW phase
            traci.trafficlight.setPhase(tlID, 5)
            self.__trafficlights[tlID]["nextGreen"] = 0

    # obs: traci.trafficlight.getPhaseDuration(tlID)
    # it is the time defined in .net file, not the current elapsed time
    def update_phaseTime(self, string, tlID):
        self.__trafficlights[tlID][string] += 1

    # for states
    def calculate_queue_size(self, tlID):
        minSpeed = 2.8  # 10km/h - 2.78m/s
        allVehicles = traci.vehicle.getIDList()

        for vehID in allVehicles:
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        info_veh = traci.vehicle.getSubscriptionResults(None)

        # VAR_LANE_ID = 81
        # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001

        qsize = []
        if len(info_veh) > 0:
            for x in info_veh.keys():
                if info_veh[x][81] in self.__trafficlights[tlID]['incline']:
                    qsize.append(x)

        return qsize

    # for the reward
    def calculate_stopped_queue_length(self, tlID):
        minSpeed = 2.8  # 10km/h - 2.78m/s
        allVehicles = traci.vehicle.getIDList()

        for vehID in allVehicles:
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        info_veh = traci.vehicle.getSubscriptionResults(None)

        # VAR_LANE_ID = 81
        # VAR_SPEED = 64 Returns the speed of the named vehicle within the last step [m/s]; error value: -1001

        qlength = []
        if len(info_veh) > 0:
            for x in info_veh.keys():
                if info_veh[x][64] <= minSpeed:
                    if info_veh[x][81] in self.__trafficlights[tlID]['incline']:
                        qlength.append(x)

        return len(qlength)

    def calculate_new_state(self, tlID):

        # 1) index of the current phase
        idPhase = traci.trafficlight.getPhase(tlID)

        # 2) the elapsed time in the current phase
        # obs: duration = traci.trafficlight.getPhaseDuration(tlID)
        # its the time defined in .net file, not the current elapsed time
        duration = self.__trafficlights[tlID]["greenTime"]

        # 3) queue size
        queue = self.calculate_queue_size(tlID)

        qsize = len(queue)

        # vehicle / capacity 百分数分子
        q_occupation = 0
        if qsize > 0:
            q_occupation = (qsize * 100) / self._edgesCapacity[int(tlID)]

        new_state = self.mixed_radix_encode(idPhase, duration, q_occupation)

        return new_state

    def run_episode(self, max_steps=-1, arq_tl='saida_tl.txt', exp=None):
        # print"start episode"

        self.__check_env()

        start = time.time()

        max_steps *= 1000  # traci returns steps in ms, not s
        self._has_episode_ended = False
        self._episodes += 1
        self.reset_episode()

        # 初始化每个红绿灯的排队容量
        self.__init_edges_capacity()

        # ----------------------------------------------------------------------------------

        current_time = 0
        previousQueue = {}
        currentQueue = {}
        new_state = {}
        state = {}
        choose = {}
        for id in self.get_trafficlights_ID_list():
            previousQueue[int(id)] = 0
            currentQueue[int(id)] = 0
            new_state[int(id)] = 0
            state[int(id)] = 0
            choose[int(id)] = 0

        maxGreenTime = 180  # maximum green time, to prevent starvation
        minGreenTime = 10
        # 动作选择间隔
        interv_action_selection = 5
        update_epsilon = maxGreenTime * 2  # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase

        # main loop

        while ((max_steps > -1 and traci.simulation.getCurrentTime() < max_steps) or max_steps <= -1) and (
                traci.simulation.getMinExpectedNumber() > 0 or traci.simulation.getArrivedNumber() > 0):
            # print(traci.simulation.getCurrentTime()/1000)

            queue = {}
            for id in self.get_trafficlights_ID_list():
                queue[int(id)] = 0
                # print("========================================")
                # print(id)
                # print(traci.trafficlight.getControlledLinks(id))
            # print("--------------------------------------------")

            learner_state_action = {}
            for tlID in self.get_trafficlights_ID_list():
                # print tlID
                # A) LEARNER ACTION
                # each traffic light makes a decision at each interv_action_selection (5s)
                if self.__trafficlights[tlID]["greenTime"] > 9 and (
                        self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0:
                    new_state[int(tlID)] = self.calculate_new_state(tlID)
                    state[int(tlID)], action = self._learners[tlID].act_last(new_state[int(tlID)])
                    learner_state_action[tlID] = [state[int(tlID)], action]
                    # if green time is equal or more than maxGreenTime, change phase
                    if self.__trafficlights[tlID]["greenTime"] >= maxGreenTime:
                        learner_state_action[tlID] = [state[int(tlID)], 'change']
                    choose[int(tlID)] = True  # flag: if choose an action
                else:
                    choose[int(tlID)] = False

            # run a single simulation step
            traci.simulationStep()
            current_time = traci.simulation.getCurrentTime() / 1000

            # update epsilon manually - traffic lights are not a episodic task
            # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase
            if update_epsilon == current_time:
                update_epsilon = update_epsilon + (maxGreenTime * 2)
                exp.update_epsilon_manually()

            # before start needs 'change' or 'keep' the phase according to the selected action
            for tlID in self.get_trafficlights_ID_list():

                # print tlID

                # green phase: idPhase = 0 or 3 (when have two phases)
                # if yellow or all red phase - do nothing
                if traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 4:
                    self.update_phaseTime('greenTime', tlID)

                    # if choose == True: run the action (change, keep)
                    # else: just calculate the queue length (reward will be the average queue length)
                    # 该信号灯已经选择过
                    if choose[int(tlID)] == True:
                        # B) RUN ACTION
                        if learner_state_action[tlID][1] == 'change':  # TODO: more phases
                            self.__trafficlights[tlID]["greenTime"] = 0

                            # this method must set yellow phase and save the next green phase
                            self.change_trafficlight(tlID)
                        else:  # if action = 'keep' just calculte queue size
                            # calculate queue size
                            # 获取当前东西南北方向的车辆长度
                            queue[int(tlID)] = self.calculate_stopped_queue_length(tlID)
                            currentQueue[int(tlID)] += queue[int(tlID)]

                    else:
                        # calculate queue size
                        queue[int(tlID)] = self.calculate_stopped_queue_length(tlID)

                        currentQueue[int(tlID)] += queue[int(tlID)]

                    # if it will select action in the next step,
                    # in the previous you need to calculate the feedback and update Q-table
                    if self.__trafficlights[tlID]["greenTime"] > (minGreenTime - 1) and (self.__trafficlights[tlID][
                                                                                             "greenTime"] % interv_action_selection) == 0 and current_time > 13:
                        # if current_time: it can enter in the beggining -  13 = 10 (minGreenTime) + 2 (yellow) + 1 (
                        # allRed)

                        # calculate the average queue length
                        # action 'change': stay minGreenTime before select new action
                        if self.__trafficlights[tlID]["greenTime"] == minGreenTime:
                            aver_currentQueue = currentQueue[int(tlID)] / float(minGreenTime)
                        # action 'keep': stay interv_action_selection before select new action
                        else:
                            aver_currentQueue = currentQueue[int(tlID)] / float(interv_action_selection)

                        # C) CALCULATE REWARD
                        trafficlight_to_proces_feedback = {}

                        # we define the reward as the difference between the previous and current average queue
                        # length (AQL) at the junction $R(s,a,s')= AQL_{s} - AQL_{s'}$
                        reward = (aver_currentQueue - (
                                previousQueue[int(tlID)] + previousQueue[int(tlID)]))
                        reward *= -1

                        # D) PROCESS FEEDBACK
                        trafficlight_to_proces_feedback[tlID] = [
                            reward,
                            new_state[int(tlID)],
                            state[int(tlID)]
                        ]

                        # print "process feedback"
                        self.__process_trafficlights_feedback(trafficlight_to_proces_feedback)

                        # update previous queue
                        previousQueue[int(tlID)] = aver_currentQueue
                        # clean current queue
                        currentQueue[int(tlID)] = 0

            # 打印Q表

            # self.metrics(arq_tl, current_time)
        tick = round(traci.simulation.getCurrentTime() / 1000)
        rho_max = []
        for i in self.get_trafficlights_ID_list():
            if int(i) < 8:
                for j in self.get_trafficlights_ID_list():
                    if int(j) > int(i):

                        df1 = pd.DataFrame(self._learners[i]._QTablebak).T
                        df2 = pd.DataFrame(self._learners[j]._QTablebak).T

                        for df1_r in range(df1.shape[0]):
                            for df1_c in range(df1.shape[1]):

                                x1 = df1.iat[df1_r, df1_c]

                                for df2_r in range(df2.shape[0]):
                                    for df2_c in range(df2.shape[1]):

                                        x2 = df2.iat[df2_r, df2_c]
                                        # print("x1=",x1)
                                        # print("x2=",x2)
                                        y = np.hstack([x1, x2])
                                        t = []
                                        for t_index in range(len(y)):
                                            t.append(t_index)
                                        data_array = np.vstack([t, y])

                                        data_frame = pd.DataFrame({'time': t, 'value': y})
                                        # print(data_array)
                                        # print(data_frame)
                                        # print(y)
                                        if x1 == 0:
                                            x1 = [0]
                                        if x2 == 0:
                                            x2 = [0]
                                        str1 = str(1) + " " + str(len(x1))
                                        str2 = str(len(x1) + 1) + " " + str(len(x1) + len(x2))
                                        # print(str1)
                                        # print(str2)
                                        try:
                                            rho = pyEDM.EmbedDimension(dataFrame=data_frame, lib=str1, pred=str2,
                                                                       columns="value", showPlot=False)
                                            max = rho['rho'].max()
                                            # print(data_frame)
                                            # print("max=",max)

                                        except:
                                            print("Noooooooooo!")

        self.__close_connection()
        self._has_episode_ended = True

    def __process_trafficlights_feedback(self, traffic_lights):
        # feedback_last
        for tlID in traffic_lights.keys():
            # print("feedback",tlID)
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

        cont_veh_per_tl = {}
        for id in self.get_trafficlights_ID_list():
            cont_veh_per_tl[int(id)] = 0

        if len(lanes) > 0:
            for x in lanes.keys():
                for tlID in self.get_trafficlights_ID_list():
                    if lanes[x][64] <= minSpeed:
                        if lanes[x][81] in self.__trafficlights[tlID]['incline']:
                            cont_veh_per_tl[int(tlID)] += 1

        # save in a file
        # how many vehicles were in queue in each timestep
        average_queue = 0
        phase = ''
        for tlID in self.get_trafficlights_ID_list():
            phase = phase + 'tl' + tlID + ":" + str(traci.trafficlight.getPhase(tlID)) + " "
            average_queue = average_queue + cont_veh_per_tl[int(tlID)]
        average_queue = average_queue / float(len(self.__trafficlights))
        arquivo.writelines(
            '%d,%s,%.1f,%d\n' % (current_time, phase, average_queue, len(allVehicles)))

    def run_step(self):
        raise Exception('run_step is not available in %s class' % self)
        return

    def has_episode_ended(self):
        return self._has_episode_ended

    def __calc_reward(self, state, action, new_state):
        raise Exception('__calc_reward is not available in %s class' % self)
        return
