'''
Created on 12/12/2017

@author: Liza L. Lemos <lllemos@inf.ufrgs.br>
'''
import socket

import pandas as pd

from environment import Environment
import traci
from Detector import DetectorETwo
# 找一个二进制文件，即寻找可执行文件
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
from replay import ReplayBuffer
import torch

from DDPM.DiffusionModel import RewardDiffusionModel
from DDPM.full_ddpm import MLPDiffusion


class SUMOTrafficLights(Environment):

    def __init__(self, cfg_file, port=8813, use_gui=False, batch_size=32):

        super(SUMOTrafficLights, self).__init__()

        self.total_NS = 0
        self.total_EW = 0
        self.total_queue_NS = 0
        self.total_queue_EW = 0

        self.replay_buffers = {}
        self.batch_size = batch_size
        self.learners = {}
        self.replay_buffers = {}
        self.__create_env(cfg_file, port, use_gui)
        self.already_counted_ids = set()
        self.counted_vehicles = set()

        # 加载训练好的扩散模型
        self.model = MLPDiffusion(100)  # 根据实际情况调整
        self.model.load_state_dict(torch.load('D:\\tsc_ddqn_prb_1Con_new\\weights\\best_model_fullddpm.pth'))
        self.model.eval()  # 切换到评估模式
        self.device = torch.device("cpu")
        self.model.to(self.device)

    '''
    Create the environment as a MDP. The MDP is modeled as follows:
    * for each traffic light:
    * the  is defined as a vector [current phase, elapsed time of current phase, queue length for each phase]
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

        d = [0, 1]
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

    def get_info_E2(decid):  # 传入检测器id信息
        quene_lenth = {}  # 定义排队长度
        occ = {}  # 定义占有率
        lane_length = {}  # 定义车道长度
        for dets in decid:  # 遍历检测器
            lane_id = traci.lanearea.getLaneID(dets)  # 通过检测器获取车道id
            lane_length[lane_id] = traci.lane.getLength(lane_id)  # 获取车道长度并记录
            quene_lenth[lane_id] = traci.lanearea.getJamLengthMeters(dets)  # 获取排队长度并记录
            occ[lane_id] = traci.lanearea.getLastStepOccupancy(dets)  # 获取占有率并记录
        quene_lenth_d = pd.DataFrame.from_dict(quene_lenth, orient='index')  # 将排队长度转换为dataframe
        quene_lenth_d.rename(columns={0: "quene_lenth"}, inplace=True)  # 更改数据标签
        occ_d = pd.DataFrame.from_dict(occ, orient='index')
        occ_d.rename(columns={0: "occ"}, inplace=True)
        lane_length_d = pd.DataFrame.from_dict(lane_length, orient='index')
        lane_length_d.rename(columns={0: "length"}, inplace=True)
        data = pd.concat([quene_lenth_d, occ_d, lane_length_d], axis=1)  # 融合获取的数据信息
        return data

    def __create_trafficlights(self):
        # set of all traffic lights in the simulation
        # each element in __trafficlights correspond to another in __learners
        self.__trafficlights = {}

        # process all trafficlights entries
        junctions_parse = minidom.parse(self.__net_file).getElementsByTagName('junction')
        for element in junctions_parse:
            if element.getAttribute('type') == "traffic_light":
                tlID = element.getAttribute('id').encode('utf-8')
                # print((str(tlID))[2:3])
                tlID = (str(tlID))[2:3]
                # print(tlID)
                # create the entry in the dictionary
                self.__trafficlights[tlID] = {
                    'greenTime': 0,
                    'nextGreen': -1,
                    'yellowTime': -1,
                    'redTime': -1,
                    'current_time': 0,
                    'step': 0,
                    'already_counted_ids': set(),
                    'total_NS': 0,
                    'total_EW': 0
                }

    def reset_episode(self):

        super(SUMOTrafficLights, self).reset_episode()

        # initialise TraCI
        traci.start([self._sumo_binary, "-c", self.__cfg_file])
        # traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

        # reset traffic lights attributes
        for tlID in self.get_trafficlights_ID_list():
            self.__trafficlights[tlID]['greenTime'] = 0
            self.__trafficlights[tlID]['nextGreen'] = -1
            self.__trafficlights[tlID]['yellowTime'] = -1
            self.__trafficlights[tlID]['redTime'] = -1
            self.__trafficlights[tlID]['current_time'] = 0
            self.__trafficlights[tlID]['step'] = 0

            self.__trafficlights[tlID]['already_counted_ids'].clear()
            self.__trafficlights[tlID]['total_NS'] = 0
            self.__trafficlights[tlID]['total_EW'] = 0

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

    # calculates the capacity for each queue of each traffic light
    def __init_edges_capacity(self):
        self._edgesNScapacity = {}
        self._edgesEWcapacity = {}

        for tlID in self.get_trafficlights_ID_list():
            # ~ print '----'
            # ~ print 'tlID', tlID
            lengthNS = 0
            lengthWE = 0
            # 获取交通环境中的信息
            for lane in self._edgesNS[int(tlID)]:
                lengthNS += traci.lane.getLength(lane)
            for lane in self._edgesEW[int(tlID)]:
                lengthWE += traci.lane.getLength(lane)
            lengthNS = lengthNS / 7.5  # vehicle length 5m + 2.5m (minGap)
            lengthWE = lengthWE / 7.5
            self._edgesNScapacity[int(tlID)] = lengthNS
            self._edgesEWcapacity[int(tlID)] = lengthWE

    # https://sourceforge.net/p/sumo/mailman/message/35824947/

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

    # change the traffic light phase
    # set yellow phase and save the next green
    def change_trafficlight(self, tlID):
        if traci.trafficlight.getPhase(tlID) == 0:  # NS phase
            traci.trafficlight.setPhase(tlID, 1)
            self.__trafficlights[tlID]["nextGreen"] = 0
        elif traci.trafficlight.getPhase(tlID) == 3:  # EW phase
            traci.trafficlight.setPhase(tlID, 4)
            self.__trafficlights[tlID]["nextGreen"] = 0

    # obs: traci.trafficlights.getPhaseDuration(tlID)
    # it is the time defined in .net file, not the current elapsed time
    def update_phaseTime(self, string, tlID):
        self.__trafficlights[tlID][string] += 1

    # for states
    # 返回的是所有车辆的信息
    def calculate_queue_size(self, tlID):
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
                if info_veh[x][81] in self._edgesNS[int(tlID)]:
                    qNS.append(x)
                if info_veh[x][81] in self._edgesEW[int(tlID)]:
                    qEW.append(x)
                    # print('qew',qEW)

        return [qNS, qEW]

    # for the reward
    def calculate_stopped_queue_length(self, tlID):
        minSpeed = 0.1
        cycle_duration = 5
        allVehicles = traci.vehicle.getIDList()

        for vehID in allVehicles:
            traci.vehicle.subscribe(vehID, [traci.constants.VAR_LANE_ID, traci.constants.VAR_SPEED])

        info_veh = traci.vehicle.getSubscriptionResults(None)

        current_queue_NS = 0
        current_queue_EW = 0

        if info_veh is not None and len(info_veh) > 0:
            qNS = []
            qEW = []

            for x in info_veh.keys():
                if info_veh[x][64] <= minSpeed:
                    if info_veh[x][81] in self._edgesNS[int(tlID)]:
                        qNS.append(x)
                    if info_veh[x][81] in self._edgesEW[int(tlID)]:
                        qEW.append(x)

            current_queue_NS = len(qNS)
            current_queue_EW = len(qEW)

            # # 累计当前周期的值
            # self.total_queue_NS += current_queue_NS
            # self.total_queue_EW += current_queue_EW
            #
            # # 计算步数
            # if step % cycle_duration == 0:
            #     avg_queue_NS = self.total_queue_NS / cycle_duration
            #     avg_queue_EW = self.total_queue_EW / cycle_duration
            #
            #     # 重置累计的值
            #     self.total_queue_NS = 0
            #     self.total_queue_EW = 0
            #
            #     return avg_queue_NS, avg_queue_EW

        return current_queue_NS, current_queue_EW

    def get_duration(self, tlID):
        duration = self.__trafficlights[tlID]["greenTime"]
        return duration

    # 获取状态中的相位
    def get_idPhase(self, tlID):
        idPhase = traci.trafficlight.getPhase(tlID)
        return idPhase

    def compute_state(self, tlID):
        idPhase = traci.trafficlight.getPhase(tlID)
        duration = self.__trafficlights[tlID]["greenTime"]
        NSque, EWque = self.control_traffic(tlID)
        # state = np.array([idPhase,duration,NSque,EWque],dtype=np.float32)
        state = [idPhase, duration, NSque, EWque]
        return state

    def compute_next_state(self, tlID):
        # # 获取当前时间并加上时间步长
        # current_time = traci.simulation.getCurrentTime() / 1000
        # time_step = traci.simulation.getDeltaT()  # 获取 SUMO 的时间步长
        # next_time = current_time + time_step

        # 获取下一时刻的状态
        traci.simulationStep()
        idPhase = traci.trafficlight.getPhase(tlID)
        duration = self.__trafficlights[tlID]["greenTime"]
        NSque, EWque = self.control_traffic(tlID)
        new_state = [idPhase, duration, NSque, EWque]
        return new_state

    # 获取当前时间间隔内通过检测器的车辆
    # 定义交通信号灯与检测器的对应关系
    def control_traffic(self, tlID):
        # 封装每个信号灯对应的检测器ID
        signal_to_detector_map = {
            '0': {
                'N': ['e1_000', 'e1_001'],
                'S': ['e1_004', 'e1_005'],
                'W': ['e1_002', 'e1_003'],
                'E': ['e1_006', 'e1_007']
            },
            '1': {
                'N': ['e1_008', 'e1_009'],
                'S': ['e1_012', 'e1_013'],
                'W': ['e1_010', 'e1_011'],
                'E': ['e1_014', 'e1_015']
            },
            '2': {
                'N': ['e1_024', 'e1_025'],
                'S': ['e1_028', 'e1_029'],
                'W': ['e1_026', 'e1_027'],
                'E': ['e1_030', 'e1_031']
            },
            '3': {
                'N': ['e1_032', 'e1_033'],
                'S': ['e1_036', 'e1_037'],
                'W': ['e1_034', 'e1_035'],
                'E': ['e1_038', 'e1_039']
            }
        }

        detectors = signal_to_detector_map[tlID]

        # 用于存储当前仿真步骤中去重后的车辆ID
        unique_vehicle_ids = {
            'N': set(),
            'S': set(),
            'W': set(),
            'E': set()
        }
        # 获取对应检测器的车辆数据
        for direction, detector_ids in detectors.items():
            for detector_id in detector_ids:
                vehicle_ids = traci.inductionloop.getLastStepVehicleIDs(detector_id)
                for vehicle_id in vehicle_ids:
                    if vehicle_id not in self.__trafficlights[tlID]['already_counted_ids']:  # 每个信号灯维护自己的ID集合
                        unique_vehicle_ids[direction].add(vehicle_id)
                        self.__trafficlights[tlID]['already_counted_ids'].add(vehicle_id)  # 将新检测到的车辆ID加入已统计集合

        # 返回去重后的车辆ID个数
        n = len(unique_vehicle_ids['N'])
        s = len(unique_vehicle_ids['S'])
        NS = n + s
        w = len(unique_vehicle_ids['W'])
        e = len(unique_vehicle_ids['E'])
        WE = w + e

        return NS, WE

    # def control_traffic(self, tlID):
    #     # 封装每个信号灯对应的检测器ID，按车道统计
    #     signal_to_detector_map = {
    #         '0': {
    #             'N': ['e1_000', 'e1_001'],  # 北向两条车道
    #             'S': ['e1_004', 'e1_005'],  # 南向两条车道
    #             'W': ['e1_002', 'e1_003'],  # 西向两条车道
    #             'E': ['e1_006', 'e1_007']  # 东向两条车道
    #         },
    #         '1': {
    #             'N': ['e1_008', 'e1_009'],
    #             'S': ['e1_012', 'e1_013'],
    #             'W': ['e1_010', 'e1_011'],
    #             'E': ['e1_014', 'e1_015']
    #         },
    #         '2': {
    #             'N': ['e1_024', 'e1_025'],
    #             'S': ['e1_028', 'e1_029'],
    #             'W': ['e1_026', 'e1_027'],
    #             'E': ['e1_030', 'e1_031']
    #         },
    #         '3': {
    #             'N': ['e1_032', 'e1_033'],
    #             'S': ['e1_036', 'e1_037'],
    #             'W': ['e1_034', 'e1_035'],
    #             'E': ['e1_038', 'e1_039']
    #         }
    #     }
    #
    #     detectors = signal_to_detector_map[tlID]
    #
    #     # 初始化每条车道的车辆ID集合，按检测器ID区分
    #     lane_vehicle_ids = {
    #         'N': {detector: set() for detector in detectors['N']},
    #         'S': {detector: set() for detector in detectors['S']},
    #         'W': {detector: set() for detector in detectors['W']},
    #         'E': {detector: set() for detector in detectors['E']}
    #     }
    #
    #     # 获取每个车道的车辆数据
    #     for direction, detector_ids in detectors.items():
    #         for detector_id in detector_ids:
    #             vehicle_ids = traci.inductionloop.getLastStepVehicleIDs(detector_id)
    #             for vehicle_id in vehicle_ids:
    #                 if vehicle_id not in self.__trafficlights[tlID]['already_counted_ids']:
    #                     lane_vehicle_ids[direction][detector_id].add(vehicle_id)
    #                     self.__trafficlights[tlID]['already_counted_ids'].add(vehicle_id)
    #
    #     # 统计每条车道的车辆数量
    #     lane_counts = {direction: {detector_id: len(vehicle_ids) for detector_id, vehicle_ids in lanes.items()} for
    #                    direction, lanes in lane_vehicle_ids.items()}
    #
    #     # 计算每个方向的车辆总数
    #     NS = sum(lane_counts['N'].values()) + sum(lane_counts['S'].values())
    #     WE = sum(lane_counts['W'].values()) + sum(lane_counts['E'].values())
    #
    #     # 返回每条车道的车辆数量和南北、东西总和
    #     return lane_counts, NS, WE

    def get_traffic_state(self, tlID, step):
        cycle_duration = 5
        # 每秒执行control_traffic并累加结果
        ns, we = self.control_traffic(tlID)
        self.__trafficlights[tlID]['total_NS'] += ns
        self.__trafficlights[tlID]['total_EW'] += we
        duration = self.get_duration(tlID)
        idPhase = self.get_idPhase(tlID)

        # 每5秒输出累加结果并重置计数器
        if step % cycle_duration == 0:
            # 在重置前保存当前累加值
            total_NS = self.__trafficlights[tlID]['total_NS']
            total_EW = self.__trafficlights[tlID]['total_EW']

            # 重置累计计数器
            self.__trafficlights[tlID]['total_NS'] = 0
            self.__trafficlights[tlID]['total_EW'] = 0

            # 返回当前周期的累计值
            return idPhase, duration, total_NS, total_EW
        else:
            # 如果不是周期结束，返回当前累加值（不重置）
            return self.get_idPhase(tlID), self.get_duration(tlID), self.__trafficlights[tlID]['total_NS'], \
                self.__trafficlights[tlID]['total_EW']

    def __init_replay_buffers(self):
        self.replay_buffers = {}
        for tlID in self.get_trafficlights_ID_list():
            self.replay_buffers[int(tlID)] = ReplayBuffer()

    def run_episode(self, max_steps=-1,  exp=None):

        self.__check_env()

        max_steps *= 1000  # traci returns steps in ms, not s
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
        currentqueNS = 0
        currentqueEW = 0
        currentqueNSlength = 0
        currentqueEWlength = 0
        new_state = [0] * len(self.get_trafficlights_ID_list())
        state = [0] * len(self.get_trafficlights_ID_list())
        choose = [0] * len(self.get_trafficlights_ID_list())  # flag: if choose an action
        maxGreenTime = 90  # maximum green time, to prevent starvation
        minGreenTime = 10
        interv_action_selection = 5  # interval for action selection

        reward_data = []

        update_epsilon = maxGreenTime * 2  # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase
        step = 0
        cycle_duration = 5
        data = []
        # # 创建socket对象
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # # 服务器端的IP地址和端口号
        # server_ip = '222.18.156.234'  # 请替换为B电脑的IP地址
        # port = 8000
        # # 连接服务，指定主机和端口
        # s.connect((server_ip, port))

        # main loop
        while ((max_steps > -1 and traci.simulation.getCurrentTime() < max_steps) or max_steps <= -1) and (
                traci.simulation.getMinExpectedNumber() > 0 or traci.simulation.getArrivedNumber() > 0):
            actlist = [0] * len(self.get_trafficlights_ID_list())

            learner_state_action = {}
            traci.simulationStep()
            for tlID in self.get_trafficlights_ID_list():
                # 进行仿真步骤
                current_phase = traci.trafficlight.getPhase(tlID)
                if current_phase == 0 or current_phase == 3:
                    current_time = traci.simulation.getTime()
                    self.__trafficlights[tlID]['step'] += 1
                    idPhase, duration, total_NS, total_EW = self.get_traffic_state(tlID,
                                                                                   self.__trafficlights[tlID]['step'])
                    queNS, queEW = self.calculate_stopped_queue_length(tlID)
                    data.append([current_time, queNS + queEW])
                    # print(f"tlid:{tlID},curentime:{current_time}, step:{self.__trafficlights[tlID]['step']},idPhase:{idPhase},NS:{total_NS}, WE:{total_EW},queNS:{queNS},queEW:{queEW},greentime:{duration}")
                    new_state[int(tlID)] = [idPhase, total_NS, total_EW, queNS, queEW]

                    # print('new_state[int(tlID)]',new_state[int(tlID)])
                if self.__trafficlights[tlID]["greenTime"] > 9 and \
                        (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 \
                        and (traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 3):

                    state[int(tlID)], action = self._learners[tlID].act_last(new_state[int(tlID)], tlID)
                    # 第一次像边缘盒子发送数据，用于返回动作
                    # if int(tlID) == 0:
                    #     # self.send_and_receive_action(s,int(tlID),new_state[int(tlID)])
                    #     message = str(new_state[int(tlID)]).encode('utf-8')
                    #     s.send(message)
                    #     # 从盒子接收动作
                    #     response = s.recv(1024)
                    #     if response:
                    #         print("动作: ", response.decode())
                    #     action = int(response.decode())

                    learner_state_action[tlID] = [state[int(tlID)], action]
                    # if green time is equal or more than maxGreenTime, change phase
                    if self.__trafficlights[tlID]["greenTime"] >= maxGreenTime:
                        learner_state_action[tlID] = [state[int(tlID)], 1]

                    choose[int(tlID)] = True  # flag: if choose an action

                    # new_state[int(tlID)] = [0, 0, 0, 0]
                else:
                    choose[int(tlID)] = False

            # before start needs 'change' or 'keep' the phase according to the selected action
            for tlID in self.get_trafficlights_ID_list():

                # 对应交通信号灯绿灯时间+1
                if traci.trafficlight.getPhase(tlID) == 0 or traci.trafficlight.getPhase(tlID) == 3:
                    self.update_phaseTime('greenTime', tlID)
                    # if choose == True: run the action (change, keep)
                    # else: just calculate the queue length (reward will be the average queue length)
                    if choose[int(tlID)]:
                        # B) RUN ACTION
                        if learner_state_action[tlID][1] == 1:  # TODO: more phases
                            self.__trafficlights[tlID]['greenTime'] = 0
                            self.__trafficlights[tlID]['step'] = 0
                            # this method must set yellow phase and save the next green phase
                            self.change_trafficlight(tlID)
                            actlist[int(tlID)] = 1
                        else:  # if action = 'keep' just calculte queue size

                            _, _, pNS, pEW = self.get_traffic_state(tlID, self.__trafficlights[tlID]['step'])
                            NSlength, EWlength = self.calculate_stopped_queue_length(tlID)

                            currentqueNS = pNS
                            currentqueEW = pEW
                            currentqueNSlength = NSlength
                            currentqueEWlength = EWlength
                    else:

                        # calculate queue size
                        # queueNS[int(tlID)], queueEW[int(tlID)] = self.calculate_stopped_queue_length(tlID)
                        # currentNSqueue[int(tlID)] += queueNS[int(tlID)]
                        # currentEWqueue[int(tlID)] += queueEW[int(tlID)]
                        # 自加
                        _, _, pNS, pEW = self.get_traffic_state(tlID, self.__trafficlights[tlID]['step'])
                        NSlength, EWlength = self.calculate_stopped_queue_length(tlID)

                        currentqueNS = pNS
                        currentqueEW = pEW
                        currentqueNSlength = NSlength
                        currentqueEWlength = EWlength

                    # if it will select action in the next step,
                    # in the previous you need to calculate the feedback and update Q-table
                    if self.__trafficlights[tlID]["greenTime"] > (minGreenTime - 1) and \
                            (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 and \
                            current_time > 10:

                        # C) CALCULATE REWARD
                        trafficlight_to_proces_feedback = {}


                        # # D) PROCESS FEEDBACK
                        # if traci.trafficlight.getPhase(tlID) == 0:
                        #     reward = (new_state[int(tlID)][1] - new_state[int(tlID)][4])
                        #     # reward = (currentqueNS - currentqueEWlength)
                        # else:
                        #     # reward = (currentqueEW - currentqueNSlength)
                        #     reward = (new_state[int(tlID)][2] - new_state[int(tlID)][3])

                        # 获取当前状态
                        state_tensor = torch.tensor(new_state[int(tlID)], dtype=torch.float32).unsqueeze(0).to(
                            self.device)
                        # 对一个batchsize样本生成随机的时刻t

                        t_tensor = torch.tensor([self.__trafficlights[tlID]['step']], dtype=torch.long).to(self.device)  # 将时间步转换为张量

                        # 使用模型预测奖励
                        with torch.no_grad():
                            reward = self.model(state_tensor,t_tensor).item()
                            print('reward',reward)

                        # 将状态和奖励存储
                        reward_data.append((tlID,new_state[int(tlID)], reward))
                        # 将奖励和相关状态存储到反馈字典中
                        trafficlight_to_proces_feedback[tlID] = [
                            reward,
                            new_state[int(tlID)],
                            state[int(tlID)]
                        ]
                        # 第二次向边缘盒子发送数据，上传经验到边缘盒子用于让智能体学习
                        # feedback_data = trafficlight_to_proces_feedback[tlID]
                        # s.send(str(feedback_data).encode())
                        # response = s.recv(1024)

                        # print('trafficlight_to_proces_feedback[tlID]', trafficlight_to_proces_feedback[tlID])

                        self.__process_trafficlights_feedback(trafficlight_to_proces_feedback)

                        # # update previous queue
                        # previousNSqueue[int(tlID)] = aver_currentNSqueue
                        # previousEWqueue[int(tlID)] = aver_currentEWqueue

                        previousNSqueue[int(tlID)] = currentqueNS
                        previousEWqueue[int(tlID)] = currentqueEW
                        # # clean current queue
                        # currentNSqueue[int(tlID)] = 0
                        # currentEWqueue[int(tlID)] = 0
                        currentqueNS = 0
                        currentqueEW = 0

            # self.metrics(arq_tl, current_time)
            queue_list = ""
            for tlID in self.get_trafficlights_ID_list():
                queue_len = round((previousNSqueue[int(tlID)] + previousEWqueue[int(tlID)]), 1)
                queue_list = queue_list + str(queue_len) + ","
                # queue_list = queue_list + "|" +str(queue_len) + "," + str(traci.trafficlight.getPhase(tlID)) + "| "
                # queue_list = queue_list + str(traci.trafficlight.getPhase(tlID)) + "," +str(traci.trafficlight.getPhaseDuration(tlID))

        # 使用 pandas 将数据转换为 DataFrame 并保存为 Excel
        # df = pd.DataFrame(data, columns=['Time', 'Total_Queue_Length'])
        # df.to_excel('data/queue_data.xlsx', index=False)

        df = pd.DataFrame(reward_data, columns=['Tlid', 'nex_state','reward'])
        df.to_excel('data/diffusion_data.xlsx', index=False)
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

    def has_episode_ended(self):
        return self._has_episode_ended

    def __calc_reward(self, state, action, new_state):
        raise Exception('__calc_reward is not available in %s class' % self)
