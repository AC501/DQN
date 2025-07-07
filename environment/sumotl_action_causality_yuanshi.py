'''
Created on 12/12/2017

@author: Liza L. Lemos <lllemos@inf.ufrgs.br>
'''
import socket

import pandas as pd
from matplotlib import pyplot as plt

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

# from DDPM.DiffusionModel import RewardDiffusionModel
from DDPM.DDPM_model_gpt import RewardDiffusionModel


from RewardFunction.FunctionFitting_model import RewardPredictionModel



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
        self.model = RewardDiffusionModel(input_dim=5, hidden_dim=128)  # 根据实际情况调整
        self.model.load_state_dict(
            torch.load('D:\\tsc_ddqn_prb_1Con_new\\weights\\best_model_ddpm_GPT.pth'))
        self.model.eval()  # 切换到评估模式
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        #加载回报函数拟合模型
        self.rewardmodel = RewardPredictionModel(input_dim = 120,output_dim = 4)
        self.rewardmodel.load_state_dict(torch.load('RewardFunction/best_reward_prediction_model.pth'))
        self.rewardmodel.eval()
        self.rewardmodel.to(self.device)

        # 新增：2023-11-01 - 添加当前时间周期属性
        self.current_period = 0  # 默认使用第0个时间周期

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
        traci.start([self._sumo_binary, "-c", self.__cfg_file,"--scale", "5","--no-warnings"])
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
        minSpeed = 1
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

        return current_queue_NS, current_queue_EW

    def get_duration(self, tlID):
        duration = self.__trafficlights[tlID]["greenTime"]
        return duration

    # 获取状态中的相位
    def get_idPhase(self, tlID):
        idPhase = traci.trafficlight.getPhase(tlID)
        return idPhase

    # def compute_state(self, tlID):
    #     idPhase = traci.trafficlight.getPhase(tlID)
    #     duration = self.__trafficlights[tlID]["greenTime"]
    #     NSque, EWque = self.control_traffic(tlID)
    #     # state = np.array([idPhase,duration,NSque,EWque],dtype=np.float32)
    #     state = [idPhase, duration, NSque, EWque]
    #     return state

    # def compute_next_state(self, tlID):
    #     # # 获取当前时间并加上时间步长
    #     # current_time = traci.simulation.getCurrentTime() / 1000
    #     # time_step = traci.simulation.getDeltaT()  # 获取 SUMO 的时间步长
    #     # next_time = current_time + time_step
    #
    #     # 获取下一时刻的状态
    #     traci.simulationStep()
    #     idPhase = traci.trafficlight.getPhase(tlID)
    #     duration = self.__trafficlights[tlID]["greenTime"]
    #     NSque, EWque = self.control_traffic(tlID)
    #     new_state = [idPhase, duration, NSque, EWque]
    #     return new_state

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

    def get_traffic_leave(self, tlID):
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
                    # 检查车辆ID是否已经被记录（在全局集合中）
                    if vehicle_id not in self.already_counted_ids:
                        unique_vehicle_ids[direction].add(vehicle_id)
                        self.already_counted_ids.add(vehicle_id)  # 将新检测到的车辆ID加入全局已统计集合

        # 返回去重后的车辆ID个数
        n = len(unique_vehicle_ids['N'])
        s = len(unique_vehicle_ids['S'])
        NS = n + s
        w = len(unique_vehicle_ids['W'])
        e = len(unique_vehicle_ids['E'])
        WE = w + e

        return NS, WE

    def control_traffic_leave(self, tlID):
        # 封装每个信号灯对应的检测器ID
        signal_to_detector_map = {
            '0': {
                'N': ['e1_0', 'e1_1', 'e1_2', 'e1_3'],

                'W': ['e1_4', 'e1_5', 'e1_6', 'e1_7']
            },
            '1': {
                'N': ['e1_28', 'e1_29', 'e1_31', 'e1_30'],
                'E': ['e1_24', 'e1_25', 'e1_26', 'e1_27']
            },
            '2': {
                'S': ['e1_12', 'e1_13', 'e1_14', 'e1_15'],
                'W': ['e1_8', 'e1_9', 'e1_10', 'e1_11']
            },
            '3': {
                'S': ['e1_16', 'e1_17', 'e1_18', 'e1_19'],
                'E': ['e1_20', 'e1_21', 'e1_22', 'e1_23']
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
        total_leave = NS + WE

        return total_leave

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

    # def save_vehicle_data(self, produced_data, departed_data, slow_vehicles_data):
    #     produced_df = pd.DataFrame(produced_data, columns=['Time', 'Produced Vehicles'])
    #     departed_df = pd.DataFrame(departed_data, columns=['Time', 'Departed Vehicles'])
    #     df_slow_vehicles = pd.DataFrame(slow_vehicles_data, columns=["Time", "SlowVehiclesCount"])
    #
    #     produced_df.to_excel('data/produced_vehicles.xlsx', index=False)
    #     departed_df.to_excel('data/departed_vehicles.xlsx', index=False)
    #     df_slow_vehicles.to_excel("data/slow_vehicles_data.xlsx", index=False)

    def save_vehicle_data(self, produced_data, departed_data):
        # 创建 DataFrame
        produced_df = pd.DataFrame(produced_data, columns=['Time', 'Produced Vehicles'])
        departed_df = pd.DataFrame(departed_data, columns=['Time', 'Departed Vehicles'])
        # df_slow_vehicles = pd.DataFrame(slow_vehicles_data, columns=["Time", "SlowVehiclesCount"])

        # 定义文件路径
        produced_file = 'data/produced_vehicles.xlsx'
        departed_file = 'data/departed_vehicles.xlsx'
        slow_vehicles_file = 'data/slow_vehicles_data.xlsx'

        # 处理 produced_vehicles 数据
        if os.path.exists(produced_file):
            produced_df_existing = pd.read_excel(produced_file)
            combined_produced_df = pd.concat([produced_df_existing, produced_df], ignore_index=True)
        else:
            combined_produced_df = produced_df

        # 限制行数为最多 12,500 行
        combined_produced_df = combined_produced_df.tail(25000000)
        combined_produced_df.to_excel(produced_file, index=False)

        # 处理 departed_vehicles 数据
        if os.path.exists(departed_file):
            departed_df_existing = pd.read_excel(departed_file)
            combined_departed_df = pd.concat([departed_df_existing, departed_df], ignore_index=True)
        else:
            combined_departed_df = departed_df

        # 限制行数为最多 12,500 行
        combined_departed_df = combined_departed_df.tail(25000000)
        combined_departed_df.to_excel(departed_file, index=False)

        # # 处理 slow_vehicles_data 数据
        # if os.path.exists(slow_vehicles_file):
        #     df_slow_vehicles_existing = pd.read_excel(slow_vehicles_file)
        #     combined_slow_vehicles_df = pd.concat([df_slow_vehicles_existing, df_slow_vehicles], ignore_index=True)
        # else:
        #     combined_slow_vehicles_df = df_slow_vehicles
        #
        # # 限制行数为最多 12,500 行
        # combined_slow_vehicles_df = combined_slow_vehicles_df.tail(25000000)
        # combined_slow_vehicles_df.to_excel(slow_vehicles_file, index=False)

    def run_episode(self, max_steps=-1, exp=None):

        global current_time_all
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
        CPMData = []

        produced_vehicles_data = []  # 用于记录每秒产生的车辆
        departed_vehicles_data = []  # 用于记录每秒离开的车辆

        # 用于记录低于 5 km/h 的车辆数量
        slow_vehicles_data = []

        tracked_vehicles = set()  # 记录已统计的车辆

        update_epsilon = maxGreenTime * 2  # maxGreenTime *2: to assure that the traffic ligth pass at least one time in each phase
        step = 0
        cycle_duration = 5
        data = []
        slow_data = []
        fit_r = 0

        # 产生的车辆
        pro_data = []
        # 离开的车辆
        leave_data = []
        # 记录每个时间步的离开车辆数据
        leave_data_summary = []
        ns = []
        we = []
        total_leave_data = []
        # 统计四个交通灯的数据
        total_leave_data = 0

        # # 创建socket对象
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # # 服务器端的IP地址和端口号
        # server_ip = '222.18.156.234'  # 请替换为B电脑的IP地址
        # port = 8000
        # # 连接服务，指定主机和端口
        # s.connect((server_ip, port))

        # main loop
        # 在循环外定义一个集合来跟踪在上一时间步骤存在的车辆
        previous_vehicles = set()

        previous_slow_vehicles = set()  # 用于记录上一步骤速度低于5km/h的车辆


        #保存拟合回报函数的数据
        fit_rewardfunction_df = []
        rewardpredictions_list = []

        # 创建一个空的 DataFrame 用于保存结果
        result = pd.DataFrame()
        # 初始化数据存储列表
        traffic_light_data = []
        num_traffic_lights = 4
        # 初始化 fit_reward 数据框，包含4个信号灯的数据，初始状态全为0
        initial_data = []
        for i in range(4):
            for tlID in range(num_traffic_lights):
                # 每个信号灯的数据： [0, 0, 0, 0, 0] 包含 [idPhase, total_NS, total_EW, queNS, queEW, fit_r]
                initial_data.append([tlID, [0, 0, 0, 0, 0, 0]])  # fit_state_reward 初始为零

            fit_rewardfunction_df = initial_data


        while ((max_steps > -1 and traci.simulation.getCurrentTime() < max_steps) or max_steps <= -1) and (
                traci.simulation.getMinExpectedNumber() > 0 or traci.simulation.getArrivedNumber() > 0):
            actlist = [0] * len(self.get_trafficlights_ID_list())

            learner_state_action = {}
            traci.simulationStep()

            # 获取当前所有车辆的ID
            all_vehicles = traci.vehicle.getIDList()

            # 记录当前时间产生的车辆
            produced_vehicles = []
            for vid in all_vehicles:
                if vid not in tracked_vehicles and traci.vehicle.getLaneID(vid) != "":
                    produced_vehicles.append(vid)
                    tracked_vehicles.add(vid)  # 标记为已统计
            if current_time >= 10:
                # 记录当前时间的车辆数量
                produced_vehicles_count = len(produced_vehicles)
                # 统计离开的车辆
                departed_vehicles = previous_vehicles - set(all_vehicles)
                departed_vehicles_count = len(departed_vehicles)
                # 记录产生和离开的车辆数量
                produced_vehicles_data.append((current_time, produced_vehicles_count))
                departed_vehicles_data.append((current_time, departed_vehicles_count))
                # 统计速度低于5km/h的车辆
                slow_vehicles = [vid for vid in all_vehicles if traci.vehicle.getSpeed(vid) < (3 / 3.6)]  # 转换为 m/s
                # 当前时间的速度低于5km/h的车辆（去重处理）
                new_slow_vehicles = set(slow_vehicles) - previous_slow_vehicles  # 新进入低速状态的车辆
                departed_slow_vehicles = previous_slow_vehicles - set(slow_vehicles)  # 离开低速状态的车辆
                # 更新记录
                slow_vehicles_count = len(new_slow_vehicles)  # 统计当前进入低速状态的车辆数
                slow_vehicles_data.append((current_time, slow_vehicles_count))
                # 更新previous_slow_vehicles为当前时间的低速车辆
                previous_slow_vehicles = set(slow_vehicles)

                # 更新 previous_vehicles 为当前时间步骤的车辆
                previous_vehicles = set(all_vehicles)

            # total_leave_data = 0

            total_slow = 0

            for tlID in self.get_trafficlights_ID_list():
                # 进行仿真步骤
                current_phase = traci.trafficlight.getPhase(tlID)
                current_time_all = traci.simulation.getTime()

                # 保存信号灯数据
                traffic_light_data.append([tlID, current_time_all, current_phase])


                #统计速度低于3.6km/h的车辆
                queNS, queEW = self.calculate_stopped_queue_length(tlID)
                total_slow += queEW+queNS

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
                # 将每个时间步的总离开车辆数据保存

            # before start needs 'change' or 'keep' the phase according to the selected action

            # 设定最大保存的行数，例如最多保留1000行数据
            max_rows = 20

            for tlID in self.get_trafficlights_ID_list():
                ##这里只需要获取state
                # 便于拟合新的回报函数
                # 拟合reward所需的数据 state reward一起
                _, act = self._learners[tlID].act_last(new_state[int(tlID)], tlID)
                fit_state = new_state[int(tlID)] + [act]
                fit_rewardfunction_df.append((int(tlID), fit_state))


                # 如果fit_rewardfunction_df的长度超过最大行数，删除最前面几行
                if len(fit_rewardfunction_df) > max_rows:
                    fit_rewardfunction_df = fit_rewardfunction_df[4:]

            # 将数据转换为DataFrame并保存为Excel文件
            fit_reward = pd.DataFrame(fit_rewardfunction_df, columns=['tlid', 'state'])

            # 保存为Excel文件
            fit_reward.to_excel('RewardFunction/fit_reward.xlsx', index=False,
                                engine='openpyxl')
            # 读取 Excel 文件
            fit_df = pd.read_excel('RewardFunction/fit_reward.xlsx', engine='openpyxl')
            # 预处理 state_reward 列，去除方括号并保留逗号
            fit_df['state'] = fit_df['state'].apply(lambda x: str(x).strip('[]'))
            # 创建一个空的 DataFrame 用于保存结果
            result = pd.DataFrame()

            # 根据 Tlid 分组，并将对应的 state 和 reward 分开
            # Tlid 只有 0, 1, 2, 3
            for tlid in range(4):
                # 生成列名
                state_column = f'state_{tlid}'
                # 按照 Tlid 对数据进行筛选
                filtered_data = fit_df[fit_df['tlid'] == tlid]
                # 将 state 和 reward 列分别加入到结果 DataFrame 中
                result[state_column] = filtered_data['state'].reset_index(drop=True)
            # 重新排列列的顺序，确保 state 列在前，reward 列在后，且顺序是按 Tlid=0, 1, 2, 3
            state_columns = [f'state_{tlid}' for tlid in range(4)]
            # 将所有 state 列排在前面，reward 列排在后面
            result = result[state_columns]

            #转换成一维数组
            # 提取状态数据（5行4列）
            state_data = result[['state_0', 'state_1', 'state_2', 'state_3']].values
            # 将每个状态的数据展开为一维数组
            flattened_states = state_data.flatten()  # 将 5行4列展平成一维数组
            # 处理字符串，将它们拆分并转化为浮点型数组
            flattened_state = np.array([list(map(np.float32, item.split(','))) for item in flattened_states])
            # 展平为一维数组
            flattened_state = flattened_state.flatten()
            flattened_state_tensor = torch.tensor(flattened_state, dtype=torch.float32).to(self.device)

            # 将转换后的 Tensor 传递给模型

            rewardpredictions = self.rewardmodel(flattened_state_tensor)
            # 将 rewardpredictions 从 CUDA 移动到 CPU
            rewardpredictions_cpu = rewardpredictions.cpu()
            # 将 tensor 转换为列表
            rewardpredictions_list = rewardpredictions_cpu.tolist()



            file_path = "data/generated_reward.xlsx"

            for tlID in self.get_trafficlights_ID_list():
                # 修改：2023-11-01 - 使用基于时间周期的奖励文件
                # 构建对应周期的奖励文件路径
                reward_file_path = f"data/generated_reward_period_{self.current_period}.xlsx"
                
                if os.path.exists(reward_file_path):
                    # 读取对应周期的奖励文件
                    GenReward = pd.read_excel(reward_file_path)
                    
                    # 计算在当前周期内的相对时间(0-99)
                    relative_time = int(current_time_all % 100)
                    if relative_time == 0:
                        relative_time = 100  # 处理第100秒的情况
                        
                    # 从文件读取奖励
                    row_index = relative_time - 1
                    row_index = min(row_index, GenReward.shape[0] - 1)
                    col_index = int(tlID)
                    r = int(GenReward.iloc[row_index, col_index])
                elif os.path.exists("data/generated_reward.xlsx"):
                    # 向后兼容：使用旧的通用奖励文件
                    GenReward = pd.read_excel("data/generated_reward.xlsx")
                    row_index = int(current_time_all) - 1
                    row_index = min(row_index, GenReward.shape[0] - 1)
                    col_index = int(tlID)
                    r = int(GenReward.iloc[row_index, col_index])
                else:
                    # 实时计算奖励
                    if traci.trafficlight.getPhase(tlID) == 0:
                        r = (new_state[int(tlID)][1] - new_state[int(tlID)][4])
                    elif traci.trafficlight.getPhase(tlID) == 3:
                        r = (new_state[int(tlID)][2] - new_state[int(tlID)][3])



                # state rward action一起
                if current_time_all > 10:
                    state_reward = new_state[int(tlID)]+[action]+[r]
                    CPMData.append((tlID, state_reward))

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

                    # if it will select action in the next step,
                    # in the previous you need to calculate the feedback and update Q-table
                    if self.__trafficlights[tlID]["greenTime"] > (minGreenTime - 1) and \
                            (self.__trafficlights[tlID]["greenTime"] % interv_action_selection) == 0 and \
                            current_time > 10:
                        # C) CALCULATE REWARD
                        trafficlight_to_proces_feedback = {}

                        # D) PROCESS FEEDBACK
                        # if traci.trafficlight.getPhase(tlID) == 0:
                        #     reward = (new_state[int(tlID)][1] - new_state[int(tlID)][4])
                        # else:
                        #     reward = (new_state[int(tlID)][2] - new_state[int(tlID)][3])

                        reward = rewardpredictions_list[int(tlID)]



                        # 将状态和奖励存储
                        reward_data.append((tlID, new_state[int(tlID)], reward))
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

                        previousNSqueue[int(tlID)] = currentqueNS
                        previousEWqueue[int(tlID)] = currentqueEW
                        currentqueNS = 0
                        currentqueEW = 0

            # self.metrics(arq_tl, current_time)
            queue_list = ""
            for tlID in self.get_trafficlights_ID_list():
                queue_len = round((previousNSqueue[int(tlID)] + previousEWqueue[int(tlID)]), 1)
                queue_list = queue_list + str(queue_len) + ","
                # queue_list = queue_list + "|" +str(queue_len) + "," + str(traci.trafficlight.getPhase(tlID)) + "| "
                # queue_list = queue_list + str(traci.trafficlight.getPhase(tlID)) + "," +str(traci.trafficlight.getPhaseDuration(tlID))


            # 保存车速小于3.6km/h的车辆
            slow_data.append((current_time_all, total_slow))
            # 使用 pandas 将数据转换为 DataFrame
            df = pd.DataFrame(slow_data, columns=['Time', 'slow'])

            # 保存 DataFrame 到 Excel 文件
            df.to_excel('slow_data.xlsx', index=False, engine='openpyxl')
        # 使用 pandas 将数据转换为 DataFrame 并保存为 Excel
        # df = pd.DataFrame(data, columns=['Time', 'Total_Queue_Length'])
        # df.to_excel('data/queue_data.xlsx', index=False)

        # 将相序数据保存为 DataFrame
        df_traffic_light = pd.DataFrame(traffic_light_data, columns=["tlID", "current_time_all", "current_phase"])
        # 保存为 Excel 文件
        df_traffic_light.to_excel("data/traffic_light_phases.xlsx", index=False)


        # 将数据保存到文件，每S在地图产生的车辆、离开地图的车辆、缓行的车辆

        self.save_vehicle_data(produced_vehicles_data, departed_vehicles_data)

        # # 条件扩散模型所需数据，覆盖的形式保存
        # df = pd.DataFrame(CPMData, columns=['Tlid', 'state_reward'])
        # # df = pd.DataFrame(CPMData, columns=['Tlid', 'state','reward'])
        # df.to_excel('data/CPMData.xlsx', index=False)

        # 追加的形式保存CPMData数据
        df = pd.DataFrame(CPMData, columns=['Tlid', 'state_reward'])
        # 定义文件路径
        file_path = 'data/CPMData.xlsx'
        # 检查文件是否存在，如果存在则读取现有数据
        if os.path.exists(file_path):
            # 读取现有的 Excel 文件
            existing_df = pd.read_excel(file_path)
            # 合并现有数据和新的数据
            combined_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            # 如果文件不存在，直接使用新的数据
            combined_df = df
        # 限制行数为最多 12000 行
        combined_df = combined_df.head(100000000)
        # 保存数据到 Excel 文件，使用追加模式
        combined_df.to_excel(file_path, index=False)


        # df = pd.DataFrame(reward_data, columns=['Tlid', 'nex_state','reward'])
        # df.to_excel('data/diffusion_data.xlsx', index=False)
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
