import pandas as pd
import traci
import os
import time
import sys
from sumolib import checkBinary

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# 相关的配置的声明,比如运行的是哪一个sumocfg，
# 是否要有GUI（Graphical User Interface 图形化用户接口）
show_gui = True
sumoconfig_path = r'D:\tsc_ddqn_prb\nets\22grid_fuben_e1_wbst.sumocfg'

if not show_gui:
    sumoBinary = checkBinary('sumo')
else:
    sumoBinary = checkBinary('sumo-gui')

traci.start([sumoBinary, '-c', sumoconfig_path])


def create_edges():
    edgesNS = {}
    edgesEW = {}

    edgesNS[0] = ['0Ni_0', '0Ni_1', '0Si_0', '0Si_1']
    edgesEW[0] = ['0Wi_0', '0Wi_1', '0Ei_0', '0Ei_1']
    edgesNS[1] = ['1Ni_0', '1Ni_1', '1Si_0', '1Si_1']
    edgesEW[1] = ['1Wi_0', '1Wi_1', '1Ei_0', '1Ei_1']
    edgesNS[2] = ['2Ni_0', '2Ni_1', '2Si_0', '2Si_1']
    edgesEW[2] = ['2Wi_0', '2Wi_1', '2Ei_0', '2Ei_1']
    edgesNS[3] = ['3Ni_0', '3Ni_1', '3Si_0', '3Si_1']
    edgesEW[3] = ['3Wi_0', '3Wi_1', '3Ei_0', '3Ei_1']

    return edgesNS, edgesEW



already_counted_ids = set()
counted_vehicles = set()

def calculate_stopped_queue_length(tlID, edgesNS, edgesEW):
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
                if info_veh[x][81] in edgesNS[int(tlID)]:
                    qNS.append(x)
                if info_veh[x][81] in edgesEW[int(tlID)]:
                    qEW.append(x)

        current_queue_NS = len(qNS)
        current_queue_EW = len(qEW)

    return current_queue_NS, current_queue_EW
def save_vehicle_data(produced_data, departed_data,slow_vehicles_data):
    import pandas as pd

    produced_df = pd.DataFrame(produced_data, columns=['Time', 'Produced Vehicles'])
    departed_df = pd.DataFrame(departed_data, columns=['Time', 'Departed Vehicles'])
    df_slow_vehicles = pd.DataFrame(slow_vehicles_data, columns=["Time", "SlowVehiclesCount"])

    produced_df.to_excel('data/produced_vehicles_wbst.xlsx', index=False)
    departed_df.to_excel('data/departed_vehicles_wbst.xlsx', index=False)
    df_slow_vehicles.to_excel("data/slow_vehicles_data_wbst.xlsx", index=False)


def get_traffic_leave(tlID):
    # 封装每个信号灯对应的检测器ID
    signal_to_detector_map = {
        '0': {
            'N': ['e1_0', 'e1_1'],
            'S': ['e1_4', 'e1_5'],
            'W': ['e1_2', 'e1_3'],
            'E': ['e1_30', 'e1_31']
        },
        '1': {
            'N': ['e1_6', 'e1_7'],
            'S': ['e1_10', 'e1_11'],
            'W': ['e1_8', 'e1_9'],
            'E': ['e1_12', 'e1_13']
        },
        '2': {
            'N': ['e1_14', 'e1_15'],
            'S': ['e1_18', 'e1_19'],
            'W': ['e1_16', 'e1_17'],
            'E': ['e1_20', 'e1_21']
        },
        '3': {
            'N': ['e1_22', 'e1_23'],
            'S': ['e1_26', 'e1_27'],
            'W': ['e1_24', 'e1_25'],
            'E': ['e1_28', 'e1_29']
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
                if vehicle_id not in already_counted_ids:
                    unique_vehicle_ids[direction].add(vehicle_id)
                    already_counted_ids.add(vehicle_id)  # 将新检测到的车辆ID加入全局已统计集合

    # 返回去重后的车辆ID个数
    n = len(unique_vehicle_ids['N'])
    s = len(unique_vehicle_ids['S'])
    NS = n + s
    w = len(unique_vehicle_ids['W'])
    e = len(unique_vehicle_ids['E'])
    WE = w + e

    return NS, WE

# 获取edgesNS和edgesEW
edgesNS, edgesEW = create_edges()
data = []
# 仿真循环
# 初始化步数为 0
step = 0
max_steps = 4000  # 设定最大步数
produced_vehicles_data = []  # 用于记录每秒产生的车辆
departed_vehicles_data = []  # 用于记录每秒离开的车辆
tracked_vehicles = set()  # 记录已统计的车辆
# 用于记录低于 5 km/h 的车辆数量
slow_vehicles_data = []
# 用于记录低于 3.6 km/h 的车辆数量
slow_data = []
previous_slow_vehicles = set()  # 用于记录上一步骤速度低于5km/h的车辆


leave_data_summary = []
previous_vehicles = set()

traffic_light_data = []
# 仿真循环：当仿真中仍有车辆且步数小于 max_steps 时继续仿真
while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
    total_leave_data = 0
    total_slow = 0

    for tlid in traci.trafficlight.getIDList():
        traci.simulationStep()
        simulation_current_time = traci.simulation.getTime()
        current_phase = traci.trafficlight.getPhase(tlid)
        traffic_light_data.append([tlid,simulation_current_time,current_phase])

        # 获取当前所有车辆的ID
        all_vehicles = traci.vehicle.getIDList()

        # 记录当前时间产生的车辆
        produced_vehicles = []
        for vid in all_vehicles:
            if vid not in tracked_vehicles and traci.vehicle.getLaneID(vid) != "":
                produced_vehicles.append(vid)
                tracked_vehicles.add(vid)  # 标记为已统计

        # 记录当前时间的车辆数量
        produced_vehicles_count = len(produced_vehicles)

        # 统计离开的车辆
        departed_vehicles = previous_vehicles - set(all_vehicles)
        departed_vehicles_count = len(departed_vehicles)

        # 记录产生和离开的车辆数量
        produced_vehicles_data.append((simulation_current_time, produced_vehicles_count))
        departed_vehicles_data.append((simulation_current_time, departed_vehicles_count))

        # 统计速度低于5km/h的车辆
        slow_vehicles = [vid for vid in all_vehicles if traci.vehicle.getSpeed(vid) < (3 / 3.6)]  # 转换为 m/s
        # 当前时间的速度低于5km/h的车辆（去重处理）
        new_slow_vehicles = set(slow_vehicles) - previous_slow_vehicles  # 新进入低速状态的车辆
        departed_slow_vehicles = previous_slow_vehicles - set(slow_vehicles)  # 离开低速状态的车辆

        queNS, queEW = calculate_stopped_queue_length(tlid,edgesNS, edgesEW)
        total_slow += queEW + queNS

        # 保存车速小于3.6km/h的车辆
        slow_data.append((simulation_current_time, total_slow))

        # 更新记录
        slow_vehicles_count = len(new_slow_vehicles)  # 统计当前进入低速状态的车辆数
        slow_vehicles_data.append((simulation_current_time, slow_vehicles_count))

        # 更新previous_slow_vehicles为当前时间的低速车辆
        previous_slow_vehicles = set(slow_vehicles)

        # 更新 previous_vehicles 为当前时间步骤的车辆
        previous_vehicles = set(all_vehicles)

        # 仿真步进
        # 获取排队长度
        nslength, ewlength = calculate_stopped_queue_length(tlid,  edgesNS, edgesEW)
        # 将当前时间和排队总长度保存到 data 列表
        data.append([simulation_current_time, nslength + ewlength])
        ns, we = get_traffic_leave(tlid)
        # Sum the leave data for this traffic light
        total_leave_data += ns + we
        # 增加步数计数器
        step += 1



    leave_data_summary.append((step, total_leave_data))






 # 保存车速低于3.6km/h的
    # 使用 pandas 将数据转换为 DataFrame
    df = pd.DataFrame(slow_data, columns=['Time', 'slow'])
    # 保存 DataFrame 到 Excel 文件
    df.to_excel('slow_data_wbst.xlsx', index=False, engine='openpyxl')

#保存相序数据
phase_logical = pd.DataFrame(traffic_light_data,columns=["tlID", "current_time_all", "current_phase"])
phase_logical.to_excel("data/traffic_light_phases_wbst.xlsx",index=False)

# 将数据保存到文件
save_vehicle_data(produced_vehicles_data, departed_vehicles_data,slow_vehicles_data)
# 保存为 Excel 文件
df = pd.DataFrame(leave_data_summary, columns=["Time", "Total Departed Vehicles"])
df.to_excel("leave_data_summary_wbst.xlsx", index=False)

# 在仿真结束后，将数据保存到 Excel
df = pd.DataFrame(data, columns=['Time', 'Total_Queue_Length'])
df.to_excel('data/que_data_wbst.xlsx', index=False)

# 确保仿真终止
traci.close()


