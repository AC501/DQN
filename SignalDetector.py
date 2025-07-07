signal_to_detector_map = {
    '0':{'N':['e2_000','e2_001'],
         'S':['e2_004','e2_005'],
         'W':['e2_002','e2_003'],
         'E':['e2_006','e2_007']},
    '1': {'N': ['e2_008', 'e2_009'],
          'S': ['e2_012', 'e2_013'],
          'W': ['e2_010', 'e2_011'],
          'E': ['e2_014', 'e2_015']},
    '2': {'N': ['e2_016', 'e2_017'],
          'S': ['e2_020', 'e2_021'],
          'W': ['e2_018', 'e2_019'],
          'E': ['e2_022', 'e2_023']},
    '3': {'N': ['e2_024', 'e2_025'],
          'S': ['e2_028', 'e2_029'],
          'W': ['e2_026', 'e2_027'],
          'E': ['e2_030', 'e2_031']},
    '4': {'N': ['e2_032', 'e2_033'],
          'S': ['e2_036', 'e2_037'],
          'W': ['e2_034', 'e2_035'],
          'E': ['e2_038', 'e2_039']},
    '5': {'N': ['e2_040', 'e2_041'],
          'S': ['e2_044', 'e2_045'],
          'W': ['e2_042', 'e2_043'],
          'E': ['e2_046', 'e2_047']},
    '6': {'N': ['e2_048', 'e2_049'],
          'S': ['e2_052', 'e2_053'],
          'W': ['e2_050', 'e2_051'],
          'E': ['e2_054', 'e2_055']},
    '7': {'N': ['e2_056', 'e2_057'],
          'S': ['e2_060', 'e2_061'],
          'W': ['e2_058', 'e2_059'],
          'E': ['e2_062', 'e2_063']},
    '8': {'N': ['e2_064', 'e2_065'],
          'S': ['e2_068', 'e2_069'],
          'W': ['e2_066', 'e2_067'],
          'E': ['e2_070', 'e2_071']}

}
# 控制器逻辑
def control_traffic():
    # 遍历信号灯列表
    for signal_id, detectors in signal_to_detector_map.items():
        vehicle_data = {}
        # 获取对应检测器的车辆数据
        for direction, detector_ids in detectors.items():
            total_vehicles = sum(traci.lanearea.getIntervalVehicleNumber(det_id) for det_id in detector_ids)
            vehicle_data[direction] = total_vehicles

        print(f"Signal {signal_id} has {vehicle_data} vehicles on its detectors")