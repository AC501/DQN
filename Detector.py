class DetectorETwo:
    def __init__(self):
        self.detector_data = {}

    def updateDetectorData(self, detector_id, data):
        self.detector_data[detector_id] = data

    def getIntervalVehicleNumber(self, detector_id):
        if detector_id in self.detector_data:
            return self.detector_data[detector_id]
        else:
            return 0  # 如果检测器数据不存在，默认返回0