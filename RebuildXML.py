import xml.etree.ElementTree as ET

# 解析 XML 文件
tree = ET.parse('./nets/22grid_e1.add.xml')
root = tree.getroot()

# # 遍历每个 laneAreaDetector 元素
# for detector in root.findall('laneAreaDetector'):
#     # 修改 length 属性为 35
#     detector.set('length', '0.01')
#     # 获取并修改 pos 属性
#     pos = float(detector.get('pos'))
#     detector.set('pos', str(pos - 0.01))
# 遍历所有的inductionLoop元素，修改period属性
for induction_loop in root.findall('inductionLoop'):
    if induction_loop.get('period') == '5.00':
        induction_loop.set('period', '10.00')

# 保存修改后的 XML 文件
tree.write('./nets/22grid_e1.add.xml')