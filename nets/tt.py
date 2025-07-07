import xml.etree.ElementTree as ET

# 解析 XML 文件
tree = ET.parse('routes.rou.xml')
root = tree.getroot()

filtered_vehicles = []

for vehicle in root.findall('vehicle'):
    route = vehicle.find('route')
    edges = route.get('edges').split()

    # 条件1: 0号口北向出发 (起始边为 '0Ni')
    if edges[0] == '0Ni':
        filtered_vehicles.append(vehicle)

    # 条件2: 2号口南向出发 (起始边为 '2Si') 并直行通过0号口
    elif edges[0] == '2Si' and '0Si' in edges:
        # 判断是否直行，比如是否依次经过 '0Si' -> '0No' 或 '0Wo'
        idx = edges.index('0Si')
        if idx + 1 < len(edges) and edges[idx + 1] in ['0No', '0Wo']:
            filtered_vehicles.append(vehicle)

# 构建新的 XML 结构
new_root = ET.Element('routes')
for v in filtered_vehicles:
    new_root.append(v)

# 写入新文件
new_tree = ET.ElementTree(new_root)
new_tree.write('routes02NS.rou.xml', encoding='utf-8', xml_declaration=True)
