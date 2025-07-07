import xml.etree.ElementTree as ET

tree = ET.parse('22Grid2lanes_junction0.net.xml')
root = tree.getroot()

# 收集所有合法 edge ID
valid_edges = set(edge.attrib['id'] for edge in root.findall('edge'))

# 清理非法 connection
for conn in root.findall('connection'):
    from_id = conn.attrib.get('from')
    to_id = conn.attrib.get('to')
    if from_id not in valid_edges or to_id not in valid_edges:
        root.remove(conn)

# 保存回原文件或新文件
tree.write('22Grid2lanes_junction0.net.xml', encoding='utf-8', xml_declaration=True)
print("✅ 已生成清理后的 net.xml 文件")
