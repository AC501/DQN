import xml.etree.ElementTree as ET

# 示例数据
traffic_flows = [600, 800, 400, 500]  # 每个相位的流量
saturation_flows = [1800, 1800, 1800, 1800]  # 每个相位的饱和流量
L = 16  # 总损失时间（4个相位，每个4秒）

# 计算每个相位的饱和度
Y_i = [flow / saturation for flow, saturation in zip(traffic_flows, saturation_flows)]
Y = sum(Y_i)

# 计算总周期时间
C = (1.5 * L + 5) / (1 - Y)

# 计算每个相位的绿灯时间
green_times = [(yi / Y) * (C - L) for yi in Y_i]


# 生成 XML 配置文件
def generate_webster_tls_file(output_file, green_times):
    root = ET.Element('additional')

    for i in range(4):  # 对于4个交叉口
        tlLogic = ET.SubElement(root, 'tlLogic', id=f"{i}", programID="0", offset="0", type="static")

        # 添加相位配置
        ET.SubElement(tlLogic, 'phase', duration=str(int(green_times[i])), state="GGGgrrrrGGGgrrrr")
        ET.SubElement(tlLogic, 'phase', duration="2", state="YYYYrrrrYYYYrrrr")
        ET.SubElement(tlLogic, 'phase', duration="1", state="rrrrrrrrrrrrrrrr")
        ET.SubElement(tlLogic, 'phase', duration=str(int(green_times[i])), state="rrrrGGGgrrrrGGGg")
        ET.SubElement(tlLogic, 'phase', duration="2", state="rrrrYYYYrrrrYYYY")
        ET.SubElement(tlLogic, 'phase', duration="1", state="rrrrrrrrrrrrrrrr")

    tree = ET.ElementTree(root)
    tree.write(output_file)

    print(f"生成的Webster配置文件已保存到: {output_file}")


# 使用方法
output_file = 'webster_tlsOffsets.add.xml'
generate_webster_tls_file(output_file, green_times)
