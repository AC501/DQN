import os
import xml.etree.ElementTree as ET
import pandas as pd

# 定义一个函数来处理单个XML文件
def process_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    results = []
    for interval in root.findall('.//interval'):
        n_veh_left = int(interval.get('nVehLeft'))  # 转换为整数
        results.append(n_veh_left)
    return results

# 指定包含XML文件的文件夹路径
folder_path = 'D:/tsc_ddqn_prb_1Con_new/nets/3x3grid/e2_result'

# 处理所有XML文件
all_results = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xml'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)
        # 处理XML文件并将结果存储到列表中
        all_results.extend(process_xml_file(file_path))

# 计算每个数据所属的组
group_size = 16
groups = [i // group_size + 1 for i in range(len(all_results))]

# 创建一个DataFrame来存储原始结果
df_original = pd.DataFrame({'Group': groups, 'Result': all_results})
df_original.index.name = 'Index'

# 对每个分组中的相邻两行进行相加
group_sums = []
for group_num, group_data in df_original.groupby('Group'):
    group_sum = group_data.groupby(group_data.index // 2 * 2).sum()  # 每两行相加
    group_sums.extend(group_sum['Result'])  # 将结果添加到列表中

# 创建一个DataFrame来存储处理后的结果
df_processed = pd.DataFrame({'Result': group_sums})
df_processed.index.name = 'Index'

# 对处理后的结果再次进行相加，每两行相加
final_results = []
for i in range(0, len(df_processed), 2):
    final_results.append(df_processed.iloc[i:i+2]['Result'].sum())

# 创建一个DataFrame来存储最终结果
df_final = pd.DataFrame({'Final Result': final_results})
df_final.index.name = 'Index'

# 将DataFrame保存为Excel文件
output_excel_file = os.path.join(folder_path, 'final_results.xlsx')
df_final.to_excel(output_excel_file)

print("最终结果已保存到:", output_excel_file)


# 读取 Excel 文件
input_excel_file = 'D:/tsc_ddqn_prb_1Con_new/nets/3x3grid/e2_result/final_results.xlsx'  # 请替换成你的 Excel 文件路径
df = pd.read_excel(input_excel_file)

# 每四行添加一列标记组号
group_number = 0
group_column = []
for i in range(len(df)):
    group_column.append(group_number)
    if (i + 1) % 4 == 0:  # 每四行更新一次组号
        group_number += 1

# 添加组号列到 DataFrame
df['Group'] = group_column

# 保存处理后的数据到新的 Excel 文件
output_excel_file = 'D:/tsc_ddqn_prb_1Con_new/nets/3x3grid/e2_result/processed_data.xlsx'  # 请替换成你想保存的文件路径
df.to_excel(output_excel_file, index=False)

print("处理后的结果已保存到:", output_excel_file)