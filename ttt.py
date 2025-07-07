import pandas as pd

# 读取 Excel 文件
input_excel_file = 'D:/tsc_ddqn_prb_1Con_new/nets/3x3grid/e2_result/final_results5.xlsx'  # 请替换成你的 Excel 文件路径
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