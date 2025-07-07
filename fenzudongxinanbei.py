import pandas as pd
import numpy as np
# 读取 Excel 文件
input_excel_file = 'D:/tsc_ddqn_prb_1Con_new/nets/3x3grid/e2_result/processed_data.xlsx'  # 请替换成你的 Excel 文件路径
df = pd.read_excel(input_excel_file)

# 将数据转换为pandas DataFrame
df = pd.DataFrame(df)

# 计算每组的奇数行和偶数行的和
odd_sums = df.groupby('Group').apply(lambda x: x.iloc[::2]['Final Result'].sum())
even_sums = df.groupby('Group').apply(lambda x: x.iloc[1::2]['Final Result'].sum())

# 将结果合并为一个DataFrame
result = pd.DataFrame({'NSqueue': odd_sums, 'EWqueue': even_sums}).reset_index()
# 将结果保存到Excel文件
result.to_excel('D:/tsc_ddqn_prb_1Con_new/nets/3x3grid/e2_result/grouped_sums1.xlsx', index=False)
print(result)