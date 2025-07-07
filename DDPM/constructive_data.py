import pandas as pd

# 读取 Excel 文件
# 假设文件名为 'data.xlsx'，如果文件名不同，请替换为实际文件名
df = pd.read_excel(r'D:\tsc_ddqn_prb\data\CPMData.xlsx')

# 预处理 state 列，去除方括号并保留逗号
df['state'] = df['state'].apply(lambda x: str(x).strip('[]'))

# 创建一个空的 DataFrame 用于保存结果
result = pd.DataFrame()

# 根据 Tlid 分组，并将对应的 state 和 reward 排列到不同的列
for tlid in range(4):  # Tlid 只有 0, 1, 2, 3
    # 生成列名
    state_column = f'state_{tlid}'
    reward_column = f'reward_{tlid}'

    # 按照 Tlid 对数据进行筛选
    filtered_data = df[df['Tlid'] == tlid]

    # 将筛选后的 state 和 reward 列加入到结果 DataFrame 中
    result[state_column] = filtered_data['state'].reset_index(drop=True)
    result[reward_column] = filtered_data['reward'].reset_index(drop=True)

# 重新排列列的顺序，确保 state 列在前，reward 列在后，且顺序是按 Tlid=0, 1, 2, 3
state_columns = [f'state_{tlid}' for tlid in range(4)]
reward_columns = [f'reward_{tlid}' for tlid in range(4)]

# 将所有 state 列排在前面，reward 列排在后面
result = result[state_columns + reward_columns]

# 查看最终结果
print(result)

# 如果需要将结果保存为新的 Excel 文件，可以使用以下代码
result.to_excel(r'D:\tsc_ddqn_prb\data\CPMData_final.xlsx', index=False)
