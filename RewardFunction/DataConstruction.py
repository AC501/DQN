import pandas as pd

# 读取 Excel 文件
df = pd.read_excel(r'D:\tsc_ddqn_prb_xiugaimainkuangjia\data\CPMData.xlsx')

# 预处理 state_reward 列，去除方括号并保留逗号
df['state_reward'] = df['state_reward'].apply(lambda x: str(x).strip('[]'))
print('df',df)
# 创建一个空的 DataFrame 用于保存结果
result = pd.DataFrame()

# 根据 Tlid 分组，并将对应的 state 和 reward 分开
for tlid in range(4):  # Tlid 只有 0, 1, 2, 3
    # 生成列名
    state_column = f'state_{tlid}'
    reward_column = f'reward_{tlid}'

    # 按照 Tlid 对数据进行筛选
    filtered_data = df[df['Tlid'] == tlid]

    # 假设 state_reward 是以逗号分隔的 state 和 reward，最后一位是 reward
    filtered_data['state'] = filtered_data['state_reward'].apply(lambda x: x.rsplit(',', 1)[0])  # 获取 state 部分
    filtered_data['reward'] = filtered_data['state_reward'].apply(
        lambda x: float(x.rsplit(',', 1)[1]))  # 获取 reward 部分，并转换为数值

    # 将 state 和 reward 列分别加入到结果 DataFrame 中
    result[state_column] = filtered_data['state'].reset_index(drop=True)
    result[reward_column] = filtered_data['reward'].reset_index(drop=True)

# 重新排列列的顺序，确保 state 列在前，reward 列在后，且顺序是按 Tlid=0, 1, 2, 3
state_columns = [f'state_{tlid}' for tlid in range(4)]
reward_columns = [f'reward_{tlid}' for tlid in range(4)]

# 将所有 state 列排在前面，reward 列排在后面
result = result[state_columns + reward_columns]

# 查看最终结果
print(result)


result.to_excel(r'D:\tsc_ddqn_prb_xiugaimainkuangjia\data\FitRewardData.xlsx', index=False)
