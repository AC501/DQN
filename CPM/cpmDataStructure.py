import pandas as pd

def process_cpm_data(input_filepath, output_filepath):
    """
    处理 CPM 数据，读取 Excel 文件，进行数据预处理，并将处理结果保存为新的 Excel 文件。

    参数:
    - input_filepath: 输入 Excel 文件的路径
    - output_filepath: 输出 Excel 文件的路径
    """
    # 读取 Excel 文件
    df = pd.read_excel(input_filepath)

    # 预处理 state_reward 列，去除方括号并保留逗号
    df['state_reward'] = df['state_reward'].apply(lambda x: str(x).strip('[]'))

    # 创建一个空的 DataFrame 用于保存结果
    result = pd.DataFrame()

    # 修改：2023-11-01 - 处理所有四个交通信号灯的数据
    for tlid in range(4):  # Tlid 有 0, 1, 2, 3
        # 生成列名
        state_reward_column = f'state_reward_{tlid}'

        # 按照 Tlid 对数据进行筛选
        filtered_data = df[df['Tlid'] == tlid]

        # 将 state_reward 列加入到结果 DataFrame 中
        result[state_reward_column] = filtered_data['state_reward'].reset_index(drop=True)

    # 将最终结果保存为 Excel 文件
    result.to_excel(output_filepath, index=False)
    print(f"处理完成，结果已保存到: {output_filepath}")

# 主函数入口
if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_filepath = 'data_0/CPMData.xlsx'
    output_filepath = 'data_0/CPMData_final.xlsx'

    # 调用处理函数
    process_cpm_data(input_filepath, output_filepath)
