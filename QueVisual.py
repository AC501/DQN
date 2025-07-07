import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 选择一个支持中文的字体
# 读取生成的 Excel 文件
df = pd.read_excel('data/que_data_wbst.xlsx')

# 计算 Total_Queue_Length 的均值
mean_queue_length = df['Total_Queue_Length'].mean()

# 绘制带连线的散点图，并适当调整点的透明度和大小
plt.figure(figsize=(30, 15))  # 设置图形的大小

# 使用 plot 绘制带连线的散点图
plt.plot(df['Time'], df['Total_Queue_Length'], marker='o', linestyle='-', color='lightblue', alpha=0.7, label='排队长度')

# 添加均值线
plt.axhline(y=mean_queue_length, color='r', linestyle='--', label=f'平均排队长度 = {mean_queue_length:.2f}')
# 添加标题和标签
plt.title('')
plt.xlabel('时间(秒)')
plt.ylabel('排队长度(辆)')

# 添加图例
plt.legend()

# 添加网格线
plt.grid(True)

plt.savefig('quewbst_plot062.png', dpi=800, bbox_inches='tight')


# 展示图形
plt.show()
