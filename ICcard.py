import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
# 针对你提供的报错（列数为1），尝试显式指定分隔符并处理可能的编码问题
try:
    # 尝试使用制表符读取，若仍为1列，则可能是逗号分隔或存在其他格式问题
    df = pd.read_csv('ICData.csv', sep='\t', encoding='utf-8')

    # 检查是否读取成功（如果列数还是1，尝试自动修正）
    if df.shape[1] == 1:
        df = pd.read_csv('ICData.csv', sep=None, engine='python', encoding='utf-8')

    print("--- 数据集前5行 ---")
    print(df.head())
    print("\n--- 数据集基本信息 ---")
    print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
    print("\n--- 各列数据类型 ---")
    print(df.dtypes)
except Exception as e:
    print(f"读取文件时出错: {e}")

# 2. 时间解析
# 确认列名是否存在后再操作，避免 KeyError
if '交易时间' in df.columns:
    df['交易时间'] = pd.to_datetime(df['交易时间'])
    df['hour'] = df['交易时间'].dt.hour
else:
    print("错误：列名中未找到 '交易时间'，请检查文件表头格式。")

# 3. 构造衍生字段
# 计算搭乘站点数：|下车站点 - 上车站点|
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

# 处理异常记录 (ride_stops 为 0)
initial_count = len(df)
df = df[df['ride_stops'] != 0].copy()
dropped_rows = initial_count - len(df)
print(f"\n--- 异常记录处理 ---")
print(f"已删除 ride_stops 为 0 的异常记录共计: {dropped_rows} 行")

# 4. 缺失值检查与处理
print("\n--- 各列缺失值数量 ---")
missing_values = df.isnull().sum()
print(missing_values)

# 处理策略：删除含缺失值的行
if missing_values.sum() > 0:
    df.dropna(inplace=True)
    print(f"\n检测到缺失值，已执行删除策略。")
else:
    print("\n未检测到缺失值。")

print("\n--- 任务1 预处理完成 ---")

# 设置绘图中文字体（解决Matplotlib中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 承接任务1的预处理结果 (假设 df 已加载并完成 hour 列提取) ---

# 任务2 (a)：早晚时段刷卡量统计（使用 numpy）
# 仅统计刷卡类型=0（上车刷卡）的记录
pickup_df = df[df['刷卡类型'] == 0].copy()
total_pickups = len(pickup_df)

# 使用 numpy.where 或布尔索引提取 hour 数组
hours_arr = pickup_df['hour'].values

# 1. 统计时段数量
# 早峰前时段：hour < 7
early_morning_mask = hours_arr < 7
early_morning_count = np.sum(early_morning_mask)

# 深夜时段：hour >= 22
late_night_mask = hours_arr >= 22
late_night_count = np.sum(late_night_mask)

# 2. 计算并打印百分比
early_pct = (early_morning_count / total_pickups) * 100
late_pct = (late_night_count / total_pickups) * 100

print("\n--- 任务2(a) 早晚时段刷卡量统计 ---")
print(f"全天总刷卡量(上车): {total_pickups} 次")
print(f"早峰前时段 (< 07:00) 刷卡量: {early_morning_count} 次, 占比: {early_pct:.2f}%")
print(f"深夜时段 (>= 22:00) 刷卡量: {late_night_count} 次, 占比: {late_pct:.2f}%")

# 任务2 (b)：24小时刷卡量分布可视化
# 3. 准备绘图数据
hour_counts = pickup_df['hour'].value_counts().sort_index()
# 补全可能缺失的小时（确保0-23都有数据，即使为0）
full_hours = pd.Series(0, index=np.arange(24))
hour_counts = (full_hours + hour_counts).fillna(0)

# 定义颜色逻辑：早峰前和深夜高亮
colors = []
for h in range(24):
    if h < 7:
        colors.append('salmon')    # 早峰前颜色
    elif h >= 22:
        colors.append('steelblue') # 深夜颜色
    else:
        colors.append('lightgray') # 其他时段颜色

# 开始绘图
plt.figure(figsize=(10, 6))
bars = plt.bar(hour_counts.index, hour_counts.values, color=colors, edgecolor='black', alpha=0.8)

# 设置图例（手动创建代理图例）
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='salmon', label='早峰前 (<7h)'),
    Patch(facecolor='steelblue', label='深夜 (≥22h)'),
    Patch(facecolor='lightgray', label='普通时段')
]
plt.legend(handles=legend_elements)

# 设置坐标轴与标题
plt.title('公交IC卡24小时刷卡量分布图', fontsize=14)
plt.xlabel('小时', fontsize=12)
plt.ylabel('刷卡量（次）', fontsize=12)
plt.xticks(np.arange(0, 24, 2))  # x轴步长为2
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图像
plt.tight_layout()
plt.savefig('hour_distribution.png', dpi=150)
print("\n--- 任务2(b) 可视化图表已保存为 hour_distribution.png ---")

# 展示图像（可选）
plt.show()


# 解决中文字体问题的通用配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# --- 任务3：线路站点分析 ---

def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    """
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，按 mean_stops 降序排列
    """
    # 使用 groupby 聚合计算均值和标准差
    route_stats = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()

    # 重命名列名
    route_stats.columns = [route_col, 'mean_stops', 'std_stops']

    # 按 mean_stops 降序排列
    route_stats = route_stats.sort_values(by='mean_stops', ascending=False)

    return route_stats


# 1. 调用函数并打印结果（前10行）
route_analysis_result = analyze_route_stops(df)
print("\n--- 任务3：线路平均搭乘站点数统计（前10行） ---")
print(route_analysis_result.head(10))

# 2. 使用 seaborn 水平条形图可视化
# 筛选前15条线路的原始数据用于绘制带误差棒的图
top_15_ids = route_analysis_result.head(15)['线路号'].tolist()
plot_df = df[df['线路号'].isin(top_15_ids)].copy()

# 将线路号转为字符串，确保 y 轴作为分类标签处理
plot_df['线路号'] = plot_df['线路号'].astype(str)
# 确立排序顺序（字符串格式）
order_list = [str(x) for x in top_15_ids]

plt.figure(figsize=(10, 8))

# 绘制水平条形图
# 修正 palette 警告：将 y 赋值给 hue 并设置 legend=False
sns.barplot(
    data=plot_df,
    y='线路号',
    x='ride_stops',
    hue='线路号',
    order=order_list,
    palette='Blues_d',
    errorbar='sd',  # 显示标准差
    capsize=0.3,
    legend=False  # 移除多余图例
)

# 设置图表细节
plt.title('前15条平均搭乘站点数最高的线路', fontsize=14)
plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.xlim(0, None)  # x 轴范围从 0 起始

# 保存图像
plt.tight_layout()
plt.show()
plt.savefig('route_stops.png', dpi=150)
print("\n--- 任务3：可视化图表已保存为 route_stops.png ---")


# --- 任务4：高峰小时系数 (PHF) 计算 (修正版) ---

print("\n--- 任务4：高峰小时系数计算 ---")

# 1. 自动寻找全天刷卡量最大的小时 (高峰小时)
peak_hour = hour_counts.idxmax()
peak_hour_count = int(hour_counts.max())

print(f"高峰小时：{peak_hour:02d}:00 ~ {peak_hour+1:02d}:00，刷卡量：{peak_hour_count} 次")

# 2. 筛选出高峰小时内的所有刷卡数据
peak_data = pickup_df[pickup_df['hour'] == peak_hour].copy()

# 确保“交易时间”是 datetime 类型并设为索引
peak_data['交易时间'] = pd.to_datetime(peak_data['交易时间'])
peak_data.set_index('交易时间', inplace=True)

# 3. 5分钟粒度统计 (PHF5)
# 将 '5T' 改为 '5min' 以兼容新版 Pandas
five_min_counts = peak_data.resample('5min').size()

# 找出最大5分钟刷卡量及其对应时间段
max_5min_val = five_min_counts.max()
max_5min_start = five_min_counts.idxmax()
max_5min_end = max_5min_start + pd.Timedelta(minutes=5)

# 计算 PHF5
phf5 = peak_hour_count / (12 * max_5min_val)

print(f"最大5分钟刷卡量（{max_5min_start.strftime('%H:%M')}~{max_5min_end.strftime('%H:%M')}）：{max_5min_val} 次")
print(f"PHF5  = {peak_hour_count} / (12 × {max_5min_val}) = {phf5:.4f}")

# 4. 15分钟粒度统计 (PHF15)
# 将 '15T' 改为 '15min' 以兼容新版 Pandas
fifteen_min_counts = peak_data.resample('15min').size()

# 找出最大15分钟刷卡量及其对应时间段
max_15min_val = fifteen_min_counts.max()
max_15min_start = fifteen_min_counts.idxmax()
max_15min_end = max_15min_start + pd.Timedelta(minutes=15)

# 计算 PHF15
phf15 = peak_hour_count / (4 * max_15min_val)

print(f"最大15分钟刷卡量（{max_15min_start.strftime('%H:%M')}~{max_15min_end.strftime('%H:%M')}）：{max_15min_val} 次")
print(f"PHF15 = {peak_hour_count} / ( 4 × {max_15min_val}) = {phf15:.4f}")

# --- 任务4 完成 ---