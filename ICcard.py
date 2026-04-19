#库的导入
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



# 任务1 数据预处理

# 1. 读取数据
'''
    尝试使用制表符读取，若仍为1列，则可能是逗号分隔或存在其他格式问题（
    注：此处是因为文件导入版本兼容问题，在本机py上无法将分隔符识别为制表符，
    所以让AI先遵循原本要求，在识别错误之后再执行pandas的自动修正，
    确保数据读入成功。 
'''
try:
    #用原本规定的\t为分隔符为判断条件进行读取
    df = pd.read_csv('ICData.csv', sep='\t', encoding='utf-8')
    # 检查是否读取成功（如果列数还是1，尝试用自动修正进行读取）
    if df.shape[1] == 1:
        df = pd.read_csv('ICData.csv', sep=None, engine='python', encoding='utf-8')

#试输出数据集前五行
    print("--- 数据集前5行 ---")
    print(df.head())
#输出数据集基本信息
    print("\n--- 数据集基本信息 ---")
    print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")
#输出各列数据类型
    print("\n--- 各列数据类型 ---")
    print(df.dtypes)
#处理读入出错情况
except Exception as e:
    print(f"读取文件时出错: {e}")

# 2. 时间解析
# 确认列名是否存在后再操作，避免KeyError
if '交易时间' in df.columns:                            #确认该列存在
    df['交易时间'] = pd.to_datetime(df['交易时间'])     #将「交易时间」列转换为pandas的datetime类型
    df['hour'] = df['交易时间'].dt.hour                 #从中提取「小时」字段（整数），新增为hour列
else:
    print("错误：列名中未找到 '交易时间'，请检查文件表头格式。")    #处理该列不存在情况

# 3. 构造衍生字段
# 计算搭乘站点数：|下车站点 - 上车站点|
df['ride_stops'] = (df['下车站点'] - df['上车站点']).abs()

# 处理异常记录 (ride_stops 为 0)
initial_count = len(df)                     #获取当前数据框的总行数
df = df[df['ride_stops'] != 0].copy()      #判断站点数是否为0，不为0的才保留，否则就不copy（删去了）
dropped_rows = initial_count - len(df)      #原来的减去删去后的数据集就是被删除的数据行
print(f"\n--- 异常记录处理 ---")
print(f"已删除 ride_stops 为 0 的异常记录共计: {dropped_rows} 行")      #打印删除行数

# 4. 缺失值检查与处理
'''
    此处采用了遇到缺失值就直接删除的策略，因为有缺失值无法进行数据填充，
    会导致后续数据分析时出现错误，所以要把异常数据剔除，
    这里采用了直接整行删除的处理策略。
'''
print("\n--- 各列缺失值数量 ---")
missing_values = df.isnull().sum()      #判断是否为空（缺失数据），非空为False，空为True，并且统计每列True数量
print(missing_values)                   #打印各列缺失值数量

#删除含缺失值的行
if missing_values.sum() > 0:
    df.dropna(inplace=True)     #如果是空（True）就删去
    print(f"\n检测到缺失值，已执行删除策略。")
else:
    print("\n未检测到缺失值。")

print("\n--- 任务1 预处理完成 ---")



#任务2 时间分布分析

# 任务2预处理：设置绘图中文字体（解决Matplotlib中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 任务2 (a)：早晚时段刷卡量统计（使用 numpy）
# 仅统计刷卡类型=0（上车刷卡）的记录
pickup_df = df[df['刷卡类型'] == 0].copy()          #只有刷卡类型=0（上车刷卡）才会被copy（记录）留下，实现删去不刷卡记录的功能
total_pickups = len(pickup_df)                      #记录刷卡数据总数据量

# 使用numpy布尔索引进行统计

# 提取hour数组
hours_arr = pickup_df['hour'].values

# 1. 统计两个特殊时段数量
# 早峰前时段：hour < 7
early_morning_mask = hours_arr < 7      #创建数组记录刷卡情况判断结果，对hours_arr里的每个数字进行判断。如果是凌晨0-6点，对应位置就是True（1），否则是False（0）。
early_morning_count = np.sum(early_morning_mask)        #统计七点前刷卡次数

# 深夜时段：hour >= 22
late_night_mask = hours_arr >= 22       #创建数组记录刷卡情况判断结果，对hours_arr里的每个数字进行判断。如果是晚上22（包括22）点后，对应位置就是True（1），否则是False（0）。
late_night_count = np.sum(late_night_mask)          #统计晚上十点后（包括22）刷卡次数

# 2. 计算并打印百分比
early_pct = (early_morning_count / total_pickups) * 100
late_pct = (late_night_count / total_pickups) * 100

print("\n--- 任务2(a) 早晚时段刷卡量统计 ---")
print(f"全天总刷卡量(上车): {total_pickups} 次")
print(f"早峰前时段 (< 07:00) 刷卡量: {early_morning_count} 次, 占比: {early_pct:.2f}%")
print(f"深夜时段 (>= 22:00) 刷卡量: {late_night_count} 次, 占比: {late_pct:.2f}%")

# 任务2 (b)：24小时刷卡量分布可视化
# 3. 准备绘图数据
hour_counts = pickup_df['hour'].value_counts().sort_index()     #统计每小时刷卡量
# 补全可能缺失的小时（确保0-23都有数据，即使为0）
'''
np.arange(24)：利用 NumPy 生成一个从 0 到 23 的整数序列。
pd.Series(0, ...)：创建空模板。
通过构造一个虚拟空表格（包含 24 行，对应 24 小时，且每一小时的初始值都设为 0），
用于防止原始数据中缺失某些时段（数据为0）
'''
full_hours = pd.Series(0, index=np.arange(24))
'''
将空模板和实际统计数据相加，确保 hour_counts 变成了一个长度固定为 24 的序列
'''
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
plt.figure(figsize=(10, 6))         #图的大小
bars = plt.bar(hour_counts.index, hour_counts.values, color=colors, edgecolor='black', alpha=0.8)   #绘图

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

# 展示图像，用于调试时直接检验是否出错
#plt.show()

'''
    解决中文字体问题的通用配置，
    原始的预处理仍然不能解决本机中文字体的配置兼容问题，
    在debug时AI继续添加了能够进一步处理兼容问题的代码段
'''
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False



#任务3 线路站点分析

"""
    按照要求编写函数功能，要求如下（方便对照）：
    计算各线路乘客的平均搭乘站点数及其标准差。
    Parameters
    ----------
    df : pd.DataFrame  预处理后的数据集
    route_col : str    线路号列名
    stops_col : str    搭乘站点数列名
    Returns
    -------
    pd.DataFrame  包含列：线路号、mean_stops、std_stops，
    按 mean_stops 降序排列
 """
def analyze_route_stops(df, route_col='线路号', stops_col='ride_stops'):
    # 使用 groupby 聚合计算均值和标准差
    route_stats = df.groupby(route_col)[stops_col].agg(['mean', 'std']).reset_index()

    # 重命名列名
    route_stats.columns = [route_col, 'mean_stops', 'std_stops']

    # 按 mean_stops 降序排列
    route_stats = route_stats.sort_values(by='mean_stops', ascending=False)

    return route_stats


# 1. 调用函数并打印结果（前10行）
route_analysis_result = analyze_route_stops(df)     #调用函数，对全天所有线路的乘车情况进行汇总，并将结果存入新表格
print("\n--- 任务3：线路平均搭乘站点数统计（前10行） ---")
print(route_analysis_result.head(10))               #输出前十行数据

# 2. 使用 seaborn 水平条形图可视化
# 筛选前15条线路的原始数据用于绘制带误差棒的图
top_15_ids = route_analysis_result.head(15)['线路号'].tolist()         #筛选前15的线路（由于已经排序，所以直接选取前15行）
plot_df = df[df['线路号'].isin(top_15_ids)].copy()         #将这15组数据复制到新的数据集用于后面的绘图

# 将线路号转为字符串，确保 y 轴作为分类标签处理
plot_df['线路号'] = plot_df['线路号'].astype(str)
# 确立排序顺序（字符串格式）
order_list = [str(x) for x in top_15_ids]

plt.figure(figsize=(10, 8))

# 使用seaborn绘制水平条形图（设定各个参数）
# 修正 palette 警告：将 y 赋值给 hue 并设置 legend=False
sns.barplot(
    data=plot_df,
    y='线路号',
    x='ride_stops',
    hue='线路号',
    order=order_list,
    palette='Blues_d',  #颜色要求
    errorbar='sd',  # 显示标准差
    capsize=0.3,
    legend=False  # 移除多余图例
)

# 设置图表细节（标题和xy轴）
plt.title('前15条平均搭乘站点数最高的线路', fontsize=14)
plt.xlabel('平均搭乘站点数', fontsize=12)
plt.ylabel('线路号', fontsize=12)
plt.xlim(0, None)  # x 轴范围从 0 起始

# 保存图像
plt.tight_layout()
#plt.show() 用于调试时直接查看输出
plt.savefig('route_stops.png', dpi=150)     #按要求保存图表

print("\n--- 任务3：可视化图表已保存为 route_stops.png ---")


#任务4 高峰小时系数 (PHF) 计算

print("\n--- 任务4：高峰小时系数计算 ---")

# 1. 自动寻找全天刷卡量最大的小时 (高峰小时)
peak_hour = hour_counts.idxmax()            #寻找前面处理得出的每小时刷卡统计数据集里面，刷卡次数最多的小时（高峰小时）的索引（index/hour）
peak_hour_count = int(hour_counts.max())    #寻找前面处理得出的每小时刷卡统计数据集里面，刷卡次数最多的小时（高峰小时）的次数（values/times）

print(f"高峰小时：{peak_hour:02d}:00 ~ {peak_hour+1:02d}:00，刷卡量：{peak_hour_count} 次")        #输出高峰小时

# 2. 筛选出高峰小时内的所有刷卡数据
peak_data = pickup_df[pickup_df['hour'] == peak_hour].copy()        #将高峰小时内的所有刷卡数据复制到新数据集用于后续处理

# 确保“交易时间”是datetime类型并设为索引
peak_data['交易时间'] = pd.to_datetime(peak_data['交易时间'])       #用pandas将交易时间转化为datetime类型
peak_data.set_index('交易时间', inplace=True)                       #把交易时间设置为索引

# 3. 5分钟粒度统计 (PHF5)
# 将 '5T' 改为 '5min' 以兼容新版 Pandas
# （注：此处是AI原始版本设置的是'T'类型，和本机版本不兼容，
# 后续debug时修改了这一错误）
'''
用resample('5min')把这1小时的数据按每5分钟划分成一组
再用.size()：统计每个5分钟桶里有多少条数据（即有多少人刷卡）。
'''
five_min_counts = peak_data.resample('5min').size()

# 找出最大5分钟刷卡量及其对应时间段
max_5min_val = five_min_counts.max()            #找到所有组别中最大的值
max_5min_start = five_min_counts.idxmax()       #找到最大值的五分钟的开始节点（用idxmax()返回最大值对应的索引）
max_5min_end = max_5min_start + pd.Timedelta(minutes=5)     #（用pd.Timedelta(minutes=5)创建一个5分钟的时间增量）计算得到最大值的五分钟的结束节点（+5min）

# 计算 PHF5
phf5 = peak_hour_count / (12 * max_5min_val)        #套公式

#按照格式输出
print(f"最大5分钟刷卡量（{max_5min_start.strftime('%H:%M')}~{max_5min_end.strftime('%H:%M')}）：{max_5min_val} 次")
print(f"PHF5  = {peak_hour_count} / (12 × {max_5min_val}) = {phf5:.4f}")

# 4. 15分钟粒度统计 (PHF15)
# 将 '15T' 改为 '15min' 以兼容新版 Pandas
# （注：此处是AI原始版本设置的是'T'类型，和本机版本不兼容，
# 后续debug时修改了这一错误）
'''
用resample('15min')把这1小时的数据按每15分钟划分成一组
再用.size()：统计每个15分钟桶里有多少条数据（即有多少人刷卡）。
'''
fifteen_min_counts = peak_data.resample('15min').size()

# 找出最大15分钟刷卡量及其对应时间段
max_15min_val = fifteen_min_counts.max()            #找到所有组别中最大的值
max_15min_start = fifteen_min_counts.idxmax()       #找到最大值的十五分钟的开始节点（用idxmax()返回最大值对应的索引）
max_15min_end = max_15min_start + pd.Timedelta(minutes=15)     #（用pd.Timedelta(minutes=15)创建一个15分钟的时间增量）计算得到最大值的五分钟的结束节点（+15min）

# 计算 PHF15
phf15 = peak_hour_count / (4 * max_15min_val)        #套公式

#按照格式输出
print(f"最大15分钟刷卡量（{max_15min_start.strftime('%H:%M')}~{max_15min_end.strftime('%H:%M')}）：{max_15min_val} 次")
print(f"PHF15 = {peak_hour_count} / ( 4 × {max_15min_val}) = {phf15:.4f}")

# --- 任务4 完成 ---



#任务 5：线路驾驶员信息批量导出


print("\n--- 任务 5：线路驾驶员信息批量导出 ---")

# 1. 定义目标文件夹名称
output_folder = "线路驾驶员信息"

# 2. 创建文件夹（如果不存在）
if not os.path.exists(output_folder):       #不存在就创建
    os.makedirs(output_folder)
    print(f"文件夹 '{output_folder}' 创建成功。")
else:
    print(f"文件夹 '{output_folder}' 已存在。")

# 3. 筛选线路号在 1101 至 1120 之间的所有记录
# 使用布尔索引进行范围筛选
target_routes = df[(df['线路号'] >= 1101) & (df['线路号'] <= 1120)].copy()        #如果满足在1101-1120之间的条件就记为True，并复制到新的数据集方便后续操作

# 获取唯一线路列表并排序（补充：按线路号排序）
route_list = sorted(target_routes['线路号'].unique())

print(f"共识别到符合范围的线路共 {len(route_list)} 条，准备开始导出...")

# 4. 循环处理每条线路并写入文件
for route_id in route_list:
    # 筛选当前线路的数据
    current_route_df = target_routes[target_routes['线路号'] == route_id]

    # 提取 (车辆编号 -> 驾驶员编号) 对应关系并去重
    driver_info = current_route_df[['车辆编号', '驾驶员编号']].drop_duplicates()

    # 构造文件名
    file_name = f"{int(route_id)}.txt"
    # 构造相对路径
    relative_path = os.path.join(output_folder, file_name)
    # 获取绝对路径（打印要求）
    absolute_path = os.path.abspath(relative_path)

    try:
        with open(relative_path, 'w', encoding='utf-8') as f:
            # 写入表头信息
            f.write(f"线路号: {int(route_id)}\n")
            f.write("车辆编号 驾驶员编号\n")

            # 遍历写入数据
            for _, row in driver_info.iterrows():
                f.write(f"{int(row['车辆编号'])}  {int(row['驾驶员编号'])}\n")

        # 打印完整生成路径，确认输出成功
        print(f"成功导出线路 {int(route_id)} -> {absolute_path}")

    except Exception as e:
        print(f"写入线路 {int(route_id)} 时出错: {e}")

print("\n--- 任务 5：所有线路驾驶员信息导出完毕 ---")



#任务6 服务绩效排名与热力图

print("\n--- 任务 6：服务绩效排名与热力图 ---")

# 1. 分维度统计 Top 10 服务人次（获取索引 ID 和数值）
top_drivers = df['驾驶员编号'].value_counts().head(10)
top_routes = df['线路号'].value_counts().head(10)
top_stations = df['上车站点'].value_counts().head(10)
top_vehicles = df['车辆编号'].value_counts().head(10)

# 打印排名结果
print("\n[Top 10 驾驶员服务人次]:")
print(top_drivers)
print("\n[Top 10 线路服务人次]:")
print(top_routes)
print("\n[Top 10 上车站点服务人次]:")
print(top_stations)
print("\n[Top 10 车辆服务人次]:")
print(top_vehicles)

# 2. 构造热力图所需的数据矩阵
# 数值矩阵：决定格子的颜色深浅
heatmap_data = pd.DataFrame([
    top_drivers.values,
    top_routes.values,
    top_stations.values,
    top_vehicles.values
])

# 标注矩阵：在格子里显示的文字（ID + 数值）
def format_labels(series):
    return [f"ID:{idx}\n{val}次" for idx, val in zip(series.index, series.values)]

heatmap_labels = pd.DataFrame([
    format_labels(top_drivers),
    format_labels(top_routes),
    format_labels(top_stations),
    format_labels(top_vehicles)
])


# 设置行标签和列标签
row_labels = ['司机', '线路', '上车站点', '车辆']
heatmap_data.index = row_labels
heatmap_labels.index = row_labels
heatmap_data.columns = [f'Top{i}' for i in range(1, 11)]

# 3. 热力图可视化
plt.figure(figsize=(15, 7))

sns.heatmap(
    heatmap_data,
    annot=heatmap_labels.values, # 使用拼好的 ID 和数值作为标注
    fmt='',                      # 因为标注是字符串，这里必须设为空
    cmap="YlOrRd",               # 使用黄-橙-红渐变
    linewidths=.5,               # 格子间距
    annot_kws={'size': 9},       # 调整字体大小以适应 ID 显示
    cbar_kws={'label': '服务人次'}
)

# 设置图表标题与轴标签
plt.title('公交服务绩效多维度 Top 10 热力图分析', fontsize=16, pad=20)
plt.suptitle('反映司机、线路、站点及车辆四个维度的客流密集度对比', fontsize=10, y=0.92)
plt.xlabel('排名层级', fontsize=12)
plt.ylabel('分析维度', fontsize=12)
plt.xticks(rotation=0)

# 保存图像
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
print("\n--- 任务 6：可视化热力图已保存为 performance_heatmap.png ---")

# 展示图像
plt.show()

# 4. 结论说明
print("\n--- 绩效规律观察结论 ---")
observation = """
通过绩效热力图可以观察到：
1. 线路维度的 Top1 (46003) 服务人次明显高于其他维度，说明该线路是整个公交系统的核心骨干。
2. 上车站点的 Top10 数值分布较为接近，反映出高客流站点存在集群效应，客流并非仅集中在单一站点。
3. 司机与车辆的服务人次高度正相关，数值相近，表明“人车相对固定”的排班模式运行稳定。
4. 头部司机的服务人次远超普通均值，体现出高峰时段或核心线路上高强度的运营工作量。
"""
print(observation)

# --- 任务 6 完成 ---