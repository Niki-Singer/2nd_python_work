import pandas as pd
import numpy as np

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