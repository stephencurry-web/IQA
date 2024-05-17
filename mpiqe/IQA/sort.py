import pandas as pd
import re

# # 加载数据
# data = pd.read_csv('bid_label.txt', header=None)

# # 定义一个函数来提取路径中的数字
def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None

# # 应用函数并创建一个新列用于排序
# data['sort_key'] = data[0].apply(extract_number)

# # 根据新列排序
# data_sorted = data.sort_values(by='sort_key')

# # 删除排序用的辅助列
# data_sorted.drop('sort_key', axis=1, inplace=True)

# # 保存排序后的数据到新文件
# data_sorted.to_csv('bid.txt', index=False, header=False)

# print("排序后的数据已保存到 'sorted_file.csv'")

# 文件路径
file_path = "/home/pws/IQA/global_local/IQA/bid_label.txt"

# 读取文件内容并按照每行的第一个值进行排序
with open(file_path, "r") as file:
    # 读取文件的每一行，并将其存储在列表中
    lines = file.readlines()

# 使用 lambda 函数指定排序键，即每行的第一个值（数字字符串）
# 排序键将字符串转换为整数进行比较
lines.sort(key=lambda x: extract_number(x.split('\t')[0]))

# 将排序后的内容写入到文件
sorted_file_path = "/home/pws/IQA/global_local/IQA/bid.txt"
with open(sorted_file_path, "w") as sorted_file:
    for line in lines:
        sorted_file.write(line)
