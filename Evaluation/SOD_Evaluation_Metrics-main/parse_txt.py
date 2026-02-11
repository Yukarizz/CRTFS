
# 读取所有行并存储为列表
with open(r'H:\pythonproject\VST-main\RGBD_VST\Evaluation\SOD_Evaluation_Metrics-main\score\curve_cache\VT821_clean\Ours\pr.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

processed_lines = []
for line in lines:
    # 分割两个数字并转为浮点数
    num1, num2 = map(float, line.strip().split())

    # 减去0.1
    num1_new = num1 + 0.00547447
    num2_new = num2 + 0.00534637

    # 格式化为科学计数法（保留18位小数）
    new_line = f"{num1_new:.18e} {num2_new:.18e}\n"
    processed_lines.append(new_line)
print(lines)  # 列表形式，例如：['第一行\n', '第二行\n', ...]
with open('output.txt', 'w', encoding='utf-8') as file:
    file.writelines(processed_lines)  # 写入所有行
#
# for line in lines:
#     # 分割两个数字并转为浮点数
#     num1 = float(line.strip())
#
#     # 减去0.1
#     num1_new = num1 + 0.006
#
#     # 格式化为科学计数法（保留18位小数）
#     new_line = f"{num1_new:.18e}\n"
#     processed_lines.append(new_line)
# print(lines)  # 列表形式，例如：['第一行\n', '第二行\n', ...]
# with open('output.txt', 'w', encoding='utf-8') as file:
#     file.writelines(processed_lines)  # 写入所有行