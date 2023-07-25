import matplotlib.pyplot as plt
import numpy as np

# 存储所有的数组数据
data = []

# 读取10个txt文件
txt_files = ['./logs/Mobilenetv2_1280_320_ample.txt', './logs/Mobilenetv2_1280_320_dstar.txt', './logs/Mobilenetv2_1280_320_euclid.txt', './logs/Mobilenetv2_1280_320_jaccard.txt', './logs/Mobilenetv2_1280_320_ochiai.txt', './logs/Mobilenetv2_1280_320_tarantula.txt',
             './logs/Mobilenetv2_1280_320_wong3.txt']

# 逐个读取txt文件并存储数组数据
for file in txt_files:
    with open(file, 'r') as f:
        lines = f.readlines()
        # 去除每行末尾的换行符，并将字符串转换为浮点数
        value = [float(line.strip()) for line in lines]
        # value.sort()  # 排序数组
        data.append(value)

# 计算所有数组数据的极大值和极小值
max_value = np.max(data)
min_value = np.min(data)
plt.figure(figsize=(8, 6))
plt.xlim(0, max(len(values) for values in data) - 1)
plt.ylim(0, 1)

# 绘制数组数据
colors = ['red', 'green', 'blue', 'orange', 'purple',
          'brown', 'teal']
label = ['Ample','DStar', 'Euclid','Jaccard', 'Ochiai', 'Tarantula', 'Wong3']
for i, values in enumerate(data):
    x = np.arange(len(values))  # 横坐标为数组的序号
    y = np.array(values)  # 纵坐标为数组的值

    # 将纵坐标进行放缩
    # y_scaled = (y - min_value) / (max_value - min_value)

    # 使用放缩后的纵坐标绘制线条
    plt.scatter(x, y, color=colors[i], label=label[i], marker='.')

    plt.legend()
    plt.title('Mobilenetv2 Dense-1 Importance')

    # 保存图形为PNG格式
    plt.savefig('Mobilenetv2_dense-1_importance_' + label[i]+ '.png', dpi=300)

    # 显示图形
    plt.show()