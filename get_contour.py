import cv2
import glob
import os
import numpy as np
input_folder = r'D:\pythonproject\TransEx (2)\train_data\VT5000\test\GT'  # 请将 'input_folder_path' 替换为你的文件夹路径
output_folder = r'D:\pythonproject\TransEx (2)\train_data\VT5000\test\contour'  # 请将 'output_folder_path' 替换为你的输出文件夹路径

for filename in glob.glob(os.path.join(input_folder, '*.png')):
    # 读取图像
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    ret, binary_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    # if "\\2.png" in filename:
    #     print(1)
    # 使用Canny边缘检测
    edges = cv2.Canny(binary_img, 1, 1)
    if edges.max() == 0:
        print(1)
    # 这里值设置为 1, 1 是由于图片的像素值只有0和1
    # # 定义用于开操作的卷积核
    # kernel_size = 3  # 可以根据需要调整此值，较大的值会删除较大的区域
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    #
    # # 应用开操作
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    save_name = filename.split('\\')[-1].split('.')[-2] + ".jpg"
    cv2.imwrite(os.path.join(output_folder, save_name), edges)