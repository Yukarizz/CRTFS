import json
import numpy as np
import cv2

# 读取JSON文件
with open('labels_my-project-name_2025-05-14-08-31-01.json') as f:
    coco_data = json.load(f)

# 获取图像尺寸信息
image_info = coco_data['images'][0]
width = image_info['width']
height = image_info['height']

# 创建全黑背景（0表示黑色）
mask = np.zeros((height, width), dtype=np.uint8)

# 遍历所有标注
for annotation in coco_data['annotations']:
    # 将浮点坐标转换为整数并重塑为OpenCV需要的格式
    segmentation = annotation['segmentation']
    for polygon in segmentation:
        # 将坐标转换为整数并重塑为(N, 1, 2)格式
        points = np.array(polygon, dtype=np.float32).reshape((-1, 2))
        points = np.round(points).astype(np.int32)
        points = points.reshape((-1, 1, 2))

        # 用白色填充多边形（255表示白色）
        cv2.fillPoly(mask, [points], color=255)

# 保存mask图像
cv2.imwrite('mask.jpg', mask)
print("Mask已保存为mask.jpg")