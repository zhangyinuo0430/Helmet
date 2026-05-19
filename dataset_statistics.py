import os
import glob
from PIL import Image

def main():
    # 数据集路径
    dataset_path = "E:/Yolov8_helmet/qq_3045834499/Helmet_42_yolo_dataset"
    
    # 类别定义
    class_names = {
        0: "person",
        1: "head",    # 未佩戴安全帽（负类）
        2: "helmet"   # 佩戴安全帽（正类）
    }
    
    # 小目标阈值
    small_obj_threshold = 32  # 框小于 32x32 算小目标
    
    # 统计结果
    total_images = 0
    total_boxes = 0
    class_boxes = {0: 0, 1: 0, 2: 0}
    class_small_boxes = {0: 0, 1: 0, 2: 0}
    
    # 获取所有分割（train/val/test）
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = os.path.join(dataset_path, "images", split)
        label_dir = os.path.join(dataset_path, "labels", split)
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"警告：{split} 目录不存在")
            continue
        
        # 获取所有图像文件
        img_files = glob.glob(os.path.join(img_dir, "*.jpg")) + \
                    glob.glob(os.path.join(img_dir, "*.png")) + \
                    glob.glob(os.path.join(img_dir, "*.jpeg"))
        
        print(f"\n正在统计 {split} 集...")
        
        for img_path in img_files:
            total_images += 1
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, img_name + ".txt")
            
            # 读取图像尺寸
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"无法读取图像 {img_path}: {e}")
                continue
            
            # 读取标签文件
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 计算实际框大小（像素）
                    box_width = width * img_width
                    box_height = height * img_height
                    
                    total_boxes += 1
                    class_boxes[class_id] += 1
                    
                    # 判断是否为小目标
                    if box_width < small_obj_threshold and box_height < small_obj_threshold:
                        class_small_boxes[class_id] += 1
            else:
                # 无标签的图像
                pass
    
    # 输出统计结果
    print("\n" + "="*60)
    print("数据集统计结果")
    print("="*60)
    print(f"总图像数: {total_images}")
    print(f"总bounding box数: {total_boxes}")
    print("\n各类别统计:")
    print(f"  佩戴安全帽 (helmet - 正类): {class_boxes[2]} 个")
    print(f"    - 其中小目标 (<32x32): {class_small_boxes[2]} 个 ({(class_small_boxes[2]/class_boxes[2]*100):.1f}%)")
    print(f"  未佩戴安全帽 (head - 负类): {class_boxes[1]} 个")
    print(f"    - 其中小目标 (<32x32): {class_small_boxes[1]} 个 ({(class_small_boxes[1]/class_boxes[1]*100):.1f}%)")
    print(f"  行人 (person): {class_boxes[0]} 个")
    if class_boxes[0] > 0:
        print(f"    - 其中小目标 (<32x32): {class_small_boxes[0]} 个 ({(class_small_boxes[0]/class_boxes[0]*100):.1f}%)")
    else:
        print(f"    - 其中小目标 (<32x32): 0 个 (0%)")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
