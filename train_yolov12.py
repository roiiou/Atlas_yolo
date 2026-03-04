"""
YOLOv12 训练脚本 (Training Script)
=====================================
功能描述：
1. 数据集增强：针对 "fishing boat" 类别进行过采样 (Rebalancing)，解决样本不平衡问题。
2. 模型加载：加载 YOLOv12 预训练模型 (yolo12n.pt)。
3. 训练配置：设置超参数 (Hyperparameters)，包括 epochs, imgsz, batch size, learning rate 等。
4. 数据增强配置：调整几何变换参数 (degrees, translate, scale, shear, flip) 以适应船只检测任务。
5. 损失函数调整：调整 box, cls, dfl 损失权重，优化小目标检测性能。
6. 模型保存：训练完成后保存最佳模型权重。

主要模块：
- rebalance_fishing_boat: 数据集平衡函数，复制指定类别的样本。
- main: 主训练流程。

"""

import os
import shutil
import torch
from ultralytics import YOLO

# --- Enhanced Class Rebalancing (增强类别平衡) ---
def rebalance_fishing_boat(dup_factor=2):
    """
    针对 "fishing boat" (类别索引 1) 进行数据增强/过采样。
    
    原因：
        数据集中 "fishing boat" 样本较少或较难检测，导致模型对其召回率低。
        通过物理复制图片和标签文件，增加其在训练集中的权重。
    
    参数:
        dup_factor (int): 复制倍数。例如 dup_factor=2 表示额外复制 2 份。
    """
    base_dir = os.path.join("datasets", "train")
    img_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels")
    # 标记文件，防止重复运行导致数据爆炸 (升级到 v4 以适应新合并的数据集)
    marker = os.path.join(base_dir, ".fishing_boat_rebalanced_v4")
    
    if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
        return
        
    if os.path.exists(marker):
        print("Dataset already rebalanced (v3). Skipping. (数据集已平衡，跳过)")
        return
        
    print(f"Rebalancing: Duplicating fishing boat samples {dup_factor} times... (正在进行类别平衡：复制 fishing boat 样本 {dup_factor} 次)")
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    count = 0
    
    for name in label_files:
        # 跳过已经是复制生成的文件 (文件名包含 _fb)
        if "_fb" in os.path.splitext(name)[0]:
            continue
            
        label_path = os.path.join(label_dir, name)
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
            
        has_fishing = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # Check for class 1 (fishing boat)
            # 检查是否包含类别 1 (fishing boat)
            if parts[0] == "1":
                has_fishing = True
                break
                
        if not has_fishing:
            continue
            
        img_name_root, _ = os.path.splitext(name)
        src_img = None
        # 寻找对应的图片文件 (支持多种后缀)
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = os.path.join(img_dir, img_name_root + ext)
            if os.path.exists(cand):
                src_img = cand
                break
                
        if src_img is None:
            continue
            
        # Duplicate (开始复制)
        for k in range(dup_factor):
            suffix = f"_fb{k+1}"
            new_img_name_root = img_name_root + suffix
            new_img_path = os.path.join(img_dir, new_img_name_root + os.path.splitext(src_img)[1])
            new_label_path = os.path.join(label_dir, new_img_name_root + ".txt")
            
            if os.path.exists(new_img_path) or os.path.exists(new_label_path):
                continue
                
            try:
                shutil.copy2(src_img, new_img_path)
                shutil.copy2(label_path, new_label_path)
                count += 1
            except OSError:
                continue
                
    try:
        # 创建标记文件，表示已完成平衡
        with open(marker, "w", encoding="utf-8") as f:
            f.write("done")
        print(f"Rebalancing complete. Added {count} samples. (平衡完成，新增 {count} 个样本)")
    except OSError:
        pass

def main():
    print("Starting YOLOv12 Ultimate Training... (开始 YOLOv12 终极训练)")
    
    # 检查 GPU 可用性
    if torch.cuda.is_available():
        device = 0
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using CPU (Warning: Slow) (警告：使用 CPU 训练会很慢)")

    # 1. Start from pretrained base to leverage general features
    # 加载预训练模型 (Transfer Learning)
    # 使用 yolo12n.pt (Nano版本) 以平衡速度和精度，适合板端部署
    model = YOLO("yolo12n.pt") 

    # 2. Prepare Data (准备数据)
    # 数据集已天然平衡 (Kayak: ~7100, Fishing Boat: ~5400)
    # 无需再进行人工过采样，避免过拟合
    # rebalance_fishing_boat(dup_factor=1) 

    # 3. Hyperparameters for Stability & Small Objects (针对稳定性与小目标的超参数优化)
    # 我们通过 train() 函数的参数覆盖默认配置
    
    print("Beginning training with optimized hyperparameters... (开始使用优化超参数进行训练)")
    results = model.train(
        data="datasets/data.yaml", # 数据集配置文件路径
        epochs=30,                # 训练轮数：紧急模式，30轮，确保11点前完成
        imgsz=640,                # 输入图像尺寸：640，速度最快
        device=device,            # 训练设备
        project="runs/train",     # 项目保存路径
        name="yolov12n_express_final", # 换个新名字，防止目录冲突
        batch=16,                 # 批次大小：16，拉满显卡
        patience=5,               # 早停
        resume=False,             # 重新开始（之前的都没跑出权重文件，无法恢复）
        # exist_ok=True,          # 已在下方定义，避免重复
        
        # --- Optimization for Small Objects & Stability (优化参数) ---
        lr0=0.01,                 # 初始学习率：0.01，快速收敛
        lrf=0.01,                 # 最终学习率比例
        momentum=0.937,           # 动量
        weight_decay=0.0005,      # 权重衰减
        
        # --- Augmentation Tweaks (数据增强微调) ---
        # 减少强烈的几何变换，避免让相似的船只类型混淆
        degrees=0.0,              # 旋转角度 (No rotation, 船只通常是直立的)
        translate=0.1,            # 平移 (Slight translation)
        scale=0.5,                # 缩放 (Scale variation, 适应远近不同的大小)
        shear=0.0,                # 剪切 (No shear)
        perspective=0.0,          # 透视 (No perspective)
        flipud=0.0,               # 上下翻转 (No upside-down flip, 船不会倒过来)
        fliplr=0.5,               # 左右翻转 (Horizontal flip is good, 常用增强)
        mosaic=1.0,               # 马赛克增强 (Keep mosaic for small object context, 对小目标有益)
        mixup=0.1,                # Mixup 增强 (Slight mixup to handle overlap, 处理遮挡)
        copy_paste=0.1,           # Copy-paste 增强 (增加实例密度)
        
        # --- Loss Tuning (损失函数调整) ---
        box=7.5,                  # 边框回归损失权重 (Higher box loss gain, 侧重定位准确度)
        cls=0.5,                  # 类别损失权重 (Standard cls gain)
        dfl=1.5,                  # 分布焦点损失权重 (Distribution Focal Loss gain)
        
        exist_ok=True             # 允许覆盖同名实验目录
    )
    print("Training complete. Best model saved to runs/train/yolov12n_ultimate/weights/best.pt (训练完成，最佳模型已保存)")

if __name__ == "__main__":
    main()
