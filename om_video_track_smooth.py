"""
YOLOv12 Ascend NPU 视频推理与跟踪脚本
=========================================
功能描述：
1. 加载华为 Ascend OM (Offline Model) 离线模型。
2. 使用 ACL (Ascend Computing Language) 接口进行模型推理。
3. 实现视频帧的预处理 (Letterbox, Normalization)。
4. 实现后处理 (NMS, 坐标还原)。
5. 集成卡尔曼滤波 (Kalman Filter) 进行目标跟踪和平滑。
6. 输出带有检测框和跟踪 ID 的结果视频。

主要类与模块：
- ModelRunner: 封装 ACL 推理流程（内存分配、H2D/D2H 拷贝、执行）。
- KalmanBoxTracker: 单个目标的卡尔曼滤波器状态管理。
- Tracker: 多目标跟踪器，负责目标匹配、生命周期管理。
- preprocess_frame / postprocess: 数据预处理与后处理。

"""

import os
import sys
import time
import acl
import cv2
import numpy as np

# --- 配置部分 ---
OM_MODEL_PATH = "bestmax.om"  # OM 模型文件路径
VIDEO_PATH = "data/test2.mp4"  # 输入视频文件路径
OUTPUT_VIDEO = "runs/test2_om.mp4" # 输出视频文件路径
DEVICE_ID = 0 # 计算设备 ID (NPU)
CLASS_NAMES = ['kayak', 'fishing boat'] # 类别名称列表

# 默认输入分辨率，稍后会根据模型信息自动调整
INPUT_HEIGHT = 960
INPUT_WIDTH = 960

# 跟踪器与后处理参数设置
CONF_THRES = 0.30
IOU_THRES = 0.2        # NMS 阈值
TRACK_IOU_THRES = 0.15 # 跟踪匹配阈值
TRACK_ALPHA = 0.4      # 平滑系数
TRACK_MIN_VISIBLE = 5  # 最小显示帧数
TRACK_HOLD_FRAMES = 8
TRACK_BUFFER_SIZE = 12 # 轨迹平滑窗口大小
MIN_AREA_RATIO = 0.00025
MAX_AREA_RATIO = 0.6   # 最大面积比：设置保守阈值 (60%)，仅过滤超大特写误检
MAX_RATIO = 7.0
SMALL_AREA_RATIO = 0.001
CONF_THRES_SMALL = 0.18
NEAR_LOCK_RATIO = 0.015
FAR_LOCK_RATIO = 0.0008
FAR_CENTER_STEP_PX = 3
FAR_SIZE_DELTA_LIMIT = 0.04
FAR_SIZE_DEADZONE_PX = 2

# --- ACL 辅助函数 ---
# 这一部分封装了华为 Ascend ACL (Ascend Computing Language) 的底层 C 接口
# 用于设备初始化、模型加载、内存管理等

# ACL 全局状态变量
_inited = False   # ACL 是否已初始化
_model_id = None  # 加载的模型 ID
_ctx = None       # ACL 上下文 (Context)

def _looks_like_ret_code(x):
    # 判断返回值是否像错误码
    if not isinstance(x, int):
        return False
    if x < 0:
        return True
    return x <= 65535

def _split_ret_value(res):
    # 处理 ACL 接口返回的 (ret_code, value) 元组
    # 背景：不同的 Python ACL 绑定版本返回格式可能不同（有的返回单值，有的返回元组）
    # 此函数尝试智能解析返回值，分离出 错误码 (ret) 和 实际结果 (value)
    if isinstance(res, int):
        return 0, res
    if isinstance(res, (tuple, list)):
        if len(res) == 2:
            a, b = res
            if isinstance(a, int) and isinstance(b, int):
                if a == 0 and b != 0:
                    return 0, b
                if b == 0 and a != 0:
                    return 0, a
                if _looks_like_ret_code(a) and not _looks_like_ret_code(b):
                    return a, b
                if _looks_like_ret_code(b) and not _looks_like_ret_code(a):
                    return b, a
                return a, b
            if isinstance(a, int) and not isinstance(b, int):
                return a, b
            if isinstance(b, int) and not isinstance(a, int):
                return b, a
            return 0, b

        ints = [x for x in res if isinstance(x, int)]
        if 0 in ints:
            ret = 0
        else:
            candidates = [x for x in ints if _looks_like_ret_code(x)]
            ret = candidates[0] if candidates else (ints[0] if ints else 0)
        val = None
        for x in res:
            if x is ret:
                continue
            if not isinstance(x, int):
                val = x
                break
        if val is None:
            for x in res:
                if isinstance(x, int) and not _looks_like_ret_code(x):
                    val = x
                    break
        if val is None and len(res) > 0:
            val = res[0]
        return ret, val
    return 0, res

def _ret_code(res):
    # 仅提取错误码
    if res is None:
        return 0
    if isinstance(res, int):
        return res
    if isinstance(res, (tuple, list)):
        ints = [x for x in res if isinstance(x, int)]
        if 0 in ints:
            return 0
        candidates = [x for x in ints if _looks_like_ret_code(x)]
        if candidates:
            return candidates[0]
        if ints:
            return ints[0]
    return 0

def acl_init():
    # 初始化 ACL 环境
    global _inited
    if _inited:
        return
    ret = acl.init()
    if ret != 0:
        raise RuntimeError(f"acl.init failed: {ret}")
    ret = acl.rt.set_device(DEVICE_ID)
    if ret != 0:
        raise RuntimeError(f"acl.rt.set_device failed: {ret}")
    if hasattr(acl.rt, "create_context"):
        ret, ctx = _split_ret_value(acl.rt.create_context(DEVICE_ID))
        if ret != 0:
            raise RuntimeError(f"acl.rt.create_context failed: {ret}")
        if hasattr(acl.rt, "set_context"):
            ret = acl.rt.set_context(ctx)
            if ret != 0:
                raise RuntimeError(f"acl.rt.set_context failed: {ret}")
        global _ctx
        _ctx = ctx
    _inited = True

def acl_release():
    # 释放 ACL 资源
    global _inited, _ctx
    if not _inited:
        return
    try:
        if _ctx is not None and hasattr(acl.rt, "destroy_context"):
            try:
                acl.rt.destroy_context(_ctx)
            except Exception:
                pass
            _ctx = None
        try:
            acl.rt.reset_device(DEVICE_ID)
        except TypeError:
            acl.rt.reset_device()
    finally:
        try:
            acl.finalize()
        finally:
            _inited = False

def load_model(model_path):
    # 加载离线模型 (.om)
    path = os.path.abspath(model_path)
    ret, model_id = _split_ret_value(acl.mdl.load_from_file(path))
    if ret != 0:
        raise RuntimeError(f"acl.mdl.load_from_file failed: {ret}, path: {path}")
    return model_id

def _get_attr_callable(objs, names):
    # 辅助函数：动态获取对象属性
    for obj in objs:
        if obj is None:
            continue
        for name in names:
            fn = getattr(obj, name, None)
            if callable(fn):
                return fn
    return None

def _mdl_create_data_buffer(ptr, size):
    # 创建模型数据缓冲区
    fn = _get_attr_callable(
        [getattr(acl, "mdl", None), acl],
        [
            "create_data_buffer",
            "create_databuffer",
            "create_data_buf",
            "create_buffer",
        ],
    )
    if fn is None:
        raise RuntimeError("create_data_buffer api not found")
    res = fn(ptr, size)
    ret, buf = _split_ret_value(res)
    if ret != 0:
        raise RuntimeError(f"create_data_buffer failed: {ret}")
    return buf

def _mdl_destroy_data_buffer(buf):
    # 销毁模型数据缓冲区
    fn = _get_attr_callable(
        [getattr(acl, "mdl", None), acl],
        [
            "destroy_data_buffer",
            "destroy_databuffer",
            "destroy_data_buf",
            "destroy_buffer",
        ],
    )
    if fn is None:
        return
    try:
        fn(buf)
    except Exception:
        pass

def _mdl_add_dataset_buffer(dataset, buf):
    # 向数据集添加缓冲区
    fn = _get_attr_callable(
        [getattr(acl, "mdl", None), acl],
        [
            "add_dataset_buffer",
            "add_buffer_to_dataset",
            "dataset_add_buffer",
        ],
    )
    if fn is None:
        raise RuntimeError("add_dataset_buffer api not found")
    res = fn(dataset, buf)
    return _ret_code(res)

def _mdl_destroy_dataset(dataset):
    # 销毁数据集
    fn = _get_attr_callable(
        [getattr(acl, "mdl", None), acl],
        [
            "destroy_dataset",
            "destroy_data_set",
        ],
    )
    if fn is None:
        return
    try:
        fn(dataset)
    except Exception:
        pass

def _malloc_device(size):
    # 在 NPU 设备上申请内存
    ret, ptr = _split_ret_value(acl.rt.malloc(size, 0))
    if ret != 0:
        raise RuntimeError(f"acl.rt.malloc failed: {ret}, size: {size}")
    return ptr

def _free_device(ptr):
    # 释放 NPU 设备内存
    try:
        acl.rt.free(ptr)
    except Exception:
        pass

def _get_memcpy_kind(name, fallback):
    # 获取内存复制类型枚举值
    for obj in (acl, getattr(acl, "rt", None)):
        if obj is None:
            continue
        for k in (name, name.upper(), f"ACL_{name.upper()}", f"ACL_MEMCPY_{name.upper()}"):
            v = getattr(obj, k, None)
            if isinstance(v, int):
                return v
    return fallback

class ModelRunner:
    # 模型运行器类，封装了模型的输入输出管理和推理执行
    def __init__(self, model_id):
        self.model_id = model_id
        self.desc = None
        self.input_size = None
        self.output_sizes = None
        self.input_dev = None
        self.output_devs = None
        self.input_ds = None
        self.output_ds = None
        self.input_buf = None
        self.output_bufs = None
        # 获取内存拷贝类型常量
        self.kind_h2d = _get_memcpy_kind("memcpy_host_to_device", 1)
        self.kind_d2h = _get_memcpy_kind("memcpy_device_to_host", 2)

    def open(self):
        # 打开模型，获取描述信息，分配内存
        ret, desc = _split_ret_value(acl.mdl.create_desc())
        if ret != 0:
            raise RuntimeError(f"acl.mdl.create_desc failed: {ret}")
        self.desc = desc

        ret = _ret_code(acl.mdl.get_desc(self.desc, self.model_id))
        if ret != 0:
            raise RuntimeError(f"acl.mdl.get_desc failed: {ret}")

        # 获取输入大小
        self.input_size = acl.mdl.get_input_size_by_index(self.desc, 0)

        # 获取输出数量
        out_count = 1
        if hasattr(acl.mdl, "get_num_outputs"):
            try:
                out_count = int(acl.mdl.get_num_outputs(self.desc))
            except Exception:
                out_count = 1

        # 获取各输出的大小
        self.output_sizes = []
        for i in range(out_count):
            self.output_sizes.append(int(acl.mdl.get_output_size_by_index(self.desc, i)))

        # 申请设备端内存
        self.input_dev = _malloc_device(self.input_size)
        self.output_devs = [_malloc_device(s) for s in self.output_sizes]

        # 创建数据集结构
        ret, ds = _split_ret_value(acl.mdl.create_dataset())
        if ret != 0:
            raise RuntimeError(f"create_dataset(input) failed: {ret}")
        self.input_ds = ds

        ret, ds = _split_ret_value(acl.mdl.create_dataset())
        if ret != 0:
            raise RuntimeError(f"create_dataset(output) failed: {ret}")
        self.output_ds = ds

        # 绑定缓冲区到数据集
        self.input_buf = _mdl_create_data_buffer(self.input_dev, self.input_size)
        ret = _mdl_add_dataset_buffer(self.input_ds, self.input_buf)
        if ret != 0:
            raise RuntimeError(f"add input buffer failed: {ret}")

        self.output_bufs = []
        for dev, size in zip(self.output_devs, self.output_sizes):
            b = _mdl_create_data_buffer(dev, size)
            self.output_bufs.append(b)
            ret = _mdl_add_dataset_buffer(self.output_ds, b)
            if ret != 0:
                raise RuntimeError(f"add output buffer failed: {ret}")

    def close(self):
        # 释放所有申请的资源
        if self.input_buf is not None:
            _mdl_destroy_data_buffer(self.input_buf)
            self.input_buf = None
        if self.output_bufs is not None:
            for b in self.output_bufs:
                _mdl_destroy_data_buffer(b)
            self.output_bufs = None
        if self.input_ds is not None:
            _mdl_destroy_dataset(self.input_ds)
            self.input_ds = None
        if self.output_ds is not None:
            _mdl_destroy_dataset(self.output_ds)
            self.output_ds = None
        if self.input_dev is not None:
            _free_device(self.input_dev)
            self.input_dev = None
        if self.output_devs is not None:
            for dev in self.output_devs:
                _free_device(dev)
            self.output_devs = None
        if self.desc is not None:
            try:
                acl.mdl.destroy_desc(self.desc)
            except Exception:
                pass
            self.desc = None

    def infer(self, input_data):
        # 执行模型推理
        # 0. 确保输入数据在内存中是连续的，这是 C 接口调用的前提
        input_data = np.ascontiguousarray(input_data)
        if input_data.nbytes != self.input_size:
            raise RuntimeError(f"input bytes mismatch: got {input_data.nbytes}, expect {self.input_size}")
        
        # 1. 将数据从 Host (CPU) 复制到 Device (NPU)
        # ACL_MEMCPY_HOST_TO_DEVICE = 1
        ret = _ret_code(
            acl.rt.memcpy(
                self.input_dev,
                self.input_size,
                input_data.ctypes.data,
                self.input_size,
                self.kind_h2d,
            )
        )
        if ret != 0:
            raise RuntimeError(f"memcpy H2D failed: {ret}")

        # 2. 执行推理
        # 模型执行是异步的，但 python 接口通常会阻塞直到完成（取决于具体实现）
        ret = _ret_code(acl.mdl.execute(self.model_id, self.input_ds, self.output_ds))
        if ret != 0:
            raise RuntimeError(f"acl.mdl.execute(dataset) failed: {ret}")

        # 3. 将结果从 Device (NPU) 复制回 Host (CPU)
        # 遍历所有输出张量，分别拷贝回 Host 内存
        outputs = []
        for dev, size in zip(self.output_devs, self.output_sizes):
            host = np.empty((size,), dtype=np.uint8)
            # ACL_MEMCPY_DEVICE_TO_HOST = 2
            ret = _ret_code(acl.rt.memcpy(host.ctypes.data, size, dev, size, self.kind_d2h))
            if ret != 0:
                raise RuntimeError(f"memcpy D2H failed: {ret}")
            outputs.append(host)
        return outputs

# --- 逻辑部分：跟踪与平滑 ---

def smooth_box(prev_box, box, alpha):
    # 简单指数平滑 (已废弃，保留作为参考)
    if prev_box is None:
        return box
    return alpha * box + (1.0 - alpha) * prev_box

def iou_xyxy(box1, box2):
    # 计算两个框的 IoU (Intersection over Union)
    # box: [x1, y1, x2, y2]
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area1 = max(0, x12 - x11) * max(0, y12 - y11)
    area2 = max(0, x22 - x21) * max(0, y22 - y21)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union

class KalmanBoxTracker:
    # 卡尔曼滤波器类：用于状态估计和平滑
    def __init__(self, box):
        # box: [cx, cy, w, h] (中心坐标 x, y 和 宽, 高)
        # 状态向量: [x, y, w, h, vx, vy, vw] (位置 + 速度)
        # 测量向量: [x, y, w, h] (观测到的位置)
        self.kf = cv2.KalmanFilter(7, 4)
        
        # 状态转移矩阵 (F)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 测量矩阵 (H)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差 (Q) - 小噪声表示模型预测较准，能产生更平滑的轨迹
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        self.kf.processNoiseCov[4:, 4:] *= 0.01 # 速度分量的噪声更小
        
        # 测量噪声协方差 (R) - 增大 R 值以更信任预测而非测量，从而平滑抖动
        # 之前是 0.1，现在设为 1.0 (远点抖动通常因为测量值不稳定)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # 误差协方差 (P)
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        
        # 初始化状态
        self.kf.statePost = np.array([box[0], box[1], box[2], box[3], 0, 0, 0], dtype=np.float32).reshape(7, 1)
        
    def update(self, box):
        # 更新步骤：用观测值修正预测值
        # box: [cx, cy, w, h]
        meas = np.array(box, dtype=np.float32).reshape(4, 1)
        self.kf.correct(meas)
        
    def predict(self):
        # 预测步骤：根据当前状态和速度预测下一帧位置
        # 返回 [cx, cy, w, h]
        pred = self.kf.predict()
        return pred[:4].flatten()

class Tracker:
    # 跟踪器类：管理所有目标的生命周期
    def __init__(self):
        self.tracks = {}  # 存储所有轨迹，key为ID
        # 轨迹数据结构: {'kf': 滤波器, 'seen': 出现次数, 'miss': 消失次数, 'label_scores': 分类得分, 'box_hist': 历史框, 'box': 当前框}
        self.next_id = 1
    
    def update(self, detections, frame_width, frame_height):
        # detections: [(box, score, class_id), ...] box为xyxy格式
        
        updated_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        matches = []  # 匹配对 (track_id, det_idx)
        to_remove = [] # 初始化待删除列表

        # --- 匹配逻辑 ---
        for tid in unmatched_tracks:
            # 1. 使用卡尔曼滤波预测当前帧的位置
            pred_box = self.tracks[tid]['kf'].predict()
            self.tracks[tid]['box'] = pred_box # 更新当前估计框
            
            t_box = pred_box # cx, cy, w, h
            # 将预测框转换为 xyxy 以计算 IoU
            tx1 = t_box[0] - t_box[2]/2
            ty1 = t_box[1] - t_box[3]/2
            tx2 = t_box[0] + t_box[2]/2
            ty2 = t_box[1] + t_box[3]/2
            t_xyxy = (tx1, ty1, tx2, ty2)
            
            best_iou = -1
            best_idx = -1
            
            # 2. IoU 匹配 (首选策略)
            for idx in unmatched_dets:
                d_xyxy = detections[idx][0]
                iou = iou_xyxy(t_xyxy, d_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou > TRACK_IOU_THRES:
                matches.append((tid, best_idx))
                unmatched_dets.remove(best_idx)
                continue
                
            # 3. 距离匹配 (备选策略，当 IoU 为 0 但距离很近时)
            best_dist = float('inf')
            best_dist_idx = -1
            tcx, tcy = t_box[0], t_box[1]
            
            for idx in unmatched_dets:
                d_xyxy = detections[idx][0]
                d_cx = (d_xyxy[0] + d_xyxy[2]) / 2
                d_cy = (d_xyxy[1] + d_xyxy[3]) / 2
                dist = (tcx - d_cx)**2 + (tcy - d_cy)**2
                if dist < best_dist:
                    best_dist = dist
                    best_dist_idx = idx
            
            # 距离阈值：最大边长的 2 倍
            limit_dist = max(t_box[2], t_box[3]) * 2.0
            if best_dist < limit_dist**2:
                matches.append((tid, best_dist_idx))
                unmatched_dets.remove(best_dist_idx)
        
        # --- 更新匹配成功的轨迹 ---
        for tid, idx in matches:
            box, score, cls_id = detections[idx]
            # 转换检测框为 cx, cy, w, h
            x1, y1, x2, y2 = box
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1, y2-y1
            cur_vec = np.array([cx, cy, w, h], dtype=np.float32)
            
            # 更新卡尔曼滤波器
            self.tracks[tid]['kf'].update(cur_vec)
            # 使用 KF 的后验状态作为平滑后的框
            smoothed = self.tracks[tid]['kf'].kf.statePost[:4].flatten()
            self.tracks[tid]['box'] = smoothed
            
            # 更新显示历史
            self.tracks[tid]['box_hist'].append(smoothed.copy())
            if len(self.tracks[tid]['box_hist']) > TRACK_BUFFER_SIZE:
                self.tracks[tid]['box_hist'].pop(0)

            self.tracks[tid]['seen'] += 1
            self.tracks[tid]['miss'] = 0
            
            # 更新类别分数 (加权投票机制)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            
            # 偏置逻辑：根据经验调整特定类别的权重
            weight = float(score)
            if label == 'fishing boat':
                weight *= 0.5 # 抑制 fishing boat (误检较多)
            elif label == 'kayak':
                weight *= 1.2 # 提升 kayak
            
            scores = self.tracks[tid]['label_scores']
            scores[label] = scores.get(label, 0.0) + weight
            
            if tid in unmatched_tracks:
                 unmatched_tracks.remove(tid)
            
        # --- 创建新轨迹 ---
        for idx in unmatched_dets:
            box, score, cls_id = detections[idx]
            x1, y1, x2, y2 = box
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1, y2-y1
            vec = np.array([cx, cy, w, h], dtype=np.float32)
            
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            
            self.tracks[self.next_id] = {
                'kf': KalmanBoxTracker(vec),
                'box': vec,
                'seen': 1,
                'miss': 0,
                'label_scores': {label: float(score)},
                'box_hist': [vec]
            }
            self.next_id += 1
            
        # --- 处理丢失的轨迹 ---
        for tid in unmatched_tracks:
            self.tracks[tid]['miss'] += 1
            if self.tracks[tid]['miss'] > TRACK_HOLD_FRAMES:
                del self.tracks[tid]
                
        # --- 清理重复轨迹 (Track Suppression) ---
        # 如果两条轨迹空间距离太近，保留历史更长或更可靠的一条
        active_tids = list(self.tracks.keys())
        # to_remove = [] # 已经在函数开头初始化，这里不再重置，保留之前静态物体过滤的结果
        for i in range(len(active_tids)):
            tid_i = active_tids[i]
            if tid_i in to_remove: continue
            
            box_i = self.tracks[tid_i]['box']
            xyxy_i = (box_i[0]-box_i[2]/2, box_i[1]-box_i[3]/2, box_i[0]+box_i[2]/2, box_i[1]+box_i[3]/2)
            
            for j in range(i + 1, len(active_tids)):
                tid_j = active_tids[j]
                if tid_j in to_remove: continue
                
                box_j = self.tracks[tid_j]['box']
                xyxy_j = (box_j[0]-box_j[2]/2, box_j[1]-box_j[3]/2, box_j[0]+box_j[2]/2, box_j[1]+box_j[3]/2)
                
                iou = iou_xyxy(xyxy_i, xyxy_j)
                if iou > 0.2: # 激进的重复轨迹阈值 (20% 重叠即视为重复)
                    # 保留 'seen' 次数更多的那条轨迹
                    seen_i = self.tracks[tid_i]['seen']
                    seen_j = self.tracks[tid_j]['seen']
                    if seen_i >= seen_j:
                        to_remove.append(tid_j)
                    else:
                        to_remove.append(tid_i)
                        break # tid_i 被删除，停止检查它
        
        for tid in to_remove:
            if tid in self.tracks:
                del self.tracks[tid]

        # --- 准备输出 ---
        draw_items = []
        for tid, info in self.tracks.items():
            if info['seen'] >= TRACK_MIN_VISIBLE and info['miss'] == 0:
                box = info['box']
                hist = info['box_hist']
                area_ratio = (box[2] * box[3]) / float(frame_width * frame_height) if frame_width > 0 and frame_height > 0 else 0.0
                if area_ratio >= NEAR_LOCK_RATIO:
                    cx, cy, w, h = box
                elif area_ratio <= FAR_LOCK_RATIO:
                    if len(hist) > 0:
                        med_vec = np.median(np.stack(hist, axis=0), axis=0)
                        cx, cy, w, h = med_vec
                    else:
                        cx, cy, w, h = box
                else:
                    if len(hist) > 0:
                        avg_vec = np.mean(hist, axis=0)
                        cx, cy, w, h = avg_vec
                    else:
                        cx, cy, w, h = box

                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
                
                # 裁剪到图像边界
                x1 = max(0, min(x1, frame_width-1))
                y1 = max(0, min(y1, frame_height-1))
                x2 = max(0, min(x2, frame_width-1))
                y2 = max(0, min(y2, frame_height-1))
                
                # 确定最终类别 (取累计得分最高的)
                scores = info['label_scores']
                best_label = max(scores, key=scores.get)

                draw_items.append((tid, x1, y1, x2, y2, best_label))
                
        return draw_items

# --- 后处理函数 ---

def xywh2xyxy(x):
    # 将 nx4 的 [cx, cy, w, h] 转换为 [x1, y1, x2, y2]
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def nms_numpy(boxes, scores, iou_thres):
    # 手写 NMS (Non-Maximum Suppression) 实现
    # boxes: [N, 4] xyxy
    # scores: [N]
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1] # 按置信度降序排列
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算当前框与其他框的重叠区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留 IoU 小于阈值的框
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
        
    return keep

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # 将 Letterbox 处理后的坐标还原回原始图像坐标
    if ratio_pad is None:  # 如果没有提供比例，重新计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # 减去 x padding
    coords[:, [1, 3]] -= pad[1]  # 减去 y padding
    coords[:, :4] /= gain        # 除以缩放比例
    
    clip_coords(coords, img0_shape) # 裁剪防止越界
    return coords

def clip_coords(boxes, shape):
    # 将坐标限制在图像范围内
    boxes[:, 0].clip(0, shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, shape[0], out=boxes[:, 3])  # y2

def postprocess(outputs, img_w, img_h, conf_thres, iou_thres, ratio=None, pad=None, min_area_ratio=0.0):
    # 后处理主函数：解析模型输出，过滤，NMS，坐标还原
    # outputs: ACL 推理的输出列表
    
    if not outputs:
        return []
        
    out = outputs[0]
    # 重新解释为 float32
    out = out.view(dtype=np.float32)
    
    nc = len(CLASS_NAMES)
    ch = 4 + nc
    anchors = out.size // ch
    
    # 调整形状为 (channels, anchors)
    out = out.reshape((ch, anchors))
    
    # 转置为 (anchors, channels) -> (N, 6)
    out = out.transpose()
    
    # 提取坐标和分数
    boxes_cxcywh = out[:, 0:4]
    class_scores = out[:, 4:]
    
    # 获取最大类分数和类别索引
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    
    # 1. 置信度过滤（面积自适应）
    areas_all = boxes_cxcywh[:, 2] * boxes_cxcywh[:, 3]
    area_ratio_all = areas_all / float(INPUT_WIDTH * INPUT_HEIGHT)
    thresh_all = np.where(area_ratio_all < SMALL_AREA_RATIO, CONF_THRES_SMALL, conf_thres)
    mask = confidences > thresh_all
    
    boxes_cxcywh = boxes_cxcywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes_cxcywh) == 0:
        return []

    # 2. 面积过滤 (过滤过小的误检目标，如船桨、石头)
    if min_area_ratio > 0:
        # 计算每个框的面积 (w * h)
        # 注意：boxes_cxcywh 是基于模型输入尺寸 (INPUT_WIDTH, INPUT_HEIGHT) 的
        areas = boxes_cxcywh[:, 2] * boxes_cxcywh[:, 3]
        min_area = INPUT_WIDTH * INPUT_HEIGHT * min_area_ratio
        max_area = INPUT_WIDTH * INPUT_HEIGHT * MAX_AREA_RATIO
        
        # 计算长宽比
        ws = boxes_cxcywh[:, 2]
        hs = boxes_cxcywh[:, 3]
        ratios1 = ws / (hs + 1e-6)
        ratios2 = hs / (ws + 1e-6)
        
        # 综合过滤：
        # 1. 面积 > 最小阈值 (过滤噪点)
        # 2. 面积 < 最大阈值 (过滤特写/巨大误检)
        # 3. 长宽比 < 最大比例 (过滤细长船桨/波纹)
        valid_mask = (areas > min_area) & (areas < max_area) & (ratios1 < MAX_RATIO) & (ratios2 < MAX_RATIO)
        
        boxes_cxcywh = boxes_cxcywh[valid_mask]
        confidences = confidences[valid_mask]
        class_ids = class_ids[valid_mask]

        if len(boxes_cxcywh) == 0:
            return []
        
    # 转换坐标格式
    boxes_xyxy = xywh2xyxy(boxes_cxcywh)
    
    # 将坐标还原到原图尺寸
    boxes_xyxy = scale_coords((INPUT_HEIGHT, INPUT_WIDTH), boxes_xyxy, (img_h, img_w), ratio_pad=(ratio, pad))
    
    # 执行 NMS 去重
    keep = nms_numpy(boxes_xyxy, confidences, iou_thres)
    
    results = []
    for i in keep:
        results.append((boxes_xyxy[i], confidences[i], class_ids[i]))
        
    return results

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # 图像预处理：保持长宽比缩放并填充 padding
    shape = img.shape[:2]  # 当前形状 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 仅缩小，不放大 (如果原图比目标小，保持原图大小以提高精度)
        r = min(r, 1.0)

    # 计算 padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # 最小矩形填充 (32倍数)
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:  # 拉伸填充 (不保持长宽比)
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # padding 分到两边
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 添加边框
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def preprocess_frame(frame):
    # 帧预处理：Letterbox -> RGB -> 归一化 -> NCHW
    img_letterboxed, ratio, pad = letterbox(frame, new_shape=(INPUT_WIDTH, INPUT_HEIGHT), auto=False, scaleup=True)
    
    img_rgb = cv2.cvtColor(img_letterboxed, cv2.COLOR_BGR2RGB)
    img_data = img_rgb.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1)) # HWC -> CHW
    img_data = np.expand_dims(img_data, axis=0)  # 添加 Batch 维度
    return img_data, ratio, pad

def main():
    global INPUT_WIDTH, INPUT_HEIGHT
    os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
    
    # 1. 初始化 ACL
    acl_init()
    model_id = None
    cap = None
    writer = None
    runner = None
    
    try:
        # 2. 加载模型
        if not os.path.exists(OM_MODEL_PATH):
            print(f"Error: Model file {OM_MODEL_PATH} not found.")
            return

        model_id = load_model(OM_MODEL_PATH)
        runner = ModelRunner(model_id)
        runner.open()
        
        # 3. 检查模型输入尺寸并自动调整
        input_size = runner.input_size
        print(f"Model Input Size: {input_size} bytes")
        # 960*960*3*4 = 11,059,200
        # 640*640*3*4 = 4,915,200
        if input_size == 11059200:
            INPUT_WIDTH = 960
            INPUT_HEIGHT = 960
            print("Detected 960x960 model input")
        elif input_size == 4915200:
            INPUT_WIDTH = 640
            INPUT_HEIGHT = 640
            print("Detected 640x640 model input")
        else:
            print(f"Warning: Unknown input size {input_size}, using default {INPUT_WIDTH}x{INPUT_HEIGHT}")
            
        # 4. 打开视频源
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"open video failed: {VIDEO_PATH}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 1.0:
            fps = 30.0
            
        print(f"Video Source: {width}x{height} @ {fps} fps")
            
        # 5. 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Failed to open VideoWriter with mp4v. Trying avc1...")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
            if not writer.isOpened():
                 raise RuntimeError(f"open VideoWriter failed: {OUTPUT_VIDEO}")
            
        print("model:", os.path.abspath(OM_MODEL_PATH))
        print("video:", os.path.abspath(VIDEO_PATH))
        print("save :", os.path.abspath(OUTPUT_VIDEO))
        
        tracker = Tracker()
        frame_idx = 0
        prev_gray = None
        no_det_frames = 0
        scene_cut_thresh = 25.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            # 场景切换检测
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, frame_gray)
                if diff.mean() > scene_cut_thresh:
                    tracker.tracks.clear()
                    tracker.next_id = 1
                    no_det_frames = 0
            prev_gray = frame_gray
            
            # 6. 预处理 (Letterbox)
            inp, ratio, pad = preprocess_frame(frame)
            
            # 7. 推理 (Inference)
            # 调试输入信息，防止字节数不匹配错误
            if frame_idx == 1:
                print(f"Inference Input: shape={inp.shape}, dtype={inp.dtype}, nbytes={inp.nbytes}")

            outs = runner.infer(inp)
            
            # 调试第一帧输出
            if frame_idx == 1:
                 for i, o in enumerate(outs):
                     print(f"Output {i} shape: {o.shape}, dtype: {o.dtype}, min: {o.min()}, max: {o.max()}")

            # 8. 后处理 (解析、过滤、NMS、坐标映射)
            detections = postprocess(outs, width, height, CONF_THRES, IOU_THRES, ratio, pad, min_area_ratio=MIN_AREA_RATIO)
            
            if len(detections) == 0:
                no_det_frames += 1
                if no_det_frames >= 2:
                    tracker.tracks.clear()
                    tracker.next_id = 1
            else:
                no_det_frames = 0
            
            # 9. 跟踪更新 (卡尔曼滤波、匹配、平滑)
            draw_items = tracker.update(detections, width, height)
            
            # 10. 绘图前的二次 NMS (去重策略)
            # 为了确保画出的框没有重叠，再次进行严格筛选
            draw_items.sort(key=lambda x: x[0]) # 按 ID 排序确保稳定
            
            kept_draw_items = []
            suppressed = [False] * len(draw_items)
            
            for i in range(len(draw_items)):
                if suppressed[i]:
                    continue
                
                tid_i, x1_i, y1_i, x2_i, y2_i, label_i = draw_items[i]
                box_i = (x1_i, y1_i, x2_i, y2_i)
                
                kept_draw_items.append(draw_items[i])
                
                for j in range(i + 1, len(draw_items)):
                    if suppressed[j]:
                        continue
                    
                    tid_j, x1_j, y1_j, x2_j, y2_j, label_j = draw_items[j]
                    box_j = (x1_j, y1_j, x2_j, y2_j)
                    
                    iou = iou_xyxy(box_i, box_j)
                    # 严格 NMS：如果绘图框重叠超过 20%，抑制 ID 靠后的框
                    if iou > 0.2:
                        suppressed[j] = True
                        continue # 已经被抑制，跳过后续检查

                    # 新增：距离抑制 (Distance Suppression)
                    # 解决远点重复框问题：如果两个小目标中心距离极近 (< 20px)，视为重复
                    
                    # 计算中心点距离平方
                    cx_i, cy_i = (box_i[0]+box_i[2])/2, (box_i[1]+box_i[3])/2
                    cx_j, cy_j = (box_j[0]+box_j[2])/2, (box_j[1]+box_j[3])/2
                    dist_sq = (cx_i - cx_j)**2 + (cy_i - cy_j)**2
                    
                    if dist_sq < 400: # 20*20 = 400
                        suppressed[j] = True
                        continue

                    # 新增：包含关系抑制 (Small-in-Large Suppression)
                    # 解决船桨/手 (box_j) 被检测在 船 (box_i) 内部的情况
                    # 仅当 box_j 面积明显小于 box_i 时生效
                    
                    area_i = (box_i[2]-box_i[0]) * (box_i[3]-box_i[1])
                    area_j = (box_j[2]-box_j[0]) * (box_j[3]-box_j[1])
                    
                    if area_j < area_i * 0.5: # j 比 i 小一半以上
                        # 计算交集
                        xx1 = max(box_i[0], box_j[0])
                        yy1 = max(box_i[1], box_j[1])
                        xx2 = min(box_i[2], box_j[2])
                        yy2 = min(box_i[3], box_j[3])
                        w_inter = max(0, xx2 - xx1)
                        h_inter = max(0, yy2 - yy1)
                        inter_area = w_inter * h_inter
                        
                        # 如果 j 70% 以上的面积在 i 内部
                        if area_j > 0 and (inter_area / area_j > 0.7):
                            suppressed[j] = True
            
            # 11. 绘制结果
            for tid, x1, y1, x2, y2, label in kept_draw_items:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label} ID:{tid}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            
            writer.write(frame)
            
            if frame_idx % 30 == 0:
                print(f"processed frames: {frame_idx} | tracks: {len(draw_items)}", end='\r')
                
        print(f"\nDone, total frames: {frame_idx}")
        writer.release()
        writer = None
        
        if os.path.exists(OUTPUT_VIDEO):
            print(f"Video saved successfully size: {os.path.getsize(OUTPUT_VIDEO)} bytes")
        else:
            print(f"Error: Video file not found at {OUTPUT_VIDEO}")

    finally:
        # 12. 资源清理
        try:
            if runner is not None:
                runner.close()
        except Exception:
            pass
        try:
            if writer is not None:
                writer.release()
        finally:
            try:
                if cap is not None:
                    cap.release()
            finally:
                if model_id is not None:
                    try:
                        acl.mdl.unload(model_id)
                    except Exception:
                        pass
                acl_release()

if __name__ == "__main__":
    main()
