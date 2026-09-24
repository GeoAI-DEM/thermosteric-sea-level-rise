#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改进版双GPU海洋热膨胀计算器 - 性能优化版
主要优化内容：
1. 深度层级并行化：多深度层同时处理（由串行改为并行）
2. 优化GPU插值算法，减少内存传输
3. 智能内存管理：按需清理，减少频繁清理导致的性能损失
4. 精确进度和时间预测：添加每层估计时间和整体完成时间预测
5. 优化I/O操作：异步写入结果，减少I/O等待
6. 动态负载均衡：更智能的GPU任务分配
7. 计算流水线优化：减少数据等待时间
8. 支持与testnew.py一致的小岛屿填充参数，先填充温盐小空洞再计算
9. 支持最终热膨胀空间分布JPG和NetCDF输出
"""

import netCDF4 as nc
import numpy as np
from osgeo import gdal
import os
import logging
import gsw
import gc
import time
import psutil
from tqdm import tqdm
import pandas as pd
import json
import warnings
import threading
import argparse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

# PyTorch imports
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.multiprocessing as mp

# Numba imports (保留但不用于深层计算)
from numba import jit, prange

# 地图输出imports - 设置无头模式
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import zoom

# 设置高性能模式
torch.backends.cudnn.benchmark = True

# 全局抑制警告
warnings.filterwarnings("ignore")

# 配置主要日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_dual_gpu_calculator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def configure_logging_for_output(output_dir, log_filename='optimized_dual_gpu_calculator.log'):
    """将当前计算任务的日志写入指定输出目录。"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    logger.info(f"日志文件: {log_path}")
    return log_path


def fill_small_enclosed_islands(data, max_island_size=5, min_ocean_ratio=0.8):
    """
    填充被有效海洋数据包围且连通面积不超过max_island_size的NaN小空洞。
    该函数与testnew.py保持同名参数和8连通判定逻辑。
    """
    from scipy.ndimage import label

    data = np.ascontiguousarray(data)
    filled_data = data.copy()
    nan_mask = np.isnan(data)

    if not np.any(nan_mask):
        return filled_data

    labeled_islands, num_islands = label(nan_mask, structure=np.ones((3, 3)))

    for island_id in range(1, num_islands + 1):
        island_mask = (labeled_islands == island_id)
        island_size = np.sum(island_mask)

        if island_size > max_island_size:
            continue

        if is_surrounded_by_ocean(island_mask, data, min_ocean_ratio=min_ocean_ratio):
            filled_data = fill_island_region(filled_data, island_mask)

    return filled_data


def is_surrounded_by_ocean(island_mask, data, buffer_size=1, min_ocean_ratio=0.8):
    """检查NaN小空洞边界是否主要被有效海洋数据包围。"""
    from scipy.ndimage import binary_dilation

    expanded_mask = binary_dilation(island_mask, iterations=buffer_size)
    boundary_mask = expanded_mask & (~island_mask)
    boundary_data = data[boundary_mask]

    if len(boundary_data) == 0:
        return False

    valid_ocean_ratio = np.sum(~np.isnan(boundary_data)) / len(boundary_data)
    return valid_ocean_ratio >= min_ocean_ratio


def fill_island_region(data, island_mask):
    """使用周围有效海洋数据的距离加权平均填充小空洞。"""
    from scipy.spatial.distance import cdist

    filled_data = data.copy()
    expanded_mask = np.zeros_like(island_mask)
    rows, cols = np.where(island_mask)
    search_radius = 3

    for r, c in zip(rows, cols):
        r_min = max(0, r - search_radius)
        r_max = min(data.shape[0], r + search_radius + 1)
        c_min = max(0, c - search_radius)
        c_max = min(data.shape[1], c + search_radius + 1)
        expanded_mask[r_min:r_max, c_min:c_max] = True

    search_mask = expanded_mask & (~island_mask) & (~np.isnan(data))

    if not np.any(search_mask):
        fill_value = np.nanmean(data)
        if np.isnan(fill_value):
            fill_value = 0.0
        filled_data[island_mask] = fill_value
        return filled_data

    valid_coords = np.column_stack(np.where(search_mask))
    valid_values = data[search_mask]
    fill_coords = np.column_stack(np.where(island_mask))

    try:
        distances = cdist(fill_coords, valid_coords)

        for i, (fill_row, fill_col) in enumerate(fill_coords):
            nearest_indices = np.argsort(distances[i])[:4]
            nearest_distances = distances[i][nearest_indices]
            nearest_values = valid_values[nearest_indices]
            weights = 1.0 / (nearest_distances + 1e-8)
            weights = weights / np.sum(weights)
            filled_data[fill_row, fill_col] = np.sum(weights * nearest_values)

    except Exception:
        filled_data[island_mask] = np.mean(valid_values)

    return filled_data


# 保留Numba函数定义（兼容性）
@jit(nopython=True)
def get_deep_water_mass_properties(lat, lon):
    """保留但不使用"""
    return -0.5, 34.9

@jit(nopython=True)
def calculate_thermal_expansion_coefficient_deep(temperature, salinity, pressure_mpa):
    """保留但不使用"""
    alpha_0 = 2.1e-4
    temp_correction = 1.0 - 0.05 * (4.0 - temperature)**2 / 16.0
    salinity_correction = 1.0 - 0.03 * (salinity - 34.7) / 3.0
    pressure_correction = 1.0 / (1.0 + 0.015 * pressure_mpa)
    alpha = alpha_0 * temp_correction * salinity_correction * pressure_correction
    return max(alpha, 1e-6)

def normalize_longitude(lon):
    """将经度标准化到-180到180范围 - 高性能版本"""
    # 使用向量化操作和原地修改提高性能
    lon = np.asarray(lon)
    return np.mod(lon + 180, 360) - 180

# 优化的坐标转换函数
def convert_coordinates_fast(coords, source_range, target_range):
    """快速线性坐标变换 - 优化内存使用"""
    source_min, source_max = source_range
    target_min, target_max = target_range

    # 内存优化的计算方式
    scale = (target_max - target_min) / (source_max - source_min)
    coords = target_min + scale * (coords - source_min)

    return coords



# 异步IO写入类 - 减少IO等待时间
class AsyncWriter:
    """异步文件写入类 - 减少IO等待"""

    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def write_file(self, filepath, content, mode='w'):
        """异步写入文件"""
        future = self.executor.submit(self._write_to_file, filepath, content, mode)
        self.futures.append(future)
        return future

    def _write_to_file(self, filepath, content, mode):
        """实际的文件写入操作"""
        try:
            with open(filepath, mode, encoding='utf-8') as f:
                f.write(content)
                f.flush()  # 强制刷新到磁盘
            return True
        except Exception as e:
            logger.error(f"文件写入失败 {filepath}: {e}")
            return False

    def write_csv_row(self, filepath, row_data, headers=None):
        """异步写入CSV行"""
        future = self.executor.submit(self._write_csv_row, filepath, row_data, headers)
        self.futures.append(future)
        return future

    def _write_csv_row(self, filepath, row_data, headers):
        """实际的CSV行写入操作"""
        try:
            # 检查文件是否存在
            file_exists = os.path.exists(filepath)

            mode = 'a' if file_exists else 'w'
            with open(filepath, mode, encoding='utf-8') as f:
                # 如果是新文件且有标题行，先写入标题
                if not file_exists and headers:
                    f.write(','.join(headers) + '\n')

                # 确保所有数据都是字符串
                row_str = ','.join(str(item) for item in row_data) + '\n'
                f.write(row_str)
                f.flush()  # 强制刷新到磁盘
            return True
        except Exception as e:
            logger.error(f"CSV行写入失败 {filepath}: {e}")
            return False

    def wait_all(self, timeout=None):
        """等待所有写入操作完成"""
        done, not_done = [], []
        if self.futures:
            done, not_done = as_completed(self.futures, timeout=timeout)

        # 清理已完成的futures
        self.futures = list(not_done)

        # 返回完成的和未完成的数量
        return len(done), len(not_done)

    def shutdown(self, wait=True):
        """关闭执行器"""
        self.executor.shutdown(wait=wait)


# 内存使用监控类 - 智能内存管理
class MemoryMonitor:
    """内存监控类 - 提供智能内存管理"""

    def __init__(self, memory_warning_threshold=80, critical_threshold=90):
        self.memory_warning_threshold = memory_warning_threshold
        self.critical_threshold = critical_threshold
        self.last_cleanup_time = time.time()
        self.cleanup_count = 0

    def check_memory(self, force_cleanup_interval=300):
        """检查内存使用情况并决定是否需要清理"""
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        percent = memory.percent

        # 获取GPU内存信息
        gpu_info = self.get_gpu_memory_info()

        # 记录当前内存状态
        logger.debug(f"系统内存: 已用{used_gb:.1f}GB, 可用{available_gb:.1f}GB, 使用率{percent:.1f}%")
        logger.debug(f"GPU内存: {gpu_info}")

        # 强制周期性清理
        current_time = time.time()
        time_since_last_cleanup = current_time - self.last_cleanup_time

        # 判断是否需要清理
        needs_cleanup = False

        if percent >= self.critical_threshold:
            logger.warning(f"内存使用率达到临界值: {percent:.1f}%，执行紧急清理")
            needs_cleanup = True
        elif percent >= self.memory_warning_threshold:
            logger.warning(f"内存使用率较高: {percent:.1f}%，执行预防性清理")
            needs_cleanup = True
        elif any(info['used_percent'] > self.memory_warning_threshold for info in gpu_info):
            logger.warning(f"GPU内存使用率较高，执行预防性清理")
            needs_cleanup = True
        elif time_since_last_cleanup >= force_cleanup_interval:
            logger.info(f"已经 {time_since_last_cleanup:.1f}秒 没有清理内存，执行周期性清理")
            needs_cleanup = True

        if needs_cleanup:
            self.cleanup_memory(critical=(percent >= self.critical_threshold))
            self.last_cleanup_time = current_time
            self.cleanup_count += 1
            return True

        return False

    def get_gpu_memory_info(self):
        """获取所有可用GPU的内存信息"""
        gpu_info = []

        if not torch.cuda.is_available():
            return gpu_info

        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory / (1024**3)  # GB
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                mem_free = mem_total - mem_reserved
                used_percent = (mem_reserved / mem_total) * 100

                gpu_info.append({
                    'device': i,
                    'name': props.name,
                    'total_gb': mem_total,
                    'reserved_gb': mem_reserved,
                    'allocated_gb': mem_allocated,
                    'free_gb': mem_free,
                    'used_percent': used_percent
                })
            except Exception as e:
                logger.error(f"获取GPU {i} 内存信息失败: {e}")

        return gpu_info

    def cleanup_memory(self, critical=False):
        """清理内存 - 基于内存紧急程度采取不同策略"""
        # 首先清理PyTorch缓存
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
            except:
                pass

        # Python垃圾回收
        for _ in range(3 if critical else 1):
            gc.collect()

        # 更激进的内存回收（如果是关键状态）
        if critical:
            # 强制触发Python的完全垃圾回收
            gc.collect(2)
            # 暂停片刻，让系统有时间回收资源
            time.sleep(1)

        logger.info(f"执行了{'紧急' if critical else '常规'}内存清理 (第{self.cleanup_count+1}次)")

        # 再次获取内存信息以验证清理效果
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        percent = memory.percent
        logger.info(f"清理后内存: 已用{used_gb:.1f}GB, 使用率{percent:.1f}%")


# 时间预测器类 - 提供准确的进度和时间估计
class TimePredictor:
    """时间预测器 - 提供准确的进度和时间估计"""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_times = []
        self.completed_steps = 0
        self.last_update_time = self.start_time

    def update(self, steps_completed=1):
        """更新完成的步骤并记录时间"""
        current_time = time.time()
        step_duration = current_time - self.last_update_time

        # 添加步骤时间到历史记录
        self.step_times.append(step_duration)

        # 保持历史记录在合理范围内
        if len(self.step_times) > min(50, self.total_steps // 2):
            self.step_times.pop(0)

        self.completed_steps += steps_completed
        self.last_update_time = current_time

        return step_duration

    def get_estimates(self):
        """获取时间和进度估计"""
        current_time = time.time()
        elapsed_seconds = current_time - self.start_time

        # 进度百分比
        progress_percent = (self.completed_steps / self.total_steps) * 100

        # 预计剩余时间
        if self.completed_steps > 0 and self.step_times:
            # 使用最近的步骤时间进行预测
            recent_avg_step_time = sum(self.step_times[-min(10, len(self.step_times)):]) / min(10, len(self.step_times))

            # 使用整体平均和最近平均的加权平均
            if len(self.step_times) >= 3:
                overall_avg_step_time = sum(self.step_times) / len(self.step_times)
                avg_step_time = 0.3 * overall_avg_step_time + 0.7 * recent_avg_step_time
            else:
                avg_step_time = recent_avg_step_time

            remaining_steps = self.total_steps - self.completed_steps
            estimated_remaining_seconds = avg_step_time * remaining_steps

            # 估计总时间和完成时间
            estimated_total_seconds = elapsed_seconds + estimated_remaining_seconds
            estimated_completion_time = time.strftime("%H:%M:%S", time.localtime(current_time + estimated_remaining_seconds))
        else:
            estimated_remaining_seconds = float('inf')
            estimated_total_seconds = float('inf')
            estimated_completion_time = "未知"

        return {
            'elapsed_seconds': elapsed_seconds,
            'elapsed_formatted': self.format_time(elapsed_seconds),
            'progress_percent': progress_percent,
            'avg_step_time': sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            'recent_step_time': self.step_times[-1] if self.step_times else 0,
            'estimated_remaining_seconds': estimated_remaining_seconds,
            'estimated_remaining_formatted': self.format_time(estimated_remaining_seconds),
            'estimated_total_seconds': estimated_total_seconds,
            'estimated_total_formatted': self.format_time(estimated_total_seconds),
            'estimated_completion_time': estimated_completion_time
        }

    @staticmethod
    def format_time(seconds):
        """格式化时间为人类可读形式"""
        if not np.isfinite(seconds):
            return "未知"

        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{int(hours)}小时{int(minutes)}分钟"
        elif minutes > 0:
            return f"{int(minutes)}分钟{int(seconds)}秒"
        else:
            return f"{seconds:.1f}秒"

# 添加优化版的多GPU计算器类
class OptimizedMultiGPUCalculator:
    """优化版多GPU海洋热膨胀计算器 - 专注于性能提升"""

    def __init__(self, config):
        self.config = config
        self.setup_logging()

        # 检测并配置GPU
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count < 1:
            raise RuntimeError("需要至少1个GPU，当前没有可用GPU")

        # 默认使用所有可用GPU
        self.gpu_devices = list(range(self.gpu_count))
        if 'gpu_devices' in config and config['gpu_devices']:
            self.gpu_devices = config['gpu_devices']

        self.setup_gpus()

        # 文件路径
        self.gebco_2008 = config.get('gebco_2008')
        self.gebco_2024 = config.get('gebco_2024')
        self.temp_2008 = config['temp_2008']
        self.temp_2024 = config['temp_2024']
        self.salt_2008 = config['salt_2008']
        self.salt_2024 = config['salt_2024']

        # 分块和并行参数
        self.min_chunk_size = config.get('min_chunk_size', 300)
        self.max_chunk_size = config.get('max_chunk_size', 600)
        self.max_layers_in_parallel = config.get('max_layers_in_parallel', self.gpu_count)
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.use_layer_parallelism = config.get('use_layer_parallelism', True)
        self.mem_efficient_interpolation = config.get('mem_efficient_interpolation', True)

        # 深度层范围控制，采用Python切片语义: [start_layer, end_layer)
        self.start_layer = config.get('start_layer', 0)
        self.end_layer = config.get('end_layer', None)

        # 小岛屿填充参数 - 与testnew.py保持一致
        self.enable_island_filling = config.get('enable_island_filling', False)
        self.max_island_size = config.get('max_island_size', 5)
        self.island_fill_ocean_ratio = config.get('island_fill_ocean_ratio', 0.8)

        # 计算参数
        self.earth_radius = 6371000
        self.ocean_area = 361.9e12

        # 禁用深层外推
        self.extrapolate_deep_layers = False

        # 输出配置
        self.output_dir = config.get('output_dir', './optimized_results/')
        os.makedirs(self.output_dir, exist_ok=True)

        # 地图和NetCDF输出配置
        self.generate_map_outputs = config.get('generate_map_outputs', True)
        self.map_downsample_factor = config.get('map_downsample_factor', 4)
        self.nc_save_full_resolution = config.get('nc_save_full_resolution', True)
        self.nc_write_block_rows = config.get('nc_write_block_rows', 256)
        self.total_volume_changes_spatial = None
        self.spatial_data_initialized = False

        # 初始化辅助组件
        self.async_writer = AsyncWriter(max_workers=2)
        self.memory_monitor = MemoryMonitor(
            memory_warning_threshold=config.get('memory_warning_threshold', 75),
            critical_threshold=config.get('critical_threshold', 90)
        )

        # 缓存设置
        self.use_coord_cache = config.get('use_coord_cache', True)
        self.coord_cache = {}

        # 初始化实时CSV写入
        self.setup_csv_output()

        # 打印配置信息
        self.log_configuration()

    def setup_logging(self):
        """配置日志"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    def setup_gpus(self):
        """设置GPU环境"""
        for gpu_id in self.gpu_devices:
            if gpu_id >= self.gpu_count:
                raise ValueError(f"GPU {gpu_id} 不存在，可用GPU数量为 {self.gpu_count}")

            # 初始化GPU
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()

            # 获取GPU信息
            props = torch.cuda.get_device_properties(gpu_id)
            memory_gb = props.total_memory / (1024**3)
            self.logger.info(f"GPU {gpu_id}: {props.name}, {memory_gb:.1f}GB 显存")

        # 启用cudNN benchmark
        torch.backends.cudnn.benchmark = True

        self.logger.info(f"已配置 {len(self.gpu_devices)} 个GPU: {self.gpu_devices}")

    def setup_csv_output(self):
        """设置CSV输出"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)

        # CSV文件路径
        self.layer_csv_file = os.path.join(self.output_dir, f'layers_data_{timestamp}.csv')
        self.summary_csv_file = os.path.join(self.output_dir, f'summary_{timestamp}.csv')

        # 初始化CSV文件
        layer_headers = [
            'depth_m', 'volume_km3', 'volume_change_km3', 'chunks_processed',
            'processing_time_seconds', 'timestamp'
        ]

        summary_headers = [
            'total_layers_processed', 'total_volume_km3', 'total_volume_change_km3',
            'sea_level_rise_mm', 'annual_rate_mm_per_year',
            'total_processing_time_minutes', 'timestamp'
        ]

        # 异步写入表头
        self.async_writer.write_csv_row(self.layer_csv_file, layer_headers)
        self.async_writer.write_csv_row(self.summary_csv_file, summary_headers)

        self.logger.info(f"CSV输出已配置: {self.output_dir}")

    def log_configuration(self):
        """记录配置信息"""
        self.logger.info("优化版多GPU海洋热膨胀计算器初始化完成")
        self.logger.info(f"GPU设备: {self.gpu_devices} (共{len(self.gpu_devices)}个)")
        self.logger.info(f"分块大小范围: {self.min_chunk_size}-{self.max_chunk_size}")
        self.logger.info(f"深度层并行数: {self.max_layers_in_parallel}")
        self.logger.info(
            f"深度层范围: start_layer={self.start_layer}, "
            f"end_layer={self.end_layer if self.end_layer is not None else '最后一层'}"
        )
        self.logger.info(f"使用混合精度: {self.use_mixed_precision}")
        self.logger.info(f"使用深度层并行: {self.use_layer_parallelism}")
        self.logger.info(f"内存高效插值: {self.mem_efficient_interpolation}")
        self.logger.info(f"小岛屿填充: {'已启用' if self.enable_island_filling else '已禁用'}")
        if self.enable_island_filling:
            self.logger.info(f"最大岛屿大小: {self.max_island_size} 网格，海洋包围度: {self.island_fill_ocean_ratio}")
        self.logger.info(f"坐标缓存: {self.use_coord_cache}")
        self.logger.info(f"地图/JPG/NetCDF输出: {'已启用' if self.generate_map_outputs else '已禁用'}")
        if self.generate_map_outputs:
            self.logger.info(f"JPG地图降采样倍数: {self.map_downsample_factor}x")
            nc_resolution_mode = 'GEBCO原始分辨率' if self.nc_save_full_resolution else 'JPG降采样分辨率'
            self.logger.info(f"NetCDF输出分辨率: {nc_resolution_mode}")
            self.logger.info(f"NetCDF分块写入行数: {self.nc_write_block_rows}")
        self.logger.info(f"深层外推: 已禁用")
        self.logger.info(f"PyTorch版本: {torch.__version__}")

        # 验证TEOS-10
        try:
            test_alpha = gsw.alpha(35.0, 15.0, 0.0)
            self.logger.info(f"GSW-TEOS10验证成功，测试值: {test_alpha:.6e} K⁻¹")
        except Exception as e:
            raise RuntimeError(f"GSW-TEOS10库不可用. 错误: {e}")

    def write_layer_to_csv(self, layer_result, processing_time):
        """异步写入层数据到CSV"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        row_data = [
            f"{float(layer_result['depth']):.1f}",
            f"{float(layer_result['volume'])/1e9:.6f}",
            f"{float(layer_result['volume_change'])/1e9:.9f}",
            f"{int(layer_result['chunks_processed'])}",
            f"{float(processing_time):.2f}",
            timestamp
        ]

        self.async_writer.write_csv_row(self.layer_csv_file, row_data)

    def write_summary_to_csv(self, results):
        """异步写入总结数据到CSV"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 计算海平面上升
        sea_level_rise = (float(results['total_volume_change']) / self.ocean_area) * 1000
        annual_rate = sea_level_rise / 1  # 保持原脚本定义不变

        row_data = [
            f"{int(results['total_layers'])}",
            f"{float(results['total_volume'])/1e9:.6f}",
            f"{float(results['total_volume_change'])/1e9:.9f}",
            f"{float(sea_level_rise):.6f}",
            f"{float(annual_rate):.6f}",
            f"{float(results['processing_time_seconds'])/60:.2f}",
            timestamp
        ]

        self.async_writer.write_csv_row(self.summary_csv_file, row_data)

    def optimized_load_netcdf(self, file_path, var_candidates):
        """优化版NetCDF数据加载 - 统一坐标系统并增强性能"""
        self.logger.info(f"加载NetCDF: {os.path.basename(file_path)}")

        try:
            with nc.Dataset(file_path, 'r') as ds:
                # 找变量
                var_name = None
                for candidate in var_candidates:
                    if candidate in ds.variables:
                        var_name = candidate
                        break

                if var_name is None:
                    for candidate in var_candidates:
                        for available_var in ds.variables.keys():
                            if candidate.lower() in available_var.lower():
                                var_name = available_var
                                break
                        if var_name:
                            break

                if var_name is None:
                    raise ValueError(f"找不到变量: {var_candidates}")

                # 获取变量对象
                var = ds.variables[var_name]
                self.logger.debug(f"读取变量: {var_name}, 形状: {var.shape}, 维度: {var.dimensions}")

                # 使用内存优化的读取方式
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # 直接转换为float32以减少内存使用
                    data = var[:].astype(np.float32)

                # 处理填充值
                if hasattr(data, 'mask'):
                    data = np.ma.filled(data, np.nan)

                # 处理各种填充值（批量方式）
                fill_values = []
                for attr in ['_FillValue', 'missing_value']:
                    if hasattr(var, attr):
                        try:
                            fill_val = getattr(var, attr)
                            if isinstance(fill_val, (list, tuple, np.ndarray)):
                                fill_val = fill_val[0] if len(fill_val) > 0 else None
                            if fill_val is not None:
                                fill_values.append(float(fill_val))
                        except:
                            pass

                fill_values.extend([-999, -9999, -99999, 1e20, 9.96921e36])

                # 优化：一次性替换所有填充值
                for fill_val in set(fill_values):
                    if not np.isnan(fill_val):
                        data = np.where(np.abs(data - fill_val) < 1e-6, np.nan, data)

                # 读取坐标 - 使用缓存提高性能
                coords = {}
                coord_maps = {
                    'depth': ['depth', 'depth_std', 'z', 'level', 'lev'],
                    'lat': ['lat', 'latitude', 'y'],
                    'lon': ['lon', 'longitude', 'x']
                }

                # 数据维度信息
                data_dimensions = var.dimensions
                data_shape = data.shape

                for coord_type, candidates in coord_maps.items():
                    for candidate in candidates:
                        if candidate in ds.variables:
                            # 使用缓存提高性能
                            cache_key = f"{file_path}_{candidate}"
                            if self.use_coord_cache and cache_key in self.coord_cache:
                                coords[coord_type] = self.coord_cache[cache_key].copy()  # 使用copy避免负步长
                                self.logger.debug(f"使用缓存的坐标 {coord_type}")
                                break

                            coord_var = ds.variables[candidate]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                coord_data = coord_var[:].astype(np.float32)

                            if hasattr(coord_data, 'mask'):
                                coord_data = np.ma.filled(coord_data, np.nan)

                            # 经度归一化
                            if coord_type == 'lon':
                                coord_data = normalize_longitude(coord_data)

                            coords[coord_type] = coord_data

                            # 缓存坐标
                            if self.use_coord_cache:
                                self.coord_cache[cache_key] = coord_data.copy()  # 存储副本

                            break

                # 确保纬度从南到北排序
                if 'lat' in coords:
                    if coords['lat'][0] > coords['lat'][-1]:
                        coords['lat'] = coords['lat'][::-1].copy()  # 添加copy避免负步长
                        # 翻转数据的lat维度
                        if len(data.shape) == 3:
                            # 根据维度名称推断顺序
                            for i, dim_name in enumerate(data_dimensions):
                                if any(k in dim_name.lower() for k in ['lat', 'y']):
                                    data = np.flip(data, axis=i).copy()  # 添加copy避免负步长
                                    break

                # 确保经度从西到东排序
                if 'lon' in coords:
                    lon_sorted_idx = np.argsort(coords['lon'])
                    if not np.array_equal(lon_sorted_idx, np.arange(len(coords['lon']))):
                        coords['lon'] = coords['lon'][lon_sorted_idx].copy()  # 添加copy避免负步长
                        # 根据维度名称推断顺序重排数据
                        if len(data.shape) == 3:
                            for i, dim_name in enumerate(data_dimensions):
                                if any(k in dim_name.lower() for k in ['lon', 'x']):
                                    # 安全检查索引
                                    if len(lon_sorted_idx) <= data.shape[i]:
                                        # 使用高级索引
                                        idx = [slice(None)] * len(data.shape)
                                        idx[i] = lon_sorted_idx
                                        data = data[tuple(idx)].copy()  # 添加copy避免负步长
                                    break

                # 标准化维度顺序为 [depth, lat, lon]
                if len(data.shape) == 3:
                    target_order = ['depth', 'lat', 'lon']
                    current_order = []

                    # 确定当前维度顺序
                    for i, dim_name in enumerate(data_dimensions):
                        if any(k in dim_name.lower() for k in ['depth', 'z', 'lev']):
                            current_order.append('depth')
                        elif any(k in dim_name.lower() for k in ['lat', 'y']):
                            current_order.append('lat')
                        elif any(k in dim_name.lower() for k in ['lon', 'x']):
                            current_order.append('lon')

                    if current_order != target_order and len(current_order) == 3:
                        # 确定转置顺序
                        axes = [current_order.index(target) for target in target_order]
                        data = np.transpose(data, axes)
                        self.logger.info(f"维度重排: {current_order} -> {target_order}")

                # 修复缺失数据层
                data = self.fix_missing_layers(data)

                # 小岛屿填充只在NetCDF原始网格上执行，避免在高分辨率GEBCO网格上放大开销。
                if self.enable_island_filling and len(data.shape) == 3:
                    self.logger.info(
                        f"开始逐层小岛屿填充: max_island_size={self.max_island_size}, "
                        f"ocean_ratio={self.island_fill_ocean_ratio}"
                    )
                    total_filled = 0

                    for layer_idx in range(data.shape[0]):
                        before_nan_count = np.sum(np.isnan(data[layer_idx]))
                        if before_nan_count == 0:
                            continue

                        filled_layer = fill_small_enclosed_islands(
                            data[layer_idx],
                            max_island_size=self.max_island_size,
                            min_ocean_ratio=self.island_fill_ocean_ratio
                        )
                        after_nan_count = np.sum(np.isnan(filled_layer))
                        layer_filled_count = int(before_nan_count - after_nan_count)

                        if layer_filled_count > 0:
                            data[layer_idx] = filled_layer
                            total_filled += layer_filled_count
                            if layer_idx < 10 or layer_idx % 20 == 0:
                                depth_label = coords.get('depth', np.arange(data.shape[0]))[layer_idx]
                                self.logger.info(
                                    f"深度层{layer_idx}({float(depth_label):.1f}m): "
                                    f"填充{layer_filled_count}个网格点"
                                )

                    self.logger.info(f"小岛屿填充完成: 累计填充 {total_filled} 个网格点")

                # 确保数据连续性
                data = np.ascontiguousarray(data)
                for coord_type in coords:
                    coords[coord_type] = np.ascontiguousarray(coords[coord_type])

                # 打印数据信息
                valid_percent = np.sum(~np.isnan(data)) / np.prod(data.shape) * 100
                self.logger.info(f"数据加载完成: 形状{data.shape}, 有效数据: {valid_percent:.1f}%")

                return data, coords

        except Exception as e:
            self.logger.error(f"加载NetCDF文件失败: {e}")
            raise

    def fix_missing_layers(self, data):
        """修复缺失的数据层 - 优化版本"""
        if len(data.shape) != 3:
            return data

        n_depths, n_lats, n_lons = data.shape
        fixed_data = data.copy()

        # 计算每层有效数据百分比 - 向量化操作
        valid_counts = np.sum(~np.isnan(data), axis=(1, 2))
        valid_percents = valid_counts / (n_lats * n_lons) * 100

        # 找出需要修复的层
        layers_to_fix = np.where(valid_percents < 1)[0]

        if len(layers_to_fix) == 0:
            return data.copy()  # 确保返回副本而不是负步长视图

        self.logger.info(f"需要修复 {len(layers_to_fix)} 个缺失数据层")

        # 对每个需要修复的层
        for depth_idx in layers_to_fix:
            # 寻找最近的有效层
            valid_layers = np.where(valid_percents >= 1)[0]

            if len(valid_layers) == 0:
                self.logger.warning(f"没有找到有效层，无法修复层 {depth_idx}")
                continue

            # 计算距离
            distances = np.abs(valid_layers - depth_idx)
            nearest_idx = valid_layers[np.argmin(distances)]

            # 使用最近的有效层替代
            fixed_data[depth_idx] = fixed_data[nearest_idx].copy()
            self.logger.info(f"深度层{depth_idx}数据不足({valid_percents[depth_idx]:.1f}%)，使用深度层{nearest_idx}数据替代")

        return fixed_data  # 已经是副本，不需要再copy

    def optimized_load_gebco(self, gebco_file):
        """优化版GEBCO数据加载 - 统一坐标系统并增强性能"""
        self.logger.info(f"加载GEBCO: {os.path.basename(gebco_file)}")

        try:
            # 使用缓存减少重复加载
            cache_key = f"gebco_{gebco_file}"
            if self.use_coord_cache and cache_key in self.coord_cache:
                return self.coord_cache[cache_key]

            dataset = gdal.Open(gebco_file)
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            geotransform = dataset.GetGeoTransform()

            # 计算坐标
            lon_min = geotransform[0]
            lon_res = geotransform[1]
            lat_max = geotransform[3]
            lat_res = geotransform[5]  # 通常是负值

            # 使用像元中心坐标，避免把边界180度归一化为-180度后制造重复经度。
            lons = (lon_min + (np.arange(width, dtype=np.float32) + 0.5) * lon_res).astype(np.float32)
            lats = (lat_max + (np.arange(height, dtype=np.float32) + 0.5) * lat_res).astype(np.float32)

            # 统一坐标系统
            lons = normalize_longitude(lons)

            # 确保纬度从南到北排序
            if lats[0] > lats[-1]:
                lats = lats[::-1].copy()  # 添加copy()避免负步长
                flip_lat = True
            else:
                flip_lat = False

            # 确保经度从西到东排序
            sorted_idx = np.argsort(lons)
            if not np.array_equal(sorted_idx, np.arange(len(lons))):
                lons = lons[sorted_idx].copy()  # 添加copy()避免负步长
                lon_reorder_idx = sorted_idx
            else:
                lon_reorder_idx = None

            # 优化bathymetry读取 - 直接读取为float32
            band = dataset.GetRasterBand(1)
            nodata_value = band.GetNoDataValue()
            bathymetry = band.ReadAsArray().astype(np.float32)

            # 释放GDAL资源
            dataset = None

            # 应用翻转并创建副本避免负步长
            if flip_lat:
                bathymetry = np.flipud(bathymetry).copy()
            if lon_reorder_idx is not None:
                bathymetry = bathymetry[:, lon_reorder_idx].copy()

            # 确保bathymetry形状正确
            if bathymetry.shape != (len(lats), len(lons)):
                if bathymetry.shape == (len(lons), len(lats)):
                    bathymetry = bathymetry.T
                    self.logger.info(f"GEBCO数据已转置为(lat, lon)格式")
                else:
                    self.logger.warning(f"GEBCO数据形状异常: {bathymetry.shape}, 预期: ({len(lats)}, {len(lons)})")

            # 纯海洋GEBCO通常用NoData标记陆地，必须先转为NaN，避免-9999被当成深海。
            if nodata_value is not None:
                nodata_mask = np.isclose(bathymetry, float(nodata_value))
                nodata_count = int(np.sum(nodata_mask))
                if nodata_count > 0:
                    bathymetry[nodata_mask] = np.nan
                    self.logger.info(f"GEBCO NoData({nodata_value})已转为NaN: {nodata_count:,} 像元")

            # 数据验证
            valid_pixels = np.sum(~np.isnan(bathymetry))
            total_pixels = np.prod(bathymetry.shape)
            ocean_pixels = np.sum((~np.isnan(bathymetry)) & (bathymetry < 0))

            self.logger.info(f"GEBCO数据: {width}x{height} -> bathymetry形状: {bathymetry.shape}")
            self.logger.info(f"坐标范围: lat[{lats.min():.2f}, {lats.max():.2f}], lon[{lons.min():.2f}, {lons.max():.2f}]")
            self.logger.info(f"有效数据覆盖率: {valid_pixels/total_pixels:.1%}")
            self.logger.info(f"海洋像元覆盖率(负高程): {ocean_pixels/total_pixels:.1%}")

            # 计算像素属性
            base_area, lat_corrections = self.calculate_pixel_properties(lats, lons, geotransform)

            # 创建海洋掩码：GEBCO陆地通常也是有效高程值，必须只保留负高程海洋。
            ocean_mask = (~np.isnan(bathymetry)) & (bathymetry < 0)

            # 缓存结果
            result = (lats, lons, bathymetry, geotransform, base_area, lat_corrections, ocean_mask)
            if self.use_coord_cache:
                self.coord_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"加载GEBCO文件失败: {e}")
            raise

    def calculate_pixel_properties(self, lats, lons, geotransform):
        """计算像素属性 - 优化版本"""
        lon_res = abs(geotransform[1])
        lat_res = abs(geotransform[5])

        # 内存优化版本
        pixel_width = self.earth_radius * np.radians(lon_res)
        pixel_height = self.earth_radius * np.radians(lat_res)
        base_area = pixel_width * pixel_height

        # 使用向量化计算纬度修正
        lat_corrections = np.cos(np.radians(lats)).astype(np.float32)

        return base_area, lat_corrections

    def initialize_spatial_map_storage(self):
        """初始化最终地图需要的空间累积数组。"""
        if not self.generate_map_outputs or self.spatial_data_initialized:
            return

        bathymetry = self.gebco_data[2]
        self.total_volume_changes_spatial = np.zeros_like(bathymetry, dtype=np.float64)
        self.spatial_data_initialized = True
        self.logger.info(f"空间热膨胀累积数组初始化完成: {bathymetry.shape}")

    def accumulate_spatial_volume_change(self, layer_volume_changes):
        """主线程累积单层空间体积变化，避免多线程同时写全局数组。"""
        if not self.generate_map_outputs or layer_volume_changes is None:
            return

        if not self.spatial_data_initialized:
            self.initialize_spatial_map_storage()

        if layer_volume_changes.shape != self.total_volume_changes_spatial.shape:
            self.logger.warning(
                f"空间体积变化形状不匹配: {layer_volume_changes.shape} vs "
                f"{self.total_volume_changes_spatial.shape}"
            )
            return

        self.total_volume_changes_spatial += np.nan_to_num(layer_volume_changes, nan=0.0)

    def downsample_map_data(self, data, factor):
        """对地图数据做双线性降采样，用于JPG显示。"""
        if factor <= 1:
            return np.ascontiguousarray(data)

        try:
            data = np.ascontiguousarray(data)
            target_shape = (
                max(1, data.shape[0] // factor),
                max(1, data.shape[1] // factor)
            )
            zoom_factors = (
                target_shape[0] / data.shape[0],
                target_shape[1] / data.shape[1]
            )
            return np.ascontiguousarray(zoom(data, zoom_factors, order=1, prefilter=False))
        except Exception as e:
            self.logger.error(f"地图降采样失败: {e}")
            return np.ascontiguousarray(data)

    def generate_final_map_outputs(self, results):
        """输出最终热膨胀空间分布JPG和NetCDF文件。"""
        if not (self.generate_map_outputs and self.spatial_data_initialized):
            return None, None

        try:
            self.logger.info("开始生成最终热膨胀JPG和NetCDF输出...")

            gebco_lats, gebco_lons, bathymetry = self.gebco_data[0], self.gebco_data[1], self.gebco_data[2]
            ocean_mask = self.gebco_data[6]

            jpg_source_km3 = np.empty(self.total_volume_changes_spatial.shape, dtype=np.float32)
            np.divide(self.total_volume_changes_spatial, 1e9, out=jpg_source_km3, casting='unsafe')
            jpg_source_km3[~ocean_mask] = np.nan

            plot_data = self.downsample_map_data(jpg_source_km3, self.map_downsample_factor)
            plot_lats = np.linspace(gebco_lats.min(), gebco_lats.max(), plot_data.shape[0], dtype=np.float32)
            plot_lons = np.linspace(gebco_lons.min(), gebco_lons.max(), plot_data.shape[1], dtype=np.float32)
            if plot_data is not jpg_source_km3:
                del jpg_source_km3
                gc.collect()

            valid_data = plot_data[~np.isnan(plot_data)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 2)
                vmax = np.percentile(valid_data, 98)
                vmax_abs = max(abs(vmin), abs(vmax))
                if vmax_abs == 0 or not np.isfinite(vmax_abs):
                    vmin, vmax = -1, 1
                else:
                    vmin, vmax = -vmax_abs, vmax_abs
            else:
                vmin, vmax = -1, 1
            del valid_data
            gc.collect()

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            jpg_file = os.path.join(self.output_dir, f'final_thermal_expansion_visual_filled_{timestamp}.jpg')
            nc_file = os.path.join(self.output_dir, f'thermal_expansion_visual_filled_{timestamp}.nc')

            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(20, 12), dpi=150)
            im = ax.imshow(
                plot_data,
                extent=[float(plot_lons.min()), float(plot_lons.max()),
                        float(plot_lats.min()), float(plot_lats.max())],
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
                origin='lower',
                interpolation='bilinear'
            )
            cbar = plt.colorbar(im, ax=ax, shrink=0.65, orientation='horizontal', pad=0.08)
            cbar.set_label('Ocean Thermal Volume Change (km3)', fontsize=13, fontweight='bold')
            cbar.ax.tick_params(labelsize=11)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.xaxis.set_major_locator(MultipleLocator(60))
            ax.yaxis.set_major_locator(MultipleLocator(30))
            ax.xaxis.set_minor_locator(MultipleLocator(30))
            ax.yaxis.set_minor_locator(MultipleLocator(15))
            ax.set_title('Global Ocean Thermal Expansion (Visual-Filled)', fontsize=18, fontweight='bold', pad=18)
            ax.set_xlabel('Longitude (degrees)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Latitude (degrees)', fontsize=13, fontweight='bold')
            ax.tick_params(labelsize=11)

            plt.savefig(jpg_file, dpi=200, bbox_inches='tight', facecolor='white', format='jpg')
            plt.close(fig)

            with nc.Dataset(nc_file, 'w', format='NETCDF4') as dataset:
                if self.nc_save_full_resolution:
                    nc_lats = gebco_lats
                    nc_lons = gebco_lons
                else:
                    nc_lats = plot_lats
                    nc_lons = plot_lons

                dataset.createDimension('lat', len(nc_lats))
                dataset.createDimension('lon', len(nc_lons))

                lat_var = dataset.createVariable('lat', 'f8', ('lat',))
                lon_var = dataset.createVariable('lon', 'f8', ('lon',))
                chunk_rows = min(128, len(nc_lats))
                chunk_cols = min(4096, len(nc_lons))
                thermal_var = dataset.createVariable(
                    'thermal_expansion_visual_filled_km3',
                    'f4',
                    ('lat', 'lon'),
                    zlib=True,
                    complevel=4,
                    chunksizes=(chunk_rows, chunk_cols),
                    fill_value=np.nan
                )

                lat_var[:] = nc_lats
                lon_var[:] = nc_lons

                if self.nc_save_full_resolution:
                    block_rows = max(1, int(self.nc_write_block_rows))
                    self.logger.info(
                        f"NetCDF原始分辨率分块写入: {len(nc_lats)} x {len(nc_lons)}, "
                        f"每块{block_rows}行"
                    )
                    for row_start in tqdm(
                        range(0, len(nc_lats), block_rows),
                        desc="写入原分辨率NetCDF",
                        leave=False
                    ):
                        row_end = min(row_start + block_rows, len(nc_lats))
                        block_km3 = np.empty(
                            self.total_volume_changes_spatial[row_start:row_end, :].shape,
                            dtype=np.float32
                        )
                        np.divide(
                            self.total_volume_changes_spatial[row_start:row_end, :],
                            1e9,
                            out=block_km3,
                            casting='unsafe'
                        )
                        block_km3[~ocean_mask[row_start:row_end, :]] = np.nan
                        thermal_var[row_start:row_end, :] = block_km3
                        del block_km3
                else:
                    thermal_var[:, :] = np.ascontiguousarray(plot_data.astype(np.float32))

                lat_var.units = 'degrees_north'
                lon_var.units = 'degrees_east'
                thermal_var.units = 'km3'
                thermal_var.long_name = 'Ocean thermal expansion volume change from island-filled input data'
                thermal_var.description = 'Temperature and salinity layers were island-filled before calculation for visual output.'

                dataset.title = 'Global Ocean Thermal Expansion Visual-Filled Map'
                dataset.source = 'TEOS-10V4 optimized multi-GPU calculator'
                dataset.history = f'Created on {time.strftime("%Y-%m-%d %H:%M:%S")}'
                dataset.base_year = int(results.get('base_year', self.config.get('base_year', 2000)))
                dataset.target_year = int(results.get('target_year', self.config.get('target_year', 2025)))
                dataset.coordinate_system = 'WGS1984'
                dataset.longitude_convention = '-180 to +180 degrees'
                dataset.input_island_filling_enabled = str(self.enable_island_filling)
                dataset.max_island_size = int(self.max_island_size)
                dataset.island_fill_ocean_ratio = float(self.island_fill_ocean_ratio)
                dataset.map_downsample_factor = int(self.map_downsample_factor)
                dataset.jpg_downsample_factor = int(self.map_downsample_factor)
                dataset.nc_full_resolution = str(self.nc_save_full_resolution)
                dataset.nc_write_block_rows = int(self.nc_write_block_rows)
                dataset.original_resolution = f'{len(gebco_lats)} x {len(gebco_lons)} pixels'
                dataset.jpg_output_resolution = f'{len(plot_lats)} x {len(plot_lons)} pixels'
                dataset.output_resolution = f'{len(nc_lats)} x {len(nc_lons)} pixels'
                dataset.total_volume_change_km3 = float(results['total_volume_change'] / 1e9)
                dataset.sea_level_rise_mm = float(results['sea_level_rise_mm'])
                dataset.note = (
                    'This output is accumulated on the GEBCO grid. JPG output may be downsampled; '
                    'NetCDF can be saved at full GEBCO resolution.'
                )

            self.logger.info(f"最终JPG已生成: {jpg_file}")
            self.logger.info(f"最终NetCDF已生成: {nc_file}")
            return jpg_file, nc_file

        except Exception as e:
            self.logger.error(f"生成最终地图输出失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None

    def enhanced_gpu_interpolation(self, data_layer, source_lats, source_lons,
                                  target_lats, target_lons, layer_idx, gpu_id):
        """增强版GPU插值 - 性能优化版本"""
        torch.cuda.set_device(gpu_id)

        # 快速验证数据有效性
        valid_mask = ~np.isnan(data_layer)
        valid_count = np.sum(valid_mask)

        if valid_count < 100:
            self.logger.warning(f"GPU{gpu_id} 深度{layer_idx} 数据太少，跳过插值")
            return np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32)

        try:
            # 检查GPU内存状态，如果内存不足则尝试使用CPU
            try:
                free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
                free_memory_gb = free_memory / (1024**3)
                if free_memory_gb < 0.5:  # 如果可用内存少于500MB
                    self.logger.warning(f"GPU{gpu_id} 内存不足({free_memory_gb:.2f}GB)，尝试使用CPU进行插值")
                    return self.cpu_fallback_interpolation(data_layer, source_lats, source_lons, target_lats, target_lons)
            except Exception as mem_e:
                self.logger.warning(f"检查GPU内存失败: {mem_e}，继续使用GPU")

            with autocast(enabled=self.use_mixed_precision):
                # 确保数据连续性和正步长
                data_layer = np.ascontiguousarray(data_layer.copy())

                # 使用半精度减少内存占用
                if self.use_mixed_precision:
                    data_layer = data_layer.astype(np.float16)

                # 内存优化：只在GPU上保存必要的数据
                # 源数据添加batch和channel维度
                source_tensor = torch.from_numpy(data_layer).unsqueeze(0).unsqueeze(0).cuda(gpu_id, non_blocking=True)

                # 坐标归一化 - 使用内存优化的计算方式
                source_lat_min, source_lat_max = source_lats.min(), source_lats.max()
                source_lon_min, source_lon_max = source_lons.min(), source_lons.max()

                # 创建网格坐标 - 不使用缓存以节省内存
                # 归一化目标坐标到[-1, 1]
                target_lat_norm = 2 * (target_lats - source_lat_min) / (source_lat_max - source_lat_min) - 1
                target_lon_norm = 2 * (target_lons - source_lon_min) / (source_lon_max - source_lon_min) - 1

                # 创建网格坐标
                lon_grid_2d, lat_grid_2d = np.meshgrid(target_lon_norm, target_lat_norm, indexing='xy')

                # 确保连续性、正步长并减少内存使用
                if self.use_mixed_precision:
                    grid_stack = np.stack([lon_grid_2d, lat_grid_2d], axis=-1).astype(np.float16).copy()
                else:
                    grid_stack = np.stack([lon_grid_2d, lat_grid_2d], axis=-1).astype(np.float32).copy()

                # 转换为PyTorch张量并发送到GPU
                grid = torch.from_numpy(grid_stack).unsqueeze(0).cuda(gpu_id, non_blocking=True)

                # 释放中间变量
                del lon_grid_2d, lat_grid_2d, grid_stack

                # 使用F.grid_sample进行插值
                result_tensor = F.grid_sample(
                    source_tensor, grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )

                # 转回numpy并移除额外维度
                result = result_tensor.squeeze().cpu().numpy()

                # 清理GPU内存
                del source_tensor, result_tensor, grid
                torch.cuda.empty_cache()

                return result.astype(np.float32)

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.warning(f"GPU{gpu_id} 内存不足，回退到CPU插值: {e}")
                torch.cuda.empty_cache()
                return self.cpu_fallback_interpolation(data_layer, source_lats, source_lons, target_lats, target_lons)
            else:
                self.logger.error(f"GPU{gpu_id} 深度{layer_idx} 插值失败: {e}")
                torch.cuda.empty_cache()
                return np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"GPU{gpu_id} 深度{layer_idx} 插值失败: {e}")
            torch.cuda.empty_cache()
            return np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32)

    def cpu_fallback_interpolation(self, data_layer, source_lats, source_lons, target_lats, target_lons):
        """CPU回退插值函数 - 当GPU内存不足时使用"""
        from scipy.interpolate import RegularGridInterpolator

        try:
            # 确保数据连续性
            data_layer = np.ascontiguousarray(data_layer.copy())

            # 创建插值函数
            valid_mask = ~np.isnan(data_layer)
            if np.sum(valid_mask) < 100:
                return np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32)

            # 填充NaN值以便插值
            filled_data = np.copy(data_layer)
            if np.any(np.isnan(filled_data)):
                filled_data[np.isnan(filled_data)] = np.nanmean(filled_data)

            # 创建插值器
            interp_func = RegularGridInterpolator(
                (source_lats, source_lons),
                filled_data,
                bounds_error=False,
                fill_value=np.nan
            )

            # 创建目标网格点
            lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
            points = np.column_stack((lat_grid.flatten(), lon_grid.flatten()))

            # 执行插值
            result_flat = interp_func(points)
            result = result_flat.reshape(lat_grid.shape)

            return result.astype(np.float32)

        except Exception as e:
            self.logger.error(f"CPU插值失败: {e}")
            return np.full((len(target_lats), len(target_lons)), np.nan, dtype=np.float32)

    def enhanced_gpu_volume_calculation(self, bathymetry_chunk, depth, thickness,
                                      base_area, lat_corrections_chunk, gpu_id):
        """增强版GPU体积计算 - 性能优化版本"""
        torch.cuda.set_device(gpu_id)

        try:
            with autocast(enabled=self.use_mixed_precision):
                # 确保数据连续性和正步长
                bathymetry_chunk = np.ascontiguousarray(bathymetry_chunk.copy())
                lat_corrections_chunk = np.ascontiguousarray(lat_corrections_chunk.copy())

                # 转换为PyTorch张量
                bathy_tensor = torch.from_numpy(bathymetry_chunk).cuda(gpu_id, non_blocking=True)
                lat_corr_tensor = torch.from_numpy(lat_corrections_chunk).cuda(gpu_id, non_blocking=True)

                # 获取形状
                lat_chunk_size, lon_chunk_size = bathy_tensor.shape

                # 扩展纬度校正
                lat_corr_expanded = lat_corr_tensor.unsqueeze(1).expand(lat_chunk_size, lon_chunk_size)

                # 计算面积校正
                areas = base_area * lat_corr_expanded

                # 海底深度：GEBCO海洋为负高程，陆地正高程不能用abs当作水深。
                ocean_mask = (~torch.isnan(bathy_tensor)) & (bathy_tensor < 0)
                seafloor_depths = torch.clamp(-bathy_tensor, min=0)

                # 有效深度掩码
                valid_depth_mask = (depth < seafloor_depths) & ocean_mask

                # 计算有效厚度
                depth_tensor = torch.tensor(depth, device=f'cuda:{gpu_id}')
                thickness_tensor = torch.tensor(thickness, device=f'cuda:{gpu_id}')

                effective_thickness = torch.clamp(
                    torch.minimum(thickness_tensor, seafloor_depths - depth_tensor),
                    min=0
                )

                # 计算体积
                volumes = torch.where(
                    valid_depth_mask,
                    areas * effective_thickness,
                    torch.tensor(0.0, device=f'cuda:{gpu_id}')
                )

                result = volumes.cpu().numpy().astype(np.float32)

                # 清理GPU内存
                del bathy_tensor, lat_corr_tensor, lat_corr_expanded, areas
                del seafloor_depths, valid_depth_mask, effective_thickness, volumes
                del depth_tensor, thickness_tensor
                torch.cuda.empty_cache()

                return result

        except Exception as e:
            self.logger.error(f"GPU{gpu_id} 体积计算失败: {e}")
            torch.cuda.empty_cache()
            return np.zeros_like(bathymetry_chunk, dtype=np.float32)

    def enhanced_gpu_thermal_expansion(self, temp_2008_chunk, temp_2024_chunk,
                                     salt_2008_chunk, salt_2024_chunk,
                                     depth, volumes_chunk, gpu_id,
                                     target_lats_chunk, target_lons_chunk):
        """增强版GPU热膨胀计算 - 性能优化版本"""
        torch.cuda.set_device(gpu_id)

        try:
            # 确保数据连续性和正步长
            temp_2008_chunk = np.ascontiguousarray(temp_2008_chunk.copy())
            temp_2024_chunk = np.ascontiguousarray(temp_2024_chunk.copy())
            salt_2008_chunk = np.ascontiguousarray(salt_2008_chunk.copy())
            salt_2024_chunk = np.ascontiguousarray(salt_2024_chunk.copy())
            volumes_chunk = np.ascontiguousarray(volumes_chunk.copy())

            # 创建有效数据掩码
            valid_mask = (
                (~np.isnan(temp_2008_chunk)) &
                (~np.isnan(temp_2024_chunk)) &
                (~np.isnan(salt_2008_chunk)) &
                (~np.isnan(salt_2024_chunk)) &
                (volumes_chunk > 0)
            )

            # 如果没有有效数据，直接返回
            if not np.any(valid_mask):
                return np.zeros_like(volumes_chunk, dtype=np.float32)

            # 准备GSW计算需要的坐标网格
            lon_grid, lat_grid = np.meshgrid(target_lons_chunk, target_lats_chunk)

            # 计算压力 (海水压力，单位：dbar，近似等于深度米数)
            pressure = np.full_like(temp_2008_chunk, depth)

            # ===== 使用GSW-TEOS10标准计算 =====

            # 1. 转换为TEOS-10标准量
            SA_2008 = salt_2008_chunk
            SA_2024 = salt_2024_chunk

            CT_2008 = gsw.CT_from_t(SA_2008, temp_2008_chunk, pressure)
            CT_2024 = gsw.CT_from_t(SA_2024, temp_2024_chunk, pressure)

            # 2. 计算热膨胀系数
            alpha_2008 = gsw.alpha(SA_2008, CT_2008, pressure)
            alpha_2024 = gsw.alpha(SA_2024, CT_2024, pressure)

            # 3. 使用平均热膨胀系数
            alpha_mean = (alpha_2008 + alpha_2024) / 2

            # 4. 计算保守温度变化
            dCT = CT_2024 - CT_2008

            # 5. 计算体积变化
            volume_changes = np.where(
                valid_mask,
                volumes_chunk * alpha_mean * dCT,
                0.0
            )

            # 处理异常值
            volume_changes = np.nan_to_num(volume_changes, nan=0.0, posinf=0.0, neginf=0.0)

            return volume_changes.astype(np.float32)

        except Exception as e:
            self.logger.error(f"GPU{gpu_id} 热膨胀计算失败: {e}")
            return np.zeros_like(volumes_chunk, dtype=np.float32)

    def process_layer_chunks_parallel(self, layer_params):
        """处理单个深度层的所有分块 - 优化版本"""
        layer_idx, depth, thickness, gpu_id = layer_params

        try:
            # 设置当前GPU
            torch.cuda.set_device(gpu_id)

            # 清理GPU内存
            torch.cuda.empty_cache()

            # 检查GPU内存状态
            free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated(gpu_id)
            free_memory_gb = free_memory / (1024**3)
            self.logger.info(f"深度层{layer_idx}(GPU{gpu_id}) 开始处理，可用内存: {free_memory_gb:.2f}GB")

            # 加载数据
            temps_2008 = self.temp_2008_data[layer_idx].copy()  # 复制以避免引用原始数据
            temps_2024 = self.temp_2024_data[layer_idx].copy()
            salts_2008 = self.salt_2008_data[layer_idx].copy()
            salts_2024 = self.salt_2024_data[layer_idx].copy()

            gebco_lats = self.gebco_data[0]
            gebco_lons = self.gebco_data[1]
            bathymetry = self.gebco_data[2]
            base_area = self.gebco_data[4]
            lat_corrections = self.gebco_data[5]
            ocean_mask = self.gebco_data[6]

            source_lats = self.nc_coords['lat']
            source_lons = self.nc_coords['lon']

            # 计算最佳分块数量 - 根据可用内存动态调整
            props = torch.cuda.get_device_properties(gpu_id)
            gpu_memory_gb = min(free_memory_gb, props.total_memory / (1024**3))

            # 如果内存不足，增加分块数量
            if free_memory_gb < 2.0:
                self.logger.warning(f"GPU{gpu_id}内存不足({free_memory_gb:.2f}GB)，增加分块数量")
                memory_factor = max(1.0, 2.0 / free_memory_gb)  # 内存越少，分块越多
                lat_chunks, lon_chunks = self.calculate_optimal_chunks(
                    bathymetry.shape,
                    gpu_memory_gb / memory_factor
                )
            else:
                lat_chunks, lon_chunks = self.calculate_optimal_chunks(bathymetry.shape, gpu_memory_gb)

            total_chunks = lat_chunks * lon_chunks
            self.logger.info(f"深度层{layer_idx} 分块: {lat_chunks}x{lon_chunks}={total_chunks}块")

            # 准备分块参数
            chunk_params = []
            for chunk_idx in range(total_chunks):
                # 计算块的索引范围
                lat_idx = chunk_idx // lon_chunks
                lon_idx = chunk_idx % lon_chunks

                lat_start = lat_idx * len(gebco_lats) // lat_chunks
                lat_end = min((lat_idx + 1) * len(gebco_lats) // lat_chunks, len(gebco_lats))
                lon_start = lon_idx * len(gebco_lons) // lon_chunks
                lon_end = min((lon_idx + 1) * len(gebco_lons) // lon_chunks, len(gebco_lons))

                # 添加到参数列表
                chunk_params.append((
                    chunk_idx, depth, thickness, gpu_id,
                    lat_start, lat_end, lon_start, lon_end
                ))

            # 记录开始时间
            chunk_start_time = time.time()

            # 限制并行线程数，避免过多线程争用资源
            max_workers = min(4, total_chunks)

            # 使用ThreadPoolExecutor处理所有分块
            chunk_results = []
            layer_volume_changes_spatial = (
                np.zeros_like(bathymetry, dtype=np.float32)
                if self.generate_map_outputs else None
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                # 分批提交任务，避免同时占用过多内存
                batch_size = max(1, total_chunks // max_workers)
                for i in range(0, len(chunk_params), batch_size):
                    batch_params = chunk_params[i:i+batch_size]

                    for params in batch_params:
                        future = executor.submit(
                            self.process_single_chunk,
                            *params,
                            temps_2008, temps_2024, salts_2008, salts_2024,
                            source_lats, source_lons, gebco_lats, gebco_lons,
                            bathymetry, base_area, lat_corrections, ocean_mask
                        )
                        futures.append((params[0], future))  # (chunk_idx, future)

                    # 等待当前批次完成，避免内存压力
                    for chunk_idx, future in list(futures):
                        try:
                            result = future.result()
                            if self.generate_map_outputs and layer_volume_changes_spatial is not None:
                                chunk_map = result.pop('volume_changes_spatial', None)
                                if chunk_map is not None:
                                    lat_range = result.get('lat_range')
                                    lon_range = result.get('lon_range')
                                    layer_volume_changes_spatial[
                                        lat_range[0]:lat_range[1],
                                        lon_range[0]:lon_range[1]
                                    ] = chunk_map
                            chunk_results.append(result)
                            futures.remove((chunk_idx, future))
                        except Exception as e:
                            self.logger.error(f"深度层{layer_idx} 块{chunk_idx} 处理失败: {e}")
                            futures.remove((chunk_idx, future))

                    # 清理内存
                    self.memory_monitor.check_memory()

                # 处理剩余的futures
                for chunk_idx, future in futures:
                    try:
                        result = future.result()
                        if self.generate_map_outputs and layer_volume_changes_spatial is not None:
                            chunk_map = result.pop('volume_changes_spatial', None)
                            if chunk_map is not None:
                                lat_range = result.get('lat_range')
                                lon_range = result.get('lon_range')
                                layer_volume_changes_spatial[
                                    lat_range[0]:lat_range[1],
                                    lon_range[0]:lon_range[1]
                                ] = chunk_map
                        chunk_results.append(result)
                    except Exception as e:
                        self.logger.error(f"深度层{layer_idx} 块{chunk_idx} 处理失败: {e}")

            # 汇总所有分块结果
            layer_volume = sum(r.get('volume', 0) for r in chunk_results)
            layer_volume_change = sum(r.get('volume_change', 0) for r in chunk_results)

            # 计算处理时间
            processing_time = time.time() - chunk_start_time

            # 清理临时数据
            del temps_2008, temps_2024, salts_2008, salts_2024
            torch.cuda.empty_cache()

            layer_result = {
                'layer_idx': layer_idx,
                'depth': depth,
                'volume': layer_volume,
                'volume_change': layer_volume_change,
                'chunks_processed': len(chunk_results),
                'processing_time': processing_time
            }

            if self.generate_map_outputs:
                layer_result['volume_changes_spatial'] = layer_volume_changes_spatial

            return layer_result

        except Exception as e:
            self.logger.error(f"深度层{layer_idx}(GPU{gpu_id})处理失败: {e}")
            torch.cuda.empty_cache()
            return {
                'layer_idx': layer_idx,
                'depth': depth,
                'volume': 0.0,
                'volume_change': 0.0,
                'chunks_processed': 0,
                'processing_time': 0.0
            }

    def process_single_chunk(self, chunk_idx, depth, thickness, gpu_id,
                            lat_start, lat_end, lon_start, lon_end,
                            temp_2008_layer, temp_2024_layer, salt_2008_layer, salt_2024_layer,
                            source_lats, source_lons, gebco_lats, gebco_lons,
                            bathymetry, base_area, lat_corrections, ocean_mask):
        """处理单个数据块 - 优化版本"""
        torch.cuda.set_device(gpu_id)

        try:
            # 提取块数据（添加copy避免负步长）
            gebco_lats_chunk = gebco_lats[lat_start:lat_end].copy()
            gebco_lons_chunk = gebco_lons[lon_start:lon_end].copy()
            bathymetry_chunk = bathymetry[lat_start:lat_end, lon_start:lon_end].copy()
            lat_corrections_chunk = lat_corrections[lat_start:lat_end].copy()
            ocean_mask_chunk = ocean_mask[lat_start:lat_end, lon_start:lon_end].copy()

            # 执行插值
            temp_2008_interp = self.enhanced_gpu_interpolation(
                temp_2008_layer, source_lats, source_lons,
                gebco_lats_chunk, gebco_lons_chunk, -1, gpu_id
            )

            temp_2024_interp = self.enhanced_gpu_interpolation(
                temp_2024_layer, source_lats, source_lons,
                gebco_lats_chunk, gebco_lons_chunk, -1, gpu_id
            )

            salt_2008_interp = self.enhanced_gpu_interpolation(
                salt_2008_layer, source_lats, source_lons,
                gebco_lats_chunk, gebco_lons_chunk, -1, gpu_id
            )

            salt_2024_interp = self.enhanced_gpu_interpolation(
                salt_2024_layer, source_lats, source_lons,
                gebco_lats_chunk, gebco_lons_chunk, -1, gpu_id
            )

            # 应用海洋掩码
            temp_2008_interp = np.where(ocean_mask_chunk, temp_2008_interp, np.nan)
            temp_2024_interp = np.where(ocean_mask_chunk, temp_2024_interp, np.nan)
            salt_2008_interp = np.where(ocean_mask_chunk, salt_2008_interp, np.nan)
            salt_2024_interp = np.where(ocean_mask_chunk, salt_2024_interp, np.nan)

            # 计算体积
            volumes = self.enhanced_gpu_volume_calculation(
                bathymetry_chunk, depth, thickness,
                base_area, lat_corrections_chunk, gpu_id
            )

            # 计算热膨胀
            volume_changes = self.enhanced_gpu_thermal_expansion(
                temp_2008_interp, temp_2024_interp, salt_2008_interp, salt_2024_interp,
                depth, volumes, gpu_id, gebco_lats_chunk, gebco_lons_chunk
            )

            if self.generate_map_outputs:
                volume_changes_spatial = np.where(volumes > 0, volume_changes, np.nan).astype(np.float32)
            else:
                volume_changes_spatial = None

            # 返回结果
            result = {
                'chunk_idx': chunk_idx,
                'lat_range': (lat_start, lat_end),
                'lon_range': (lon_start, lon_end),
                'volume': np.nansum(volumes),
                'volume_change': np.nansum(volume_changes)
            }

            if self.generate_map_outputs:
                result['volume_changes_spatial'] = volume_changes_spatial

            return result

        except Exception as e:
            self.logger.error(f"块{chunk_idx}(GPU{gpu_id})处理失败: {e}")
            result = {
                'chunk_idx': chunk_idx,
                'lat_range': (lat_start, lat_end),
                'lon_range': (lon_start, lon_end),
                'volume': 0.0,
                'volume_change': 0.0
            }

            if self.generate_map_outputs:
                result['volume_changes_spatial'] = np.full(
                    (lat_end - lat_start, lon_end - lon_start),
                    np.nan,
                    dtype=np.float32
                )

            return result

    def calculate_optimal_chunks(self, target_shape, gpu_memory_gb):
        """计算最优分块大小 - 优化版本"""
        target_height, target_width = target_shape

        # 内存估算
        bytes_per_pixel = 4 * 20  # 20个float32作为安全系数

        # 保留50%内存给模型和中间计算
        available_bytes = gpu_memory_gb * 0.5 * (1024**3)

        # 计算可处理的最大像素数
        max_pixels = int(available_bytes / bytes_per_pixel)

        # 计算分块数量
        total_pixels = target_height * target_width
        min_chunks = max(1, int(np.ceil(total_pixels / max_pixels)))

        # 尽量均匀分配行列
        lat_chunks = int(np.ceil(np.sqrt(min_chunks)))
        lon_chunks = int(np.ceil(min_chunks / lat_chunks))

        # 根据配置限制块大小
        while (target_height // lat_chunks > self.max_chunk_size or
               target_width // lon_chunks > self.max_chunk_size):
            if target_height // lat_chunks > target_width // lon_chunks:
                lat_chunks += 1
            else:
                lon_chunks += 1

        # 确保至少有足够的块
        if lat_chunks * lon_chunks < min_chunks:
            # 增加块数
            if target_height / lat_chunks > target_width / lon_chunks:
                lat_chunks = max(lat_chunks, int(np.ceil(target_height / self.max_chunk_size)))
            else:
                lon_chunks = max(lon_chunks, int(np.ceil(target_width / self.max_chunk_size)))

        return lat_chunks, lon_chunks

    def run_optimized_analysis(self):
        """运行优化版分析"""
        memory_monitor = self.memory_monitor
        try:
            self.logger.info("开始优化版多GPU海洋热膨胀分析")

            # 记录开始时间
            start_time = time.time()

            # 清理所有GPU内存
            for gpu_id in self.gpu_devices:
                torch.cuda.set_device(gpu_id)
                torch.cuda.empty_cache()

            # 加载数据
            self.logger.info("加载并预处理数据...")

            # 首先加载GEBCO数据
            gebco_file = (self.gebco_2024 if self.gebco_2024 and os.path.exists(self.gebco_2024)
                         else self.gebco_2008)
            self.gebco_data = self.optimized_load_gebco(gebco_file)
            if self.generate_map_outputs:
                self.initialize_spatial_map_storage()

            # 加载NetCDF数据 - 使用float32减少内存占用
            self.logger.info("加载温度和盐度数据...")
            temp_2008_data, temp_coords = self.optimized_load_netcdf(
                self.temp_2008, ['temperature', 'temp', 'sea_temp'])
            temp_2024_data, _ = self.optimized_load_netcdf(
                self.temp_2024, ['temperature', 'temp', 'sea_temp'])
            salt_2008_data, _ = self.optimized_load_netcdf(
                self.salt_2008, ['salinity', 'salt', 'sea_salt'])
            salt_2024_data, _ = self.optimized_load_netcdf(
                self.salt_2024, ['salinity', 'salt', 'sea_salt'])

            # 保存坐标信息
            self.nc_coords = temp_coords

            # 保存数据引用
            self.temp_2008_data = temp_2008_data
            self.temp_2024_data = temp_2024_data
            self.salt_2008_data = salt_2008_data
            self.salt_2024_data = salt_2024_data

            # 准备深度层信息
            depths = temp_coords['depth']
            n_layers = len(depths)
            start_layer = int(self.start_layer or 0)
            end_layer = n_layers if self.end_layer is None else int(self.end_layer)

            if start_layer < 0:
                start_layer = max(0, n_layers + start_layer)
            if end_layer < 0:
                end_layer = max(0, n_layers + end_layer)

            start_layer = max(0, min(start_layer, n_layers))
            end_layer = max(start_layer, min(end_layer, n_layers))
            selected_layer_indices = list(range(start_layer, end_layer))
            selected_layers = len(selected_layer_indices)

            if selected_layers == 0:
                raise ValueError(
                    f"深度层范围为空: start_layer={self.start_layer}, "
                    f"end_layer={self.end_layer}, 可用范围=0-{n_layers - 1}"
                )

            self.logger.info(f"可用深度层总数: {n_layers}")
            self.logger.info(
                f"实际处理层范围: [{start_layer}, {end_layer})，共{selected_layers}层，"
                f"深度{depths[start_layer]:.1f}m-{depths[end_layer - 1]:.1f}m"
            )

            # 检查是否需要减少处理的层数以节省内存
            if selected_layers > 50:  # 如果层数很多
                # 检查总内存占用
                memory = psutil.virtual_memory()
                if memory.percent > 70:  # 如果系统内存使用率较高
                    self.logger.warning(f"系统内存使用率高({memory.percent}%)，考虑减少处理的层数")
                    if self.max_layers_in_parallel > 2:
                        self.max_layers_in_parallel = 2
                        self.logger.info(f"已将并行层数减少到 {self.max_layers_in_parallel}")

            if self.generate_map_outputs and self.max_layers_in_parallel > 2:
                self.max_layers_in_parallel = 2
                self.logger.info("启用地图输出时将最大并行层数限制为2，以控制空间图内存占用")

            # 准备层参数 - 为并行处理做准备
            layer_params = []
            for layer_idx in selected_layer_indices:
                depth = depths[layer_idx]

                # 计算层厚度
                if layer_idx == 0:
                    thickness = depths[1] - depth if len(depths) > 1 else 50
                elif layer_idx == len(depths) - 1:
                    thickness = depth - depths[layer_idx - 1]
                else:
                    thickness = (depths[layer_idx + 1] - depths[layer_idx - 1]) / 2

                # 选择GPU - 循环使用可用GPU
                gpu_id = self.gpu_devices[layer_idx % len(self.gpu_devices)]

                layer_params.append((layer_idx, depth, thickness, gpu_id))

            # 创建时间预测器
            time_predictor = TimePredictor(selected_layers)

            # 强制执行一次内存清理
            memory_monitor.cleanup_memory(critical=True)

            # 初始化结果列表
            all_results = []

            # 根据是否使用层并行选择处理方式
            if self.use_layer_parallelism:
                self.logger.info(f"使用多层并行处理: 最大并行层数={self.max_layers_in_parallel}")

                # 分批处理深度层
                for batch_start in range(0, selected_layers, self.max_layers_in_parallel):
                    batch_end = min(batch_start + self.max_layers_in_parallel, selected_layers)
                    batch_size = batch_end - batch_start

                    # 获取当前批次的参数
                    batch_params = layer_params[batch_start:batch_end]

                    # 显示批次信息
                    batch_abs_start = batch_params[0][0]
                    batch_abs_end = batch_params[-1][0]
                    depths_range = f"{depths[batch_abs_start]:.1f}m-{depths[batch_abs_end]:.1f}m"
                    self.logger.info(
                        f"处理深度层批次 {batch_start+1}-{batch_end}/{selected_layers} "
                        f"(绝对层{batch_abs_start}-{batch_abs_end}, {depths_range})"
                    )

                    # 使用ProcessPoolExecutor处理批次
                    batch_results = []
                    with ThreadPoolExecutor(max_workers=batch_size) as executor:
                        futures = [executor.submit(self.process_layer_chunks_parallel, params)
                                  for params in batch_params]

                        # 创建进度条
                        with tqdm(total=batch_size, desc=f"批次 {batch_start+1}-{batch_end}/{selected_layers}",
                                 unit="层") as pbar:
                            # 收集结果
                            for future in as_completed(futures):
                                try:
                                    result = future.result()
                                    self.accumulate_spatial_volume_change(
                                        result.pop('volume_changes_spatial', None)
                                    )
                                    batch_results.append(result)

                                    # 更新时间预测
                                    time_predictor.update()
                                    estimates = time_predictor.get_estimates()

                                    # 更新进度条
                                    pbar.update(1)
                                    pbar.set_postfix({
                                        '体积': f"{result['volume']/1e9:.3f}km³",
                                        '变化': f"{result['volume_change']/1e9:.6f}km³",
                                        '耗时': f"{result['processing_time']:.1f}s",
                                        '预计剩余': estimates['estimated_remaining_formatted'],
                                        '完成': f"{estimates['progress_percent']:.1f}%"
                                    })

                                    # 异步写入CSV
                                    self.write_layer_to_csv(result, result['processing_time'])

                                except Exception as e:
                                    self.logger.error(f"处理层失败: {e}")
                                    pbar.update(1)

                    # 添加到总结果
                    all_results.extend(batch_results)

                    # 检查内存状态
                    memory_monitor.check_memory()

                    # 显示批次摘要
                    batch_volume = sum(r['volume'] for r in batch_results)
                    batch_change = sum(r['volume_change'] for r in batch_results)
                    self.logger.info(f"批次 {batch_start+1}-{batch_end} 完成: "
                                    f"体积={batch_volume/1e9:.3f}km³, "
                                    f"变化={batch_change/1e9:.6f}km³")

            else:
                # 串行处理每个深度层（但每层内部使用多GPU）
                self.logger.info("使用层内并行处理（层之间串行）")

                with tqdm(layer_params, desc="处理深度层", unit="层") as pbar:
                    for params in pbar:
                        layer_idx, depth, thickness, gpu_id = params

                        # 显示当前层信息
                        pbar.set_description(f"处理深度 {depth:.1f}m (GPU{gpu_id})")

                        # 处理层
                        result = self.process_layer_chunks_parallel(params)
                        self.accumulate_spatial_volume_change(
                            result.pop('volume_changes_spatial', None)
                        )
                        all_results.append(result)

                        # 更新时间预测
                        time_predictor.update()
                        estimates = time_predictor.get_estimates()

                        # 更新进度条
                        pbar.set_postfix({
                            '体积': f"{result['volume']/1e9:.3f}km³",
                            '变化': f"{result['volume_change']/1e9:.6f}km³",
                            '耗时': f"{result['processing_time']:.1f}s",
                            '预计剩余': estimates['estimated_remaining_formatted'],
                            '完成': f"{estimates['progress_percent']:.1f}%"
                        })

                        # 异步写入CSV
                        self.write_layer_to_csv(result, result['processing_time'])

                        # 定期检查内存
                        if (layer_idx + 1) % 5 == 0:
                            memory_monitor.check_memory()

            # 汇总所有结果
            all_results.sort(key=lambda x: x['layer_idx'])  # 按层索引排序

            # 计算总体积和体积变化
            total_volume = sum(r['volume'] for r in all_results)
            total_volume_change = sum(r['volume_change'] for r in all_results)

            # 计算总处理时间
            total_processing_time = time.time() - start_time

            # 计算海平面上升
            sea_level_rise = (total_volume_change / self.ocean_area) * 1000  # mm
            base_year = self.config.get('base_year', 2000)
            target_year = self.config.get('target_year', 2025)

            # 准备结果
            final_results = {
                'total_volume': total_volume,
                'total_volume_change': total_volume_change,
                'sea_level_rise_mm': sea_level_rise,
                'annual_rate_mm': sea_level_rise / 1,  # 保持原脚本定义不变
                'processing_time_seconds': total_processing_time,
                'results_by_depth': all_results,
                'total_layers': selected_layers,
                'total_available_layers': n_layers,
                'start_layer': start_layer,
                'end_layer': end_layer,
                'base_year': base_year,
                'target_year': target_year,
                'version': 'optimized_multi_gpu'
            }

            if self.generate_map_outputs:
                jpg_file, nc_file = self.generate_final_map_outputs(final_results)
                final_results['map_jpg'] = jpg_file
                final_results['map_netcdf'] = nc_file

            # 保存结果
            self.save_results(final_results)

            # 写入总结到CSV
            self.write_summary_to_csv(final_results)

            # 显示完成信息
            self.logger.info(f"分析完成: 总处理时间 {total_processing_time/60:.1f} 分钟")
            self.logger.info(f"海平面上升: {sea_level_rise:.4f} mm ({base_year}-{target_year})")
            self.logger.info(f"年均上升率: {sea_level_rise/1:.4f} mm/年")

            return final_results

        except Exception as e:
            self.logger.error(f"优化版多GPU分析失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            # 最终清理
            memory_monitor.cleanup_memory(critical=True)
            self.async_writer.shutdown(wait=True)

    def save_results(self, results):
        """保存结果到JSON和CSV"""
        # 辅助函数：转换numpy类型为Python原生类型
        def convert_to_native_type(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native_type(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_type(item) for item in obj]
            else:
                return obj

        # 处理结果字典
        json_results = {
            'analysis_summary': {
                'base_year': int(results.get('base_year', self.config.get('base_year', 2000))),
                'target_year': int(results.get('target_year', self.config.get('target_year', 2025))),
                'total_sea_level_rise_mm': float(results['sea_level_rise_mm']),
                'annual_rate_mm_per_year': float(results['annual_rate_mm'])
            },
            'volume_analysis': {
                'total_volume_km3': float(results['total_volume'] / 1e9),
                'total_volume_change_km3': float(results['total_volume_change'] / 1e9)
            },
            'processing_info': {
                'total_time_seconds': float(results['processing_time_seconds']),
                'total_time_minutes': float(results['processing_time_seconds'] / 60),
                'total_layers': int(results['total_layers']),
                'total_available_layers': int(results.get('total_available_layers', results['total_layers'])),
                'start_layer': int(results.get('start_layer', 0)),
                'end_layer': int(results.get('end_layer', results['total_layers'])),
                'parallel_strategy': 'optimized_multi_gpu',
                'version': 'optimized_multi_gpu',
                'island_filling_enabled': bool(self.enable_island_filling),
                'max_island_size': int(self.max_island_size) if self.enable_island_filling else None,
                'island_fill_ocean_ratio': float(self.island_fill_ocean_ratio) if self.enable_island_filling else None,
                'map_outputs_enabled': bool(self.generate_map_outputs),
                'map_downsample_factor': int(self.map_downsample_factor) if self.generate_map_outputs else None,
                'jpg_downsample_factor': int(self.map_downsample_factor) if self.generate_map_outputs else None,
                'nc_full_resolution': bool(self.nc_save_full_resolution) if self.generate_map_outputs else None,
                'nc_write_block_rows': int(self.nc_write_block_rows) if self.generate_map_outputs else None
            }
        }

        if results.get('map_jpg') or results.get('map_netcdf'):
            json_results['map_outputs'] = {
                'jpg': results.get('map_jpg'),
                'netcdf': results.get('map_netcdf'),
                'note': 'Map output is based on island-filled input data when island filling is enabled.'
            }

        # 处理深度分析数据
        depth_analysis = []
        for r in results['results_by_depth']:
            depth_analysis.append({
                'depth_m': float(r['depth']),
                'volume_km3': float(r['volume'] / 1e9),
                'volume_change_km3': float(r['volume_change'] / 1e9),
                'chunks_processed': int(r['chunks_processed'])
            })

        json_results['depth_analysis'] = depth_analysis

        # 转换为原生类型
        json_results = convert_to_native_type(json_results)

        # 保存JSON
        json_file = os.path.join(self.output_dir, 'optimized_results.json')
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"JSON结果保存成功: {json_file}")
        except Exception as e:
            self.logger.error(f"保存JSON结果失败: {e}")

        # 保存CSV
        csv_file = os.path.join(self.output_dir, 'data.csv')
        try:
            df = pd.DataFrame(depth_analysis)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            self.logger.info(f"CSV结果保存成功: {csv_file}")
        except Exception as e:
            self.logger.error(f"保存CSV结果失败: {e}")

        # 生成性能报告
        self.generate_performance_report(results)

    def generate_performance_report(self, results):
        """生成性能报告"""
        try:
            # 提取性能数据
            processing_time = results['processing_time_seconds']
            n_layers = results['total_layers']
            avg_layer_time = processing_time / n_layers if n_layers > 0 else 0
            base_year = results.get('base_year', self.config.get('base_year', 2000))
            target_year = results.get('target_year', self.config.get('target_year', 2025))

            # 获取每层的处理时间
            layer_times = [r.get('processing_time', 0) for r in results['results_by_depth']]
            max_layer_time = max(layer_times) if layer_times else 0
            min_layer_time = min(layer_times) if layer_times else 0

            # 生成报告内容
            report = f"""
========== 优化版多GPU海洋热膨胀分析性能报告 ==========
版本: 优化版多GPU海洋热膨胀计算器
对比年份: {target_year} 相对于 {base_year}
总处理时间: {processing_time/60:.1f} 分钟

GPU配置:
├─ 使用GPU数量: {len(self.gpu_devices)}
├─ GPU设备: {self.gpu_devices}
├─ 混合精度: {'启用' if self.use_mixed_precision else '禁用'}
└─ 坐标缓存: {'启用' if self.use_coord_cache else '禁用'}

小岛屿填充:
├─ 填充状态: {'已启用' if self.enable_island_filling else '已禁用'}
├─ 最大岛屿大小: {self.max_island_size} 网格
└─ 海洋包围度阈值: {self.island_fill_ocean_ratio}

并行策略:
├─ 多深度层并行: {'启用' if self.use_layer_parallelism else '禁用'}
├─ 最大并行层数: {self.max_layers_in_parallel}
├─ 块大小范围: {self.min_chunk_size}-{self.max_chunk_size}
└─ 内存高效插值: {'启用' if self.mem_efficient_interpolation else '禁用'}

地图输出:
├─ JPG/NetCDF输出: {'已启用' if self.generate_map_outputs else '已禁用'}
├─ JPG降采样倍数: {self.map_downsample_factor}x
├─ NetCDF原始分辨率: {'是' if self.nc_save_full_resolution else '否'}
├─ NetCDF写入块行数: {self.nc_write_block_rows}
├─ JPG文件: {results.get('map_jpg', '未生成')}
└─ NetCDF文件: {results.get('map_netcdf', '未生成')}

性能统计:
├─ 可用总深度层数: {results.get('total_available_layers', n_layers)}
├─ 实际处理层数: {n_layers}
├─ 处理层范围: [{results.get('start_layer', 0)}, {results.get('end_layer', n_layers)})
├─ 平均每层处理时间: {avg_layer_time:.1f}秒
├─ 最长层处理时间: {max_layer_time:.1f}秒
├─ 最短层处理时间: {min_layer_time:.1f}秒
└─ 处理速度: {n_layers/(processing_time/60):.2f}层/分钟

内存管理:
├─ 智能内存监控: 启用
├─ 动态内存清理: 启用
└─ 异步I/O操作: 启用

计算结果:
├─ 总海洋体积: {results['total_volume']/1e15:.2f} 百万立方千米
├─ 热膨胀体积变化: {results['total_volume_change']/1e9:.6f} 立方千米
├─ 海平面上升 ({base_year}-{target_year}): {results['sea_level_rise_mm']:.4f} 毫米
└─ 年均上升率: {results['annual_rate_mm']:.4f} 毫米/年

优化亮点:
├─ 多深度层并行: 同时处理多个深度层
├─ 内存优化插值: 减少GPU内存使用
├─ 坐标缓存: 避免重复计算和传输
├─ 智能内存管理: 基于使用率的动态清理
├─ 异步I/O: 减少文件写入等待时间
├─ 精确进度预测: 提供剩余时间估计
└─ 高效错误处理: 容错机制确保稳定运行
"""

            # 保存报告
            report_file = os.path.join(self.output_dir, 'performance_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            self.logger.info(f"性能报告已保存: {report_file}")

        except Exception as e:
            self.logger.error(f"生成性能报告失败: {e}")


def annual_temperature_file(annual_dir, year):
    return os.path.join(annual_dir, f"Annual_Temp_{year}.nc")


def annual_salinity_file(annual_dir, year):
    return os.path.join(annual_dir, f"Annual_Salinity_{year}.nc")


def create_optimized_config(base_year=2000, target_year=2025, output_dir='./outputs/',
                            annual_dir=None, gebco_file=None):
    """创建优化版配置"""
    if annual_dir is None or gebco_file is None:
        raise ValueError("annual_dir and gebco_file must be provided explicitly")
    return {
        'gebco_2008': gebco_file,
        'gebco_2024': gebco_file,
        'temp_2008': annual_temperature_file(annual_dir, base_year),
        'temp_2024': annual_temperature_file(annual_dir, target_year),
        'salt_2008': annual_salinity_file(annual_dir, base_year),
        'salt_2024': annual_salinity_file(annual_dir, target_year),
        'base_year': base_year,
        'target_year': target_year,

        # GPU配置
        'gpu_devices': [0, 1],  # 使用的GPU设备ID

        # 并行和分块参数
        'min_chunk_size': 180,  # 减小块大小以降低内存使用
        'max_chunk_size': 400,  # 减小最大块大小
        'max_layers_in_parallel': 4,  # 减少同时处理的深度层数，降低内存压力
        'use_layer_parallelism': True,  # 启用多层并行
        'use_mixed_precision': True,  # 使用混合精度计算
        'mem_efficient_interpolation': True,  # 内存高效插值

        # 深度层范围控制，采用[start_layer, end_layer)。测试单层用0-1；全量运行将end_layer改为None
        'start_layer': 0,
        'end_layer': None,

        # 小岛屿填充配置 - 与testnew.py保持一致
        'enable_island_filling': True,  # 先填充温盐数据小空洞，再计算热膨胀图
        'max_island_size': 5,  # 最大填充岛屿大小（网格数）
        'island_fill_ocean_ratio': 0.8,  # 被海洋包围的最小比例

        # 内存管理参数
        'memory_warning_threshold': 70,  # 降低内存使用率警告阈值，提前清理
        'critical_threshold': 85,  # 降低内存使用率临界阈值

        # 缓存配置
        'use_coord_cache': True,  # 禁用坐标缓存以节省内存

        # 输出配置
        'output_dir': output_dir,  # 输出目录
        'generate_map_outputs': True,  # 输出最终JPG和NetCDF地图
        'map_downsample_factor': 4,  # JPG地图输出降采样倍数；NetCDF不受该参数影响
        'nc_save_full_resolution': True,  # NetCDF保存GEBCO原始15弧秒分辨率
        'nc_write_block_rows': 256,  # 原分辨率NetCDF按纬度分块写入，避免一次性写入
    }


def parse_command_line_args():
    parser = argparse.ArgumentParser(
        description=(
            "批量运行TEOS-10热膨胀计算；可按逐年相邻或起止年份直接计算。"
        )
    )
    parser.add_argument('start_year', type=int, nargs='?', help='起始年份，例如1940')
    parser.add_argument('end_year', type=int, nargs='?', help='结束年份，例如2025')
    parser.add_argument(
        '--mode',
        choices=['yearly', 'direct'],
        default='yearly',
        help='yearly: 逐年相邻计算；direct: 结束年相对于起始年直接计算'
    )
    parser.add_argument(
        '--direct',
        action='store_true',
        help='等同于 --mode direct'
    )
    parser.add_argument(
        '--resume-from',
        type=int,
        default=None,
        help='yearly模式下从指定基准年份继续计算，例如1980表示从1981_vs_1980开始'
    )
    parser.add_argument(
        '--annual-dir',
        required=True,
        help='年平均温盐NetCDF所在目录'
    )
    parser.add_argument(
        '--gebco',
        required=True,
        help='GEBCO 15 arcsecond ocean DEM GeoTIFF'
    )
    parser.add_argument(
        '--output-root',
        default='./Results/',
        help='批量计算输出根目录'
    )

    args = parser.parse_args()
    if (args.start_year is None) != (args.end_year is None):
        parser.error('start_year 和 end_year 必须同时提供')
    if args.direct:
        args.mode = 'direct'
    if args.start_year is not None and args.end_year <= args.start_year:
        parser.error('end_year 必须大于 start_year')
    if args.resume_from is not None:
        if args.start_year is None:
            parser.error('--resume-from 必须和 start_year/end_year 一起使用')
        if args.mode == 'direct':
            parser.error('--resume-from 只适用于 yearly 模式')
        if args.resume_from < args.start_year or args.resume_from >= args.end_year:
            parser.error('--resume-from 必须满足 start_year <= resume_from < end_year')
    return args


def build_year_pairs(start_year, end_year, mode, resume_from=None):
    if mode == 'direct':
        return [(start_year, end_year)]
    first_base_year = start_year if resume_from is None else resume_from
    return [(year, year + 1) for year in range(first_base_year, end_year)]


def build_output_dir(output_root, mode, start_year, end_year, base_year, target_year):
    group_dir = f"{mode}_{start_year}_{end_year}"
    task_dir = f"{target_year}_vs_{base_year}"
    return os.path.join(output_root, group_dir, task_dir)


def validate_input_files(config):
    required_files = [
        config['temp_2008'],
        config['temp_2024'],
        config['salt_2008'],
        config['salt_2024'],
    ]
    return [file_path for file_path in required_files if not os.path.exists(file_path)]


def run_single_calculation(config):
    calculator = None

    try:
        configure_logging_for_output(config['output_dir'])
        base_year = config.get('base_year')
        target_year = config.get('target_year')
        logger.info(f"当前计算: {target_year} 相对于 {base_year}")
        logger.info(f"温度输入: {config['temp_2008']} -> {config['temp_2024']}")
        logger.info(f"盐度输入: {config['salt_2008']} -> {config['salt_2024']}")
        logger.info(f"输出目录: {config['output_dir']}")

        missing_files = validate_input_files(config)
        if missing_files:
            logger.error("输入年平均温盐文件缺失:")
            for missing_file in missing_files:
                logger.error(f"  {missing_file}")
            return False, None

        calculator = OptimizedMultiGPUCalculator(config)

        # 运行分析
        print(f"开始优化版多GPU海洋热膨胀分析: {target_year} 相对于 {base_year}")
        print(f"输出目录: {config['output_dir']}")
        results = calculator.run_optimized_analysis()

        # 打印结果摘要
        print("\n分析结果已保存到:")
        print(f"- {os.path.join(calculator.output_dir, 'optimized_results.json')} (完整结果)")
        print(f"- {os.path.join(calculator.output_dir, 'data.csv')} (深度层数据)")
        print(f"- {os.path.join(calculator.output_dir, 'performance_report.txt')} (性能报告)")
        if results.get('map_jpg'):
            print(f"- {results['map_jpg']} (最终热膨胀JPG地图)")
        if results.get('map_netcdf'):
            print(f"- {results['map_netcdf']} (最终热膨胀NetCDF数据)")

        print(f"\n========== 分析结果摘要 ==========")
        print(f"可用深度层数: {results['total_available_layers']}")
        print(f"实际处理层范围: [{results['start_layer']}, {results['end_layer']})")
        print(f"处理的深度层数: {results['total_layers']}")
        print(f"总处理时间: {results['processing_time_seconds']/60:.1f} 分钟")
        print(f"海平面上升: {results['sea_level_rise_mm']:.4f} mm ({base_year}-{target_year})")
        print(f"年均上升率: {results['annual_rate_mm']:.4f} mm/年")

        print("=" * 50)
        print("优化版多GPU分析完成！")
        return True, results

    except KeyboardInterrupt:
        print("\n用户中断了计算...")
        if calculator:
            calculator.logger.info("用户中断了计算")
        raise

    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")
        if calculator:
            calculator.logger.error(f"分析失败: {e}")
            import traceback
            calculator.logger.error(traceback.format_exc())
        return False, None

    finally:
        # 清理资源
        if calculator:
            try:
                calculator.memory_monitor.cleanup_memory(critical=True)
                calculator.async_writer.shutdown(wait=True)
                calculator.logger.info("资源清理完成")
                print("GPU资源已清理")
            except:
                pass

        # 强制Python垃圾回收
        gc.collect()


def run_batch_calculations(args):
    year_pairs = build_year_pairs(args.start_year, args.end_year, args.mode, args.resume_from)
    successful_tasks = []
    failed_tasks = []

    print("=" * 80)
    print(f"批量计算模式: {args.mode}")
    print(f"年份范围: {args.start_year}-{args.end_year}")
    if args.resume_from is not None:
        print(f"恢复起始年份: {args.resume_from}，首个任务为 {args.resume_from + 1}_vs_{args.resume_from}")
    print(f"任务数量: {len(year_pairs)}")
    print(f"年平均温盐目录: {args.annual_dir}")
    print(f"GEBCO DEM: {args.gebco}")
    print(f"输出根目录: {args.output_root}")
    print("=" * 80)

    for task_index, (base_year, target_year) in enumerate(year_pairs, start=1):
        output_dir = build_output_dir(
            args.output_root, args.mode, args.start_year, args.end_year,
            base_year, target_year
        )
        config = create_optimized_config(
            base_year=base_year,
            target_year=target_year,
            output_dir=output_dir,
            annual_dir=args.annual_dir,
            gebco_file=args.gebco
        )

        print("\n" + "=" * 80)
        print(f"当前任务: {task_index}/{len(year_pairs)}")
        print(f"当前计算: {target_year} 相对于 {base_year}")
        print(f"温度输入: {config['temp_2008']} -> {config['temp_2024']}")
        print(f"盐度输入: {config['salt_2008']} -> {config['salt_2024']}")
        print(f"输出目录: {output_dir}")
        print("=" * 80)

        success, _ = run_single_calculation(config)
        task_label = f"{target_year}_vs_{base_year}"
        if success:
            successful_tasks.append(task_label)
        else:
            failed_tasks.append(task_label)

    print("\n" + "=" * 80)
    print("批量计算完成")
    print(f"成功任务数量: {len(successful_tasks)}")
    print(f"失败任务数量: {len(failed_tasks)}")
    if failed_tasks:
        print(f"失败任务: {failed_tasks}")
    print("=" * 80)


if __name__ == "__main__":
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)

    args = parse_command_line_args()

    try:
        if args.start_year is None:
            config = create_optimized_config(annual_dir=args.annual_dir, gebco_file=args.gebco)
            run_single_calculation(config)
        else:
            run_batch_calculations(args)
    finally:
        print("程序结束")
