#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海洋温度盐度年平均数据处理器 - 单年版本
功能：
1. 使用明确指定的文件路径读取单年12个月的温度和盐度NetCDF文件
2. 计算年平均值
3. 保持原有的文件结构和nodata值
4. 输出年平均NetCDF文件
"""

import argparse

import netCDF4 as nc
import numpy as np
import os
import logging
import warnings
from tqdm import tqdm
import gc

# 抑制警告
warnings.filterwarnings("ignore")

class OceanAnnualAverager:
    """海洋温度盐度年平均处理器 - 单年版本"""

    def __init__(self, config):
        self.config = config
        self.setup_logging()

        # 输出目录
        self.output_dir = config['output_dir']

        # 年份和文件路径
        self.year = config['year']
        self.temp_files = config['temperature_files']
        self.salt_files = config['salinity_files']

        # 变量名候选列表（从您的代码中提取）
        self.temp_var_candidates = ['temperature', 'temp', 'sea_temp']
        self.salt_var_candidates = ['salinity', 'salt', 'sea_salt']

        # 坐标变量名候选列表
        self.coord_candidates = {
            'depth': ['depth', 'depth_std', 'z', 'level', 'lev'],
            'lat': ['lat', 'latitude', 'y'],
            'lon': ['lon', 'longitude', 'x']
        }

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info("海洋年平均数据处理器初始化完成（单年版本）")
        self.logger.info(f"处理年份: {self.year}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"温度文件数量: {len(self.temp_files)}")
        self.logger.info(f"盐度文件数量: {len(self.salt_files)}")

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ocean_annual_averager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_file_paths(self):
        """验证文件路径的有效性"""
        missing_files = []

        # 检查温度文件
        for month, file_path in self.temp_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"温度-{month}月: {file_path}")

        # 检查盐度文件
        for month, file_path in self.salt_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f"盐度-{month}月: {file_path}")

        return missing_files

    def load_netcdf_variable(self, file_path, var_candidates):
        """加载NetCDF变量，参考您的代码逻辑"""
        self.logger.debug(f"加载文件: {os.path.basename(file_path)}")

        with nc.Dataset(file_path, 'r') as ds:
            # 找到变量
            var_name = None
            for candidate in var_candidates:
                if candidate in ds.variables:
                    var_name = candidate
                    break

            if var_name is None:
                # 尝试模糊匹配
                for candidate in var_candidates:
                    for available_var in ds.variables.keys():
                        if candidate.lower() in available_var.lower():
                            var_name = available_var
                            break
                    if var_name:
                        break

            if var_name is None:
                self.logger.error(f"可用变量: {list(ds.variables.keys())}")
                raise ValueError(f"找不到变量: {var_candidates}")

            # 获取变量
            var = ds.variables[var_name]
            self.logger.debug(f"读取变量: {var_name}, 形状: {var.shape}, 维度: {var.dimensions}")

            # 读取数据
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = var[:]

            # 获取变量属性
            var_attrs = {}
            for attr_name in var.ncattrs():
                var_attrs[attr_name] = getattr(var, attr_name)

            # 处理填充值
            if hasattr(data, 'mask'):
                data = np.ma.filled(data, np.nan)

            # 处理各种填充值
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

            # 转换为float32
            data = data.astype(np.float32)

            # 替换填充值为NaN
            for fill_val in set(fill_values):
                if not np.isnan(fill_val):
                    mask = np.abs(data - fill_val) < 1e-6
                    if np.any(mask):
                        data[mask] = np.nan

            self.logger.debug(f"处理后数据形状: {data.shape}")

            # 读取坐标信息
            coords = {}
            for coord_type, candidates in self.coord_candidates.items():
                for candidate in candidates:
                    if candidate in ds.variables:
                        coord_var = ds.variables[candidate]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            coord_data = coord_var[:]

                        if hasattr(coord_data, 'mask'):
                            coord_data = np.ma.filled(coord_data, np.nan)

                        coord_data = coord_data.astype(np.float32)

                        # 获取坐标属性
                        coord_attrs = {}
                        for attr_name in coord_var.ncattrs():
                            coord_attrs[attr_name] = getattr(coord_var, attr_name)

                        coords[coord_type] = {
                            'data': coord_data,
                            'attrs': coord_attrs,
                            'name': candidate
                        }
                        self.logger.debug(f"坐标 {coord_type} ({candidate}): 长度={len(coord_data)}")
                        break

            # 获取全局属性
            global_attrs = {}
            for attr_name in ds.ncattrs():
                global_attrs[attr_name] = getattr(ds, attr_name)

            return data, var_name, var_attrs, coords, global_attrs

    def validate_and_fix_data_shapes(self, monthly_data_list):
        """验证和修复数据形状一致性"""
        if not monthly_data_list:
            raise ValueError("月度数据列表为空")

        # 获取第一个数据的形状作为参考
        reference_shape = monthly_data_list[0].shape
        self.logger.info(f"参考数据形状: {reference_shape}")

        # 检查所有数据形状是否一致
        for i, data in enumerate(monthly_data_list):
            if data.shape != reference_shape:
                self.logger.warning(f"第{i+1}个月数据形状不一致: {data.shape} vs {reference_shape}")

                # 尝试修复形状不一致问题
                if len(data.shape) == len(reference_shape):
                    # 维度数相同，但大小不同
                    # 裁剪到最小公共形状
                    min_shape = tuple(min(data.shape[j], reference_shape[j]) for j in range(len(reference_shape)))
                    self.logger.info(f"裁剪到最小公共形状: {min_shape}")

                    # 更新所有数据到公共形状
                    fixed_data_list = []
                    for j, month_data in enumerate(monthly_data_list):
                        if len(month_data.shape) == 3:
                            fixed_data = month_data[:min_shape[0], :min_shape[1], :min_shape[2]]
                        elif len(month_data.shape) == 2:
                            fixed_data = month_data[:min_shape[0], :min_shape[1]]
                        else:
                            fixed_data = month_data
                        fixed_data_list.append(fixed_data)

                    return fixed_data_list
                else:
                    raise ValueError(f"第{i+1}个月数据维度不匹配: {len(data.shape)}D vs {len(reference_shape)}D")

        return monthly_data_list

    def calculate_annual_average(self, monthly_data_list):
        """计算年平均值，正确处理NaN值"""
        self.logger.info(f"计算年平均值，共{len(monthly_data_list)}个月的数据")

        # 验证和修复数据形状
        monthly_data_list = self.validate_and_fix_data_shapes(monthly_data_list)

        # 将所有月度数据堆叠
        try:
            monthly_stack = np.stack(monthly_data_list, axis=0)
        except ValueError as e:
            self.logger.error(f"数据堆叠失败: {e}")
            # 打印每个月数据的形状用于调试
            for i, data in enumerate(monthly_data_list):
                self.logger.error(f"第{i+1}月数据形状: {data.shape}")
            raise

        # 计算有效数据的掩码
        valid_mask = ~np.isnan(monthly_stack)
        valid_count = np.sum(valid_mask, axis=0)

        # 计算平均值（忽略NaN）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            annual_mean = np.nanmean(monthly_stack, axis=0)

        # 对于完全没有有效数据的位置，设为NaN
        annual_mean = np.where(valid_count > 0, annual_mean, np.nan)

        # 统计信息
        total_pixels = np.prod(annual_mean.shape)
        valid_pixels = np.sum(~np.isnan(annual_mean))
        self.logger.info(f"年平均值计算完成，有效数据覆盖率: {valid_pixels/total_pixels:.1%}")
        self.logger.info(f"最终数据形状: {annual_mean.shape}")

        return annual_mean.astype(np.float32)

    def normalize_longitude(self, lon):
        """将经度标准化到-180到180范围"""
        lon = np.asarray(lon)
        return ((lon + 180) % 360) - 180

    def process_coordinates(self, coords):
        """处理坐标，确保符合标准格式"""
        processed_coords = {}

        for coord_type, coord_info in coords.items():
            coord_data = coord_info['data'].copy()

            # 经度标准化
            if coord_type == 'lon':
                coord_data = self.normalize_longitude(coord_data)
                self.logger.info(f"经度归一化: [{coord_data.min():.2f}, {coord_data.max():.2f}]")

            processed_coords[coord_type] = {
                'data': coord_data,
                'attrs': coord_info['attrs'],
                'name': coord_info['name']
            }

        return processed_coords

    def create_output_netcdf(self, data, var_name, var_attrs, coords, global_attrs,
                           output_path, data_type):
        """创建输出NetCDF文件，保持原有结构"""
        self.logger.info(f"创建输出文件: {output_path}")
        self.logger.info(f"数据形状: {data.shape}")

        # 打印坐标信息用于调试
        coord_info_str = []
        for coord_type, coord_info in coords.items():
            coord_info_str.append(f"{coord_type}: {len(coord_info['data'])}")
        self.logger.info(f"坐标长度: {', '.join(coord_info_str)}")

        with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
            # 设置全局属性
            for attr_name, attr_value in global_attrs.items():
                try:
                    setattr(ds, attr_name, attr_value)
                except:
                    pass  # 忽略无法设置的属性

            # 添加处理信息
            ds.setncattr('processing_info', f'Annual average of {self.year} monthly data')
            ds.setncattr('created_by', 'Ocean Annual Averager - Single Year Version')
            ds.setncattr('processing_date', np.datetime64('now').astype(str))
            ds.setncattr('source_year', self.year)
            ds.setncattr('data_type', data_type)

            # 创建维度 - 按照数据的实际形状顺序
            dimension_mapping = {}

            # 根据数据形状确定维度顺序
            if len(data.shape) == 3:
                expected_order = ['depth', 'lat', 'lon']
                actual_shapes = [len(coords[coord_type]['data']) for coord_type in expected_order if coord_type in coords]

                # 检查形状是否匹配
                if len(actual_shapes) == 3 and tuple(actual_shapes) == data.shape:
                    # 形状匹配，按照预期顺序创建维度
                    for i, coord_type in enumerate(expected_order):
                        if coord_type in coords:
                            coord_info = coords[coord_type]
                            dim_name = coord_info['name']
                            ds.createDimension(dim_name, len(coord_info['data']))
                            dimension_mapping[i] = dim_name
                            self.logger.info(f"创建维度 {dim_name}: {len(coord_info['data'])}")
                else:
                    # 形状不匹配，尝试重新排列
                    self.logger.warning(f"数据形状 {data.shape} 与坐标长度 {actual_shapes} 不匹配，尝试自动调整")

                    # 尝试找到匹配的维度顺序
                    coord_sizes = {}
                    for coord_type, coord_info in coords.items():
                        coord_sizes[coord_type] = len(coord_info['data'])

                    # 为每个数据维度找到对应的坐标
                    dimension_order = []
                    for data_dim_size in data.shape:
                        for coord_type, coord_size in coord_sizes.items():
                            if coord_size == data_dim_size and coord_type not in dimension_order:
                                dimension_order.append(coord_type)
                                break

                    if len(dimension_order) == 3:
                        for i, coord_type in enumerate(dimension_order):
                            coord_info = coords[coord_type]
                            dim_name = coord_info['name']
                            ds.createDimension(dim_name, len(coord_info['data']))
                            dimension_mapping[i] = dim_name
                            self.logger.info(f"创建维度 {dim_name}: {len(coord_info['data'])} (自动匹配)")
                    else:
                        raise ValueError(f"无法匹配数据形状 {data.shape} 与可用坐标")
            else:
                # 非3维数据，按照坐标顺序创建
                coord_order = ['depth', 'lat', 'lon'] if 'depth' in coords else ['lat', 'lon']
                for i, coord_type in enumerate(coord_order):
                    if coord_type in coords:
                        coord_info = coords[coord_type]
                        dim_name = coord_info['name']
                        if i < len(data.shape):
                            ds.createDimension(dim_name, len(coord_info['data']))
                            dimension_mapping[i] = dim_name

            # 创建坐标变量
            for coord_type, coord_info in coords.items():
                coord_data = coord_info['data']
                coord_attrs = coord_info['attrs']
                coord_name = coord_info['name']

                if coord_name in [dim for dim in ds.dimensions]:
                    # 创建坐标变量
                    coord_var = ds.createVariable(coord_name, 'f4', (coord_name,))
                    coord_var[:] = coord_data

                    # 设置坐标属性
                    for attr_name, attr_value in coord_attrs.items():
                        try:
                            setattr(coord_var, attr_name, attr_value)
                        except:
                            pass

            # 创建数据变量 - 使用实际的维度顺序
            dimensions = [dimension_mapping[i] for i in sorted(dimension_mapping.keys())]
            self.logger.info(f"数据变量维度顺序: {dimensions}")

            # 确定填充值
            original_fill_value = var_attrs.get('_FillValue', -9999.0)

            # 创建主要数据变量
            data_var = ds.createVariable(var_name, 'f4', dimensions,
                                       fill_value=original_fill_value, zlib=True, complevel=4)

            # 处理NaN值 - 设置为原始的填充值
            output_data = data.copy()
            output_data = np.where(np.isnan(output_data), original_fill_value, output_data)

            # 验证形状匹配
            expected_shape = tuple(ds.dimensions[dim].size for dim in dimensions)
            if output_data.shape != expected_shape:
                self.logger.error(f"形状不匹配: 数据形状 {output_data.shape}, 期望形状 {expected_shape}")
                raise ValueError(f"数据形状 {output_data.shape} 与维度定义 {expected_shape} 不匹配")

            data_var[:] = output_data

            # 设置变量属性
            for attr_name, attr_value in var_attrs.items():
                try:
                    setattr(data_var, attr_name, attr_value)
                except:
                    pass

            # 添加处理特定的属性
            data_var.setncattr('temporal_resolution', 'annual_average')
            data_var.setncattr('source_months', '1-12')
            data_var.setncattr('processing_year', self.year)

        self.logger.info(f"文件创建完成: {output_path}")

    def run_processing(self):
        """运行完整的处理流程"""
        self.logger.info(f"开始处理{self.year}年年平均数据")

        # 验证文件路径
        missing_files = self.validate_file_paths()
        if missing_files:
            self.logger.error(f"存在缺失文件:")
            for missing_file in missing_files:
                self.logger.error(f"  {missing_file}")
            return False

        try:
            # 处理温度数据
            self.logger.info(f"处理{self.year}年温度数据...")
            temp_monthly_data = []
            temp_coords = None
            temp_var_name = None
            temp_var_attrs = None
            temp_global_attrs = None

            for month in tqdm(range(1, 13), desc="加载温度数据"):
                if month not in self.temp_files:
                    self.logger.error(f"{month}月温度文件路径未配置")
                    return False

                file_path = self.temp_files[month]
                data, var_name, var_attrs, coords, global_attrs = self.load_netcdf_variable(
                    file_path, self.temp_var_candidates
                )
                temp_monthly_data.append(data)

                # 保存第一个文件的元数据
                if temp_coords is None:
                    temp_coords = self.process_coordinates(coords)
                    temp_var_name = var_name
                    temp_var_attrs = var_attrs
                    temp_global_attrs = global_attrs

            # 计算温度年平均
            temp_annual = self.calculate_annual_average(temp_monthly_data)

            # 处理盐度数据
            self.logger.info(f"处理{self.year}年盐度数据...")
            salt_monthly_data = []
            salt_coords = None
            salt_var_name = None
            salt_var_attrs = None
            salt_global_attrs = None

            for month in tqdm(range(1, 13), desc="加载盐度数据"):
                if month not in self.salt_files:
                    self.logger.error(f"{month}月盐度文件路径未配置")
                    return False

                file_path = self.salt_files[month]
                data, var_name, var_attrs, coords, global_attrs = self.load_netcdf_variable(
                    file_path, self.salt_var_candidates
                )
                salt_monthly_data.append(data)

                # 保存第一个文件的元数据
                if salt_coords is None:
                    salt_coords = self.process_coordinates(coords)
                    salt_var_name = var_name
                    salt_var_attrs = var_attrs
                    salt_global_attrs = global_attrs

            # 计算盐度年平均
            salt_annual = self.calculate_annual_average(salt_monthly_data)

            # 创建输出文件
            temp_output = os.path.join(self.output_dir, f'Annual_Temp_{self.year}.nc')
            salt_output = os.path.join(self.output_dir, f'Annual_Salinity_{self.year}.nc')

            # 保存温度年平均
            self.create_output_netcdf(
                temp_annual, temp_var_name, temp_var_attrs, temp_coords,
                temp_global_attrs, temp_output, 'temperature'
            )

            # 保存盐度年平均
            self.create_output_netcdf(
                salt_annual, salt_var_name, salt_var_attrs, salt_coords,
                salt_global_attrs, salt_output, 'salinity'
            )

            # 清理内存
            del temp_monthly_data, salt_monthly_data, temp_annual, salt_annual
            gc.collect()

            self.logger.info(f"{self.year}年处理完成:")
            self.logger.info(f"  温度: {temp_output}")
            self.logger.info(f"  盐度: {salt_output}")

            return True

        except Exception as e:
            self.logger.error(f"{self.year}年处理失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


def get_decade_dir(year):
    """根据年份返回数据所在的年代目录名。"""
    decade_start = (year // 10) * 10
    decade_end = decade_start + 9
    if 2020 <= year <= 2025:
        return "2020_2024"
    return f"{decade_start}_{decade_end}"


def create_config(year, temperature_root, salinity_root, output_dir):
    """创建配置 - 单年版本"""
    decade_dir = get_decade_dir(year)
    temp_dir = os.path.join(temperature_root, decade_dir)
    salt_dir = os.path.join(salinity_root, decade_dir)

    return {
        # 处理年份
        'year': year,

        # 输出目录
        'output_dir': output_dir,

        # 温度文件路径（12个月）
        'temperature_files': {
            month: os.path.join(
                temp_dir,
                f"IAPv4_Temp_monthly_1_6000m_year_{year}_month_{month:02d}.nc"
            )
            for month in range(1, 13)
        },

        # 盐度文件路径（12个月）
        'salinity_files': {
            month: os.path.join(
                salt_dir,
                f"IAPv2_Salinity_monthly_1_6000m_year_{year}_month_{month:02d}.nc"
            )
            for month in range(1, 13)
        }
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate annual IAP temperature and salinity averages without changing monthly-input handling."
    )
    parser.add_argument("--start-year", type=int, default=1940)
    parser.add_argument("--end-year", type=int, default=1998)
    parser.add_argument("--temperature-root", required=True, help="Root containing decade temperature directories")
    parser.add_argument("--salinity-root", required=True, help="Root containing decade salinity directories")
    parser.add_argument("--output-dir", required=True, help="Directory for annual NetCDF outputs")
    args = parser.parse_args()
    if args.end_year < args.start_year:
        parser.error("--end-year must be greater than or equal to --start-year")
    return args


if __name__ == "__main__":
    args = parse_args()
    start_year = args.start_year
    end_year = args.end_year
    successful_years = []
    failed_years = []

    print(f"开始批量处理年平均温盐数据: {start_year}-{end_year}")

    for year in range(start_year, end_year + 1):
        config = create_config(year, args.temperature_root, args.salinity_root, args.output_dir)
        temp_output = os.path.join(config['output_dir'], f"Annual_Temp_{year}.nc")
        salt_output = os.path.join(config['output_dir'], f"Annual_Salinity_{year}.nc")

        print("\n" + "=" * 80)
        print(f"当前输入年份: {year}")
        print(f"温度输入目录: {os.path.join(args.temperature_root, get_decade_dir(year))}")
        print(f"盐度输入目录: {os.path.join(args.salinity_root, get_decade_dir(year))}")
        print(f"当前输出年份: {year}")
        print(f"温度输出文件: {temp_output}")
        print(f"盐度输出文件: {salt_output}")
        print("=" * 80)

        averager = OceanAnnualAverager(config)
        success = averager.run_processing()

        if success:
            successful_years.append(year)
            print(f"\n{year}年年平均数据处理完成！")
        else:
            failed_years.append(year)
            print(f"\n{year}年年平均数据处理失败！")
            print("请检查日志文件 ocean_annual_averager.log 获取详细错误信息。")

    print("\n" + "=" * 80)
    print("批量处理完成")
    print(f"成功年份数量: {len(successful_years)}")
    print(f"失败年份数量: {len(failed_years)}")
    if failed_years:
        print(f"失败年份: {failed_years}")
    print("=" * 80)
