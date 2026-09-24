#!/usr/bin/env python3
"""流式计算 1940--2023 年全球、大洋和纬度区域热膨胀不确定性。

本程序不写入热膨胀地图，只保存逐成员和汇总 CSV。温度使用 IAPv4-LE
集合成员，盐度固定使用主 IAP 年平均盐度；热膨胀计算沿用现有 TEOS-10
流程。月度成员 nc 文件读取后立即删除，支持按成员断点续算。
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gsw
import netCDF4 as nc
import numpy as np
from osgeo import gdal, ogr

PILOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PILOT_DIR))
import iapv4_le_1deg_pilot as pilot  # noqa: E402


OUTPUT_DEFAULT = Path(__file__).resolve().parent / "results"
SHAPEFILE_ROOT = None
BASINS = (
    ("Arctic", "Arctic_Ocean/Arctic_Ocean.shp"),
    ("Atlantic", "Atlantic_Ocean_Shp/Atlantic_Ocean.shp"),
    ("Indian", "Indian_Ocean/Indian_Ocean.shp"),
    ("Pacific", "Pacific_Ocean_Shp/Pacific_Ocean.shp"),
    ("Southern", "Southern Ocean/Antarctic_Ocean.shp"),
)
DEPTH_BANDS = (
    ("0-700m", 0.0, 700.0),
    ("700-2000m", 700.0, 2000.0),
    ("2000-6000m", 2000.0, 6000.1),
    ("0-2000m", 0.0, 2000.0),
    ("0-6000m", 0.0, 6000.1),
)
LAT10 = tuple(
    [(f"{abs(lo):g}S-{abs(hi):g}S", lo, hi) for lo, hi in zip(range(-90, 0, 10), range(-80, 10, 10))]
    + [(f"{lo:g}N-{hi:g}N", lo, hi) for lo, hi in zip(range(0, 90, 10), range(10, 100, 10))]
)
LAT_CUMULATIVE = (
    ("15S-15N", -15.0, 15.0),
    ("30S-30N", -30.0, 30.0),
    ("45S-45N", -45.0, 45.0),
    ("60S-60N", -60.0, 60.0),
    ("75S-75N", -75.0, 75.0),
    ("86S-90N", -86.0, 90.0),
)
# 12123116 的官方 IAPv4-LE 目录持续返回 HTTP 404；不纳入正式样本。
ENSEMBLE_MEMBERS = tuple(member for member in pilot.PILOT_MEMBERS if member != "12123116")
BASE_DOWNLOAD_FILE = pilot.download_file
BASE_MEMBER_FILE_MAP = pilot.member_monthly_file_map
DOWNLOAD_RETRY_DELAYS_SECONDS = (5, 15, 30, 60)


def member_supports_depth_band(member, depth_band):
    """WOA 背景场在约1500 m以下无有效值，不能用于包含1500 m以下的深度带。"""
    return depth_band == "0-700m" or not member.startswith("3")


def logger_for(path):
    logger = logging.getLogger("full_uncertainty")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def download_with_retries(url, destination, timeout=300):
    """IAP 服务器偶发断开连接时重试；404 等永久错误立即上抛。"""
    last_error = None
    for attempt in range(1, len(DOWNLOAD_RETRY_DELAYS_SECONDS) + 2):
        try:
            return BASE_DOWNLOAD_FILE(url, destination, timeout=timeout)
        except Exception as error:
            last_error = error
            if "HTTP Error 404" in str(error) or attempt > len(DOWNLOAD_RETRY_DELAYS_SECONDS):
                raise
            delay = DOWNLOAD_RETRY_DELAYS_SECONDS[attempt - 1]
            logging.getLogger("full_uncertainty").warning(
                "下载失败，第%s/%s次重试将在%s秒后进行: %s",
                attempt, len(DOWNLOAD_RETRY_DELAYS_SECONDS) + 1, delay, error,
            )
            time.sleep(delay)
    raise last_error


def member_file_map_with_retries(member, cache_dir):
    """成员目录列表请求同样允许短暂网络故障后恢复。"""
    for attempt in range(1, len(DOWNLOAD_RETRY_DELAYS_SECONDS) + 2):
        try:
            return BASE_MEMBER_FILE_MAP(member, cache_dir)
        except Exception as error:
            if "HTTP Error 404" in str(error) or attempt > len(DOWNLOAD_RETRY_DELAYS_SECONDS):
                raise
            delay = DOWNLOAD_RETRY_DELAYS_SECONDS[attempt - 1]
            logging.getLogger("full_uncertainty").warning(
                "成员%s目录请求失败，第%s/%s次重试将在%s秒后进行: %s",
                member, attempt, len(DOWNLOAD_RETRY_DELAYS_SECONDS) + 1, delay, error,
            )
            time.sleep(delay)


def write_rows(path, rows, fields, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w" or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def basin_labels(lat, lon, volumes):
    """用现有 IHO 五大洋 shp 栅格化到 1°网格，并限制在有水体积的像元。"""
    if not SHAPEFILE_ROOT.is_dir():
        raise FileNotFoundError(f"缺少大洋边界目录: {SHAPEFILE_ROOT}")
    h, w = len(lat), len(lon)
    dataset = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform((float(lon[0] - 0.5), 1.0, 0.0, float(lat[0] - 0.5), 0.0, 1.0))
    band = dataset.GetRasterBand(1)
    band.Fill(0)
    for index, (_, relative) in enumerate(BASINS, 1):
        source = ogr.Open(str(SHAPEFILE_ROOT / relative), 0)
        if source is None:
            raise FileNotFoundError(f"无法打开大洋边界: {SHAPEFILE_ROOT / relative}")
        if gdal.RasterizeLayer(dataset, [1], source.GetLayer(0), burn_values=[index]) != 0:
            raise RuntimeError(f"栅格化失败: {relative}")
        source = None
    labels = band.ReadAsArray().astype(np.int8)
    labels[np.sum(volumes, axis=0) <= 0.0] = 0
    dataset = None
    return labels


def region_masks(lat, lon, volumes, labels):
    masks = {"Global": np.sum(volumes, axis=0) > 0.0}
    lat_grid = np.broadcast_to(lat[:, None], masks["Global"].shape)
    for index, (name, _) in enumerate(BASINS, 1):
        masks[name] = labels == index
    for name, lower, upper in LAT10:
        masks[name] = ((lat_grid >= lower) & (lat_grid < upper) & masks["Global"])
    for name, lower, upper in LAT_CUMULATIVE:
        masks[name] = ((lat_grid >= lower) & (lat_grid < upper) & masks["Global"])
    return masks


def prepare_basin_label_cache(output_dir, lat, lon, volumes, log):
    """只在主进程栅格化一次五大洋边界，避免各集合成员重复执行。"""
    path = output_dir / "basin_labels_1deg.npy"
    if path.exists():
        labels = np.load(path)
        expected = (len(lat), len(lon))
        if labels.shape == expected:
            log.info("复用五大洋1°标签缓存: %s", path)
            return path
        log.warning("五大洋标签缓存形状错误 %s，重新生成", labels.shape)
    log.info("首次栅格化五大洋 shp 到1°网格，约需1分钟，仅执行一次")
    labels = basin_labels(lat, lon, volumes)
    np.save(path, labels)
    log.info("五大洋1°标签缓存已写入: %s", path)
    return path


def all_region_specs():
    return (
        [("global", "Global", None)]
        + [("basin", name, None) for name, _ in BASINS]
        + [("latitude_10deg", name, None) for name, _, _ in LAT10]
        + [("latitude_cumulative", name, None) for name, _, _ in LAT_CUMULATIVE]
    )


def compute_grid_results(temp0, temp1, salt0, salt1, depth, volumes, masks, ocean_area):
    totals = {(kind, name, band): 0.0 for kind, name, _ in all_region_specs() for band, _, _ in DEPTH_BANDS}
    for depth_index, depth_m in enumerate(depth):
        t0, t1 = temp0[depth_index], temp1[depth_index]
        s0, s1 = salt0[depth_index], salt1[depth_index]
        valid = np.isfinite(t0) & np.isfinite(t1) & np.isfinite(s0) & np.isfinite(s1)
        valid &= volumes[depth_index] > 0.0
        if not np.any(valid):
            continue
        ct0 = gsw.CT_from_t(s0[valid], t0[valid], depth_m)
        ct1 = gsw.CT_from_t(s1[valid], t1[valid], depth_m)
        alpha0 = gsw.alpha(s0[valid], ct0, depth_m)
        alpha1 = gsw.alpha(s1[valid], ct1, depth_m)
        change = np.zeros_like(t0, dtype=np.float64)
        change[valid] = volumes[depth_index][valid] * ((alpha0 + alpha1) / 2.0) * (ct1 - ct0)
        for kind, name, _ in all_region_specs():
            mask = masks[name]
            for band, lower, upper in DEPTH_BANDS:
                if lower <= depth_m < upper:
                    amount = float(np.nansum(change[mask]))
                    totals[(kind, name, band)] += amount
    rows = []
    for kind, name, _ in all_region_specs():
        grid_cell_count = int(np.count_nonzero(masks[name]))
        for band, _, _ in DEPTH_BANDS:
            amount_m3 = totals[(kind, name, band)]
            rows.append({
                "region_type": kind,
                "region": name,
                "depth_band": band,
                "volume_change_km3": amount_m3 / 1.0e9,
                "global_contribution_mm": amount_m3 / ocean_area * 1000.0,
                "grid_cell_count": grid_cell_count,
            })
    return rows


def member_worker(member, years, cache_dir, output_dir, basin_label_file, overwrite, keep_member_files):
    log = logger_for(output_dir / "processing.log")
    pilot.download_file = download_with_retries
    volumes, ocean_area, depth, lat, lon = pilot.read_volume_data()
    masks = region_masks(lat, lon, volumes, np.load(basin_label_file))
    climate_files = []
    for month in range(1, 13):
        filename, variable = pilot.climatology_spec(member, month)
        climate_files.append((cache_dir / "climatology" / filename, variable))
    file_map = member_file_map_with_retries(member, cache_dir)
    result_file = output_dir / "member_results" / f"member_{member}.csv"
    fields = ["member", "base_year", "target_year", "region_type", "region", "depth_band", "volume_change_km3", "global_contribution_mm", "grid_cell_count"]
    existing = read_rows(result_file) if not overwrite else []
    completed = {(int(r["base_year"]), int(r["target_year"])) for r in existing}
    if overwrite or not result_file.exists():
        write_rows(result_file, [], fields, mode="w")
    pending_targets = [year for year in years[1:] if (year - 1, year) not in completed]
    if not pending_targets:
        log.info("成员%s的全部年度配对已存在，跳过", member)
        return member
    first_target = pending_targets[0]
    previous_temp = None
    previous_year = first_target - 1
    previous_salt = None
    try:
        previous_temp = pilot.read_member_annual_temperature(member, previous_year, climate_files, file_map, cache_dir / "member_monthly", keep_member_files)
        previous_salt = pilot.prepare_fixed_salinity((previous_year,), depth, lat, lon)[previous_year]
        for target_year in range(first_target, years[-1] + 1):
            base_year = target_year - 1
            current_temp = pilot.read_member_annual_temperature(member, target_year, climate_files, file_map, cache_dir / "member_monthly", keep_member_files)
            current_salt = pilot.prepare_fixed_salinity((target_year,), depth, lat, lon)[target_year]
            if (base_year, target_year) not in completed:
                rows = compute_grid_results(previous_temp, current_temp, previous_salt, current_salt, depth, volumes, masks, ocean_area)
                rows = [row for row in rows if member_supports_depth_band(member, row["depth_band"])]
                for row in rows:
                    row.update({"member": member, "base_year": base_year, "target_year": target_year})
                write_rows(result_file, rows, fields, mode="a")
                log.info("成员%s完成 %s 相对于 %s", member, target_year, base_year)
            del previous_temp, previous_salt
            previous_temp, previous_salt = current_temp, current_salt
            previous_year = target_year
    finally:
        if previous_temp is not None:
            del previous_temp
        if previous_salt is not None:
            del previous_salt
    return member


def summarize(member_rows, output_dir):
    grouped = {}
    for row in member_rows:
        key = (int(row["base_year"]), int(row["target_year"]), row["region_type"], row["region"], row["depth_band"])
        grouped.setdefault(key, []).append(float(row["global_contribution_mm"]))
    fields = ["base_year", "target_year", "region_type", "region", "depth_band", "member_count", "ensemble_mean_mm", "ensemble_sd_mm", "p05_mm", "p95_mm"]
    rows = []
    for key, values in sorted(grouped.items()):
        values = np.asarray(values, dtype=np.float64)
        rows.append({
            "base_year": key[0], "target_year": key[1], "region_type": key[2], "region": key[3], "depth_band": key[4],
            "member_count": len(values), "ensemble_mean_mm": float(np.mean(values)),
            "ensemble_sd_mm": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "p05_mm": float(np.percentile(values, 5)), "p95_mm": float(np.percentile(values, 95)),
        })
    write_rows(output_dir / "uncertainty_summary.csv", rows, fields, mode="w")
    for region_type in ("global", "basin", "latitude_10deg", "latitude_cumulative"):
        subset = [row for row in rows if row["region_type"] == region_type]
        write_rows(output_dir / f"uncertainty_summary_{region_type}.csv", subset, fields, mode="w")


RESULT_FIELDS = [
    "member", "base_year", "target_year", "region_type", "region", "depth_band",
    "volume_change_km3", "global_contribution_mm", "grid_cell_count",
]


def temperature_state_path(state_dir, member, year):
    return state_dir / f"temperature_{member}_{year}.npy"


def salinity_state_path(state_dir, year):
    return state_dir / f"salinity_{year}.npy"


def save_array(path, values):
    """完整写入后再替换，避免中断留下半个年度状态文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(str(temporary), values)
    temporary.replace(path)


def completed_pair(result_file, base_year, target_year):
    return any(
        int(row["base_year"]) == base_year and int(row["target_year"]) == target_year
        for row in read_rows(result_file)
    )


def load_or_download_member_temperature(member, year, cache_dir, state_dir, climate_files, file_map, keep_member_files):
    state_file = temperature_state_path(state_dir, member, year)
    if state_file.exists():
        return np.load(state_file), True
    values = pilot.read_member_annual_temperature(
        member, year, climate_files, file_map, cache_dir / "member_monthly", keep_member_files,
    )
    return values, False


def prepare_salinity_state(year, state_dir, depth, lat, lon, log):
    path = salinity_state_path(state_dir, year)
    if path.exists():
        return path
    log.info("准备年度盐度状态: %s", year)
    values = pilot.prepare_fixed_salinity((year,), depth, lat, lon)[year]
    save_array(path, values)
    del values
    return path


def year_pair_worker(member, base_year, target_year, cache_dir_str, output_dir_str,
                     basin_label_file_str, state_dir_str, salt_base_file_str,
                     salt_target_file_str, keep_member_files):
    """计算一个成员的一个年度配对；倒序运行时保留较早年份供下一步复用。"""
    output_dir = Path(output_dir_str)
    cache_dir = Path(cache_dir_str)
    state_dir = Path(state_dir_str)
    log = logger_for(output_dir / "processing.log")
    pilot.download_file = download_with_retries
    result_file = output_dir / "member_results" / f"member_{member}.csv"
    pair_done = completed_pair(result_file, base_year, target_year)
    base_state = temperature_state_path(state_dir, member, base_year)

    volumes, ocean_area, depth, lat, lon = pilot.read_volume_data()
    masks = region_masks(lat, lon, volumes, np.load(basin_label_file_str))
    climate_files = [
        (cache_dir / "climatology" / pilot.climatology_spec(member, month)[0], pilot.climatology_spec(member, month)[1])
        for month in range(1, 13)
    ]
    file_map = member_file_map_with_retries(member, cache_dir)

    if pair_done and base_state.exists():
        return member
    if pair_done:
        base_temp, _ = load_or_download_member_temperature(
            member, base_year, cache_dir, state_dir, climate_files, file_map, keep_member_files,
        )
        save_array(base_state, base_temp)
        del base_temp
        return member

    target_temp, _ = load_or_download_member_temperature(
        member, target_year, cache_dir, state_dir, climate_files, file_map, keep_member_files,
    )
    base_temp, base_from_state = load_or_download_member_temperature(
        member, base_year, cache_dir, state_dir, climate_files, file_map, keep_member_files,
    )
    salt_base = np.load(salt_base_file_str)
    salt_target = np.load(salt_target_file_str)
    try:
        rows = compute_grid_results(
            base_temp, target_temp, salt_base, salt_target, depth, volumes, masks, ocean_area,
        )
        rows = [row for row in rows if member_supports_depth_band(member, row["depth_band"])]
        for row in rows:
            row.update({"member": member, "base_year": base_year, "target_year": target_year})
        if not result_file.exists():
            write_rows(result_file, [], RESULT_FIELDS, mode="w")
        write_rows(result_file, rows, RESULT_FIELDS, mode="a")
        if not base_from_state:
            save_array(base_state, base_temp)
        log.info("成员%s完成 %s 相对于 %s", member, target_year, base_year)
    finally:
        del target_temp, base_temp, salt_base, salt_target
    return member


def write_all_outputs(output_dir):
    all_rows = []
    for path in sorted((output_dir / "member_results").glob("member_*.csv")):
        all_rows.extend(read_rows(path))
    write_rows(output_dir / "member_results.csv", all_rows, RESULT_FIELDS, mode="w")
    for region_type in ("global", "basin", "latitude_10deg", "latitude_cumulative"):
        rows = [row for row in all_rows if row["region_type"] == region_type]
        write_rows(output_dir / f"member_results_{region_type}.csv", rows, RESULT_FIELDS, mode="w")
    summarize(all_rows, output_dir)
    return len(all_rows)


def remove_superseded_states(state_dir, target_year):
    """倒序完成当前配对后，target_year 不再被下一步使用，删除其年度状态。"""
    for path in state_dir.glob(f"temperature_*_{target_year}.npy"):
        path.unlink()
    salinity_path = salinity_state_path(state_dir, target_year)
    if salinity_path.exists():
        salinity_path.unlink()


def main():
    global SHAPEFILE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1940)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--annual-dir", type=Path, required=True, help="Directory containing annual central IAP T/S NetCDF files")
    parser.add_argument("--volume-file", type=Path, required=True, help="1-degree GEBCO layer-volume NetCDF")
    parser.add_argument("--basin-shapefile-root", type=Path, required=True, help="Root directory of the IHO basin shapefiles")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-member-files", action="store_true", help="仅调试使用；默认处理后删除月度nc")
    args = parser.parse_args()
    if args.start_year < 1940 or args.end_year > 2023 or args.start_year >= args.end_year:
        raise ValueError("当前 IAPv4-LE 文件范围为1940--2023，且需要至少两个年份")
    pilot.ANNUAL_DIR = args.annual_dir.resolve()
    pilot.VOLUME_FILE = args.volume_file.resolve()
    SHAPEFILE_ROOT = args.basin_shapefile_root.resolve()
    if not pilot.ANNUAL_DIR.is_dir():
        raise FileNotFoundError(f"Annual IAP directory not found: {pilot.ANNUAL_DIR}")
    if not pilot.VOLUME_FILE.is_file():
        raise FileNotFoundError(f"1-degree volume file not found: {pilot.VOLUME_FILE}")
    if not SHAPEFILE_ROOT.is_dir():
        raise FileNotFoundError(f"Basin shapefile directory not found: {SHAPEFILE_ROOT}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = logger_for(output / "processing.log")
    years = tuple(range(args.start_year, args.end_year + 1))
    cache = output / "cache"
    state_dir = output / "annual_state"
    log.info("开始范围: %s--%s；年度变化数: %s；按年份倒序处理", args.start_year, args.end_year, len(years) - 1)
    pilot.prepare_climatology_cache(cache, log)
    volumes, _, _, lat, lon = pilot.read_volume_data()
    basin_label_file = prepare_basin_label_cache(output, lat, lon, volumes, log)
    del volumes
    members = ENSEMBLE_MEMBERS
    try:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for target_year in range(args.end_year, args.start_year, -1):
                base_year = target_year - 1
                log.info("开始年度配对: %s 相对于 %s", target_year, base_year)
                volumes, _, depth, lat, lon = pilot.read_volume_data()
                salt_base = prepare_salinity_state(base_year, state_dir, depth, lat, lon, log)
                salt_target = prepare_salinity_state(target_year, state_dir, depth, lat, lon, log)
                del volumes
                futures = {
                    executor.submit(
                        year_pair_worker, member, base_year, target_year, str(cache), str(output),
                        str(basin_label_file), str(state_dir), str(salt_base), str(salt_target),
                        args.keep_member_files,
                    ): member
                    for member in members
                }
                failures = []
                for future in as_completed(futures):
                    member = futures[future]
                    try:
                        log.info("成员%s完成年度配对 %s 相对于 %s", future.result(), target_year, base_year)
                    except Exception:
                        failures.append(member)
                        log.exception("成员%s未完成年度配对 %s 相对于 %s", member, target_year, base_year)
                if failures:
                    log.error("年度配对%s相对于%s未完整完成，保留状态并停止；重启后将从该年度继续。失败成员: %s", target_year, base_year, ", ".join(failures))
                    return
                row_count = write_all_outputs(output)
                log.info("年度配对%s相对于%s汇总完成；当前累计成员区域记录: %s", target_year, base_year, row_count)
                remove_superseded_states(state_dir, target_year)
    except KeyboardInterrupt:
        log.warning("用户中断运行；已完成的年度成员结果已保留，可用相同命令继续。")
        return
    row_count = write_all_outputs(output)
    metadata = {
        "start_year": args.start_year, "end_year": args.end_year,
        "members": list(members), "workers": args.workers,
        "geometry": str(pilot.VOLUME_FILE), "basin_shapefile_root": str(SHAPEFILE_ROOT),
        "depth_bands": [x[0] for x in DEPTH_BANDS],
        "latitude_10deg_bands": [x[0] for x in LAT10],
        "latitude_cumulative_bands": [x[0] for x in LAT_CUMULATIVE],
        "salinity": "fixed annual central IAP salinity; no salinity ensemble perturbation",
        "deep_member_rule": "WOA members are retained only for 0-700m; all bands reaching below 1500m use IAP/WAGHC members only",
        "processing_order": "year-first, descending from target_year to start_year + 1",
        "temporary_monthly_files": "deleted after each annual temperature is read by default",
        "annual_state": "retained after successful completion; superseded yearly state is deleted during normal reverse progression",
        "cache": "retained after successful completion until manually removed",
        "uncertainty": "sample standard deviation across available members",
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    log.info("全部处理结束，共写入 %s 条成员区域记录；annual_state与cache已保留，等待手动清理。", row_count)


if __name__ == "__main__":
    main()
