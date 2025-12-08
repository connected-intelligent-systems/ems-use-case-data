import config as cf
import os
import numpy as np
from plotly import express as px
import plotly.io as pio
from pathlib import Path
from clearml import Dataset
import pandas as pd
from tqdm.notebook import tqdm
import logging

pio.renderers.default = "browser"
log = logging.getLogger(__name__)


def save_intermediate_csv(data, name):
    if cf.save_intermediates:
        path = Path(cf.saving_dir, name + ".csv")
        log.info(f"Saving data to {path} ...")
        data.to_csv(path)


def intermediate_exists(name):
    path = Path(cf.saving_dir, name + ".csv")
    return path.exists() and path.is_file()


def read_intermediate_csv(name, is_device=False):
    path = Path(cf.saving_dir, name + ".csv")
    log.info(f"Reading data from {path} ...")
    if is_device:
        df = pd.read_csv(
            path,
            usecols=["_time", "power"],
            index_col="_time",
            parse_dates=True,
            date_format="ISO8601",
        )
    else:
        df = pd.read_csv(
            path,
            usecols=["_time", "total"],
            index_col="_time",
            parse_dates=True,
            date_format="ISO8601",
        )

    return df


def load_total_from_clearml(data_root):
    if cf.use_cached and intermediate_exists("total_raw"):
        return read_intermediate_csv("total_raw")
    else:
        raw_data_path = Path(data_root, cf.total_path_clearml)

        data = pd.read_csv(
            raw_data_path,
            usecols=["_time", "_value"],
            index_col="_time",
            parse_dates=True,
            date_format="ISO8601",
        )
        data.rename(columns={"_value": "total"}, inplace=True)

        save_intermediate_csv(data, "total_raw")

        return data


def process_total(data):
    resampled_data = data.resample(cf.process_freq).max()
    save_intermediate_csv(resampled_data, "total_processed")
    return resampled_data


def clean_cols(thing_name: str, data: pd.DataFrame):
    """CAUTION: Side effects!"""
    match thing_name:
        case "DECT210 Waschmaschine":
            data.rename(columns={"sensor_160-value": "power"}, inplace=True)
            data.drop(["sensor_163-value"], axis=1, inplace=True)
        case "Smart Switch 6 Spülmaschine":
            data.rename(columns={"sensor_582-value": "power"}, inplace=True)
    return data


def bulk_process_devices(data_root):
    ts_dirs = {k: Path(data_root, fn) for k, fn in cf.devices_dict_clearml.items()}

    devices = {}
    for thing_name in ts_dirs.keys():
        device_df = load_device_from_clreaml(
            data_root, thing_name, clearml_url=ts_dirs[thing_name]
        )
        processed_device_df = process_device(device_df)
        save_intermediate_csv(processed_device_df, thing_name + "_processed")
        devices[thing_name] = processed_device_df

    return devices


def load_device_from_clreaml(data_root, thing_name, clearml_url):
    if cf.use_cached and intermediate_exists(thing_name + "_raw"):
        return read_intermediate_csv(thing_name + "_raw", is_device=True)
    else:
        frames = []
        for p in tqdm(list(clearml_url.iterdir())):  # +
            if p.suffix == ".csv" and any(
                [
                    p.stem == "energy",
                    p.stem == "power",
                    p.stem.startswith("sensor"),
                ]
            ):
                df = pd.read_csv(
                    p,
                    usecols=["_time", "_value"],
                    index_col="_time",
                    parse_dates=True,
                    date_format="ISO8601",
                )
                df.columns = [p.stem]
                frames.append(df)
        bulk = pd.concat(frames, axis=1)
        bulk = clean_cols(thing_name, bulk)

        save_intermediate_csv(bulk, thing_name + "_raw")

        return bulk


def process_device(df):
    df.astype(float)
    df = df.resample(cf.process_freq).max()
    df_process = df.copy()

    # Restrict ffill to 1 hour
    bins_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(cf.process_freq))  # 360
    df_process = df_process.ffill(limit=bins_per_hour)

    return df_process


def substract_devices_from_total(total, devices, filter_negatives=True):
    joint_frame = total.copy()
    joint_frame.columns = ["total"]
    for name in devices.keys():
        joint_frame[name] = devices[name]["power"]

    wms = ["DECT200 Waschmaschine", "DECT210 Waschmaschine"]
    joint_frame["WM"] = joint_frame["DECT200 Waschmaschine"].fillna(0)
    joint_frame.drop(wms, axis=1, inplace=True)

    sms = ["DECT200 Spülmaschine", "Smart Switch 6 Spülmaschine"]
    joint_frame["SM"] = joint_frame[sms].max(axis=1).fillna(0)
    joint_frame.drop(sms, axis=1, inplace=True)

    joint_frame["base"] = joint_frame["total"] - joint_frame["SM"] - joint_frame["WM"]
    downsampled_frame = joint_frame.resample(cf.output_freq).mean()
    if filter_negatives:
        downsampled_frame["base"][downsampled_frame["base"] < 0] = np.nan

    save_intermediate_csv(downsampled_frame, "output")
    return downsampled_frame


def plot_output(df):
    fig = px.line(df)
    fig.update_layout(height=400)
    fig.update_layout(
        legend=dict(
            title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        )
    )
    fig.show()


def fetch_and_preprocess():
    os.makedirs(cf.saving_dir, exist_ok=True)

    data_root = Dataset.get(
        dataset_project="ForeSightNEXT/Electric Load Forecasting",
        dataset_name="household-1235",
        dataset_version="2.0.0",
    ).get_local_copy()

    total_processed = process_total(load_total_from_clearml(data_root))
    devices_processed = bulk_process_devices(data_root)
    baseload_output = substract_devices_from_total(
        total_processed, devices_processed, filter_negatives=True
    )
    return baseload_output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    baseload_output = fetch_and_preprocess()
    plot_output(baseload_output["base"])
