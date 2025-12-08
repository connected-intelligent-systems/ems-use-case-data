from pathlib import Path

import pandas as pd

import config as cf
from imputation import impute_seasonally
from preprocessing import intermediate_exists, fetch_and_preprocess, plot_output

if __name__ == "__main__":
    pp_file_name = "output"
    if cf.use_cached and intermediate_exists(pp_file_name):
        path = Path(cf.saving_dir, pp_file_name + ".csv")
        aligned_data = pd.read_csv(
            path,
            parse_dates=True,
            index_col=0,
        )
    else:
        aligned_data = fetch_and_preprocess()

    freq = aligned_data.index.inferred_freq
    if freq is None:
        raise ValueError("Failed to parse frequency of the preprocessed data.")
    aligned_data = aligned_data[["base"]].asfreq(freq)

    imputed_base_load = impute_seasonally(aligned_data, 2022)
    plot_output(imputed_base_load)
