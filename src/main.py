from pathlib import Path

import pandas as pd

import config as cf
from imputation import impute_seasonally
from preprocessing import fetch_and_preprocess, intermediate_exists, plot_output

if __name__ == "__main__":
    pp_file_name = "output"
    ip_file_prefix = "imputed_base_load_"
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

    year = 2022
    imputed_base_load = impute_seasonally(aligned_data, year)
    imputed_base_load.to_csv(Path(cf.saving_dir, ip_file_prefix + str(year) + ".csv"))
    plot_output(imputed_base_load)
