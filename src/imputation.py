import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def closest_substitute(df, day):  # mind the side effect
    """
    Finds the closest suitable substitute for each incomplete day

    The closest suitable substitute is
        - the same weekday
        - from another year
        - that has no gaps
        - and is as close as possible in terms of time in year

    Caveat: The current implementation is not cyclical,
    meaning that when replacing Jan 1st, Jan 10th is considered closer
    than December 31st
    """
    wd = day.weekday()
    indicator_df = df[
        (df.index.weekday == wd)  # same weekday
        & (df.index.year != day.year)  # other year
        & ~(df["has_gap"])  # no gaps
        & ~(df["used"])  # not yet used elsewhere
    ].copy()
    indicator_df["offset"] = np.abs(indicator_df.index.day_of_year - day.day_of_year)
    substitute = indicator_df.idxmin()["offset"]
    df.at[substitute, "used"] = True  # side effect
    return substitute


def impute_seasonally(
    data: pd.DataFrame,
    target_year: int,
    freq: str | pd.Timedelta | None = None,
):
    """
    TODO make sure the UTC conversion is done correctly,
      i.e., the series is not actually cut at 10/11 pm
    TODO consider adding the possibility to specify a cutoff time
    """
    idx = data.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("data must be a pandas DataFrame with a DatetimeIndex.")
    freq = freq or data.index.inferred_freq

    # finding out what to substitute
    substitution_mapping = data.groupby(idx.date).apply(lambda g: g.isna().any())
    substitution_mapping.set_index(
        pd.to_datetime(substitution_mapping.index, utc=True),
        inplace=True,
    )
    substitution_mapping.columns = ["has_gap"]
    substitution_mapping["replacement"] = substitution_mapping.index
    substitution_mapping["used"] = False
    log.debug(f"substitution_mapping:\n{substitution_mapping}")

    # finding out what to substitute by
    for idx in substitution_mapping[
        (substitution_mapping["has_gap"])
        & (substitution_mapping.index.year == target_year)
    ].index:
        substitution_mapping.at[idx, "replacement"] = closest_substitute(
            substitution_mapping, idx
        )

    target_substitution_mapping = substitution_mapping[
        (substitution_mapping.index.year == target_year)
        & (substitution_mapping["has_gap"])
    ]
    log.debug(f"target_substitution_mapping:\n{target_substitution_mapping}")
    log.debug(
        "days used for substitution:\n"
        f"{substitution_mapping[substitution_mapping["used"]]}"
    )

    # performing the actual substitution
    imputed_data = pd.DataFrame(
        index=pd.date_range(
            start=pd.Timestamp(target_year, 1, 1),
            end=pd.Timestamp(target_year, 12, 31, 23, 59, 59),
            freq=freq,
            name="time",
            tz="utc",
        ),
    )

    imputed_data = pd.concat([imputed_data, data.loc[imputed_data.index]], axis=1)
    log.debug(f"imputed_data:\n{imputed_data}")

    # replace entire days according to the substitution mapping
    def dayspan_after(ts):
        return slice(ts, ts + pd.Timedelta(24 * 60 * 60 - 1, "s"))

    for gap_date, substitution in target_substitution_mapping.iterrows():
        imputed_data.loc[dayspan_after(gap_date)] = data.loc[
            dayspan_after(substitution["replacement"])
        ].values

    # sanity check
    if imputed_data.isna().any(axis=None):
        log.warning("Imputation failed to fill all the gaps!")

    return imputed_data


if __name__ == "__main__":
    pd.set_option("display.max_rows", 12)
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)

    freq = "15min"
    resampled_data_path = Path(
        __file__,
        f"../../data/resampled-{freq}/total_power.csv",
    ).resolve()

    csv_contents = pd.read_csv(resampled_data_path, index_col=0, parse_dates=True)

    log.info(f"read data \n{csv_contents}")

    year = 2022

    imputed_data = impute_seasonally(csv_contents, year, freq)

    log.info(f"imputed_data:\n{imputed_data}")
