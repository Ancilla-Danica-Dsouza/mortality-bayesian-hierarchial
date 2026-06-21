"""
data_prep.py
============
Ingests and harmonizes two data sources into a single country-age-year panel:

1. Human Mortality Database (HMD)   -> age/year/country death rates & exposures
2. World Bank Open Data             -> GDP per capita, health expenditure, education

Output: a single tidy DataFrame saved to data/processed/panel.csv with columns:
    country, age, year, mx (central death rate), exposure,
    gdp_per_capita, health_expenditure_pct_gdp, mean_years_schooling

Run directly:
    python src/data_prep.py
"""

import os
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def _find_hmd_file(country_dir: str, stem: str) -> str:
    """
    Locate a file like 'Mx_1x1.txt' inside a country folder, tolerating
    HMD's occasional country-code prefix (e.g. 'SWE.Mx_1x1.txt').

    Raises FileNotFoundError with a clear message if nothing matches.
    """
    candidates = [
        f for f in os.listdir(country_dir)
        if f.lower().endswith(f"{stem.lower()}.txt") or f.lower() == f"{stem.lower()}.txt"
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a file matching '{stem}.txt' in {country_dir}. "
            f"Files present: {os.listdir(country_dir)}"
        )
    return os.path.join(country_dir, candidates[0])


def _parse_hmd_txt(filepath: str, value_name: str) -> pd.DataFrame:
    """
    Parse a single HMD '_1x1' text file (e.g. Mx_1x1.txt or Exposures_1x1.txt)
    into a tidy long-format DataFrame: columns = [year, age, value_name].

    HMD format reminders (confirmed against real downloaded files):
        Line 1      : title/metadata, e.g. "Sweden, Death rates (period 1x1), ..."
        Line 2      : blank
        Line 3      : column headers -> Year   Age   Female   Male   Total
        Line 4+     : whitespace-delimited data rows
        Age column  : mostly integers, but the oldest age group is the
                      open-ended string "110+" (or similar) -> must be handled
                      specially, not cast straight to int.
        Missing data: shown as "." in some files (e.g. early cohort years) ->
                      pandas' na_values handles this.

    Only the 'Total' (both-sexes combined) column is kept for this project.
    """
    df = pd.read_csv(
        filepath,
        skiprows=2,          # skip title line + blank line; row 3 becomes the header
        sep=r"\s+",          # whitespace-delimited, robust to variable column widths
        na_values=["."],     # HMD's missing-data marker
        engine="python",
    )

    # Normalize column names in case of stray capitalization/whitespace
    df.columns = [c.strip() for c in df.columns]

    # The oldest age group is an open interval like "110+". Strip the "+" and
    # treat it as a plain integer age (start of the open interval), keeping
    # a flag in case the model needs to treat it differently later.
    df["age_is_open_interval"] = df["Age"].astype(str).str.endswith("+")
    df["age"] = (
        df["Age"].astype(str).str.replace("+", "", regex=False).astype(int)
    )

    out = df[["Year", "age", "age_is_open_interval", "Total"]].rename(
        columns={"Year": "year", "Total": value_name}
    )
    return out


def download_hmd_data(countries: list[str]) -> None:
    """
    Verify that the expected raw HMD files exist for each country.

    HMD requires a registered (free) account, and bulk files are gated behind
    login, so for now this function does NOT auto-download -- it validates
    that the manually-downloaded files are in place at:
        data/raw/hmd/<COUNTRY_CODE>/Mx_1x1.txt
        data/raw/hmd/<COUNTRY_CODE>/Exposures_1x1.txt

    Raises FileNotFoundError early (with a clear message) if anything is missing,
    rather than failing deep inside the parsing step.
    """
    missing = []
    for country in countries:
        country_dir = os.path.join(RAW_DIR, "hmd", country)
        if not os.path.isdir(country_dir):
            missing.append(f"{country}: folder not found at {country_dir}")
            continue
        for stem in ["Mx_1x1", "Exposures_1x1"]:
            try:
                _find_hmd_file(country_dir, stem)
            except FileNotFoundError as e:
                missing.append(str(e))

    if missing:
        raise FileNotFoundError(
            "Missing HMD files:\n" + "\n".join(f"  - {m}" for m in missing)
        )
    print(f"All HMD files found for: {', '.join(countries)}")


def load_hmd_mortality_tables(countries: list[str]) -> pd.DataFrame:
    """
    Parse raw HMD .txt files into a single tidy DataFrame across all countries:
        columns = [country, year, age, age_is_open_interval, mx, exposure]
    """
    frames = []
    for country in countries:
        country_dir = os.path.join(RAW_DIR, "hmd", country)

        mx_path = _find_hmd_file(country_dir, "Mx_1x1")
        exp_path = _find_hmd_file(country_dir, "Exposures_1x1")

        mx_df = _parse_hmd_txt(mx_path, value_name="mx")
        exp_df = _parse_hmd_txt(exp_path, value_name="exposure")

        merged = mx_df.merge(
            exp_df[["year", "age", "exposure"]],
            on=["year", "age"],
            how="left",
        )
        merged["country"] = country
        frames.append(merged)

    full = pd.concat(frames, ignore_index=True)
    full = full[["country", "year", "age", "age_is_open_interval", "mx", "exposure"]]
    return full


def download_worldbank_indicators(countries: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    """
    Pull socioeconomic indicators from the World Bank API:
        - GDP per capita (NY.GDP.PCAP.CD)
        - Health expenditure % of GDP (SH.XPD.CHEX.GD.ZS)
        - Mean years of schooling (proxy via Barro-Lee or UNDP HDI components)

    Returns tidy DataFrame: columns = [country, year, gdp_per_capita,
                                        health_expenditure_pct_gdp, mean_years_schooling]

    TODO: implement requests-based calls to World Bank API v2 (JSON format).
    """
    raise NotImplementedError


def merge_panel(mortality_df: pd.DataFrame, socioeconomic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge mortality and socioeconomic data on (country, year).
    Socioeconomic covariates are broadcast across all ages within a country-year.

    TODO: implement merge, handle missing years (interpolate or forward-fill),
    and flag countries/years with insufficient data for exclusion.
    """
    raise NotImplementedError


def save_processed_panel(df: pd.DataFrame, filename: str = "panel.csv") -> None:
    """Save the final tidy panel to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, filename), index=False)


def main():
    # Folder names match what's actually on disk in data/raw/hmd/
    countries = ["USA", "JPN", "SWE", "UK", "FRN"]

    download_hmd_data(countries)
    mortality_df = load_hmd_mortality_tables(countries)

    print(mortality_df.head(10))
    print(f"\nLoaded {len(mortality_df)} rows across {mortality_df['country'].nunique()} countries.")
    print(f"Year range: {mortality_df['year'].min()}–{mortality_df['year'].max()}")

    # TODO: once World Bank step is implemented, wire in:
    # socio_df = download_worldbank_indicators(countries, start_year=1970, end_year=2020)
    # panel = merge_panel(mortality_df, socio_df)
    # save_processed_panel(panel)


if __name__ == "__main__":
    main()