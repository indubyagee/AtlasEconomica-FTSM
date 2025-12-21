import pandas as pd
import holidays
from holidays.constants import PUBLIC, UNOFFICIAL
from pathlib import Path



# Configuration
start_ts = pd.Timestamp("1990-01-01")
end_ts = pd.Timestamp("2024-12-31")
years = range(start_ts.year, end_ts.year + 1)

base = Path(__file__).resolve().parents[1]
out_dir = base / "data" / "holidays" / "raw" 
out_dir.mkdir(parents=True, exist_ok=True)



# Data retrieval
us_public_holidays = holidays.country_holidays("US", categories=PUBLIC, years=years)
us_unofficial_holidays = holidays.country_holidays("US", categories=UNOFFICIAL, years=years)
nyse_holidays = holidays.financial_holidays("NYSE", years=years)

all_dates = sorted(
    set(us_public_holidays.keys())
    | set(us_unofficial_holidays.keys())
    | set(nyse_holidays.keys())
)
df = pd.DataFrame({"timestamp": pd.to_datetime(all_dates)})
df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].reset_index(drop=True)

date_keys = df["timestamp"].map(lambda ts: ts.date())
df["publicHolidaysUS"] = date_keys.map(us_public_holidays.get)
df["unofficialHolidaysUS"] = date_keys.map(us_unofficial_holidays.get)
df["financialHolidaysNYSE"] = date_keys.map(nyse_holidays.get)

# Data splitting
us_df = df[["timestamp", "publicHolidaysUS", "unofficialHolidaysUS"]]
nyse_df = df[["timestamp", "financialHolidaysNYSE"]]



# Export dataframes to csv and parquet files
us_df.to_csv(out_dir / "us_holidays.csv", index=False)
us_df.to_parquet(out_dir / "us_holidays.parquet", index=False)
nyse_df.to_csv(out_dir / "nyse_holidays.csv", index=False)
nyse_df.to_parquet(out_dir / "nyse_holidays.parquet", index=False)
print(f"[Output] Exported holiday data to: {out_dir}\n")