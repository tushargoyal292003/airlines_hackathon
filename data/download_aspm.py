"""
ASPM Data Downloader for TFT-DCP
Pulls Aviation System Performance Metrics from FAA ASPM system.

ASPM provides airport-level operational data that BTS/NOAA don't cover:
  - Arrival/Departure rates (AAR/ADR)
  - Taxi times, gate delays
  - GDP/EDCT ground delay programs
  - Airport efficiency metrics

FAA ASPM web interface: https://aspm.faa.gov/apm/sys/AnalysisAP.asp
Requires an FAA ASPM account (free registration).

Usage:
  # Interactive login (browser-based):
  python download_aspm.py --airports DFW --start 2019-01-01 --end 2024-12-31

  # With saved credentials:
  python download_aspm.py --airports DFW --start 2019-01-01 --end 2024-12-31 \
      --username YOUR_FAA_USER --password YOUR_FAA_PASS

  # If you already have ASPM CSVs exported manually:
  python download_aspm.py --from-exports ./my_aspm_exports/
"""
import argparse
import os
import time
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from getpass import getpass


# FAA ASPM endpoints
ASPM_BASE = "https://aspm.faa.gov"
ASPM_LOGIN = f"{ASPM_BASE}/apm/sys/login.asp"
ASPM_AIRPORT_QUERY = f"{ASPM_BASE}/apm/sys/AnalysisAP.asp"
ASPM_EXPORT = f"{ASPM_BASE}/apm/sys/APMExport.asp"

# Output directory
DEFAULT_OUT_DIR = "./data/raw/aspm"

# Airports to pull — DFW + top connecting airports
DFW_CONNECTIONS = [
    "DFW",  # hub
    # Top DFW connections by traffic (for A->DFW->B pairs)
    "LAX", "ORD", "ATL", "DEN", "SFO", "JFK", "MIA", "PHX", "SEA", "CLT",
    "LAS", "MCO", "EWR", "IAH", "BOS", "MSP", "DTW", "FLL", "SAN", "AUS",
    "SAT", "STL", "OKC", "TUL", "ABQ", "ELP", "LIT", "COS", "SHV", "MAF",
]


class ASPMDownloader:
    """Download ASPM airport performance data from FAA."""

    def __init__(self, output_dir: str = DEFAULT_OUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (research/academic use)"
        })
        self.authenticated = False

    def login(self, username: str = None, password: str = None) -> bool:
        """Authenticate with FAA ASPM system."""
        if not username:
            username = input("FAA ASPM username: ")
        if not password:
            password = getpass("FAA ASPM password: ")

        payload = {
            "txtLogin": username,
            "txtPassword": password,
            "Submit": "Login",
        }

        resp = self.session.post(ASPM_LOGIN, data=payload, allow_redirects=True)
        if resp.status_code == 200 and "Invalid" not in resp.text:
            self.authenticated = True
            print("  Authenticated with FAA ASPM")
            return True
        else:
            print("  ERROR: FAA ASPM authentication failed.")
            print("  Register at: https://aspm.faa.gov/apm/sys/register.asp")
            return False

    def download_airport_data(
        self,
        airport: str,
        start_date: str,
        end_date: str,
        granularity: str = "H",  # H=hourly, Q=quarter-hour, D=daily
    ) -> pd.DataFrame:
        """
        Download ASPM data for a single airport and date range.

        Args:
            airport: IATA code (e.g. "DFW")
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            granularity: H (hourly), Q (15-min), D (daily)

        Returns:
            DataFrame with ASPM metrics
        """
        if not self.authenticated:
            raise RuntimeError("Not authenticated. Call login() first.")

        # FAA ASPM limits queries to ~90 days at a time
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_chunks = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=89), end)

            params = {
                "Ession": "",
                "GrpOpt": "Airport",
                "Airport": airport,
                "StartDate": chunk_start.strftime("%m/%d/%Y"),
                "EndDate": chunk_end.strftime("%m/%d/%Y"),
                "TimeBasis": granularity,
                "Format": "CSV",
                "AvgOpt": "No",
                "CompareAirport": "",
                "CompareStartDate": "",
                "CompareEndDate": "",
            }

            print(f"    Pulling {airport} {chunk_start.date()} - {chunk_end.date()}...")

            try:
                resp = self.session.get(ASPM_EXPORT, params=params, timeout=120)
                if resp.status_code == 200 and len(resp.content) > 100:
                    from io import StringIO
                    chunk_df = pd.read_csv(StringIO(resp.text))
                    chunk_df["airport"] = airport
                    all_chunks.append(chunk_df)
                    print(f"      -> {len(chunk_df):,} records")
                else:
                    print(f"      -> No data or error (HTTP {resp.status_code})")
            except Exception as e:
                print(f"      -> Error: {e}")

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(2)  # rate-limit courtesy

        if all_chunks:
            return pd.concat(all_chunks, ignore_index=True)
        return pd.DataFrame()

    def download_all(
        self,
        airports: list,
        start_date: str,
        end_date: str,
        granularity: str = "H",
    ) -> pd.DataFrame:
        """Download ASPM data for multiple airports."""
        all_dfs = []

        for airport in airports:
            print(f"\n  [{airport}]")
            df = self.download_airport_data(airport, start_date, end_date, granularity)
            if len(df) > 0:
                # Save per-airport file
                out_file = self.output_dir / f"aspm_{airport}_{start_date[:4]}_{end_date[:4]}.csv"
                df.to_csv(out_file, index=False)
                print(f"    Saved: {out_file} ({len(df):,} rows)")
                all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined_path = self.output_dir / "aspm_combined.csv"
            combined.to_csv(combined_path, index=False)
            print(f"\n  Combined ASPM: {len(combined):,} rows -> {combined_path}")
            return combined
        return pd.DataFrame()


def process_manual_exports(export_dir: str, output_dir: str = DEFAULT_OUT_DIR) -> pd.DataFrame:
    """
    Process ASPM CSV files that were manually exported from the FAA web interface.

    If you can't use the automated downloader, manually export from:
      https://aspm.faa.gov/apm/sys/AnalysisAP.asp

    Steps:
      1. Select airport (e.g. DFW)
      2. Set date range (max ~90 days per export)
      3. Set time basis to "Hourly"
      4. Click "Export to CSV"
      5. Save files into export_dir

    This function reads all CSVs from that directory and combines them.
    """
    export_path = Path(export_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(export_path.glob("*.csv"))
    if not files:
        print(f"  No CSV files found in {export_path}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        # Try to infer airport from filename or content
        fname = f.stem.upper()
        for code in DFW_CONNECTIONS:
            if code in fname:
                df["airport"] = code
                break
        dfs.append(df)
        print(f"  Loaded {f.name}: {len(df):,} records, {len(df.columns)} cols")

    combined = pd.concat(dfs, ignore_index=True)

    # Standardize column names (FAA ASPM exports vary slightly)
    col_map = {
        "Facility": "airport",
        "Airport": "airport",
        "Date": "date",
        "Hour": "hour",
        "A_Demand": "arr_demand",
        "A_Rate": "arr_rate",
        "A_Efficiency": "arr_efficiency",
        "D_Demand": "dep_demand",
        "D_Rate": "dep_rate",
        "D_Efficiency": "dep_efficiency",
        "Avg Gate Arr Delay": "avg_gate_arr_delay",
        "Avg Gate Dep Delay": "avg_gate_dep_delay",
        "Avg Taxi In": "avg_taxi_in",
        "Avg Taxi Out": "avg_taxi_out",
        "A GDP": "gdp_arrivals",
        "D GDP": "gdp_departures",
        "EDCT": "edct_count",
        "MIT": "mit_count",
        "GDP": "gdp_active",
        "GS": "ground_stop_active",
        "AvgTaxiInTime": "avg_taxi_in",
        "AvgTaxiOutTime": "avg_taxi_out",
        "ArrDemand": "arr_demand",
        "DepDemand": "dep_demand",
        "ArrRate": "arr_rate",
        "DepRate": "dep_rate",
        "ArrEff": "arr_efficiency",
        "DepEff": "dep_efficiency",
        "OAGDep": "oag_departures",
        "OAGArr": "oag_arrivals",
        "AvgArrDelay": "avg_gate_arr_delay",
        "AvgDepDelay": "avg_gate_dep_delay",
    }

    combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns},
                    inplace=True)

    out_file = out_path / "aspm_combined.csv"
    combined.to_csv(out_file, index=False)
    print(f"\n  Combined: {len(combined):,} rows -> {out_file}")
    print(f"  Columns: {list(combined.columns)}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Download FAA ASPM data for TFT-DCP")
    parser.add_argument("--airports", nargs="+", default=["DFW"],
                        help="Airport IATA codes to download")
    parser.add_argument("--start", default="2019-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--granularity", default="H", choices=["H", "Q", "D"],
                        help="Time granularity: H=hourly, Q=15min, D=daily")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--username", default=None, help="FAA ASPM username")
    parser.add_argument("--password", default=None, help="FAA ASPM password")
    parser.add_argument("--all-connections", action="store_true",
                        help="Download all DFW connection airports")
    parser.add_argument("--from-exports", type=str, default=None,
                        help="Process manually exported CSVs from this directory instead")

    args = parser.parse_args()

    if args.from_exports:
        print("Processing manual ASPM exports...")
        process_manual_exports(args.from_exports, args.output_dir)
        return

    airports = DFW_CONNECTIONS if args.all_connections else args.airports

    print(f"ASPM Download")
    print(f"  Airports: {airports}")
    print(f"  Range: {args.start} to {args.end}")
    print(f"  Granularity: {args.granularity}")
    print(f"  Output: {args.output_dir}")

    downloader = ASPMDownloader(args.output_dir)
    if not downloader.login(args.username, args.password):
        print("\nTo register for FAA ASPM access (free):")
        print("  https://aspm.faa.gov/apm/sys/register.asp")
        return

    downloader.download_all(airports, args.start, args.end, args.granularity)
    print("\nDone. Next: integrate into preprocessing pipeline.")


if __name__ == "__main__":
    main()
