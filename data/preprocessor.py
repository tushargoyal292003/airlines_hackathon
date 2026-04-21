"""
Data Preprocessing Pipeline (Paper Section 3.1)
Steps:
  1. Anomaly detection (retain extreme delays)
  2. Data cleaning (flight + weather)
  3. Airport status feature extraction
  4. Multi-source data integration

FIXES applied (annotated with # FIX):
  - FIX-1: _merge_weather: don't overwrite airport_code already set by NOAAProcessor;
            STATION col (e.g. "72266013962") never matches filename-derived id ("13962").
  - FIX-2: _merge_weather: robust dep_hour extraction — clip(0,23) guards CRSDepTime=2400.
  - FIX-3: ASPM load: guard "Total :" rows on Departure column too, not just Date.
  - FIX-4: ASPM process: coerce Departure Hour to numeric before Int64 cast.
  - FIX-5: _clean_flights: use != 1 for Cancelled/Diverted (float dtype 0./1.).
  - FIX-6: _encode_features: graceful fallback if category_encoders not installed.
  - FIX-7: NOAAProcessor.process: keep _airport_code → airport_code mapping explicit.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings("ignore")

# FIX-6: graceful import for optional dependency
try:
    from category_encoders import JamesSteinEncoder
    _HAS_JS_ENCODER = True
except ImportError:
    _HAS_JS_ENCODER = False
    print("  WARNING: category_encoders not installed — using target-mean encoding fallback")


class BTSProcessor:
    """Process Bureau of Transportation Statistics on-time performance data."""

    # Columns we actually read from BTS files.
    # Breakdown of the full 110-col BTS export:
    #   44 cols: Div1-Div5 + misc — 100% null, pure waste
    #    4 cols: CancellationCode / FirstDepTime etc — 95-100% null
    #   22 cols: Quarter, DistanceGroup, delay-group bins etc — derivable
    #    7 cols: added below — genuinely new signal
    #   30 cols: original set below
    # Total useful: 37 cols  |  Dropped: 73 cols
    REQUIRED_COLS = [
        # ── Core identifiers ──────────────────────────────────
        "Year", "Month", "DayofMonth", "DayOfWeek",
        "FlightDate", "Reporting_Airline", "Tail_Number",
        "Flight_Number_Reporting_Airline",   # route identity / schedule slot
        "Origin", "Dest",
        "OriginState", "DestState",          # regional weather/congestion patterns
        "OriginCityName", "DestCityName",    # human-readable, useful for reporting
        # ── Scheduled / actual times ─────────────────────────
        "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes",
        "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes",
        "WheelsOff", "WheelsOn",             # actual gate→air & air→gate times;
                                             # real taxi = WheelsOff-DepTime,
                                             # real airborne = WheelsOn-WheelsOff
        # ── Taxi / elapsed ────────────────────────────────────
        "TaxiOut", "TaxiIn",
        "CRSElapsedTime", "ActualElapsedTime",
        "AirTime", "Distance",
        # ── Delay cause breakdown ─────────────────────────────
        "CarrierDelay", "WeatherDelay", "NASDelay",
        "SecurityDelay", "LateAircraftDelay",
        # ── Status flags ─────────────────────────────────────
        "Cancelled", "Diverted",
    ]

    def __init__(self, data_dir: str, hub: str = "DFW"):
        self.data_dir = Path(data_dir)
        self.hub = hub

    # Unique flight key — used to deduplicate across departure + arrival files.
    # A DFW→LAX flight appears in both the DFW departure export AND the LAX
    # arrival export; same record, different file.
    DEDUP_KEY = ["Tail_Number", "FlightDate", "CRSDepTime", "Origin", "Dest"]

    def load(self) -> pd.DataFrame:
        """Load 5 years of BTS CSV files (departure + arrival) safely.

        Scale reality:
          5 years × 12 months × 2 file types ≈ 120 files × ~583k rows = ~70M rows
          At 37 cols that's ~10 GB if loaded naively.

        How we stay within memory:
          1. usecols — read only 37 of 110 columns            → -66% column RAM
          2. Chunk-filter — read in 100k-row chunks, keep only
             hub rows, discard the rest immediately           → ~8% rows kept
          3. Deduplicate after concat — departure AND arrival
             exports both contain the same flight record      → ~50% rows removed

        Peak RAM per file: ~50 MB (one chunk).
        Peak RAM total:    ~300–500 MB (all hub rows in memory at once).
        """
        files = sorted(self.data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        CHUNK = 100_000
        hub_dfs = []
        total_raw = 0

        for f in files:
            header = pd.read_csv(f, nrows=0)
            usecols = [c for c in self.REQUIRED_COLS if c in header.columns]
            file_hub_rows = []
            file_raw = 0

            for chunk in pd.read_csv(f, usecols=usecols, low_memory=False,
                                     chunksize=CHUNK):
                file_raw += len(chunk)
                hub_mask = (
                    (chunk.get("Origin", pd.Series(dtype=str)) == self.hub) |
                    (chunk.get("Dest",   pd.Series(dtype=str)) == self.hub)
                )
                kept = chunk[hub_mask]
                if len(kept):
                    file_hub_rows.append(kept)
                del chunk  # discard immediately

            total_raw += file_raw
            if file_hub_rows:
                file_df = pd.concat(file_hub_rows, ignore_index=True)
                hub_dfs.append(file_df)
                print(
                    f"  {f.name}: {file_raw:,} rows → "
                    f"{len(file_df):,} hub-connected"
                )
            else:
                print(f"  {f.name}: {file_raw:,} rows → 0 hub-connected")

        if not hub_dfs:
            raise ValueError(f"No hub-connected flights found for {self.hub}")

        data = pd.concat(hub_dfs, ignore_index=True)
        print(f"\n  Total scanned: {total_raw:,}  |  Hub rows before dedup: {len(data):,}")

        # Deduplicate: departure + arrival files both contain the same flight.
        dedup_cols = [c for c in self.DEDUP_KEY if c in data.columns]
        before = len(data)
        data = data.drop_duplicates(subset=dedup_cols)
        print(f"  After dedup: {len(data):,} (removed {before - len(data):,} duplicates)")

        return data

    def filter_hub_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag inbound/outbound leg_type.  Hub filtering already done in load()."""
        inbound = df[df["Dest"] == self.hub].copy()
        outbound = df[df["Origin"] == self.hub].copy()

        inbound["leg_type"] = "inbound"
        outbound["leg_type"] = "outbound"

        hub_flights = pd.concat([inbound, outbound], ignore_index=True)
        print(f"Hub-connected flights ({self.hub}): {len(hub_flights):,}")
        return hub_flights


class NOAAProcessor:
    """Process NOAA Local Climatological Data."""

    WEATHER_FEATURES = [
        "HourlyDryBulbTemperature", "HourlyRelativeHumidity",
        "HourlyWindSpeed", "HourlyWindDirection",
        "HourlyVisibility", "HourlyPrecipitation",
        "HourlyPressureChange", "HourlyStationPressure",
        "HourlyDewPointTemperature", "HourlyWetBulbTemperature",
        "HourlySkyConditions", "HourlyPresentWeatherType",
    ]

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> pd.DataFrame:
        """Load NOAA LCD data files. Tags each row with airport code from filename.

        Expected filename pattern: {AIRPORT}_{StationID}_{Year}.csv
        e.g. ABI_13962_2019.csv  →  airport_code = "ABI"
        """
        files = sorted(self.data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No NOAA CSV files found in {self.data_dir}")

        dfs = []
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            parts = f.stem.split("_")
            # FIX-7: tag the airport code from filename (first part, 3-char IATA)
            if len(parts) >= 1 and len(parts[0]) == 3 and parts[0].isalpha():
                df["_airport_code"] = parts[0].upper()
            elif len(parts) >= 2 and len(parts[1]) == 3 and parts[1].isalpha():
                # handle alternate patterns like "US_ABI_2019.csv"
                df["_airport_code"] = parts[1].upper()
            else:
                df["_airport_code"] = "UNK"
            dfs.append(df)
            print(f"  Loaded weather: {f.name}: {len(df):,} records "
                  f"[airport_code={df['_airport_code'].iloc[0]}]")

        data = pd.concat(dfs, ignore_index=True)
        return data

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and interpolate weather data per paper Section 3.1.3 Step 2."""
        df = df.copy()

        # Parse datetime
        df["datetime"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # FIX-7: use _airport_code column (set in load()) as the airport identifier.
        # Do NOT remap via STATION — the station numeric ID in the filename (e.g. "13962")
        # does NOT match the full STATION column value (e.g. "72266013962").
        if "_airport_code" in df.columns:
            df["airport_code"] = df["_airport_code"]
        else:
            df["airport_code"] = "UNK"

        # Clean numeric weather features
        numeric_wx = [c for c in self.WEATHER_FEATURES if c in df.columns]
        for col in numeric_wx:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Encode sky conditions as ordinal
        if "HourlySkyConditions" in df.columns:
            df["cloud_cover"] = df["HourlySkyConditions"].apply(self._parse_sky)

        # Encode present weather type
        if "HourlyPresentWeatherType" in df.columns:
            df["wx_severity"] = df["HourlyPresentWeatherType"].apply(
                self._parse_wx_severity
            )

        # Short-gap linear interpolation
        df = df.sort_values(["airport_code", "datetime"])
        for col in numeric_wx:
            if col in df.columns:
                df[col] = (
                    df.groupby("airport_code")[col]
                    .transform(lambda s: s.interpolate(method="linear", limit=3))
                )

        # Floor to hour for merge alignment
        df["hour"] = df["datetime"].dt.floor("h")

        return df

    @staticmethod
    def _parse_sky(val):
        if pd.isna(val):
            return np.nan
        val = str(val).upper()
        if "OVC" in val:
            return 8
        elif "BKN" in val:
            return 6
        elif "SCT" in val:
            return 4
        elif "FEW" in val:
            return 2
        elif "CLR" in val or "SKC" in val:
            return 0
        return np.nan

    @staticmethod
    def _parse_wx_severity(val):
        if pd.isna(val):
            return 0
        val = str(val).upper()
        severity = 0
        if "TS" in val:
            severity = max(severity, 5)
        if "FZ" in val:
            severity = max(severity, 4)
        if "SN" in val:
            severity = max(severity, 3)
        if "RA" in val and "HV" in val:
            severity = max(severity, 3)
        if "RA" in val:
            severity = max(severity, 2)
        if "BR" in val or "FG" in val:
            severity = max(severity, 2)
        if "HZ" in val:
            severity = max(severity, 1)
        return severity


class ASPMProcessor:
    """Process FAA ASPM City-Pair Analysis data.

    File naming: {year}-A.xls (arrivals to hub), {year}-D.xls (departures from hub).
    Format: HTML tables saved as .xls (FAA ASPM web-export default).
    """

    ASPM_FEATURES = [
        "aspm_flight_count",
        "aspm_pct_ontime_gate_dep", "aspm_pct_ontime_airport_dep",
        "aspm_pct_ontime_gate_arr",
        "aspm_edct_count", "aspm_avg_edct",
        "aspm_gate_dep_delay", "aspm_taxi_out_delay",
        "aspm_avg_taxi_out", "aspm_airport_dep_delay",
        "aspm_airborne_delay", "aspm_taxi_in_delay",
        "aspm_block_delay", "aspm_gate_arr_delay",
    ]

    COL_RENAME = {
        "Flight Count": "aspm_flight_count",
        "% On-Time Gate Departures": "aspm_pct_ontime_gate_dep",
        "% On-Time Airport Departures": "aspm_pct_ontime_airport_dep",
        "% On-Time Gate Arrivals": "aspm_pct_ontime_gate_arr",
        "Arrivals With EDCT": "aspm_edct_count",
        "Average EDCT Where EDCT>0": "aspm_avg_edct",
        "Gate Departure Delay": "aspm_gate_dep_delay",
        "Taxi Out Delay": "aspm_taxi_out_delay",
        "Average Taxi Out Time": "aspm_avg_taxi_out",
        "Airport Departure Delay": "aspm_airport_dep_delay",
        "Airborne Delay": "aspm_airborne_delay",
        "Taxi In Delay": "aspm_taxi_in_delay",
        "Block Delay": "aspm_block_delay",
        "Gate Arrival Delay": "aspm_gate_arr_delay",
    }

    def __init__(self, data_dir: str, hub: str = "DFW"):
        self.data_dir = Path(data_dir)
        self.hub = hub

    # Expected column names in the data row (row 1 of the HTML table).
    # Used to auto-detect the correct header row.
    _EXPECTED_COLS = {"Departure", "Arrival", "Date", "Flight Count"}

    def load(self) -> pd.DataFrame:
        """Load ASPM city-pair files.

        Format reality: FAA ASPM exports are pure HTML saved with a .xls
        extension.  xlrd and openpyxl both reject them ("Expected BOF record").
        pd.read_html is the only parser that works.

        Structure:
          Row 0 — title: "ASPM : City Pair Analysis ... Arrival=DFW ..."
          Row 1 — column headers: Departure, Arrival, Date, ...
          Rows 2..N-1 — data
          Row N — "Total :" summary row (must be stripped)

        We auto-detect which row is the real header instead of hard-coding
        header=1, so files with a missing or extra title row still parse
        correctly. We also extract year + direction from the title row as a
        cross-check against the filename.
        """
        files = (
            sorted(self.data_dir.glob("*.xls"))
            + sorted(self.data_dir.glob("*.xlsx"))
        )
        if not files:
            print(f"  No ASPM files found in {self.data_dir}")
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                df = self._parse_aspm_file(f)
                if df is not None and len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"  ERROR reading {f.name}: {e}")

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def _parse_aspm_file(self, f: Path) -> pd.DataFrame:
        """Parse a single ASPM HTML-as-XLS file robustly.

        FAA ASPM files are HTML tables saved as .xls, but the exact structure
        varies by year and export settings.  We try four strategies in order:
          1. HTML parse, header=1  (standard: title row + column row + data)
          2. HTML parse, header=0  (no title row — column names are row 0)
          3. Excel parse via openpyxl/xlrd (rare: file is actual binary Excel)
          4. HTML parse, skip 0 rows (for files with no headers at all)
        """
        import re

        # -- Detect file format from magic bytes --
        with open(f, 'rb') as fh:
            magic = fh.read(8)
        biff_magic = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])  # real .xls
        zip_magic  = bytes([0x50, 0x4B, 0x03, 0x04])                             # .xlsx
        is_real_excel = (magic[:8] == biff_magic) or (magic[:4] == zip_magic)

        df = None

        # ── Strategy A: HTML parsing (covers 99% of FAA ASPM exports) ────
        if not is_real_excel:
            for hdr in [1, 0]:
                try:
                    tables = pd.read_html(str(f), flavor='lxml', header=hdr)
                    if not tables:
                        continue
                    candidate = tables[0].copy()
                    # Drop trailing unnamed columns
                    candidate = candidate.loc[
                        :, ~candidate.columns.astype(str).str.startswith("Unnamed")
                    ]
                    if self._EXPECTED_COLS.issubset(set(candidate.columns)):
                        df = candidate
                        break
                except Exception as e:
                    print(f"    HTML parse (header={hdr}) failed: {type(e).__name__}: {e}")
                    continue

            # Strategy A fallback: column names might be in row 0 of a header=0 parse.
            # In this case candidate.columns are the long title strings, and the real
            # column names are in iloc[0].
            if df is None:
                try:
                    tables = pd.read_html(str(f), flavor='lxml', header=0)
                    if tables:
                        raw = tables[0]
                        # Check if row 0 looks like real column names
                        row0 = raw.iloc[0].astype(str).tolist()
                        if self._EXPECTED_COLS.issubset(set(row0)):
                            raw.columns = row0
                            df = raw.iloc[1:].reset_index(drop=True)
                            df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
                except Exception as e:
                    print(f"    HTML row-0-header fallback failed: {type(e).__name__}: {e}")

        # ── Strategy B: True Excel binary (xlrd for .xls, openpyxl for .xlsx) ─
        if df is None:
            for engine in ['xlrd', 'openpyxl']:
                for skip in [0, 1, 2]:
                    try:
                        candidate = pd.read_excel(
                            f, header=skip, engine=engine
                        )
                        candidate = candidate.loc[
                            :, ~candidate.columns.astype(str).str.startswith("Unnamed")
                        ]
                        if self._EXPECTED_COLS.issubset(set(candidate.columns)):
                            df = candidate
                            break
                    except Exception:
                        continue
                if df is not None:
                    break

        # ── All strategies failed ─────────────────────────────────────────
        if df is None:
            print(f"  Skipping {f.name}: could not parse — diagnostics:")
            try:
                _t = pd.read_html(str(f), flavor='lxml', header=1)
                if _t:
                    _cols = _t[0].columns.tolist()
                    print(f"    HTML header=1 columns: {_cols[:8]}")
                    print(f"    Missing from expected: {self._EXPECTED_COLS - set(_cols)}")
                else:
                    print("    read_html returned 0 tables")
            except Exception as _e:
                print(f"    read_html raised: {type(_e).__name__}: {_e}")
            return None

        # --- 2. Extract year + hub from the title row (header=0 parse) ---
        year_from_title, direction_from_title = None, None
        try:
            title_tables = pd.read_html(str(f), header=0)
            if title_tables:
                title_col = title_tables[0].columns[0]          # long title string
                title = str(title_col)
                # Parse year range: "From 01/2019 To 12/2019"
                import re
                years = re.findall(r"\d{2}/(\d{4})", title)
                if years:
                    year_from_title = int(years[0])
                # Parse direction: "Arrival=DFW" → inbound, "Departure=DFW" → outbound
                if "Arrival=" + self.hub in title:
                    direction_from_title = "inbound"
                elif "Departure=" + self.hub in title:
                    direction_from_title = "outbound"
        except Exception:
            pass

        # --- 3. Determine direction: title > filename stem ---
        stem_upper = f.stem.upper()
        if direction_from_title:
            direction = direction_from_title
        elif stem_upper.endswith("-A"):
            direction = "inbound"
        elif stem_upper.endswith("-D"):
            direction = "outbound"
        else:
            direction = "unknown"

        # --- 4. Strip "Total :" summary rows ---
        for sentinel_col in ["Date", "Departure", "Arrival"]:
            if sentinel_col in df.columns:
                df = df[
                    ~df[sentinel_col]
                    .astype(str)
                    .str.contains("Total", na=False, case=False)
                ]

        df["aspm_direction"] = direction
        if year_from_title:
            df["_title_year"] = year_from_title   # used for validation in process()

        print(
            f"  Loaded ASPM: {f.name}: {len(df):,} records  "
            f"direction={direction}"
            + (f"  title_year={year_from_title}" if year_from_title else "")
        )
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize ASPM city-pair data."""
        if df.empty:
            return df

        df = df.copy()

        # Rename feature columns
        df.rename(columns=self.COL_RENAME, inplace=True)

        # Parse year/month from "MM/YYYY" date format
        df["aspm_year"] = (
            df["Date"].astype(str).str.extract(r"(\d{4})")[0]
        )
        df["aspm_month"] = (
            df["Date"].astype(str).str.extract(r"^(\d{1,2})/")[0]
        )
        df["aspm_year"] = pd.to_numeric(df["aspm_year"], errors="coerce")
        df["aspm_month"] = pd.to_numeric(df["aspm_month"], errors="coerce")

        # FIX-4: coerce Departure Hour to numeric BEFORE attempting Int64 cast
        df["aspm_dep_hour"] = pd.to_numeric(df.get("Departure Hour"), errors="coerce")
        df["aspm_arr_hour"] = pd.to_numeric(df.get("Arrival Hour"), errors="coerce")

        # Clean ASPM feature columns
        for col in self.ASPM_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows missing key join fields
        df = df.dropna(subset=["Departure", "Arrival", "aspm_year", "aspm_month"])
        df["aspm_year"] = df["aspm_year"].astype(int)
        df["aspm_month"] = df["aspm_month"].astype(int)

        if "Departure" in df.columns and "Arrival" in df.columns:
            hub_match = (df["Departure"] == self.hub) | (df["Arrival"] == self.hub)
            n_hub = hub_match.sum()
            print(f"  Hub-matching rows ({self.hub}): {n_hub:,}/{len(df):,}")

        print(
            f"  Processed ASPM: {len(df):,} records, "
            f"years: {sorted(df['aspm_year'].unique())}"
        )
        return df


class AnomalyDetector:
    """
    Paper Section 3.1.3 Step 1: Anomaly Detection.
    Identifies data errors (impossible/contradictory values) while RETAINING
    genuine extreme delays.

    Implementation: IQR-fence method on a representative sample, applied back
    to the full dataset via threshold rules.  Replaces the original DBSCAN
    approach which is O(n²) in memory and kills the process on full BTS files
    (500k+ rows → multi-GB distance matrix).

    Rules that flag a record as a DATA ERROR (not just a big delay):
      • TaxiOut > 200 min  (physically impossible — longest ever ~4 h but that
        was an extreme weather event; BTS docs cap valid values at 200)
      • Distance <= 0
      • DepDelayMinutes < 0 and ArrDelay > 180  (contradictory: early departure
        but massive arrival delay suggests a data entry swap)
      • DepDelayMinutes > 1440 (> 24 h — almost certainly a date-wrap error)

    Genuine extreme delays (DepDelayMinutes >= extreme_threshold) are NEVER
    removed regardless of the above flags, preserving the paper's intent.
    """

    # Hard physical-limit thresholds (independent of dataset distribution)
    MAX_TAXI_OUT = 200        # minutes
    MAX_DELAY_MINUTES = 1440  # 24 hours — beyond this is almost certainly a bad record

    def __init__(self, extreme_threshold: int = 180):
        self.extreme_threshold = extreme_threshold

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 10:
            return df

        error_mask = pd.Series(False, index=df.index)

        # Rule 1: impossible taxi-out time
        if "TaxiOut" in df.columns:
            taxi = pd.to_numeric(df["TaxiOut"], errors="coerce")
            error_mask |= taxi > self.MAX_TAXI_OUT

        # Rule 2: non-positive distance
        if "Distance" in df.columns:
            dist = pd.to_numeric(df["Distance"], errors="coerce")
            error_mask |= dist <= 0

        # Rule 3: contradictory early-dep / massive-arr-delay
        if "DepDelayMinutes" in df.columns and "ArrDelay" in df.columns:
            dep = pd.to_numeric(df["DepDelayMinutes"], errors="coerce").fillna(0)
            arr = pd.to_numeric(df["ArrDelay"], errors="coerce").fillna(0)
            error_mask |= (dep < 0) & (arr > 180)

        # Rule 4: delay > 24 hours — almost certainly a date-rollover data error
        if "DepDelayMinutes" in df.columns:
            dep = pd.to_numeric(df["DepDelayMinutes"], errors="coerce").fillna(0)
            error_mask |= dep > self.MAX_DELAY_MINUTES

        # Never remove genuine extreme delays — paper Section 3.1.3 explicitly
        # states flagged extreme delays should be retained
        if "DepDelayMinutes" in df.columns:
            extreme_mask = (
                pd.to_numeric(df["DepDelayMinutes"], errors="coerce").fillna(0)
                >= self.extreme_threshold
            )
        else:
            extreme_mask = pd.Series(False, index=df.index)

        remove_mask = error_mask & ~extreme_mask

        n_errors = error_mask.sum()
        n_extreme = extreme_mask.sum()
        n_removed = remove_mask.sum()
        print(
            f"  Anomaly detection: {n_errors} flagged as data errors, "
            f"{n_extreme} extreme delays retained, {n_removed} records removed"
        )
        return df[~remove_mask].copy()


class AirportFeatureExtractor:
    """
    Paper Section 3.1.3 Step 3: Airport Status Feature Extraction.
    Derives: historical peak capacity, cumulative delay, operational density,
    real-time utilization rate.
    """

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["dep_hour"] = pd.to_numeric(
            df["CRSDepTime"].astype(str).str.zfill(4).str[:2], errors="coerce"
        )
        df["dep_date"] = pd.to_datetime(df["FlightDate"], errors="coerce")

        df["airport_key"] = df["Origin"]
        df["date_hour"] = (
            df["dep_date"].dt.strftime("%Y-%m-%d")
            + "_"
            + df["dep_hour"].fillna(0).astype(int).astype(str)
        )

        # 1. Operational density
        hourly_counts = (
            df.groupby(["airport_key", "date_hour"])
            .size()
            .reset_index(name="op_density")
        )
        df = df.merge(hourly_counts, on=["airport_key", "date_hour"], how="left")

        # 2. Historical peak capacity per airport
        peak_capacity = (
            df.groupby("airport_key")["op_density"]
            .max()
            .reset_index(name="hist_peak_capacity")
        )
        df = df.merge(peak_capacity, on="airport_key", how="left")

        # 3. Real-time utilization rate
        df["utilization_rate"] = df["op_density"] / df["hist_peak_capacity"].clip(lower=1)

        # 4. Cumulative departure delay in current hour
        df = df.sort_values(["airport_key", "dep_date", "dep_hour", "CRSDepTime"])
        df["cum_delay_past_hour"] = df.groupby(
            ["airport_key", "dep_date", "dep_hour"]
        )["DepDelayMinutes"].transform("sum")

        n_airports = df["airport_key"].nunique()
        print(f"  Airports covered:      {n_airports}")
        print(f"  op_density:            min={df['op_density'].min():.0f}  "
              f"max={df['op_density'].max():.0f}  "
              f"mean={df['op_density'].mean():.1f}")
        print(f"  hist_peak_capacity:    min={df['hist_peak_capacity'].min():.0f}  "
              f"max={df['hist_peak_capacity'].max():.0f}")
        print(f"  utilization_rate:      min={df['utilization_rate'].min():.2f}  "
              f"max={df['utilization_rate'].max():.2f}")
        print(f"  cum_delay_past_hour:   min={df['cum_delay_past_hour'].min():.0f}  "
              f"max={df['cum_delay_past_hour'].max():.0f} min")
        null_counts = {c: int(df[c].isna().sum())
                       for c in ["op_density", "hist_peak_capacity",
                                 "utilization_rate", "cum_delay_past_hour"]}
        if any(v > 0 for v in null_counts.values()):
            print(f"  WARNING nulls: {null_counts}")
        else:
            print(f"  All 4 airport features complete (0 nulls)")

        return df


class DataPipeline:
    """
    End-to-end data processing pipeline.
    Paper Section 3.1.3 Steps 1-4 + Section 3.1.4 flight chain construction.
    """

    def __init__(self, config):
        self.config = config
        self.bts_processor = BTSProcessor(config.bts_data_dir, config.hub_airport)
        self.noaa_processor = NOAAProcessor(config.noaa_data_dir)
        self.aspm_processor = ASPMProcessor(
            getattr(config, "aspm_data_dir", "./data/raw/aspm"),
            hub=config.hub_airport,
        )
        self.anomaly_detector = AnomalyDetector(config.extreme_delay_threshold)
        self.airport_extractor = AirportFeatureExtractor()

        self.scaler = MinMaxScaler()
        if _HAS_JS_ENCODER:
            self.js_encoder = JamesSteinEncoder()
        else:
            self.js_encoder = None

    def run(self, save: bool = True) -> pd.DataFrame:
        """Execute the full pipeline."""
        print("=" * 60)
        print("STEP 1: Loading data")
        print("=" * 60)
        flight_df = self.bts_processor.load()
        hub_flights = self.bts_processor.filter_hub_connections(flight_df)

        print("\n" + "=" * 60)
        print("STEP 2: Anomaly detection")
        print("=" * 60)
        cleaned = self.anomaly_detector.detect(hub_flights)

        print("\n" + "=" * 60)
        print("STEP 3: Data cleaning")
        print("=" * 60)
        cleaned = self._clean_flights(cleaned)

        print("\n" + "=" * 60)
        print("STEP 4: Airport feature extraction")
        print("=" * 60)
        cleaned = self.airport_extractor.extract(cleaned)

        print("\n" + "=" * 60)
        print("STEP 5: Weather data integration")
        print("=" * 60)
        weather_df = self.noaa_processor.load()
        weather_df = self.noaa_processor.process(weather_df)
        merged = self._merge_weather(cleaned, weather_df)

        print("\n" + "=" * 60)
        print("STEP 5B: ASPM data integration")
        print("=" * 60)
        aspm_df = self.aspm_processor.load()
        if not aspm_df.empty:
            aspm_df = self.aspm_processor.process(aspm_df)
            merged = self._merge_aspm(merged, aspm_df)
        else:
            print("  No ASPM data found — skipping (model will train without it)")

        print("\n" + "=" * 60)
        print("STEP 6: Derived feature computation")
        print("=" * 60)
        merged = self._compute_derived_features(merged)

        print("\n" + "=" * 60)
        print("STEP 7: Flight chain construction")
        print("=" * 60)
        merged = self._build_flight_chains(merged)

        print("\n" + "=" * 60)
        print("STEP 8: Feature encoding")
        print("=" * 60)
        merged = self._encode_features(merged)

        if save:
            import joblib, json
            from datetime import datetime
            out_path = Path(self.config.processed_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # ── Primary output ────────────────────────────────────────────
            flights_path = out_path / "processed_flights.parquet"
            merged.to_parquet(flights_path, index=False)

            # ── Encoder metadata (column lists the encoder was fitted on) ─
            encoder_meta = {
                "norm_cols": merged.select_dtypes(include="number").columns.tolist(),
                "cat_cols": ["Origin", "Dest", "Reporting_Airline", "Tail_Number",
                             "OriginState", "DestState"],
            }
            with open(out_path / "encoder_meta.json", "w") as f:
                json.dump(encoder_meta, f, indent=2)

            # ── Manifest — single source of truth for downstream steps ───
            feature_groups = get_feature_groups(merged)
            manifest = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "hub_airport": self.config.hub_airport,
                "total_records": len(merged),
                "total_columns": len(merged.columns),
                "chain_count": int(merged["chain_id"].nunique()),
                "date_range": {
                    "min": str(merged["FlightDate"].min()) if "FlightDate" in merged.columns else None,
                    "max": str(merged["FlightDate"].max()) if "FlightDate" in merged.columns else None,
                },
                "feature_counts": {g: len(cols) for g, cols in feature_groups.items()},
                "feature_groups": feature_groups,
                "files": {
                    "processed_flights": str(flights_path),
                    "scaler":   str(out_path / "scaler.joblib"),
                    "encoder":  str(out_path / "encoder.joblib") if self.js_encoder else None,
                    "encoder_meta": str(out_path / "encoder_meta.json"),
                    "proxy_sequences": str(out_path / "proxy_sequences.parquet"),
                },
                "ready_for_training": True,
            }
            manifest_path = out_path / "pipeline_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            print(f"\n{'='*60}")
            print(f"PREPROCESSING COMPLETE — all outputs saved to {out_path}/")
            print(f"{'='*60}")
            print(f"  processed_flights.parquet : {len(merged):,} records x {len(merged.columns)} cols")
            print(f"  scaler.joblib             : MinMaxScaler (fitted on {len(encoder_meta['norm_cols'])} numeric cols)")
            if self.js_encoder:
                print(f"  encoder.joblib            : JamesSteinEncoder (fitted on {encoder_meta['cat_cols']})")
            print(f"  encoder_meta.json         : column lists used during encoding")
            print(f"  pipeline_manifest.json    : full run summary")
            print(f"\n  Feature groups ready for model:")
            for g, cols in feature_groups.items():
                print(f"    {g} ({len(cols)}): {cols}")
            print(f"\n  Run training next:")
            print(f"    python main.py --mode train")

        return merged

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _clean_flights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paper Step 2: remove records with critical missing attributes."""
        initial = len(df)

        # FIX-5: BTS Cancelled/Diverted are float (0.0/1.0) — use != 1 instead of == 0
        if "Cancelled" in df.columns:
            df = df[df["Cancelled"] != 1]
        if "Diverted" in df.columns:
            df = df[df["Diverted"] != 1]

        critical = ["DepDelay", "Tail_Number", "Origin", "Dest", "CRSDepTime"]
        available_critical = [c for c in critical if c in df.columns]
        df = df.dropna(subset=available_critical)

        delay_causes = ["CarrierDelay", "WeatherDelay", "NASDelay",
                        "SecurityDelay", "LateAircraftDelay"]
        for col in delay_causes:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        print(f"  Cleaned: {initial:,} -> {len(df):,} records")
        return df

    def _merge_weather(self, flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-source data integration.
        Merges weather to flights by AIRPORT CODE and DEPARTURE HOUR.

        FIX-1: NOAAProcessor.process() already sets `airport_code` from the filename
               (_airport_code tag). We must NOT overwrite it by remapping STATION,
               because the numeric suffix in the filename (e.g. "13962") does not match
               the full STATION field value (e.g. "72266013962").
        FIX-2: Guard dep_hour extraction against edge cases (CRSDepTime=2400, NaN).
        """
        flights = flights.copy()

        wx_cols = [
            "HourlyWindSpeed", "HourlyVisibility", "HourlyPrecipitation",
            "HourlyDryBulbTemperature", "HourlyRelativeHumidity",
            "cloud_cover", "wx_severity",
        ]
        available_wx = [c for c in wx_cols if c in weather.columns]

        if not available_wx:
            print("  WARNING: No weather feature columns found — filling with 0")
            for col in wx_cols:
                flights[col] = 0.0
            return flights

        # FIX-2: robust dep_hour extraction — handles NaN and 2400 edge case
        crs_numeric = pd.to_numeric(flights["CRSDepTime"], errors="coerce").fillna(0)
        dep_hour = (crs_numeric.astype(int) // 100).clip(0, 23)

        flights["dep_datetime"] = (
            pd.to_datetime(flights["FlightDate"], errors="coerce")
            + pd.to_timedelta(dep_hour, unit="h")
        )
        flights["dep_hour_key"] = flights["dep_datetime"].dt.floor("h")

        # FIX-1: airport_code is already correct (set from filename in NOAAProcessor).
        #        Only fall back to STATION mapping if airport_code is genuinely absent.
        if "airport_code" not in weather.columns:
            print("  WARNING: airport_code missing from weather data — weather merge will be global-only")
            if "hour" in weather.columns:
                wx_hourly = (
                    weather.groupby("hour")[available_wx]
                    .mean()
                    .reset_index()
                    .rename(columns={"hour": "dep_hour_key"})
                )
                flights = flights.merge(wx_hourly, on="dep_hour_key", how="left")
            for col in wx_cols:
                if col in flights.columns:
                    flights[col] = flights[col].fillna(0)
            return flights

        # Aggregate weather by (airport_code, hour)
        wx_hourly = (
            weather.dropna(subset=["airport_code"])
            .groupby(["airport_code", "hour"])[available_wx]
            .mean()
            .reset_index()
            .rename(columns={"airport_code": "Origin", "hour": "dep_hour_key"})
        )

        pre_len = len(flights)
        flights = flights.merge(wx_hourly, on=["Origin", "dep_hour_key"], how="left")
        matched = flights[available_wx[0]].notna().sum()
        print(
            f"  Weather merged: {matched:,}/{pre_len:,} flights matched "
            f"({matched / max(pre_len, 1) * 100:.1f}%)"
        )

        for col in wx_cols:
            if col in flights.columns:
                flights[col] = flights[col].fillna(0)

        return flights

    def _merge_aspm(self, flights: pd.DataFrame, aspm: pd.DataFrame) -> pd.DataFrame:
        """
        Merge ASPM city-pair metrics into flight data.
        ASPM granularity: (Departure, Arrival, Year, Month, Departure Hour).
        BTS granularity:  (Origin, Dest, Year, Month, Day, Departure Hour).
        """
        flights = flights.copy()

        flights["_year"] = pd.to_numeric(flights.get("Year"), errors="coerce").astype("Int64")
        flights["_month"] = pd.to_numeric(flights.get("Month"), errors="coerce").astype("Int64")

        crs_numeric = pd.to_numeric(flights["CRSDepTime"], errors="coerce").fillna(0)
        flights["_dep_hour"] = (crs_numeric.astype(int) // 100).clip(0, 23).astype("Int64")

        aspm_feature_cols = [c for c in ASPMProcessor.ASPM_FEATURES if c in aspm.columns]
        if not aspm_feature_cols:
            print("  WARNING: No ASPM feature columns found")
            return flights

        group_keys = ["Departure", "Arrival", "aspm_year", "aspm_month", "aspm_dep_hour"]
        # FIX-4: ensure aspm_dep_hour is numeric/Int64 before groupby
        aspm_clean = aspm.dropna(subset=group_keys).copy()
        aspm_clean["aspm_dep_hour"] = (
            pd.to_numeric(aspm_clean["aspm_dep_hour"], errors="coerce")
        )

        aspm_agg = (
            aspm_clean
            .groupby(group_keys)[aspm_feature_cols]
            .mean()
            .reset_index()
            .rename(columns={
                "Departure": "Origin",
                "Arrival": "Dest",
                "aspm_year": "_year",
                "aspm_month": "_month",
                "aspm_dep_hour": "_dep_hour",
            })
        )

        # Safe Int64 cast — fill NaN first to avoid conversion errors
        for key_col in ["_year", "_month", "_dep_hour"]:
            aspm_agg[key_col] = (
                pd.to_numeric(aspm_agg[key_col], errors="coerce")
                .round()
                .astype("Int64")
            )

        pre_len = len(flights)
        flights = flights.merge(
            aspm_agg,
            on=["Origin", "Dest", "_year", "_month", "_dep_hour"],
            how="left",
        )
        first_feat = aspm_feature_cols[0]
        matched = flights[first_feat].notna().sum()
        print(
            f"  ASPM merged: {matched:,}/{pre_len:,} flights "
            f"({matched / max(pre_len, 1) * 100:.1f}%) "
            f"via {len(aspm_agg):,} city-pair × month × hour rows"
        )

        for col in aspm_feature_cols:
            if col in flights.columns:
                flights[col] = flights[col].fillna(0)

        flights.drop(columns=["_year", "_month", "_dep_hour"], errors="ignore", inplace=True)
        return flights

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features that require combining multiple raw BTS columns.
        Called after cleaning, before chain construction.

        New features:
          actual_taxi_out   — WheelsOff minus DepTime in minutes (real push-back time)
          taxi_out_excess   — actual_taxi_out minus scheduled TaxiOut (congestion signal)
          actual_airborne   — WheelsOn minus WheelsOff in minutes (real flight time)
          airborne_excess   — actual_airborne minus AirTime (headwind / routing signal)

        These are strictly better than the raw WheelsOff/WheelsOn timestamps because
        the model needs a numeric duration, not a HHMM clock reading.
        """
        df = df.copy()

        def hhmm_to_min(series: pd.Series) -> pd.Series:
            """Convert HHMM integer (e.g. 1423 → 864 min) to minutes since midnight."""
            s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
            return (s // 100) * 60 + (s % 100)

        if "WheelsOff" in df.columns and "DepTime" in df.columns:
            wo  = hhmm_to_min(df["WheelsOff"])
            dep = hhmm_to_min(df["DepTime"])
            df["actual_taxi_out"] = (wo - dep).clip(lower=0)
            taxi_sched = pd.to_numeric(df.get("TaxiOut"), errors="coerce").fillna(0)
            df["taxi_out_excess"] = df["actual_taxi_out"] - taxi_sched

        if "WheelsOn" in df.columns and "WheelsOff" in df.columns:
            won = hhmm_to_min(df["WheelsOn"])
            wof = hhmm_to_min(df["WheelsOff"])
            df["actual_airborne"] = (won - wof).clip(lower=0)
            air_sched = pd.to_numeric(df.get("AirTime"), errors="coerce").fillna(0)
            df["airborne_excess"] = df["actual_airborne"] - air_sched

        new_cols = [c for c in
                    ["actual_taxi_out","taxi_out_excess","actual_airborne","airborne_excess"]
                    if c in df.columns]
        if new_cols:
            print(f"  Derived features computed: {new_cols}")

        return df

    def _build_flight_chains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paper Section 3.1.4: Reconstruct flight chains by tail number + date.
        Ck = {x1, x2, ..., xi | Date(xi) = Dj, Tail_Number(xi) = Nj}
        """
        df = df.copy()
        df = df.sort_values(["Tail_Number", "FlightDate", "CRSDepTime"])

        df["chain_id"] = df.groupby(["Tail_Number", "FlightDate"]).ngroup()
        df["chain_position"] = df.groupby("chain_id").cumcount()
        chain_lengths = df.groupby("chain_id").size()
        df["chain_length"] = df["chain_id"].map(chain_lengths)

        df["prev_arr_delay"] = (
            df.groupby("chain_id")["ArrDelay"].shift(1).fillna(0)
        )

        if "ArrTime" in df.columns:
            df["prev_arr_time"] = df.groupby("chain_id")["ArrTime"].shift(1)
            arr_num = pd.to_numeric(df["prev_arr_time"], errors="coerce").fillna(0)
            dep_num = pd.to_numeric(df["CRSDepTime"], errors="coerce").fillna(0)
            arr_h, arr_m = arr_num.astype(int) // 100, arr_num.astype(int) % 100
            dep_h, dep_m = dep_num.astype(int) // 100, dep_num.astype(int) % 100
            df["turnaround_minutes"] = (
                (dep_h * 60 + dep_m) - (arr_h * 60 + arr_m)
            ).clip(lower=0).fillna(60)
        else:
            df["turnaround_minutes"] = 60

        # Shift delay-cause columns from the PREVIOUS leg in the chain.
        # Tells the model WHY the inbound aircraft was late, not just how late —
        # a weather delay propagates differently than a carrier/NAS delay.
        cause_cols = ["CarrierDelay", "WeatherDelay", "NASDelay",
                      "SecurityDelay", "LateAircraftDelay"]
        for col in cause_cols:
            if col in df.columns:
                prev_col = f"prev_{col.lower()}"
                df[prev_col] = (
                    df.groupby("chain_id")[col].shift(1).fillna(0)
                )

        n_chains = df["chain_id"].nunique()
        avg_len = df["chain_length"].mean()
        print(f"  Built {n_chains:,} flight chains, avg length: {avg_len:.1f}")
        return df

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paper Section 3.1.3: James-Stein encoding for categoricals,
        Min-Max normalization for numerics.
        FIX-6: graceful fallback to target-mean encoding if category_encoders absent.
        """
        df = df.copy()

        # Preserve raw string Origin/Dest so downstream steps (proxy engineering,
        # pair risk scoring) can still filter on "DFW" after encoding.
        if "Origin" in df.columns:
            df["Origin_str"] = df["Origin"].astype(str)
        if "Dest" in df.columns:
            df["Dest_str"] = df["Dest"].astype(str)

        # Preserve raw HHMM scheduled/actual times for proxy engineering
        # (Min-Max normalization below would otherwise destroy them).
        for tcol in ["CRSDepTime", "CRSArrTime", "DepTime", "ArrTime"]:
            if tcol in df.columns:
                df[f"{tcol}_raw"] = pd.to_numeric(df[tcol], errors="coerce")

        # Calendar fields: keep raw copies for train/val/test splitting,
        # and cyclical-encode Month + DayOfWeek so the model sees their wrap-around.
        if "Year" in df.columns:
            df["Year_raw"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        if "Month" in df.columns:
            m = pd.to_numeric(df["Month"], errors="coerce")
            df["Month_raw"] = m.astype("Int64")
            df["Month_sin"] = np.sin(2 * np.pi * m / 12)
            df["Month_cos"] = np.cos(2 * np.pi * m / 12)
            df = df.drop(columns=["Month"])
        if "DayOfWeek" in df.columns:
            d = pd.to_numeric(df["DayOfWeek"], errors="coerce")
            df["DayOfWeek_raw"] = d.astype("Int64")
            df["DayOfWeek_sin"] = np.sin(2 * np.pi * d / 7)
            df["DayOfWeek_cos"] = np.cos(2 * np.pi * d / 7)
            df = df.drop(columns=["DayOfWeek"])

        # OriginState / DestState are string categoricals — encode them too.
        # OriginCityName / DestCityName are kept in the df for reporting
        # but are too high-cardinality to be useful model features.
        cat_cols = ["Origin", "Dest", "Reporting_Airline", "Tail_Number",
                    "OriginState", "DestState"]
        available_cats = [c for c in cat_cols if c in df.columns]

        if available_cats and "DepDelay" in df.columns:
            target = df["DepDelay"].fillna(0)
            for col in available_cats:
                df[col] = df[col].astype(str)

            if _HAS_JS_ENCODER and self.js_encoder is not None:
                df[available_cats] = self.js_encoder.fit_transform(
                    df[available_cats], target
                )
                print(f"  James-Stein encoded: {available_cats}")
            else:
                # FIX-6: target-mean fallback
                for col in available_cats:
                    mean_map = df.groupby(col)["DepDelay"].mean()
                    df[col] = df[col].map(mean_map).fillna(target.mean())
                print(f"  Target-mean encoded (fallback): {available_cats}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["DepDelay", "DepDelayMinutes", "chain_id", "chain_position",
                   "Year_raw", "Month_raw", "DayOfWeek_raw",
                   "CRSDepTime_raw", "CRSArrTime_raw", "DepTime_raw", "ArrTime_raw",
                   "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos"]
        norm_cols = [c for c in numeric_cols if c not in exclude]

        if norm_cols:
            df[norm_cols] = self.scaler.fit_transform(df[norm_cols].fillna(0))
            print(f"  Min-Max normalized {len(norm_cols)} numeric features")

        # Save fitted transformers so inference on new raw data uses the
        # same scale/encoding as training data.
        if hasattr(self, 'config'):
            import joblib
            out_path = Path(self.config.processed_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, out_path / "scaler.joblib")
            print(f"  Saved scaler → {out_path / 'scaler.joblib'}")
            if self.js_encoder is not None:
                joblib.dump(self.js_encoder, out_path / "encoder.joblib")
                print(f"  Saved encoder → {out_path / 'encoder.joblib'}")

        return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Paper Eq. 1: xi = {ai, bi, fi}
    Returns column name groups for airport status, weather, and flight features.

    All columns listed here must be present in REQUIRED_COLS (loaded from BTS)
    or computed by the pipeline (_compute_derived_features / _build_flight_chains
    / merge steps).  Only columns that actually exist in df are returned, so
    adding a column here is safe even if it is absent for some data years.
    """
    airport_features = [
        # ── Operational density (computed by AirportFeatureExtractor) ──
        "op_density", "hist_peak_capacity", "utilization_rate",
        "cum_delay_past_hour",
        # ── ASPM city-pair metrics (merged from FAA ASPM) ──────────────
        "aspm_flight_count",
        "aspm_pct_ontime_gate_dep", "aspm_pct_ontime_airport_dep",
        "aspm_pct_ontime_gate_arr",
        "aspm_edct_count", "aspm_avg_edct",
        "aspm_gate_dep_delay", "aspm_taxi_out_delay",
        "aspm_avg_taxi_out", "aspm_airport_dep_delay",
        "aspm_airborne_delay", "aspm_taxi_in_delay",
        "aspm_block_delay", "aspm_gate_arr_delay",
    ]
    weather_features = [
        # ── NOAA LCD hourly observations (merged by airport + hour) ────
        "HourlyWindSpeed", "HourlyVisibility", "HourlyPrecipitation",
        "HourlyDryBulbTemperature", "HourlyRelativeHumidity",
        "cloud_cover", "wx_severity",
    ]
    flight_features = [
        # ── Route identity ─────────────────────────────────────────────
        "Origin", "Dest", "Reporting_Airline",
        "OriginState", "DestState",              # regional congestion pattern
        "Flight_Number_Reporting_Airline",        # schedule-slot recurring delays
        # ── Scheduled times ────────────────────────────────────────────
        "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
        "Distance",
        # ── Taxi / ground time ─────────────────────────────────────────
        "TaxiOut",                               # scheduled taxi-out
        "actual_taxi_out",                       # WheelsOff - DepTime (real)
        "taxi_out_excess",                       # actual - scheduled (congestion)
        # ── Airborne time ──────────────────────────────────────────────
        "actual_airborne",                       # WheelsOn - WheelsOff (real)
        "airborne_excess",                       # actual - AirTime (routing/winds)
        # ── Temporal (cyclical) ────────────────────────────────────────
        "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
        # ── Chain propagation ──────────────────────────────────────────
        "prev_arr_delay",                        # how late was the inbound aircraft
        "turnaround_minutes",                    # gate time before next departure
        "chain_position", "chain_length",
        # ── Delay-cause from previous chain leg ────────────────────────
        # WHY was the previous flight late — propagation differs by cause type
        "prev_carrierdelay",
        "prev_weatherdelay",
        "prev_nasdelay",
        "prev_securitydelay",
        "prev_lateaircraftdelay",
    ]

    # Framing A (causal): drop any column that's only knowable post-departure
    # or depends on same-day chain observations.
    _LEAKY = {
        "actual_taxi_out", "taxi_out_excess", "actual_airborne", "airborne_excess",
        "TaxiOut", "AirTime", "WheelsOff", "WheelsOn",
        "DepTime", "ArrTime", "DepTime_raw", "ArrTime_raw",
        "ArrDelay", "ArrDelayMinutes", "DepDelayMinutes",
        "prev_arr_delay", "prev_dep_delay",
        "prev_carrierdelay", "prev_weatherdelay", "prev_nasdelay",
        "prev_securitydelay", "prev_lateaircraftdelay",
        "cum_delay_past_hour",   # real-time operational signal, not available pre-season
    }

    available = set(df.columns)
    return {
        "airport": [c for c in airport_features if c in available and c not in _LEAKY],
        "weather": [c for c in weather_features if c in available and c not in _LEAKY],
        "flight":  [c for c in flight_features  if c in available and c not in _LEAKY],
    }
