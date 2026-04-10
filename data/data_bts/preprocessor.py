"""
Data Preprocessing Pipeline (Paper Section 3.1)
Steps:
  1. Anomaly detection (retain extreme delays)
  2. Data cleaning (flight + weather)
  3. Airport status feature extraction
  4. Multi-source data integration
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from category_encoders import JamesSteinEncoder
from sklearn.cluster import DBSCAN
import warnings

warnings.filterwarnings("ignore")


class BTSProcessor:
    """Process Bureau of Transportation Statistics on-time performance data."""

    REQUIRED_COLS = [
        "Year", "Month", "DayofMonth", "DayOfWeek",
        "FlightDate", "Reporting_Airline", "Tail_Number",
        "Origin", "Dest",
        "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes",
        "CRSArrTime", "ArrTime", "ArrDelay", "ArrDelayMinutes",
        "TaxiOut", "TaxiIn",
        "CRSElapsedTime", "ActualElapsedTime",
        "AirTime", "Distance",
        "CarrierDelay", "WeatherDelay", "NASDelay",
        "SecurityDelay", "LateAircraftDelay",
        "Cancelled", "Diverted",
    ]

    def __init__(self, data_dir: str, hub: str = "DFW"):
        self.data_dir = Path(data_dir)
        self.hub = hub

    def load(self) -> pd.DataFrame:
        """Load and concatenate all BTS CSV files."""
        files = sorted(self.data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        dfs = []
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            # Keep only columns that exist
            cols = [c for c in self.REQUIRED_COLS if c in df.columns]
            dfs.append(df[cols])
            print(f"  Loaded {f.name}: {len(df):,} records")

        data = pd.concat(dfs, ignore_index=True)
        print(f"Total raw records: {len(data):,}")
        return data

    def filter_hub_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to flights involving the hub airport (DFW).
        For the hackathon: A -> DFW -> B sequences.
        """
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
        """Load NOAA LCD data files. Tags each row with airport code from filename."""
        files = sorted(self.data_dir.glob("*.csv"))
        dfs = []
        for f in files:
            df = pd.read_csv(f, low_memory=False)
            # Extract airport code from filename pattern: ATL_USW00013874_2023.csv
            parts = f.stem.split("_")
            if len(parts) >= 2 and len(parts[0]) == 3:
                df["_airport_code"] = parts[0]
            dfs.append(df)
            print(f"  Loaded weather: {f.name}: {len(df):,} records")

        data = pd.concat(dfs, ignore_index=True)
        return data

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and interpolate weather data per paper Section 3.1.3 Step 2."""
        # Parse datetime
        df["datetime"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        # Set airport_code from STATION mapping or filename tag
        if "_airport_code" in df.columns:
            df["airport_code"] = df["_airport_code"]
        elif "STATION" in df.columns:
            df["station_id"] = df["STATION"]

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

        # Short-gap linear interpolation (paper: short gaps interpolated linearly)
        df = df.sort_values("datetime")
        for col in numeric_wx:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit=3)

        # Resample to hourly (align with flight data)
        df["hour"] = df["datetime"].dt.floor("h")

        return df

    @staticmethod
    def _parse_sky(val):
        """Convert sky condition string to numeric cloud cover (0-8 oktas)."""
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
        """Map present weather to severity score."""
        if pd.isna(val):
            return 0
        val = str(val).upper()
        severity = 0
        if "TS" in val:
            severity = max(severity, 5)  # thunderstorm
        if "FZ" in val:
            severity = max(severity, 4)  # freezing
        if "SN" in val:
            severity = max(severity, 3)  # snow
        if "RA" in val and "HV" in val:
            severity = max(severity, 3)  # heavy rain
        if "RA" in val:
            severity = max(severity, 2)  # rain
        if "BR" in val or "FG" in val:
            severity = max(severity, 2)  # mist/fog
        if "HZ" in val:
            severity = max(severity, 1)  # haze
        return severity


class AnomalyDetector:
    """
    Paper Section 3.1.3 Step 1: Anomaly Detection
    Uses DBSCAN to distinguish genuine extreme delays from erroneous records.
    Extreme delays are RETAINED, only data errors are removed.
    """

    def __init__(self, extreme_threshold: int = 180):
        self.extreme_threshold = extreme_threshold

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag anomalies but retain genuine extreme delays."""
        features_for_clustering = ["DepDelayMinutes", "TaxiOut", "Distance"]
        available = [c for c in features_for_clustering if c in df.columns]

        if len(available) < 2:
            print("  Not enough features for anomaly detection, skipping")
            return df

        subset = df[available].dropna()
        if len(subset) < 100:
            return df

        # Normalize for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(subset)

        # DBSCAN - noise points (label=-1) are potential anomalies
        clustering = DBSCAN(eps=2.0, min_samples=10).fit(X)
        labels = pd.Series(-2, index=df.index)
        labels[subset.index] = clustering.labels_

        # Only remove noise points that are NOT extreme delays
        # (Paper: "delays flagged as statistically extreme are not removed")
        noise_mask = labels == -1
        extreme_mask = df["DepDelayMinutes"] >= self.extreme_threshold
        remove_mask = noise_mask & ~extreme_mask

        n_removed = remove_mask.sum()
        print(f"  Anomaly detection: removed {n_removed} erroneous records, "
              f"retained {extreme_mask.sum()} extreme delays")

        return df[~remove_mask].copy()


class AirportFeatureExtractor:
    """
    Paper Section 3.1.3 Step 3: Airport Status Feature Extraction
    Derives: historical peak capacity, cumulative delay, operational density,
    real-time utilization rate.
    """

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute airport-level operational features."""
        df = df.copy()

        # Parse departure hour
        df["dep_hour"] = pd.to_numeric(
            df["CRSDepTime"].astype(str).str.zfill(4).str[:2], errors="coerce"
        )
        df["dep_date"] = pd.to_datetime(df["FlightDate"], errors="coerce")

        # Group key: airport + date + hour
        df["airport_key"] = df["Origin"]
        df["date_hour"] = df["dep_date"].dt.strftime("%Y-%m-%d") + "_" + df["dep_hour"].astype(str)

        # 1. Operational density: departures in current hour
        hourly_counts = df.groupby(["airport_key", "date_hour"]).size().reset_index(name="op_density")
        df = df.merge(hourly_counts, on=["airport_key", "date_hour"], how="left")

        # 2. Historical peak capacity per airport
        peak_capacity = df.groupby("airport_key")["op_density"].max().reset_index(
            name="hist_peak_capacity"
        )
        df = df.merge(peak_capacity, on="airport_key", how="left")

        # 3. Real-time utilization rate
        df["utilization_rate"] = df["op_density"] / df["hist_peak_capacity"].clip(lower=1)

        # 4. Cumulative departure delay in past hour
        # Sort and compute rolling sum within airport-date groups
        df = df.sort_values(["airport_key", "dep_date", "dep_hour", "CRSDepTime"])
        df["cum_delay_past_hour"] = (
            df.groupby(["airport_key", "dep_date", "dep_hour"])["DepDelayMinutes"]
            .transform("sum")
        )

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
        self.anomaly_detector = AnomalyDetector(config.extreme_delay_threshold)
        self.airport_extractor = AirportFeatureExtractor()

        self.scaler = MinMaxScaler()
        self.js_encoder = JamesSteinEncoder()

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
        print("STEP 6: Flight chain construction")
        print("=" * 60)
        merged = self._build_flight_chains(merged)

        print("\n" + "=" * 60)
        print("STEP 7: Feature encoding")
        print("=" * 60)
        merged = self._encode_features(merged)

        if save:
            out_path = Path(self.config.processed_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(out_path / "processed_flights.parquet", index=False)
            print(f"\nSaved processed data: {len(merged):,} records")

        return merged

    def _clean_flights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paper Step 2: remove records with critical missing attributes."""
        initial = len(df)

        # Remove cancelled flights (can't model delay propagation)
        if "Cancelled" in df.columns:
            df = df[df["Cancelled"] == 0]

        # Remove diverted flights
        if "Diverted" in df.columns:
            df = df[df["Diverted"] == 0]

        # Drop rows missing critical fields
        critical = ["DepDelay", "Tail_Number", "Origin", "Dest", "CRSDepTime"]
        available_critical = [c for c in critical if c in df.columns]
        df = df.dropna(subset=available_critical)

        # Fill delay cause columns with 0 (no delay = 0 minutes)
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
        Merges weather to flights by AIRPORT (via station mapping) and HOUR.
        """
        import json

        flights = flights.copy()

        # Load airport-to-station mapping
        mapping_path = Path(self.config.bts_data_dir).parent.parent / "airport_station_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                airport_to_station = json.load(f)
            # Invert: station -> airport
            station_to_airport = {v: k for k, v in airport_to_station.items()}
        else:
            # Try to infer from NOAA filenames (e.g., ATL_USW00013874_2023.csv)
            station_to_airport = {}
            airport_to_station = {}
            noaa_dir = Path(self.config.noaa_data_dir)
            for f in noaa_dir.glob("*.csv"):
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    airport_to_station[parts[0]] = parts[1]
                    station_to_airport[parts[1]] = parts[0]
            print(f"  Inferred {len(airport_to_station)} airport-station mappings from filenames")

        # Tag weather rows with airport code
        if "STATION" in weather.columns:
            weather["airport_code"] = weather["STATION"].map(station_to_airport)
        elif station_to_airport:
            # Try to extract from filename tags added during loading
            weather["airport_code"] = weather.get("_airport_code", None)

        wx_cols = ["HourlyWindSpeed", "HourlyVisibility", "HourlyPrecipitation",
                   "HourlyDryBulbTemperature", "HourlyRelativeHumidity",
                   "cloud_cover", "wx_severity"]
        available_wx = [c for c in wx_cols if c in weather.columns]

        if not available_wx:
            print("  WARNING: No weather feature columns found")
            for col in wx_cols:
                flights[col] = 0.0
            return flights

        # Build departure hour key for flights
        flights["dep_datetime"] = pd.to_datetime(flights["FlightDate"]) + \
            pd.to_timedelta(
                flights["CRSDepTime"].astype(str).str.zfill(4).str[:2].astype(int),
                unit="h"
            )
        flights["dep_hour_key"] = flights["dep_datetime"].dt.floor("h")

        # Aggregate weather by airport + hour
        if "airport_code" in weather.columns and "hour" in weather.columns:
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
            print(f"  Weather merged: {matched:,}/{pre_len:,} flights matched "
                  f"({matched/pre_len*100:.1f}%) using {len(airport_to_station)} stations")
        else:
            print("  WARNING: Missing airport_code or hour in weather data — falling back to global avg")
            if "hour" in weather.columns:
                wx_hourly = weather.groupby("hour")[available_wx].mean().reset_index()
                wx_hourly = wx_hourly.rename(columns={"hour": "dep_hour_key"})
                flights = flights.merge(wx_hourly, on="dep_hour_key", how="left")

        # Fill NaN weather with 0
        for col in wx_cols:
            if col in flights.columns:
                flights[col] = flights[col].fillna(0)

        return flights

    def _build_flight_chains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paper Section 3.1.4: Reconstruct flight chains by tail number + date.
        Ck = {x1, x2, ..., xi | Date(xi) = Dj, Tail_Number(xi) = Nj}
        """
        df = df.copy()
        df = df.sort_values(["Tail_Number", "FlightDate", "CRSDepTime"])

        # Assign chain ID
        df["chain_id"] = df.groupby(["Tail_Number", "FlightDate"]).ngroup()

        # Position within chain
        df["chain_position"] = df.groupby("chain_id").cumcount()
        chain_lengths = df.groupby("chain_id").size()
        df["chain_length"] = df["chain_id"].map(chain_lengths)

        # Compute preceding flight delay (LateAircraftDelay feature)
        # For each flight in a chain (except first), get the previous leg's delay
        df["prev_arr_delay"] = df.groupby("chain_id")["ArrDelay"].shift(1).fillna(0)

        # Compute time gap to previous flight (for decay calculation)
        if "ArrTime" in df.columns and "CRSDepTime" in df.columns:
            df["prev_arr_time"] = df.groupby("chain_id")["ArrTime"].shift(1)
            arr_mins = pd.to_numeric(df["prev_arr_time"], errors="coerce")
            dep_mins = pd.to_numeric(df["CRSDepTime"], errors="coerce")
            # Convert HHMM to minutes
            arr_h = arr_mins // 100
            arr_m = arr_mins % 100
            dep_h = dep_mins // 100
            dep_m = dep_mins % 100
            df["turnaround_minutes"] = (dep_h * 60 + dep_m) - (arr_h * 60 + arr_m)
            df["turnaround_minutes"] = df["turnaround_minutes"].clip(lower=0).fillna(60)
        else:
            df["turnaround_minutes"] = 60  # default

        n_chains = df["chain_id"].nunique()
        avg_len = df["chain_length"].mean()
        print(f"  Built {n_chains:,} flight chains, avg length: {avg_len:.1f}")

        return df

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paper Section 3.1.3: James-Stein encoding for categoricals,
        Min-Max normalization for numerics.
        """
        df = df.copy()

        # Categorical features -> James-Stein encoding
        cat_cols = ["Origin", "Dest", "Reporting_Airline", "Tail_Number"]
        available_cats = [c for c in cat_cols if c in df.columns]

        if available_cats and "DepDelay" in df.columns:
            target = df["DepDelay"].fillna(0)
            for col in available_cats:
                df[col] = df[col].astype(str)
            df[available_cats] = self.js_encoder.fit_transform(
                df[available_cats], target
            )
            print(f"  James-Stein encoded: {available_cats}")

        # Numeric features -> Min-Max normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target and IDs from normalization
        exclude = ["DepDelay", "DepDelayMinutes", "chain_id", "chain_position"]
        norm_cols = [c for c in numeric_cols if c not in exclude]

        if norm_cols:
            df[norm_cols] = self.scaler.fit_transform(df[norm_cols].fillna(0))
            print(f"  Min-Max normalized {len(norm_cols)} numeric features")

        return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Paper Eq. 1: xi = {ai, bi, fi}
    Returns column name groups for airport status, weather, and flight features.
    """
    airport_features = [
        "op_density", "hist_peak_capacity", "utilization_rate",
        "cum_delay_past_hour",
    ]
    weather_features = [
        "HourlyWindSpeed", "HourlyVisibility", "HourlyPrecipitation",
        "HourlyDryBulbTemperature", "HourlyRelativeHumidity",
        "cloud_cover", "wx_severity",
    ]
    flight_features = [
        "Origin", "Dest", "Reporting_Airline",
        "CRSDepTime", "CRSArrTime", "CRSElapsedTime",
        "Distance", "TaxiOut",
        "DayOfWeek", "Month",
        "prev_arr_delay", "turnaround_minutes",
        "chain_position", "chain_length",
    ]

    # Filter to columns that actually exist
    available = set(df.columns)
    return {
        "airport": [c for c in airport_features if c in available],
        "weather": [c for c in weather_features if c in available],
        "flight": [c for c in flight_features if c in available],
    }
