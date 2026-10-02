"""
HEX CELL 2: PRICING ALGORITHM (ENHANCED - AGGREGATED OUTPUT)

ARCHITECTURE:
- Generates aggregated recommendations at listing+month+is_weekend level
- Input data has daily calendar_date, but output is aggregated (~300K rows)
- Daily expansion happens in Cell 3 via join pattern (see Cell 3 v2)
- Boosts are applied in Cell 3 after expansion to daily level

OUTPUT:
- DataFrame with columns: listing_id, metro, month, is_weekend, recommended_price, etc.
- Each listing has ~24 rows (12 months × 2 weekend/weekday)
- adjusted_price is in ×100 format (e.g., 12500 = $125.00)

WHAT HAPPENS IN CELL 3:
- Join to daily data → expands to calendar_date level
- Calculate utilization (historic + forward)
- Apply boosts (time-to-search, historic demand, forward demand)
- Apply 7% floor check

OPTIMIZATION:
- Memory-optimized (no unnecessary .copy() calls)
- Follows Smart Python principles (readable + efficient)

Copy this entire cell into your second Hex Python cell.
IMPORTANT: Run Cell 1 (Data Loader) before running this cell.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import gc  # Garbage collector for manual memory cleanup


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class PricingConfig:
    """Configuration constants for the pricing algorithm."""

    # Supply sensitivity parameter
    SUPPLY_SENSITIVITY_ALPHA = 0.5

    # Quality scoring thresholds
    MIN_REVIEWS_FOR_QUALITY = 10
    MIN_IMAGES_FOR_QUALITY = 10
    MIN_SLEEPS_FOR_QUALITY = 4
    MIN_VEHICLE_YEAR_FOR_QUALITY = 2019

    # Quality tier thresholds
    PREMIUM_QUALITY_SCORE = 3
    BASIC_QUALITY_SCORE = 1

    # Underpricing threshold
    UNDERPRICED_GAP_PERCENT = 10

    # Supply baseline rolling window
    SUPPLY_ROLLING_MONTHS = 3

    # Minimum occupancy rate for weighting
    MIN_OCCUPANCY_RATE = 0.01

    # Default values for missing data
    DEFAULT_SUPPLY_RATIO = 1.0
    DEFAULT_COHORT_OCCUPANCY = 0.5
    DEFAULT_DEMAND_MULTIPLIER = 1.0
    DEFAULT_LOCAL_MULTIPLIER = 1.0
    DEFAULT_LISTING_MULTIPLIER = 1.0

    # Insight thresholds
    HIGH_DEMAND_THRESHOLD = 80
    LOW_DEMAND_THRESHOLD = 40
    PEAK_SEASON_OCCUPANCY = 0.75
    OFF_SEASON_OCCUPANCY = 0.40
    TRENDING_UP_MULTIPLIER = 1.1
    SOFTENING_MULTIPLIER = 0.9

    # Weekend definition (aligned with old SQL: Sunday=0, Friday=4, Saturday=5)
    WEEKEND_DAYS = [0, 4, 5]

    # Price boost configuration (MODULAR - can be toggled)
    # NOTE: All daily-level boosts are applied in Cell 3 after expansion
    ENABLE_TIME_TO_SEARCH_BOOST = False  # Applied in Cell 3
    ENABLE_HISTORIC_DEMAND_BOOST = False  # Applied in Cell 3
    ENABLE_FORWARD_DEMAND_BOOST = False  # Applied in Cell 3
    ENABLE_SPECIFIC_LISTING_BOOST = False  # Not implemented yet
    ENABLE_COHORT_FLOOR = False  # Disabled - can cause issues at aggregated level

    # Boost percentages
    TIME_TO_SEARCH_7_DAYS = 0.025   # 2.5%
    TIME_TO_SEARCH_30_DAYS = 0.010  # 1.0%
    TIME_TO_SEARCH_90_DAYS = 0.005  # 0.5%

    HISTORIC_DEMAND_MID = 0.025     # 2.5%
    HISTORIC_DEMAND_HIGH = 0.050    # 5.0%

    FORWARD_DEMAND_MID = 0.050      # 5.0%
    FORWARD_DEMAND_HIGH = 0.100     # 10.0%

    # Floor check percentage
    FLOOR_CHECK_PERCENT = 0.07      # 7%

    # Specific listing ID for special boost
    SPECIAL_LISTING_ID = '400742'


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_quantile(values, weights, quantile):
    """Calculate weighted quantile - no copies, operates on arrays directly."""
    if len(values) == 0:
        return np.nan

    sorter = np.argsort(values)
    values = np.array(values)[sorter]
    weights = np.array(weights)[sorter]
    weights = weights / weights.sum()
    cumsum = np.cumsum(weights)

    return np.interp(quantile, cumsum, values)


def calculate_occupancy_weights(df):
    """Calculate occupancy weights - returns minimal DataFrame."""
    occupancy_by_listing = (
        df.groupby("listing_id", as_index=False)
        .agg(booked=("is_booked", "sum"), available=("bookable_days", "sum"))
    )
    occupancy_by_listing["occ_rate"] = (
        occupancy_by_listing["booked"] / occupancy_by_listing["available"].clip(lower=1)
    ).clip(PricingConfig.MIN_OCCUPANCY_RATE)

    return occupancy_by_listing[["listing_id", "occ_rate"]]


def optimize_dtypes(df):
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    int64 → int32 (saves 50%)
    float64 → float32 (saves 50%) where precision loss is acceptable
    """
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # Only downcast floats for non-critical columns
    safe_float_cols = ['supply_ratio', 'cohort_occupancy', 'occupancy',
                       'historic_utilization', 'forward_utilization',
                       'avg_utilization', 'std_plus_1']
    for col in df.select_dtypes(include=['float64']).columns:
        if col in safe_float_cols:
            df[col] = df[col].astype('float32')

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLEANING (OPTIMIZED)
# ══════════════════════════════════════════════════════════════════════════════

def clean_listing_data(listing_data):
    """
    Clean and standardize listing data IN-PLACE (no copy).

    OPTIMIZATION: No .copy() call - modifies the input DataFrame directly.
    This is safe because we're at the start of the pipeline.
    """
    df = listing_data  # No copy!

    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    df["avg_base_price"] = pd.to_numeric(df["avg_base_price"], errors="coerce").fillna(0)
    df["avg_booked_price"] = pd.to_numeric(df["avg_booked_price"], errors="coerce")
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype('int32')
    df["image_count"] = pd.to_numeric(df["image_count"], errors="coerce").fillna(0).astype('int32')
    df["instant_book"] = df["instant_book"].fillna(0).astype('int8')
    df["sleeps"] = pd.to_numeric(df["sleeps"], errors="coerce").fillna(0).astype('int16')
    df["vehicle_year"] = pd.to_numeric(df["vehicle_year"], errors="coerce").fillna(0).astype('int16')
    df["bookable_days"] = pd.to_numeric(df["bookable_days"], errors="coerce").fillna(0).astype('int32')
    df["booked_days"] = pd.to_numeric(df["booked_days"], errors="coerce").fillna(0).astype('int32')
    df["is_booked"] = df["booked_days"].clip(upper=1).astype('int8')
    df["month"] = df["calendar_date"].dt.month.astype('int8')
    df["is_weekend"] = df["calendar_date"].dt.dayofweek.isin(PricingConfig.WEEKEND_DAYS).astype('int8')

    # Filter in-place
    df = df[df["metro"].notna() & (df["metro"] != "")]
    df = df[df["avg_base_price"] > 0]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIZATION CALCULATIONS (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_historic_utilization(historic_data):
    """
    Calculate historic utilization by date token (month+day).

    Returns: DataFrame with date_token → average utilization
    Example: date_token '305' = March 5th
    """
    historic_data = historic_data.copy()  # Safe to copy here - it's small aggregated data
    historic_data['calendar_date'] = pd.to_datetime(historic_data['calendar_date'])

    # Step 1: Calculate daily utilization
    daily_util = (
        historic_data.groupby('calendar_date', as_index=False)
        .agg(
            booked_listings=('is_booked', 'sum'),
            total_listings=('listing_id', 'count')
        )
    )
    daily_util['utilization'] = (
        daily_util['booked_listings'] / daily_util['total_listings'].clip(lower=1)
    ).astype('float32')

    # Step 2: Create date token (month+day, e.g., "305" for March 5)
    daily_util['date_token'] = (
        daily_util['calendar_date'].dt.month.astype(str) +
        daily_util['calendar_date'].dt.day.astype(str)
    )

    # Step 3: Average by date token
    historic_util = (
        daily_util.groupby('date_token', as_index=False)['utilization'].mean()
    )

    return historic_util


def calculate_forward_utilization(future_data):
    """
    Calculate forward utilization by calendar_date + metro.

    Returns: DataFrame with calendar_date, metro, is_weekend → utilization
    """
    future_data = future_data.copy()  # Safe to copy - it's aggregated data
    future_data['calendar_date'] = pd.to_datetime(future_data['calendar_date'])
    future_data['is_weekend'] = future_data['calendar_date'].dt.dayofweek.isin(PricingConfig.WEEKEND_DAYS).astype('int8')

    forward_util = (
        future_data.groupby(['calendar_date', 'metro', 'is_weekend'], as_index=False)
        .agg(
            booked_listings=('is_booked', 'sum'),
            total_listings=('listing_id', 'count')
        )
    )
    forward_util['utilization'] = (
        forward_util['booked_listings'] / forward_util['total_listings'].clip(lower=1)
    ).astype('float32')

    return forward_util[['calendar_date', 'metro', 'is_weekend', 'utilization']]


def calculate_demand_flags(historic_utilization):
    """
    Calculate statistical thresholds for demand detection.

    Returns: Single-row DataFrame with avg_utilization, std_plus_1, std_plus_2, std_plus_3
    """
    avg_util = historic_utilization['utilization'].mean()
    std_util = historic_utilization['utilization'].std()

    demand_flags = pd.DataFrame([{
        'avg_utilization': avg_util,
        'std_plus_1': avg_util + std_util,
        'std_plus_2': avg_util + (2 * std_util),
        'std_plus_3': avg_util + (3 * std_util)
    }])

    return demand_flags


# ══════════════════════════════════════════════════════════════════════════════
#  BENCHMARK CALCULATOR (OPTIMIZED)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_benchmarks(booked_data):
    """
    Calculate all three benchmark tiers at once.

    OPTIMIZATION: No .apply() - uses manual loop to avoid recursion errors.
    """
    def calc_weighted_benchmarks(groupby_keys):
        groups = booked_data.groupby(groupby_keys)

        results = []
        for name, group in groups:
            # name is tuple of group key values
            row = {key: val for key, val in zip(groupby_keys, name if isinstance(name, tuple) else (name,))}
            row['bookings'] = len(group)
            row['p25'] = weighted_quantile(group['avg_booked_price'].values, group['occ_rate'].values, 0.25)
            row['p50'] = weighted_quantile(group['avg_booked_price'].values, group['occ_rate'].values, 0.50)
            row['p75'] = weighted_quantile(group['avg_booked_price'].values, group['occ_rate'].values, 0.75)
            results.append(row)

        return pd.DataFrame(results)

    tier1 = calc_weighted_benchmarks(["metro", "vehicle_type_group", "month", "is_weekend"])
    tier2 = calc_weighted_benchmarks(["metro", "month", "is_weekend"])
    tier3 = calc_weighted_benchmarks(["vehicle_type_group", "month", "is_weekend"])

    return tier1, tier2, tier3


# ══════════════════════════════════════════════════════════════════════════════
#  MULTIPLIER CALCULATOR (OPTIMIZED)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_demand_multiplier(booked_data):
    """Calculate national demand multiplier - minimal memory footprint."""
    national_avg = (
        booked_data.groupby(["vehicle_type_group", "month", "is_weekend"], as_index=False)
        ["avg_booked_price"].mean()
        .rename(columns={"avg_booked_price": "national_avg_price"})
    )

    national_baseline = (
        booked_data.groupby("vehicle_type_group", as_index=False)
        ["avg_booked_price"].mean()
        .rename(columns={"avg_booked_price": "national_baseline"})
    )

    national_avg = national_avg.merge(national_baseline, on="vehicle_type_group")
    national_avg["demand_multiplier"] = (
        national_avg["national_avg_price"] / national_avg["national_baseline"].clip(lower=1)
    ).round(4).astype('float32')

    return national_avg[["vehicle_type_group", "month", "is_weekend", "demand_multiplier"]]


def calculate_local_multiplier(full_data, booked_data):
    """Calculate local multiplier - optimized for memory."""
    # Metro averages
    metro_avg = (
        booked_data.groupby(["metro", "vehicle_type_group", "month", "is_weekend"], as_index=False)
        ["avg_booked_price"].mean()
        .rename(columns={"avg_booked_price": "metro_avg_price"})
    )

    # National averages
    national_avg = (
        booked_data.groupby(["vehicle_type_group", "month", "is_weekend"], as_index=False)
        ["avg_booked_price"].mean()
        .rename(columns={"avg_booked_price": "national_avg_price"})
    )

    # Supply ratios
    full_data["year_month"] = full_data["calendar_date"].dt.to_period("M")

    listing_counts = (
        full_data[full_data["bookable_days"] > 0]
        .groupby(["metro", "vehicle_type_group", "year_month"], as_index=False)
        ["listing_id"].nunique()
        .rename(columns={"listing_id": "listing_count"})
        .sort_values(["metro", "vehicle_type_group", "year_month"])
    )

    listing_counts["listing_count_baseline"] = (
        listing_counts.groupby(["metro", "vehicle_type_group"])["listing_count"]
        .transform(lambda x: x.shift(1).rolling(PricingConfig.SUPPLY_ROLLING_MONTHS, min_periods=1).mean())
        .fillna(listing_counts["listing_count"])
    )

    listing_counts["supply_ratio"] = (
        listing_counts["listing_count"] / listing_counts["listing_count_baseline"].clip(lower=1)
    ).round(4).astype('float32')

    listing_counts["month"] = listing_counts["year_month"].dt.month
    listing_counts = listing_counts.sort_values("year_month").groupby(["metro", "vehicle_type_group", "month"]).last().reset_index()

    # Cohort occupancy
    cohort_occ = (
        full_data.groupby(["metro", "vehicle_type_group", "month"], as_index=False)
        .agg(total_booked=("is_booked", "sum"), total_bookable=("bookable_days", "sum"))
    )
    cohort_occ["cohort_occupancy"] = (
        cohort_occ["total_booked"] / cohort_occ["total_bookable"].clip(lower=1)
    ).round(4).astype('float32')

    # Merge components
    metro_avg = metro_avg.merge(national_avg, on=["vehicle_type_group", "month", "is_weekend"], how="left")
    metro_avg = metro_avg.merge(
        listing_counts[["metro", "vehicle_type_group", "month", "listing_count", "listing_count_baseline", "supply_ratio"]],
        on=["metro", "vehicle_type_group", "month"], how="left"
    )
    metro_avg = metro_avg.merge(
        cohort_occ[["metro", "vehicle_type_group", "month", "cohort_occupancy"]],
        on=["metro", "vehicle_type_group", "month"], how="left"
    )

    # Fill and calculate
    metro_avg["supply_ratio"] = metro_avg["supply_ratio"].fillna(PricingConfig.DEFAULT_SUPPLY_RATIO).astype('float32')
    metro_avg["cohort_occupancy"] = metro_avg["cohort_occupancy"].fillna(PricingConfig.DEFAULT_COHORT_OCCUPANCY).astype('float32')

    metro_avg["adjusted_supply_pressure"] = (
        metro_avg["supply_ratio"] * (1 - metro_avg["cohort_occupancy"])
    ).round(4).astype('float32')

    metro_avg["local_multiplier"] = (
        (metro_avg["metro_avg_price"] / metro_avg["national_avg_price"].clip(lower=1))
        * (1 / metro_avg["supply_ratio"].clip(lower=0.1) ** PricingConfig.SUPPLY_SENSITIVITY_ALPHA)
    ).round(4).astype('float32')

    return metro_avg


def calculate_cohort_averages(listing_data):
    """
    Calculate cohort average prices for flooring logic.

    Returns: DataFrame with vehicle_type_group, month → cohort_avg

    NOTE: Returns month as integer (1-12) to match recs DataFrame
    """
    booked_only = listing_data[listing_data['booked_days'] == 1]

    # recs has 'month' as integer (1-12), so group by that
    if 'month' not in booked_only.columns:
        booked_only['month'] = booked_only['calendar_date'].dt.month

    cohort_avg = (
        booked_only.groupby(['month', 'vehicle_type_group'], as_index=False)
        ['avg_base_price'].mean()
        .rename(columns={'avg_base_price': 'cohort_avg'})
    )

    return cohort_avg


# ══════════════════════════════════════════════════════════════════════════════
#  LISTING PROFILER (OPTIMIZED)
# ══════════════════════════════════════════════════════════════════════════════

def build_listing_profiles(listing_data):
    """Build listing profiles - single pass, minimal copies."""
    listings = (
        listing_data.groupby(["listing_id", "metro", "vehicle_type_group"], as_index=False)
        .agg(
            current_price=("avg_day_price", "median"),
            vehicle_year=("vehicle_year", "max"),
            review_count=("review_count", "max"),
            image_count=("image_count", "max"),
            sleeps=("sleeps", "max"),
            instant_book=("instant_book", "max"),
            booked_days=("is_booked", "sum"),
            bookable_days=("bookable_days", "sum"),
        )
    )

    listings["occupancy"] = (listings["booked_days"] / listings["bookable_days"].clip(lower=1)).round(3).astype('float32')

    listings["quality_score"] = (
        (listings["review_count"] >= PricingConfig.MIN_REVIEWS_FOR_QUALITY).astype('int8') +
        (listings["image_count"] >= PricingConfig.MIN_IMAGES_FOR_QUALITY).astype('int8') +
        (listings["sleeps"] >= PricingConfig.MIN_SLEEPS_FOR_QUALITY).astype('int8') +
        (listings["vehicle_year"] >= PricingConfig.MIN_VEHICLE_YEAR_FOR_QUALITY).astype('int8')
    )

    listings["quality"] = np.select(
        [listings["quality_score"] >= PricingConfig.PREMIUM_QUALITY_SCORE,
         listings["quality_score"] <= PricingConfig.BASIC_QUALITY_SCORE],
        ["premium", "basic"],
        default="standard"
    )

    return listings


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE BOOST FUNCTIONS (NEW - MODULAR)
# ══════════════════════════════════════════════════════════════════════════════

def apply_time_to_search_boost(recs, run_date=None):
    """
    Apply time-to-search proximity boost.

    Boosts:
    - ≤7 days out: +2.5%
    - ≤30 days out: +1.0%
    - ≤90 days out: +0.5%
    """
    if not PricingConfig.ENABLE_TIME_TO_SEARCH_BOOST:
        recs['time_to_search_boost'] = 0
        return recs

    if run_date is None:
        run_date = date.today()
    elif isinstance(run_date, str):
        run_date = pd.to_datetime(run_date).date()

    recs['days_out'] = (pd.to_datetime(recs['calendar_date']).dt.date - run_date).dt.days

    recs['time_to_search_boost'] = np.select(
        [
            recs['days_out'] <= 7,
            recs['days_out'] <= 30,
            recs['days_out'] <= 90
        ],
        [
            PricingConfig.TIME_TO_SEARCH_7_DAYS * recs['adjusted_price'],
            PricingConfig.TIME_TO_SEARCH_30_DAYS * recs['adjusted_price'],
            PricingConfig.TIME_TO_SEARCH_90_DAYS * recs['adjusted_price']
        ],
        default=0
    ).round(0).astype('int32')

    return recs


def apply_historic_demand_boost(recs, demand_flags):
    """
    Apply historic demand boost.

    Boosts:
    - historic_util ≤ avg: 0%
    - avg < historic_util < avg+std: +2.5%
    - historic_util ≥ avg+std: +5.0%
    """
    if not PricingConfig.ENABLE_HISTORIC_DEMAND_BOOST:
        recs['historic_demand_boost'] = 0
        return recs

    avg_util = demand_flags['avg_utilization'].iloc[0]
    std_plus_1 = demand_flags['std_plus_1'].iloc[0]

    recs['historic_demand_boost'] = np.select(
        [
            recs['historic_utilization'] <= avg_util,
            (recs['historic_utilization'] > avg_util) & (recs['historic_utilization'] < std_plus_1)
        ],
        [
            0,
            PricingConfig.HISTORIC_DEMAND_MID * recs['adjusted_price']
        ],
        default=PricingConfig.HISTORIC_DEMAND_HIGH * recs['adjusted_price']
    ).round(0).astype('int32')

    return recs


def apply_forward_demand_boost(recs, demand_flags):
    """
    Apply forward demand boost.

    Boosts:
    - forward_util ≤ avg: 0%
    - avg < forward_util < avg+std: +5.0%
    - forward_util ≥ avg+std: +10.0%
    """
    if not PricingConfig.ENABLE_FORWARD_DEMAND_BOOST:
        recs['forward_demand_boost'] = 0
        return recs

    avg_util = demand_flags['avg_utilization'].iloc[0]
    std_plus_1 = demand_flags['std_plus_1'].iloc[0]

    recs['forward_demand_boost'] = np.select(
        [
            recs['forward_utilization'] <= avg_util,
            (recs['forward_utilization'] > avg_util) & (recs['forward_utilization'] < std_plus_1)
        ],
        [
            0,
            PricingConfig.FORWARD_DEMAND_MID * recs['adjusted_price']
        ],
        default=PricingConfig.FORWARD_DEMAND_HIGH * recs['adjusted_price']
    ).round(0).astype('int32')

    return recs


def apply_specific_listing_boost(recs):
    """
    Apply hardcoded boost for specific listing (400742).

    Boost: historic_utilization * 100
    """
    if not PricingConfig.ENABLE_SPECIFIC_LISTING_BOOST:
        recs['specific_listing_boost'] = 0
        return recs

    recs['specific_listing_boost'] = np.where(
        recs['listing_id'] == PricingConfig.SPECIAL_LISTING_ID,
        recs['historic_utilization'] * 100,
        0
    ).round(0).astype('int32')

    return recs


def apply_cohort_floor(recs, cohort_averages):
    """
    Apply cohort average flooring logic.

    Logic:
    - If no recommendation OR price < cohort_avg: use cohort_avg
    - Else if price ≥ recommended_price: use current price
    - Else: use recommended_price

    Then multiply by 100 (client format requirement)

    NOTE: recs is at listing+month+is_weekend level (NOT daily)
    """
    if not PricingConfig.ENABLE_COHORT_FLOOR:
        recs['adjusted_price'] = (recs['recommended_price'] * 100).round(0).astype('int32')
        return recs

    # Merge cohort averages - join on month + vehicle_type_group
    # (recs has 'month' as integer 1-12, cohort_averages now matches that)
    recs = recs.merge(
        cohort_averages,
        on=['month', 'vehicle_type_group'],
        how='left'
    )

    # Apply flooring logic
    conditions = [
        recs['recommended_price'].isna() & (recs['current_price'] < recs['cohort_avg']),
        recs['recommended_price'].isna() & (recs['current_price'] >= recs['cohort_avg']),
        recs['current_price'] >= recs['recommended_price']
    ]

    choices = [
        recs['cohort_avg'],
        recs['current_price'],
        recs['current_price']
    ]

    recs['adjusted_price'] = np.select(
        conditions,
        choices,
        default=recs['recommended_price']
    )

    # Multiply by 100 (client format)
    recs['adjusted_price'] = (recs['adjusted_price'] * 100).round(0).astype('int32')

    return recs


def apply_floor_check(recs):
    """
    Apply 7% floor check.

    If price increase < 7% over historic price, force to exactly +7%.
    """
    recs['historic_price'] = recs['current_price']  # Assuming current_price is historic

    # Calculate move percentage
    recs['move_pct'] = (
        (recs['adjusted_price'] / 100 - recs['historic_price']) /
        recs['historic_price'].clip(lower=1)
    )

    # Apply floor check
    forced_price = (recs['historic_price'] * (1 + PricingConfig.FLOOR_CHECK_PERCENT) * 100).round(0).astype('int32')

    recs['adjusted_price'] = np.where(
        recs['move_pct'] < PricingConfig.FLOOR_CHECK_PERCENT,
        forced_price,
        recs['adjusted_price']
    ).astype('int32')

    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RECOMMEND FUNCTION (ENHANCED)
# ══════════════════════════════════════════════════════════════════════════════

def recommend(listing_data):
    """
    Generate aggregated pricing recommendations at listing+month+is_weekend level.

    Parameters:
    -----------
    listing_data : DataFrame
        Combined historic + future data with daily rows

    Returns:
    --------
    DataFrame with aggregated recommendations (~300K rows)
    Columns include: listing_id, metro, month, is_weekend, recommended_price, adjusted_price, etc.
    adjusted_price is in ×100 format (e.g., 12500 = $125.00)

    NOTE: Boosts are NOT applied here - they happen in Cell 3 after daily expansion
    """
    print("  Cleaning data...")
    df = clean_listing_data(listing_data)
    print(f"  {len(df):,} rows | {df['listing_id'].nunique():,} listings | "
          f"{df['metro'].nunique():,} metros | {df['is_booked'].mean():.1%} booking rate")

    # Prepare booked data
    print("  Preparing booked data...")
    booked = df[(df["is_booked"] == 1) & (df["avg_booked_price"].notna())]  # No .copy()!
    occupancy_weights = calculate_occupancy_weights(df)
    booked = booked.merge(occupancy_weights, on="listing_id", how="left")
    del occupancy_weights
    gc.collect()

    # Calculate benchmarks
    print("  Calculating benchmarks...")
    tier1, tier2, tier3 = calculate_benchmarks(booked)
    print(f"  {tier1.groupby(['metro','vehicle_type_group']).ngroups:,} exact cohorts | "
          f"{tier2['metro'].nunique():,} metro-only cohorts | "
          f"{tier3['vehicle_type_group'].nunique():,} national vehicle cohorts")

    # Calculate multipliers
    print("  Calculating multipliers...")
    demand_multiplier = calculate_demand_multiplier(booked)
    local_multiplier = calculate_local_multiplier(df, booked)

    # Calculate cohort averages for flooring
    print("  Calculating cohort averages...")
    cohort_averages = calculate_cohort_averages(df)

    del booked
    gc.collect()

    # Build listing profiles
    print("  Building listing profiles...")
    listings = build_listing_profiles(df)

    # Match listings to benchmarks (SAME AS OPTIMIZED - listing + month + weekend level)
    print("  Matching listings to benchmarks...")
    recs_t1 = listings.merge(tier1, on=["metro", "vehicle_type_group"], how="inner")
    recs_t1["benchmark_tier"] = "metro_x_vehicle"
    matched_t1 = recs_t1["listing_id"].unique()

    unmatched = listings[~listings["listing_id"].isin(matched_t1)]
    recs_t2 = unmatched.merge(tier2, on=["metro"], how="inner")
    recs_t2["benchmark_tier"] = "metro_only"
    matched_t2 = recs_t2["listing_id"].unique()

    unmatched = unmatched[~unmatched["listing_id"].isin(matched_t2)]
    recs_t3 = unmatched.merge(tier3, on=["vehicle_type_group"], how="inner")
    recs_t3["benchmark_tier"] = "national_vehicle"

    recs = pd.concat([recs_t1, recs_t2, recs_t3], ignore_index=True)
    del recs_t1, recs_t2, recs_t3, tier1, tier2, tier3, unmatched, listings
    gc.collect()

    print(f"  Benchmark coverage: {len(matched_t1):,} exact | "
          f"{len(matched_t2):,} metro fallback")

    # Calculate prices
    print("  Calculating recommended prices...")
    recs["benchmark"] = np.select(
        [recs["quality"] == "premium", recs["quality"] == "basic"],
        [recs["p75"], recs["p25"]],
        default=recs["p50"]
    ).round(0)

    recs["gap_dollars"] = (recs["benchmark"] - recs["current_price"]).round(0)
    recs["gap_pct"] = (recs["gap_dollars"] / recs["current_price"].clip(lower=1) * 100).round(1)
    recs["is_underpriced"] = (recs["gap_pct"] >= PricingConfig.UNDERPRICED_GAP_PERCENT).astype('int8')

    # Attach multipliers
    recs = recs.merge(demand_multiplier, on=["vehicle_type_group", "month", "is_weekend"], how="left")
    recs["demand_multiplier"] = recs["demand_multiplier"].fillna(PricingConfig.DEFAULT_DEMAND_MULTIPLIER)
    del demand_multiplier

    recs = recs.merge(
        local_multiplier[["metro", "vehicle_type_group", "month", "is_weekend", "local_multiplier",
                         "listing_count", "listing_count_baseline", "supply_ratio",
                         "cohort_occupancy", "adjusted_supply_pressure"]],
        on=["metro", "vehicle_type_group", "month", "is_weekend"], how="left"
    )
    del local_multiplier
    gc.collect()

    recs["local_multiplier"] = recs["local_multiplier"].fillna(PricingConfig.DEFAULT_LOCAL_MULTIPLIER)
    recs["supply_ratio"] = recs["supply_ratio"].fillna(PricingConfig.DEFAULT_SUPPLY_RATIO)
    recs["cohort_occupancy"] = recs["cohort_occupancy"].fillna(PricingConfig.DEFAULT_COHORT_OCCUPANCY)

    # Calculate listing multiplier and recommended price
    recs["listing_multiplier"] = np.select(
        [recs["quality"] == "premium", recs["quality"] == "basic"],
        [(recs["p75"] / recs["p50"].clip(lower=1)).round(4),
         (recs["p25"] / recs["p50"].clip(lower=1)).round(4)],
        default=PricingConfig.DEFAULT_LISTING_MULTIPLIER
    )

    recs["base_price"] = recs["current_price"]
    recs["recommended_price"] = (
        recs["base_price"] * recs["demand_multiplier"] *
        recs["local_multiplier"] * recs["listing_multiplier"]
    ).round(2)

    # Convert to ×100 format (client requirement)
    # NOTE: Cohort floor, boosts, and floor check are skipped here
    # They will be applied in Cell 3 after expansion to daily level
    print("  Converting to ×100 format...")
    recs['adjusted_price'] = (recs['recommended_price'] * 100).round(0).astype('int32')

    del cohort_averages
    gc.collect()

    # Optimize final DataFrame
    recs = optimize_dtypes(recs)
    recs = recs.sort_values(["metro", "listing_id", "month", "is_weekend"]).reset_index(drop=True)

    n_under = recs[recs["is_underpriced"] == 1]["listing_id"].nunique()
    print(f"  {n_under:,} underpriced listings identified")

    return recs


# Simpler score and translate_to_insights functions (optimized similarly)
def score(listing_data, recommendations):
    """Simplified scoring - main memory savings are in recommend()."""
    print("  Scoring not yet optimized in this version - use recommend() results")
    return pd.DataFrame()


def translate_to_insights(recommendations):
    """Add insights - minimal memory overhead."""
    recs = recommendations  # No copy!

    recs["local_demand_text"] = np.select(
        [(recs["local_multiplier"] * 100) > 80, (recs["local_multiplier"] * 100) < 40],
        ["High", "Low"],
        default="Average"
    )

    recs["seasonal_trend_text"] = np.select(
        [recs["cohort_occupancy"] > 0.75, recs["cohort_occupancy"] < 0.40],
        ["Peak season", "Off-season"],
        default="Stable"
    )

    recs["day_type_text"] = np.where(recs["is_weekend"] == 1, "Weekend", "Weekday")
    recs["vehicle_quality_text"] = recs["quality"].str.capitalize()

    return recs


print("✓ ENHANCED Pricing Algorithm loaded successfully")
print("  New features: Utilization calculations, modular boosts, cohort floor, 7% floor check")
print("  Configure boosts in PricingConfig class")
