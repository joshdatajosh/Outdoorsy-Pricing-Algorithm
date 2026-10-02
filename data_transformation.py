"""
HEX CELL 3: RUN ANALYSIS (ENHANCED - DAILY EXPANSION)

ARCHITECTURE (matches SQL pattern):
1. Generate aggregated recommendations (listing+month+is_weekend) - memory efficient
2. Join to daily listing_data to expand - creates calendar_date column
3. Calculate utilization (historic and forward)
4. Apply boosts (time-to-search, historic demand, forward demand)
5. Apply floor checks

This matches the pattern in reworked_one.sql and reworked_two.sql

MEMORY OPTIMIZATION:
- Aggregated recs: ~300K rows
- Daily expansion: ~1.85M rows (only after recommend())
- Aggressive cleanup between steps

Copy this entire cell into your third Hex Python cell.
IMPORTANT: Run Cell 1 (Data Loader) and Cell 2 (ENHANCED Algorithm) first!
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import gc
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.postgresql import psycopg2 as pg_psycopg2


# ══════════════════════════════════════════════════════════════════════════════
#  REDSHIFT COMPATIBILITY PATCH
# ══════════════════════════════════════════════════════════════════════════════

# Only patch once - check if already patched to prevent recursion when re-running cell
if not hasattr(pg_psycopg2.PGDialect_psycopg2.initialize, '_redshift_patched'):
    _original_initialize = pg_psycopg2.PGDialect_psycopg2.initialize

    def patched_initialize(self, connection):
        """Override initialize to handle Redshift's missing standard_conforming_strings."""
        self.default_isolation_level = 'AUTOCOMMIT'
        try:
            _original_initialize(self, connection)
        except Exception as e:
            if 'standard_conforming_strings' in str(e):
                # Redshift doesn't support this query, set defaults manually
                self.supports_sane_rowcount = True
                self.supports_native_decimal = True
                self._backslash_escapes = False
                self.supports_sane_multi_rowcount = False
                self.implicit_returning = False
            else:
                raise

    # Mark as patched
    patched_initialize._redshift_patched = True
    pg_psycopg2.PGDialect_psycopg2.initialize = patched_initialize


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

# Redshift connection details
REDSHIFT_HOST = "segmentevents.cbbinbqccto6.us-west-2.redshift.amazonaws.com"
REDSHIFT_PORT = "5439"
REDSHIFT_DATABASE = "segmentevents"
REDSHIFT_USERNAME = "rylan_mclaughlin_preql"
REDSHIFT_PASSWORD = "Pr3qL#9mR4$xK8nB"

DATABASE_URL = (
    f"postgresql+psycopg2://{REDSHIFT_USERNAME}:{REDSHIFT_PASSWORD}"
    f"@{REDSHIFT_HOST}:{REDSHIFT_PORT}/{REDSHIFT_DATABASE}"
)

# Create engine (patch already applied above)
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    execution_options={"isolation_level": "AUTOCOMMIT"}
)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIZATION & BOOST FUNCTIONS (FROM CELL 2)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_historic_utilization(historic_data):
    """Calculate historic utilization by date token (month+day)."""
    historic_data = historic_data.copy()
    historic_data['calendar_date'] = pd.to_datetime(historic_data['calendar_date'])

    # Derive is_booked from booked_days (is_booked = 1 if booked_days > 0)
    historic_data['is_booked'] = (historic_data['booked_days'] > 0).astype('int8')

    # Daily utilization
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

    # Create date token (month+day)
    daily_util['date_token'] = (
        daily_util['calendar_date'].dt.month.astype(str) +
        daily_util['calendar_date'].dt.day.astype(str)
    )

    # Average by date token
    historic_util = (
        daily_util.groupby('date_token', as_index=False)['utilization'].mean()
    )

    return historic_util


def calculate_forward_utilization(future_data):
    """Calculate forward utilization by calendar_date + metro."""
    future_data = future_data.copy()
    future_data['calendar_date'] = pd.to_datetime(future_data['calendar_date'])
    future_data['is_weekend'] = future_data['calendar_date'].dt.dayofweek.isin([0, 4, 5]).astype('int8')

    # Derive is_booked from booked_days (is_booked = 1 if booked_days > 0)
    future_data['is_booked'] = (future_data['booked_days'] > 0).astype('int8')

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
    """Calculate statistical thresholds for demand detection."""
    avg_util = historic_utilization['utilization'].mean()
    std_util = historic_utilization['utilization'].std()

    demand_flags = pd.DataFrame([{
        'avg_utilization': avg_util,
        'std_plus_1': avg_util + std_util,
        'std_plus_2': avg_util + (2 * std_util),
        'std_plus_3': avg_util + (3 * std_util)
    }])

    return demand_flags


def apply_daily_boosts(daily_recs, historic_util, forward_util, demand_flags, run_date):
    """
    Apply all boosts to daily-expanded recommendations.

    Boosts:
    - Time-to-search proximity
    - Historic demand
    - Forward demand
    """
    avg_util = demand_flags['avg_utilization'].iloc[0]
    std_plus_1 = demand_flags['std_plus_1'].iloc[0]

    # Join historic utilization (by date token)
    daily_recs['date_token'] = (
        pd.to_datetime(daily_recs['calendar_date']).dt.month.astype(str) +
        pd.to_datetime(daily_recs['calendar_date']).dt.day.astype(str)
    )
    daily_recs = daily_recs.merge(
        historic_util.rename(columns={'utilization': 'historic_utilization'}),
        on='date_token', how='left'
    )
    daily_recs['historic_utilization'] = daily_recs['historic_utilization'].fillna(0).astype('float32')

    # Join forward utilization (by calendar_date + metro)
    daily_recs = daily_recs.merge(
        forward_util.rename(columns={'utilization': 'forward_utilization'}),
        on=['calendar_date', 'metro', 'is_weekend'], how='left'
    )
    daily_recs['forward_utilization'] = daily_recs['forward_utilization'].fillna(0).astype('float32')

    # Time-to-search boost
    daily_recs['days_out'] = (pd.to_datetime(daily_recs['calendar_date']) - pd.Timestamp(run_date)).dt.days
    daily_recs['time_to_search_boost'] = np.select(
        [
            daily_recs['days_out'] <= 7,
            daily_recs['days_out'] <= 30,
            daily_recs['days_out'] <= 90
        ],
        [
            0.025 * daily_recs['adjusted_price'],
            0.010 * daily_recs['adjusted_price'],
            0.005 * daily_recs['adjusted_price']
        ],
        default=0
    ).round(0).astype('int32')

    # Historic demand boost
    daily_recs['historic_demand_boost'] = np.select(
        [
            daily_recs['historic_utilization'] <= avg_util,
            (daily_recs['historic_utilization'] > avg_util) & (daily_recs['historic_utilization'] < std_plus_1)
        ],
        [
            0,
            0.025 * daily_recs['adjusted_price']
        ],
        default=0.050 * daily_recs['adjusted_price']
    ).round(0).astype('int32')

    # Forward demand boost
    daily_recs['forward_demand_boost'] = np.select(
        [
            daily_recs['forward_utilization'] <= avg_util,
            (daily_recs['forward_utilization'] > avg_util) & (daily_recs['forward_utilization'] < std_plus_1)
        ],
        [
            0,
            0.050 * daily_recs['adjusted_price']
        ],
        default=0.100 * daily_recs['adjusted_price']
    ).round(0).astype('int32')

    # Sum all boosts
    daily_recs['total_boost'] = (
        daily_recs['time_to_search_boost'] +
        daily_recs['historic_demand_boost'] +
        daily_recs['forward_demand_boost']
    ).astype('int32')

    # Apply boosts to adjusted price
    # Handle NaN values (from join mismatches) before converting to int32
    daily_recs['adjusted_price'] = daily_recs['adjusted_price'].fillna(0)
    daily_recs['adjusted_price'] = (daily_recs['adjusted_price'] + daily_recs['total_boost']).round(0).astype('int32')

    return daily_recs


def apply_floor_check_daily(daily_recs):
    """Apply 7% floor check to daily recommendations."""
    # Handle NaN values in base_price
    daily_recs['historic_price'] = daily_recs['base_price'].fillna(0)

    # Calculate move percentage
    daily_recs['move_pct'] = (
        (daily_recs['adjusted_price'] / 100 - daily_recs['historic_price']) /
        daily_recs['historic_price'].clip(lower=1)
    )

    # Apply floor check - handle NaN before converting to int32
    forced_price = (daily_recs['historic_price'] * 1.07 * 100).fillna(0).round(0).astype('int32')

    daily_recs['adjusted_price'] = np.where(
        daily_recs['move_pct'] < 0.07,
        forced_price,
        daily_recs['adjusted_price']
    ).astype('int32')

    return daily_recs


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# MEMORY TIP: Reduce these numbers if you still hit memory limits
HISTORICAL_DAYS = 90   # Last 3 months
FUTURE_DAYS = 60       # Next 2 months

print("="*80)
print("OUTDOORSY PRICING ANALYSIS (ENHANCED - DAILY EXPANSION)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Historical period: Last {HISTORICAL_DAYS} days")
print(f"  Future period: Next {FUTURE_DAYS} days")
print(f"  Expansion: Aggregated recs → Daily via join")


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("[1/5] Loading data from Redshift...")
print("="*80)

loader = OutdoorsyDataLoader(engine)

# Historical data
start_date = (datetime.now() - timedelta(days=HISTORICAL_DAYS)).strftime('%Y-%m-%d')
print(f"  Loading historical data from {start_date}...")
historic_data = loader.load_historical_data(start_date=start_date)
print(f"  ✓ Historical: {len(historic_data):,} rows, {historic_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Future data
print(f"  Loading future data (next {FUTURE_DAYS} days)...")
future_data = loader.load_future_data(days_ahead=FUTURE_DAYS)
print(f"  ✓ Future: {len(future_data):,} rows, {future_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Combine for recommend function
combined_data = pd.concat([historic_data, future_data], ignore_index=True)
print(f"  ✓ Combined: {len(combined_data):,} rows")


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE AGGREGATED RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("[2/5] Generating aggregated recommendations...")
print("="*80)

recs_aggregated = recommend(combined_data)
print(f"  ✓ Generated {len(recs_aggregated):,} aggregated rows (listing×month×is_weekend)")
print(f"  ✓ Memory usage: {recs_aggregated.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Free combined data
del combined_data
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
#  EXPAND TO DAILY (JOIN PATTERN FROM SQL)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("[3/5] Expanding to daily recommendations...")
print("="*80)

# Prepare future_data for join (matches SQL pattern)
future_data['calendar_date'] = pd.to_datetime(future_data['calendar_date'])
future_data['month'] = future_data['calendar_date'].dt.month.astype('int8')
future_data['is_weekend'] = future_data['calendar_date'].dt.dayofweek.isin([0, 4, 5]).astype('int8')

# Join aggregated recs to daily data (matches reworked_one.sql pattern)
daily_recs = future_data.merge(
    recs_aggregated,
    on=['listing_id', 'month', 'is_weekend'],
    how='left',
    suffixes=('', '_rec')
)

print(f"  ✓ Expanded to {len(daily_recs):,} daily rows")
print(f"  ✓ Memory usage: {daily_recs.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Free aggregated recs
del recs_aggregated
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
#  CALCULATE UTILIZATION & APPLY BOOSTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("[4/5] Calculating utilization and applying boosts...")
print("="*80)

# Calculate utilization
print("  Calculating historic utilization...")
historic_util = calculate_historic_utilization(historic_data)
print(f"  ✓ Historic utilization: {len(historic_util):,} date tokens")

print("  Calculating forward utilization...")
forward_util = calculate_forward_utilization(future_data)
print(f"  ✓ Forward utilization: {len(forward_util):,} future dates")

# Calculate demand flags
demand_flags = calculate_demand_flags(historic_util)
print(f"  ✓ Demand thresholds: avg={demand_flags['avg_utilization'].iloc[0]:.3f}, "
      f"std+1={demand_flags['std_plus_1'].iloc[0]:.3f}")

# Free historic data
del historic_data, future_data
gc.collect()

# Apply boosts
print("  Applying boosts...")
daily_recs = apply_daily_boosts(daily_recs, historic_util, forward_util, demand_flags, datetime.now().date())
print(f"  ✓ Average boost: ${daily_recs['total_boost'].mean():.2f}")

# Apply floor check
print("  Applying 7% floor check...")
daily_recs = apply_floor_check_daily(daily_recs)

# Cleanup
del historic_util, forward_util, demand_flags
gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE TRANSFORMATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_explanation_vectorized(df):
    """Generate explanations for price changes."""
    dates = pd.to_datetime(df['calendar_date'])
    day_of_week = dates.dt.dayofweek
    month = dates.dt.month

    is_weekend = day_of_week >= 4
    is_seasonal = month.isin([12, 4, 6, 7, 8])
    is_premium = df['quality'].str.lower() == 'premium'

    # Convert adjusted_price from ×100 format back to regular for comparison
    adjusted_price_regular = df['adjusted_price'] / 100
    historic_price = df['historic_price']

    price_increased = adjusted_price_regular > historic_price
    price_unchanged = adjusted_price_regular == historic_price
    beats_weekday_avg = adjusted_price_regular > df['weekday_avg_price']

    conditions = [
        price_unchanged,
        price_increased & is_weekend & beats_weekday_avg,
        price_increased & is_weekend & ~beats_weekday_avg,
        price_increased & ~is_weekend & is_seasonal,
        price_increased & ~is_weekend & ~is_seasonal & is_premium,
        price_increased & ~is_weekend & ~is_seasonal & ~is_premium,
    ]

    choices = [
        'No Change',
        'weekend_premium',
        'seasonal_demand',
        'seasonal_demand',
        'premium_quality',
        'underpriced',
    ]

    return np.select(conditions, choices, default='No Change')


def generate_insights_for_db(df):
    """Generate insights dictionary for each row."""
    insights = {}

    if 'local_multiplier' in df.columns:
        local_demand_index = (df['local_multiplier'] * 100).round(0)
        insights['local_demand'] = np.select(
            [local_demand_index > 80, local_demand_index < 40],
            ["high", "low"],
            default="average"
        )
    else:
        insights['local_demand'] = "average"

    if 'cohort_occupancy' in df.columns:
        insights['season_trends'] = np.select(
            [df['cohort_occupancy'] > 0.75, df['cohort_occupancy'] < 0.40],
            ["peak_season", "off_season"],
            default="stable"
        )
    else:
        insights['season_trends'] = "stable"

    if 'is_weekend' in df.columns:
        insights['dow'] = np.where(df['is_weekend'] == 1, "weekend", "weekday")
    else:
        dates = pd.to_datetime(df['calendar_date'])
        insights['dow'] = np.where(dates.dt.dayofweek >= 4, "weekend", "weekday")

    if 'quality' in df.columns:
        insights['vehicle_traits'] = df['quality'].str.lower()
    else:
        insights['vehicle_traits'] = "standard"

    if 'demand_multiplier' in df.columns:
        insights['national_demand'] = np.select(
            [df['demand_multiplier'] > 1.1, df['demand_multiplier'] < 0.9],
            ["trending_up", "trending_down"],
            default="stable"
        )
    else:
        insights['national_demand'] = "stable"

    return pd.DataFrame(insights)


def transform_for_database(daily_recs):
    """
    Transform daily recommendations into database write format.

    Output format:
    - rental_id: listing_id
    - start_date: current date
    - prices: PostgreSQL array '{100,105,110}'
    - explanations: PostgreSQL array '{"weekend_premium","No Change"}'
    - insights: JSON array '[{...}, {...}]'
    """
    daily_recs = daily_recs.copy()

    # Ensure calendar_date is datetime
    daily_recs['calendar_date'] = pd.to_datetime(daily_recs['calendar_date'])

    # Sort by listing and date
    daily_recs = daily_recs.sort_values(['listing_id', 'calendar_date'])

    # Calculate weekday average price per listing
    weekday_mask = daily_recs['calendar_date'].dt.dayofweek < 4
    weekday_avgs = (
        daily_recs[weekday_mask]
        .groupby('listing_id')['adjusted_price']
        .mean()
        .rename('weekday_avg_price')
    )
    # Convert from ×100 format to regular
    weekday_avgs = weekday_avgs / 100

    daily_recs = daily_recs.merge(weekday_avgs, on='listing_id', how='left')
    daily_recs['weekday_avg_price'] = daily_recs['weekday_avg_price'].fillna(
        daily_recs['adjusted_price'] / 100
    )

    # Generate explanations
    daily_recs['explanation'] = get_explanation_vectorized(daily_recs)

    # Generate insights
    insights_df = generate_insights_for_db(daily_recs)
    for col in insights_df.columns:
        daily_recs[col] = insights_df[col]

    # Create insights object
    daily_recs['insights_obj'] = daily_recs.apply(
        lambda row: {
            "dow": row['dow'],
            "local_demand": row['local_demand'],
            "season_trends": row['season_trends'],
            "vehicle_traits": row['vehicle_traits'],
            "national_demand": row['national_demand']
        },
        axis=1
    )

    # Round adjusted price to integer (it's in ×100 format already)
    daily_recs['adjusted_price'] = daily_recs['adjusted_price'].round().astype(int)

    # Aggregate by listing
    result = daily_recs.groupby('listing_id').agg(
        prices=('adjusted_price', lambda x: '{' + ','.join(map(str, x)) + '}'),
        explanations=('explanation', lambda x: '{"' + '","'.join(x) + '"}'),
        insights=('insights_obj', lambda x: json.dumps(list(x)))
    ).reset_index()

    result = result.rename(columns={'listing_id': 'rental_id'})
    result['start_date'] = datetime.now().strftime('%Y-%m-%d')

    return result[['rental_id', 'start_date', 'prices', 'explanations', 'insights']]


# ══════════════════════════════════════════════════════════════════════════════
#  ADD INSIGHTS & TRANSFORM FOR DATABASE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("[5/5] Transforming for database...")
print("="*80)

print("  Creating database-ready format...")
write_ready_df = transform_for_database(daily_recs)
print(f"  ✓ Database format: {len(write_ready_df):,} listings")


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

# Summary Statistics
print("\nSUMMARY STATISTICS:")
print("-"*80)
total_listings = daily_recs['listing_id'].nunique()
underpriced_listings = daily_recs[daily_recs['is_underpriced'] == 1]['listing_id'].nunique()
underpriced_pct = (underpriced_listings / total_listings * 100) if total_listings > 0 else 0

print(f"Total listings: {total_listings:,}")
print(f"Total daily rows: {len(daily_recs):,}")
print(f"Underpriced listings: {underpriced_listings:,} ({underpriced_pct:.1f}%)")
print(f"Average base price: ${daily_recs['base_price'].mean():.2f}")
print(f"Average final price: ${(daily_recs['adjusted_price'] / 100).mean():.2f}")
print(f"Average total boost: ${daily_recs['total_boost'].mean():.2f}")
print(f"  Time-to-search: ${daily_recs['time_to_search_boost'].mean():.2f}")
print(f"  Historic demand: ${daily_recs['historic_demand_boost'].mean():.2f}")
print(f"  Forward demand: ${daily_recs['forward_demand_boost'].mean():.2f}")
print("-"*80)

# Top Metros
print("\nTOP METROS BY UNDERPRICED LISTINGS:")
print("-"*80)
underpriced_by_metro = (
    daily_recs[daily_recs['is_underpriced'] == 1]
    .groupby('metro')['listing_id']
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)
for metro, count in underpriced_by_metro.items():
    print(f"  {metro}: {count:,} listings")
print("-"*80)

# Sample
print(f"\nSAMPLE DAILY RECOMMENDATIONS (first 20 rows):")
print("-"*80)
sample = daily_recs.head(20)
if len(sample) > 0:
    display_cols = [
        "listing_id", "calendar_date", "metro", "vehicle_type_group",
        "base_price", "adjusted_price", "total_boost"
    ]
    display_df = sample[display_cols].copy()
    display_df['adjusted_price'] = (display_df['adjusted_price'] / 100).round(2)
    display_df['total_boost'] = (display_df['total_boost'] / 100).round(2)
    print(display_df.to_string(index=False))
print("-"*80)

# Database Write-Ready Format
print("\nDATABASE WRITE-READY FORMAT:")
print("-"*80)
print(f"Total listings in write_ready_df: {len(write_ready_df):,}")
print("\nSample (first 3 listings):")
sample_db = write_ready_df.head(3)
for idx, row in sample_db.iterrows():
    print(f"\nListing {row['rental_id']}:")
    print(f"  start_date: {row['start_date']}")
    prices_preview = row['prices'][:50] + '...' if len(row['prices']) > 50 else row['prices']
    print(f"  prices: {prices_preview}")
    exp_preview = row['explanations'][:50] + '...' if len(row['explanations']) > 50 else row['explanations']
    print(f"  explanations: {exp_preview}")
print("-"*80)

print("\n" + "="*80)
print("EXPLORE THE DATA:")
print("="*80)
print("Two DataFrames are available:")
print("  1. 'daily_recs' - Daily recommendations with boosts (~1.85M rows)")
print("  2. 'write_ready_df' - Database-ready format (one row per listing)")
print("\nExamples:")
print("  # View specific listing's daily recommendations")
print("  daily_recs[daily_recs['listing_id'] == '12345']")
print("")
print("  # View specific date range")
print("  daily_recs[(daily_recs['calendar_date'] >= '2024-12-01') & (daily_recs['calendar_date'] <= '2024-12-31')]")
print("")
print("  # Check boost breakdown")
print("  daily_recs[['time_to_search_boost', 'historic_demand_boost', 'forward_demand_boost', 'total_boost']].describe()")
print("")
print("  # View database-ready data")
print("  write_ready_df.head()")
print("")
print("  # Insert into database (example)")
print("  # write_ready_df.to_sql('pricing_recommendations', engine, if_exists='append', index=False)")
print("="*80)

# Final memory report
print(f"\nDaily DataFrame: {len(daily_recs):,} rows, {daily_recs.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"Write-ready DataFrame: {len(write_ready_df):,} rows, {write_ready_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print("Analysis complete! ✓")

# Store final outputs
recs = daily_recs  # For compatibility with other code that expects 'recs'
