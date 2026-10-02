"""
HEX CELL 1: DATA LOADER

Copy this entire cell into your first Hex Python cell.
This defines all the data loading functions.
"""

import pandas as pd
from datetime import datetime, timedelta


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class DataLoaderConfig:
    """Configuration constants for data loading."""

    # Date range defaults
    HISTORICAL_START_DATE = '2023-01-01'
    FUTURE_DAYS_AHEAD = 730

    # Filter criteria
    DEFAULT_COUNTRY = 'US'
    BOOKING_SOURCES = ['outdoorsy', 'combo']

    # High season months (May through September)
    HIGH_SEASON_MONTHS = [5, 6, 7, 8, 9]

    # Note: The SQL defines weekend as (0, 6) = Sunday & Saturday
    # However, the pricing algorithm recalculates this as Friday & Saturday
    # in the clean_listing_data() function, so this discrepancy is handled
    SQL_WEEKEND_DAYS = [0, 6]  # Sunday=0, Saturday=6 in SQL


# ══════════════════════════════════════════════════════════════════════════════
#  SQL QUERY TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def _build_listing_query():
    """
    Build the standardized listing data query template.

    Returns a SQL query string with a {date_filter} placeholder
    that can be customized for different date ranges.
    """
    query = """
        select
            dll.calendar_date as calendar_date,
            cast(date_trunc('month', dll.calendar_date) as date) as calendar_date_month,
            case when date_part('month',dll.calendar_date) in (5,6,7,8,9) then 1 else 0 end as is_high_season,
            dll.home_country,
            dll.home_state,
            r.metro_area as metro,
            dll.vehicle_type_group,
            dll.vehicle_type,
            dll.rental_id as listing_id,
            r.sleeps,
            r.vehicle_year,
            r.instant_book,
            r.owner_user_id as host_id,
            r.parent_images as image_count,
            r.parent_description_length as description_length,
            r.reviews_num as review_count,
            r.reviews_sum as total_review_score,
            case when date_part('dow', dll.calendar_date) in (0, 6) then 1 else 0 end as is_weekend,
            count(case when bookable = TRUE then calendar_date else null end) as bookable_days,
            count(dll.calendar_date) as gross_days,
            count(case when booked = TRUE and booking_source in ('outdoorsy', 'combo')
                  then calendar_date else null end) as booked_days,
            avg(base_price) as avg_base_price,
            avg(day_price) as avg_day_price,
            avg(case when booked = TRUE and booking_source in ('outdoorsy','combo')
                then booked_price else null end) as avg_booked_price

        from public.daily_listing_log dll
        join segmentevents.dbt.rentals r on dll.rental_id = r.id

        where dll.published = TRUE
          and dll.hidden = FALSE
          and dll.deleted = FALSE
          and dll.home_country = '{country}'
          {date_filter}

        group by 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
        order by 2 asc, 3 asc, 4 asc, 5 asc, 6 asc, 1 asc
    """
    return query


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class OutdoorsyDataLoader:
    """
    Loads listing data from the Outdoorsy database.

    Provides methods for loading historical, future, and custom date ranges
    using a single parameterized SQL query.
    """

    def __init__(self, engine, country=None):
        """
        Initialize the data loader.

        Args:
            engine: SQLAlchemy engine connected to the database
            country: Country code to filter by (default: 'US')
        """
        self.engine = engine
        self.country = country or DataLoaderConfig.DEFAULT_COUNTRY
        self.query_template = _build_listing_query()

    def load_historical_data(self, start_date=None):
        """
        Load historical listing data from past to yesterday.

        High-level steps:
        1. Determine start date (default: 2023-01-01)
        2. Build date filter for historical range
        3. Execute query and return data

        Args:
            start_date: Start date for historical data (default: 2023-01-01)

        Returns:
            DataFrame with historical listing data
        """
        start = start_date or DataLoaderConfig.HISTORICAL_START_DATE
        date_filter = self._build_historical_date_filter(start)
        return self._execute_query(date_filter)

    def load_future_data(self, days_ahead=None):
        """
        Load future listing data from today onwards.

        High-level steps:
        1. Determine how many days ahead to load (default: 730)
        2. Build date filter for future range
        3. Execute query and return data

        Args:
            days_ahead: Number of days ahead to load (default: 730)

        Returns:
            DataFrame with future listing data
        """
        days = days_ahead or DataLoaderConfig.FUTURE_DAYS_AHEAD
        date_filter = self._build_future_date_filter(days)
        return self._execute_query(date_filter)

    def load_custom_date_range(self, start_date, end_date):
        """
        Load listing data for a custom date range.

        High-level steps:
        1. Validate date parameters
        2. Build date filter for custom range
        3. Execute query and return data

        Args:
            start_date: Start date (str or datetime)
            end_date: End date (str or datetime)

        Returns:
            DataFrame with listing data for the specified range
        """
        date_filter = self._build_custom_date_filter(start_date, end_date)
        return self._execute_query(date_filter)

    def load_both_datasets(self):
        """
        Load both historical and future data in one call.

        Convenience method for the common use case of loading both datasets.

        Returns:
            Tuple of (historical_data, future_data) DataFrames
        """
        historical = self.load_historical_data()
        future = self.load_future_data()
        return historical, future

    # ─────────────────────────────────────────────────────────────────────────
    #  PRIVATE HELPER METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_historical_date_filter(self, start_date):
        """Build SQL date filter for historical data."""
        return f"""
            and calendar_date >= '{start_date}'
            and calendar_date < current_date
        """

    def _build_future_date_filter(self, days_ahead):
        """Build SQL date filter for future data."""
        return f"""
            and calendar_date >= current_date
            and calendar_date <= current_date + interval '{days_ahead} days'
        """

    def _build_custom_date_filter(self, start_date, end_date):
        """Build SQL date filter for custom date range."""
        start = self._format_date(start_date)
        end = self._format_date(end_date)
        return f"""
            and calendar_date >= '{start}'
            and calendar_date <= '{end}'
        """

    def _format_date(self, date):
        """Format date to YYYY-MM-DD string."""
        if isinstance(date, str):
            # Validate it's a proper date string
            pd.to_datetime(date)  # Will raise if invalid
            return date
        elif isinstance(date, (datetime, pd.Timestamp)):
            return date.strftime('%Y-%m-%d')
        else:
            raise ValueError(f"Invalid date format: {date}. Expected str or datetime.")

    def _execute_query(self, date_filter):
        """
        Execute the query with the given date filter.

        Args:
            date_filter: SQL WHERE clause for date filtering

        Returns:
            DataFrame with query results
        """
        query = self.query_template.format(
            country=self.country,
            date_filter=date_filter
        )

        print(f"  Loading data for country={self.country}...")
        df = pd.read_sql(query, self.engine)
        print(f"  Loaded {len(df):,} rows")

        return df


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_data_for_pricing(engine, country='US'):
    """
    Convenience function to load both datasets for pricing algorithm.

    High-level steps:
    1. Create data loader
    2. Load historical data
    3. Load future data
    4. Return both datasets

    Args:
        engine: SQLAlchemy database engine
        country: Country code to filter by (default: 'US')

    Returns:
        Tuple of (historical_data, future_data) DataFrames

    Usage:
        historic, future = load_data_for_pricing(engine)
        recs = recommend(future)
    """
    loader = OutdoorsyDataLoader(engine, country)
    return loader.load_both_datasets()


print("✓ Data Loader functions loaded successfully")
