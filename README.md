# Outdoorsy Pricing Algorithm

A sophisticated dynamic pricing algorithm for RV rentals that generates optimized price recommendations based on demand signals, utilization patterns, and market conditions.

## Overview

This algorithm analyzes historical booking data and market trends to generate daily pricing recommendations for RV listings. It uses a multi-factor approach combining:
- Benchmark pricing (3-tier fallback: metro → region → national)
- Demand multipliers (national trends)
- Local market conditions (metro-level)
- Listing quality scoring
- Utilization-based boosts
- Time-to-search proximity adjustments

## Architecture

The system is designed as three modular Python scripts for use in Hex notebooks:

```
┌─────────────────────────────────────────────────────────┐
│ 1. data_loader.py                                       │
│    - Loads historical and future listing data           │
│    - Connects to Redshift                               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. pricing_algorithm.py                                 │
│    - Generates aggregated recommendations               │
│    - Calculates benchmarks and multipliers              │
│    - Detects underpriced listings                       │
│    - Output: listing × month × weekend level            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. data_transformation.py                               │
│    - Expands to daily recommendations                   │
│    - Applies utilization-based boosts                   │
│    - Applies 7% floor check                             │
│    - Transforms to database-ready format                │
│    - Output: PostgreSQL-compatible arrays               │
└─────────────────────────────────────────────────────────┘
```

## Features

### Core Pricing Logic
- **Benchmark Calculation**: 3-tier fallback (metro → region → national)
- **Demand Multiplier**: National booking trends analysis
- **Local Multiplier**: Metro-level market conditions
- **Listing Multiplier**: Quality, reviews, instant book, images
- **Quality Scoring**: Premium/standard/basic classification

### Dynamic Boosts
- **Time-to-Search Boost**:
  - ≤7 days: +2.5%
  - ≤30 days: +1.0%
  - ≤90 days: +0.5%
- **Historic Demand Boost**: Based on seasonal utilization patterns
  - Low: 0%
  - Medium: +2.5%
  - High: +5.0%
- **Forward Demand Boost**: Based on future booking velocity
  - Low: 0%
  - Medium: +5.0%
  - High: +10.0%

### Safety Features
- **7% Minimum Floor**: Ensures price increases meet minimum threshold
- **Underpriced Detection**: Flags listings priced below market benchmarks
- **NaN Handling**: Robust handling of missing data
- **Memory Optimization**: Efficient processing of large datasets

## Requirements

- Python 3.8+
- pandas
- numpy
- sqlalchemy
- psycopg2
- Redshift database access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joshdatajosh/Outdoorsy-Pricing-Algorithm.git
cd Outdoorsy-Pricing-Algorithm
```

2. Install dependencies:
```bash
pip install pandas numpy sqlalchemy psycopg2-binary
```

3. Configure database credentials in `data_transformation.py` (lines 62-67)

## Usage in Hex

### Cell 1: Load Data Loader
```python
# Paste entire contents of data_loader.py
# This defines OutdoorsyDataLoader class
```

### Cell 2: Load Pricing Algorithm
```python
# Paste entire contents of pricing_algorithm.py
# This defines recommend() function and all pricing logic
```

### Cell 3: Generate Recommendations
```python
# Paste entire contents of data_transformation.py
# This loads data, generates recs, applies boosts, and outputs database-ready format
```

**Output:**
- `daily_recs` - Daily recommendations with all boosts applied (~1.85M rows)
- `write_ready_df` - Database-ready format (one row per listing)

### Database Format

The `write_ready_df` DataFrame contains:
- `rental_id`: Listing ID
- `start_date`: Current date
- `prices`: PostgreSQL array `{100,105,110,...}` (prices in cents)
- `explanations`: PostgreSQL array `{"weekend_premium","seasonal_demand",...}`
- `insights`: JSON array of insight objects

Example:
```json
{
  "rental_id": 12345,
  "start_date": "2024-10-02",
  "prices": "{12500,13000,13500,...}",
  "explanations": "{\"weekend_premium\",\"No Change\",\"seasonal_demand\",...}",
  "insights": "[{\"dow\":\"weekend\",\"local_demand\":\"high\",...},...]"
}
```

## Module Details

### data_loader.py
Handles all data extraction from Redshift:
- `OutdoorsyDataLoader`: Main class for loading listing data
- `load_historical_data()`: Loads past booking history
- `load_future_data()`: Loads future calendar availability
- Parameterized SQL queries with date filtering

### pricing_algorithm.py
Core pricing logic with aggregated output:
- `recommend()`: Main function generating recommendations
- `calculate_benchmarks()`: 3-tier benchmark calculation
- `calculate_demand_multiplier()`: National demand trends
- `calculate_local_multiplier()`: Metro-level market conditions
- `calculate_listing_multiplier()`: Listing-specific factors
- `detect_underpriced()`: Identifies underpriced listings
- Output: listing × month × is_weekend level (~300K rows)

### data_transformation.py
Expands recommendations to daily level and prepares for database:
- Daily expansion via join pattern (aggregated → daily)
- Historic utilization calculation (by date token)
- Forward utilization calculation (by date + metro)
- Boost application (time-to-search, demand-based)
- 7% floor check enforcement
- Database transformation (PostgreSQL arrays + JSON)
- Output: Daily recs + database-ready format

## Performance

- **Memory Usage**: ~1.5 GB peak (down from 32GB+ in previous versions)
- **Runtime**: ~2-3 minutes for full pipeline
- **Data Volume**:
  - Input: ~500 MB (historical + future data)
  - Aggregated: ~300K rows (45 MB)
  - Daily expanded: ~1.85M rows (160 MB)
  - Database-ready: ~12K listings (5 MB)

## Technical Notes

### Redshift Compatibility
The code includes a SQLAlchemy dialect patch to handle Redshift's lack of `standard_conforming_strings` support. The patch is applied once with a recursion guard to prevent errors when re-running cells.

### Aggregated → Daily Pattern
Recommendations are generated at `listing × month × is_weekend` level for memory efficiency, then expanded to daily level via join. This matches the SQL pattern in `date_transformation.sql`.

### ×100 Price Format
All prices are stored in cents (×100 format) as int32 for precision and memory efficiency. Divide by 100 for display.

## Contributing

This is a private repository for Outdoorsy's internal use.

## License

Proprietary - Outdoorsy Inc.

## Authors

- Joshua Haase (Data Team)
- With assistance from Claude (Anthropic)

## Version History

- **v2.0** (2024-10-02): Complete rewrite with 3-module architecture, daily expansion, and database transformation
- **v1.0** (2024-09): Initial aggregated pricing algorithm
