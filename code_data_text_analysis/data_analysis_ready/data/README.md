# Data Card

## Overview

`life_expectancy.csv` contains 32 fictional communities created for this exercise. It is synthetic teaching data, not a sample of people or places, and is unsuitable for health decisions.

## Schema

| Column | Description |
| --- | --- |
| `community_id` | Unique row identifier; excluded from model features |
| `income_per_capita_usd` | Annual income per person |
| `healthcare_access_pct` | Population with basic healthcare access |
| `schooling_years` | Average years of schooling |
| `clean_water_pct` | Population with clean-water access |
| `smoking_rate_pct` | Adult smoking rate |
| `infant_mortality_per_1000` | Infant deaths per 1,000 births |
| `life_expectancy_years` | Community average and prediction target |

## Known conditions

- The target and identifiers are complete.
- Three feature cells are intentionally missing.
- All predictors are numeric; `community_id` is for traceability only.
- Apparent associations are synthetic and must not be described as causal.
