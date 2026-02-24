# Spatio-Temporal Predictive Analytics Benchmark

## What This Is

A runnable demo that backs the BigBear.ai resume bullet:
> *"Predictive analytics precision +22% using geospatial features and transformer temporal encoders"*

Builds a complete spatio-temporal prediction pipeline from scratch, engineers meaningful features out of raw lat/lon + timestamps, and measures the precision improvement from each technique.

## Quick Start

```bash
# Upload SpatioTemporal_Benchmark.ipynb to Google Colab
# Runtime → CPU (no GPU needed)
# Run All Cells → ~3 minutes
```

**Dependencies** (auto-installed in Colab):
```
h3, python-geohash, scikit-learn, torch, matplotlib, pandas
```

## The Problem

You have events with lat/lon coordinates and timestamps. You need to predict which location-time combinations are high-risk. Raw coordinates are terrible features because:

- 1° longitude = 111km at the equator but 78km at 45°N (not linear)
- Coordinates 38.900 and 38.901 look numerically far apart but are 100m apart
- Hour 23 and hour 0 look 23 apart but are 1 hour apart
- The model has no concept of "near a hotspot" or "late at night" from raw numbers

## What the Notebook Builds

### 1. Synthetic Event Data (20K events)
Simulates defense/intelligence spatio-temporal incidents across a DC-area region:
- 4 spatial hotspots with different risk profiles
- Temporal patterns (night = riskier, weekends = riskier)
- Spatio-temporal interactions (a location can be safe by day, dangerous at night)
- ~25% incident rate with realistic noise

### 2. Geospatial Feature Engineering

| Feature | What It Does | Why It Helps |
|---------|-------------|-------------|
| **Sinusoidal spatial encoding** | `sin(lat * freq)`, `cos(lat * freq)` at 6 frequencies | Same idea as transformer positional encoding — smooth continuous spatial awareness at multiple scales |
| **H3 hex binning** | Uber's hexagonal grid at resolutions 3, 4, 5 | Groups nearby points into cells; multi-resolution so model sees both "which neighborhood" and "which block" |
| **Distance to hotspots** | Euclidean distance to each known risk zone | Direct spatial context the model can't infer from raw coordinates |
| **Local event density** | Count of events in same grid cell | Captures "busy area" vs "quiet area" |
| **Local incident rate** | Historical incident rate per cell | The strongest spatial signal — past behavior predicts future behavior |

### 3. Temporal Feature Engineering

| Feature | What It Does | Why It Helps |
|---------|-------------|-------------|
| **Cyclical encoding** | `sin(2π × hour/24)`, `cos(2π × hour/24)` | Model understands 23:00 and 01:00 are close (1 hour apart, not 22) |
| **Context flags** | `is_night`, `is_weekend`, `is_rush_hour` | Binary signals for known risk windows |
| **Spatio-temporal interactions** | `is_night × distance_to_hotspot` | Captures "a location near a hotspot AT NIGHT is riskier than the same location at noon" |

### 4. Transformer Temporal Encoder

A small transformer that processes the **sequence of recent events** at each location:

```
Input:  [time_delta, hour_sin, hour_cos, was_incident, density, hotspot_dist] × 8 steps
Output: 32-dim embedding capturing temporal dynamics
```

This captures patterns like:
- "3 incidents in this cell in the last 2 hours" → escalation pattern → high risk
- "This cell has been quiet for 3 days" → low risk
- "Events here follow a daily cycle" → periodic pattern

Architecture: 2-layer transformer encoder with 4 attention heads, learned positional encoding, mean pooling to fixed-size output.

### 5. Model Comparison

Trains GradientBoosting on 4 feature sets and compares:

| Configuration | Features | Expected Precision Lift |
|--------------|----------|----------------------|
| Raw Baseline | lat, lon, hour, day_of_week, month | — |
| + Geospatial | + sinusoidal encoding, H3 cells, distances, density | +10-15% |
| + Geo + Temporal | + cyclical time, context flags, interactions | +15-20% |
| Full Pipeline | + transformer temporal embeddings | +20-25% |
