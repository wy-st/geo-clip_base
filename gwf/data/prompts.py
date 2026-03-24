"""
gwf/data/prompts.py
====================
Prompt templates for benchmark datasets.

The LLM channel uses these prompts to extract world knowledge that
complements the numeric tabular features. Each prompt describes:
  - The geographic region and time period
  - The prediction target and its real-world meaning
  - Key spatial drivers of variation in the target
  - The feature columns and their semantic meaning

Usage:
    from gwf.data.prompts import PROMPT_TEMPLATES
    prompt = PROMPT_TEMPLATES["airbnb_sandiego"]
    # or build your own with build_prompt()
"""

# =============================================================================
# Pre-defined prompts for common benchmark datasets
# =============================================================================

PROMPT_TEMPLATES: dict[str, str] = {

    # ------------------------------------------------------------------
    # Airbnb short-term rental prices
    # ------------------------------------------------------------------
    "airbnb_sandiego": """
This dataset contains Airbnb short-term rental listings in San Diego,
California, USA. The task is to predict the log-transformed nightly price
(log_price) from property and location features.

Key spatial drivers:
- Proximity to the Pacific Ocean and beaches (coastal premium)
- Distance to Balboa Park (major urban amenity)
- Neighborhood type: Entire home/apt commands premium over private/shared rooms
- Property type: Houses and condos differ from apartments in pricing
- Accommodation capacity: more beds and bathrooms → higher price
- Pool availability adds premium in warm climate

Feature semantics:
  accommodates: number of guests the property can host
  bathrooms: number of bathrooms
  bedrooms: number of bedrooms
  beds: number of beds
  pool: 1 if the listing has a pool
  d2balboa: distance to Balboa Park in km
  coastal: 1 if within 2km of the coastline
  pg_*: property group indicators (Apartment, Condominium, House, Townhouse)
  rt_*: room type indicators (Entire home/apt, Private room, Shared room)

San Diego has a Mediterranean climate, strong beach tourism, and highly
spatially heterogeneous pricing — coastal areas and Gaslamp Quarter are
premium, while inland suburban areas are more affordable.
""".strip(),

    # ------------------------------------------------------------------
    # Air quality / PM2.5
    # ------------------------------------------------------------------
    "pm25_china": """
This dataset contains PM2.5 (fine particulate matter) concentrations at
monitoring stations across China. The task is to predict annual mean PM2.5
(μg/m³) from satellite-derived and environmental features.

Key spatial drivers:
- North China Plain has extremely high PM2.5 due to coal heating, heavy
  industry (steel, cement), and unfavorable topography (basin effects)
- Tibetan Plateau and western deserts have low PM2.5 but high dust events
- Coastal cities benefit from sea breezes that disperse pollutants
- Seasonal heating in northern cities causes winter PM2.5 spikes

Feature semantics:
  AOD: aerosol optical depth from MODIS satellite (proxy for PM2.5)
  NDVI: vegetation index — greener areas often have lower PM2.5
  DEM: elevation — higher altitudes generally have cleaner air
  pop_density: population density — urbanisation drives emissions
  temperature: warm weather reduces coal heating demand
  wind_speed: higher wind disperses pollutants
  humidity: affects aerosol formation and scattering
  precipitation: rain washes out PM2.5

China's air pollution is strongly spatially heterogeneous — Beijing-Tianjin-
Hebei (BTH) region is one of the most polluted globally, while Yunnan and
Hainan provinces have pristine air.
""".strip(),

    # ------------------------------------------------------------------
    # House prices
    # ------------------------------------------------------------------
    "house_prices_london": """
This dataset contains residential property transaction prices in Greater
London, UK. The task is to predict log-transformed sale price from
property and location features.

Key spatial drivers:
- Distance to Central London / City of London financial district
- London Underground (Tube) accessibility — transit premium
- School catchment quality (Ofsted ratings)
- Borough-level wealth and gentrification patterns
- Thames riverside premium
- Heritage conservation areas limit supply, inflating prices

Feature semantics:
  floor_area: property size in m²
  num_rooms: number of habitable rooms
  property_type: Detached/Semi/Terraced/Flat
  tenure: Freehold vs Leasehold (freehold is premium)
  dist_tube: distance to nearest Tube station in km
  dist_city: distance to Bank/Canary Wharf in km
  borough: London borough code (categorical)
  school_score: average Ofsted score of schools in catchment

London's housing market is one of the most spatially heterogeneous in the
world, with prices ranging from £200k in outer zones to £10M+ in Mayfair.
""".strip(),

    # ------------------------------------------------------------------
    # Generic template for user-defined datasets
    # ------------------------------------------------------------------
    "generic": """
This is a spatial regression dataset. The task is to predict a continuous
target variable from tabular features at geographically distributed locations.

The model should learn spatially varying regression coefficients that capture
how feature-target relationships change across space (Tobler's First Law:
nearby things are more alike than distant things).

Please use the spatial coordinates (latitude, longitude) and tabular features
provided to generate accurate, geographically aware predictions.
""".strip(),

}


# =============================================================================
# Builder function: construct a prompt from dataset metadata
# =============================================================================

def build_prompt(
    dataset_name: str,
    region:       str,
    time_period:  str,
    target_desc:  str,
    feature_descs: dict[str, str],
    spatial_drivers: list[str],
) -> str:
    """
    Programmatically build a dataset prompt.

    Args:
        dataset_name    : short name of the dataset
        region          : geographic region (e.g. "San Diego, CA, USA")
        time_period     : time period (e.g. "2019-2023")
        target_desc     : description of the target variable
        feature_descs   : dict mapping feature names to descriptions
        spatial_drivers : list of key spatial factors affecting the target

    Returns:
        Formatted prompt string.
    """
    feat_lines = "\n".join(
        f"  {k}: {v}" for k, v in feature_descs.items()
    )
    driver_lines = "\n".join(f"- {d}" for d in spatial_drivers)

    return f"""
Dataset: {dataset_name}
Region: {region}
Time period: {time_period}

Task: {target_desc}

Key spatial drivers of variation:
{driver_lines}

Feature semantics:
{feat_lines}
""".strip()
