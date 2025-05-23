"""Exposes tools to query GeoContex-related data."""

import aiohttp
import asyncio

from shapely.geometry import shape

import numpy as np
from scipy.stats import t

from langchain_core.tools import StructuredTool

from typing import Callable, Any
from functools import partial

from provider.GeoSessionProvider import GeoSessionProvider

async def _fetch_solar_potential_roofing(municipality_name: str, confidence_level=0.8) -> tuple[float, float, float]:
    """Asynchronously estimates the roofing solar potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate solar potential for.
        confidence_level (float, optional): The confidence level for the margin of error (default is 0.8).

    Returns:
        tuple[float, float, float]: A tuple containing the estimated solar potential in GWh/year, the estimated solar potential margin in GWh/year and the confidence level.
    """
    async def _fetch_solar_potential(session, tile) -> float:
        minx, miny, maxx, maxy = tile.bounds
        geometry_str = f"{minx},{miny},{maxx},{maxy}"

        identify_url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometry": geometry_str,
            "geometryType": "esriGeometryEnvelope",
            "layers": "all:ch.bfe.solarenergie-eignung-daecher",
            "returnGeometry": "false",
            "tolerance": "0",
            "sr": "2056",
        }

        try:
            # retrieve and aggregate partial result
            # of tile from retrieved roofs
            async with session.get(identify_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tile_potential = 0  # kWh/year

                    # sum up the potential for all features
                    for result in data.get("results", []):
                        potential = result.get("attributes", {}).get("stromertrag", 0)
                        tile_potential += potential

                    return tile_potential
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return 0

    # await needed GeoSession
    # for further computing
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=0.3)
    await provider.wait_until_ready()

    # create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # launch all tile potential fetches concurrently
        tasks = [_fetch_solar_potential(session, tile) for tile in provider.sampled_tiles]
        sampled_potentials = await asyncio.gather(*tasks)

    # aggregate all partial potentials
    # and compute estimate potential
    # for municipality
    sampled_potentials = np.array(sampled_potentials)
    n = len(sampled_potentials)
    mean_tile_value = np.mean(sampled_potentials)
    std_dev = np.std(sampled_potentials, ddof=1)

    # compute t-value for the
    # given confidence level
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1)
    total_estimate_kwh = mean_tile_value * provider.total_tiles
    margin_kwh = t_value * (std_dev / np.sqrt(n)) * provider.total_tiles

    # convert to GWh/year
    total_estimate_gwh = total_estimate_kwh / 1e6
    margin_gwh = float(margin_kwh / 1e6)
    # handle full aggregation
    if margin_gwh == float("+inf"):
        margin_gwh = 0.0

    return (float(total_estimate_gwh), margin_gwh, confidence_level)

async def _fetch_solar_potential_facades(municipality_name: str, confidence_level=0.8) -> tuple[float, float, float]:
    """Asynchronously estimates the facades solar potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate solar potential for.
        confidence_level (float, optional): The confidence level for the margin of error (default is 0.8).

    Returns:
        tuple[float, float, float]: A tuple containing the estimated solar potential in GWh/year, the estimated solar potential margin in GWh/year and the confidence level.
    """
    async def _fetch_solar_potential(session, tile) -> float:
        minx, miny, maxx, maxy = tile.bounds
        geometry_str = f"{minx},{miny},{maxx},{maxy}"

        identify_url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometry": geometry_str,
            "geometryType": "esriGeometryEnvelope",
            "layers": "all:ch.bfe.solarenergie-eignung-fassaden",
            "returnGeometry": "false",
            "tolerance": "0",
            "sr": "2056",
        }

        try:
            # retrieve and aggregate partial result
            # of tile from retrieved roofs
            async with session.get(identify_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    tile_potential = 0  # kWh/year

                    # sum up the potential for all features
                    for result in data.get("results", []):
                        potential = result.get("attributes", {}).get("stromertrag", 0)
                        tile_potential += potential

                    return tile_potential
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return 0

    # await needed GeoSession
    # for further computing
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=0.3)
    await provider.wait_until_ready()

    # create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # launch all tile potential fetches concurrently
        tasks = [_fetch_solar_potential(session, tile) for tile in provider.sampled_tiles]
        sampled_potentials = await asyncio.gather(*tasks)

    # aggregate all partial potentials
    # and compute estimate potential
    # for municipality
    sampled_potentials = np.array(sampled_potentials)
    n = len(sampled_potentials)
    mean_tile_value = np.mean(sampled_potentials)
    std_dev = np.std(sampled_potentials, ddof=1)

    # compute t-value for the
    # given confidence level
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1)
    total_estimate_kwh = mean_tile_value * provider.total_tiles
    margin_kwh = t_value * (std_dev / np.sqrt(n)) * provider.total_tiles

    # convert to GWh/year
    total_estimate_gwh = total_estimate_kwh / 1e6
    margin_gwh = float(margin_kwh / 1e6)
    # handle full aggregation
    if margin_gwh == float("+inf"):
        margin_gwh = 0.0

    return (float(total_estimate_gwh), margin_gwh, confidence_level)

async def _fetch_small_hydro_potential(municipality_name: str, efficiency: float = 0.3) -> tuple[float, float]:
    """Asynchronously fetches the small hydroelectricity potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate small hydroelectricity for.
        efficiency (float, optional): The hydroelectricity production efficiency used to compute the total potential (default is 0.3).

    Returns:
        tuple[float, float]: A tuple containing the estimated small hydroelectricity potential in GWh/year and the estimated efficiency.
    """
    # await GeoSession geometry
    # fetching for further computing ;
    # as no tiling is needed, await
    # most common GeoSession
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=0.3)
    await provider.wait_until_ready()

    minx, miny, maxx, maxy = provider.geometry.bounds
    geometry_str = f"{minx},{miny},{maxx},{maxy}"

    url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
    params = {
        "geometry": geometry_str,
        "geometryType": "esriGeometryEnvelope",
        "layers": "all:ch.bfe.kleinwasserkraftpotentiale",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        # retrieve and aggregate partial result
        # of tile from retrieved roofs
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    total_potential = 0  # kWh

                    # aggregate all features
                    for feature in data.get("results", []):
                        geometry = shape(feature.get("geometry"))
                        props = feature.get("properties", {})
                        kw_per_meter = float(props.get("kwprometer", 0))

                        # intersect and clip features
                        # to municipality geometry
                        if geometry.intersects(provider.geometry):
                            clipped = geometry.intersection(provider.geometry)
                            if not clipped.is_empty:
                                length = clipped.length
                                potential_kw = length * kw_per_meter

                                total_potential += potential_kw

                    # convert to GWh/year
                    hours_per_year = 8760
                    total_potential_GWh = total_potential * hours_per_year * efficiency / 1e6

                    return (total_potential_GWh, efficiency)
                else:
                    print(f"Could not retrieve small hydroelectricity data: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0, 0

# expose all tools with
# associated description
class GeoDataTool(StructuredTool):
    def __init__(
            self,
            municipality_name: str,
            func: Callable[..., Any],
            name: str,
            layer_id: str,
            description: str,
            **kwargs
        ):
            # wrap the function so it always
            # returns data with the name of the
            # tool that was invoked and extra
            # layer_id
            async def wrapped_func(*args, **inner_kwargs) -> dict[str, tuple[str, Any]]:
                return {
                    name: (layer_id, await func(*args, municipality_name=municipality_name, **inner_kwargs, **kwargs))
                }

            super().__init__(
                coroutine=wrapped_func,
                name=name,
                description=description,
                args_schema=None,
                **kwargs
            )

    @classmethod
    def factory(cls, municipality_name: str, **kwargs) -> "GeoDataTool":
        """
        Factory method to instantiate the tool with the given municipality_name.

        Args:
            municipality_name (str): The name of the municipality.
            **kwargs: Additional keyword arguments.

        Returns:
            GeoDataTool: An instance of the GeoDataTool subclass.
        """
        return cls(municipality_name, **kwargs)

class RoofingSolarPotentialEstimatorTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_solar_potential_roofing, confidence_level=0.8),
            name="estimate_solar_potential_roofing",
            layer_id="ch.bfe.solarenergie-eignung-daecher",
            description="ESTIMATES THE ROOFING SOLAR POTENTIAL for a given municipality in Switzerland. Returns the estimated solar potential in GWh/year, the margin of error, and the confidence level. Useful for assessing renewable energy potential at the municipal level.",
        )

class RoofingSolarPotentialAggregatorTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_solar_potential_roofing, confidence_level=1.0),
            name="aggregate_solar_potential_roofing",
            layer_id="ch.bfe.solarenergie-eignung-daecher",
            description="Returns the EXACT ROOFING SOLAR POTENTIAL for a given municipality in Switzerland. Provides the total solar potential in GWh/year, the margin of error, and the confidence level. Useful for obtaining precise renewable energy data at the municipal level at the COST OF GREATER COMPUTE TIME. ONLY USE IF USER EXPLICITLY ASKS FOR PRECISE DATA.",
        )

class FacadesSolarPotentialEstimatorTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_solar_potential_facades, confidence_level=0.8),
            name="estimate_solar_potential_facades",
            layer_id="ch.bfe.solarenergie-eignung-fassaden",
            description="ESTIMATES THE FACADES SOLAR POTENTIAL for a given municipality in Switzerland. Returns the estimated solar potential in GWh/year, the margin of error, and the confidence level. Useful for assessing renewable energy potential at the municipal level.",
        )

class FacadesSolarPotentialAggregatorTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_solar_potential_facades, confidence_level=1.0),
            name="aggregate_solar_potential_facades",
            layer_id="ch.bfe.solarenergie-eignung-fassaden",
            description="Returns the EXACT FACADES SOLAR POTENTIAL for a given municipality in Switzerland. Provides the total solar potential in GWh/year, the margin of error, and the confidence level. Useful for obtaining precise renewable energy data at the municipal level at the COST OF GREATER COMPUTE TIME. ONLY USE IF USER EXPLICITLY ASKS FOR PRECISE DATA.",
        )

class SmallHydroPotentialTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_small_hydro_potential, efficiency=0.3),
            name="small_hydro_potential",
            layer_id="ch.bfe.kleinwasserkraftpotentiale",
            description="Returns the potential of small hydroelectricity due to small hydro sources in the municipality in GWh/year and the efficiency used to compute it.",
        )
