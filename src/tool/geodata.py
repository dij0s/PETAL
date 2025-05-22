"""Exposes tools to query GeoContex-related data."""

import aiohttp
import asyncio

import numpy as np
from scipy.stats import t

from langchain_core.tools import StructuredTool

from typing import Callable, Any
from functools import partial

from provider.GeoSessionProvider import GeoSessionProvider

async def _estimate_solar_potential(municipality_name: str, confidence_level=0.8) -> tuple[float, float, float]:
    """Asynchronously estimates the solar potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate solar potential for.
        confidence_level (float, optional): The confidence level for the margin of error (default is 0.8).

    Returns:
        tuple[float, float]: A tuple containing the estimated solar potential in GWh/year, the estimated solar potential margin in GWh/year and the confidence level.
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
            "sr": "2056"
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
                        potential = result.get("properties", {}).get("stromertrag", 0)
                        tile_potential += potential

                    return tile_potential
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"error for tile at {geometry_str}: {e}")

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
    margin_gwh = margin_kwh / 1e6

    return total_estimate_gwh, margin_gwh, confidence_level

# expose all tools with
# associated description
class GeoDataTool(StructuredTool):
    def __init__(
            self,
            municipality_name: str,
            func: Callable[..., Any],
            name: str,
            description: str,
            **kwargs
        ):
            self.municipality_name = municipality_name
            # wrap the function so it always
            # gets municipality_name as a kwarg
            async def wrapped_func(*args, **inner_kwargs):
                return await func(*args, municipality_name=self.municipality_name, **inner_kwargs)

            super().__init__(
                coroutine=wrapped_func,
                name=name,
                description=description,
                **kwargs
            )

class SolarPotentialEstimatorTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_estimate_solar_potential, confidence_level=0.8),
            name="estimate_solar_potential",
            description="Estimates the solar potential for a given municipality in Switzerland. Returns the estimated solar potential in GWh/year, the margin of error, and the confidence level. Useful for assessing renewable energy potential at the municipal level.",
        )

class SolarPotentialAggregatorTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_estimate_solar_potential, confidence_level=1.0),
            name="aggregate_solar_potential",
            description="Returns the exact solar potential for a given municipality in Switzerland. Provides the total solar potential in GWh/year, the margin of error, and the confidence level. Useful for obtaining precise renewable energy data at the municipal level.",
        )
