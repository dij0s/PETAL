"""Exposes tools to query GeoContext-related data."""

import aiohttp
import asyncio

from shapely.geometry import shape

import re
import numpy as np
from scipy.stats import t

from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer

from typing import Callable, Any
from functools import partial, reduce
from collections import defaultdict

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
            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with session.get(identify_url, params=params, headers=headers) as response:
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
    # sampling_rate depends
    # on if the user wants an
    # estimation, or not
    sampling_rate = 0.3 if confidence_level < 1.0 else 1.0
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=sampling_rate)
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
            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with session.get(identify_url, params=params, headers=headers) as response:
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
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_potential = 0  # kWh

                    # aggregate all features
                    for feature in data.get("results", []):
                        geometry = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if geometry.intersects(provider.geometry):
                            clipped = geometry.intersection(provider.geometry)
                            if not clipped.is_empty:
                                props = feature.get("properties", {})
                                kw_per_meter = float(props.get("kwprometer", 0))

                                length = clipped.length
                                potential_kw = length * kw_per_meter

                                total_potential += potential_kw

                    # convert to GWh/year
                    hours_per_year = 24 * 365
                    total_potential_GWh = total_potential * hours_per_year * efficiency / 1e6

                    return (total_potential_GWh, efficiency)
                else:
                    print(f"Could not retrieve small hydroelectricity data: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0, 0

async def _fetch_big_hydro_potential(municipality_name: str) -> tuple[float, float]:
    """Asynchronously fetches the big hydro heating/cooling potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate big hydro for.

    Returns:
        tuple[float, float]: A tuple containing the heating and cooling potential in GWh/year.
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
        "layers": "all:ch.bfe.waermepotential-gewaesser",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    heating_potential = [] # GWh/year
                    cooling_potential = [] # GWh/year

                    # aggregate all features
                    for feature in data.get("results", []):
                        geometry = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if geometry.intersects(provider.geometry):
                            clipped = geometry.intersection(provider.geometry)
                            if not clipped.is_empty:
                                props = feature.get("properties", {})
                                heating = float(props.get("heat_extraction_gwha", 0))
                                cooling = float(props.get("heat_disposal_gwha", 0))

                                heating_potential.append(heating)
                                cooling_potential.append(cooling)

                    # average over municipality
                    # attention, this is very
                    # difficult to properly exploit
                    # as a single body of water
                    # may be split into "sections"
                    # with different estimate
                    # as they are very rare in CH,
                    # not a big issue to average
                    # (Rhône and other smaller rivers)
                    heating_potential_GWh = sum(heating_potential) / len(heating_potential) if len(heating_potential) > 0 else 0.0
                    cooling_potential_GWh = sum(cooling_potential) / len(cooling_potential) if len(cooling_potential) > 0 else 0.0

                    return (heating_potential_GWh, cooling_potential_GWh)
                else:
                    print(f"Could not retrieve big hydroelectricity data: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0, 0

async def _fetch_available_biomass(municipality_name: str) -> tuple[float, float]:
    """Asynchronously fetches the available biomass for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.

    Returns:
        tuple[float, float]: A tuple containing the available woody and non-woody biomass in GWh.
    """
    # await GeoSession SFSO number
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=0.3)
    await provider.wait_until_sfso_ready()

    url = "https://api3.geo.admin.ch/rest/services/api/MapServer/find"
    params = {
        "layer": "ch.bfe.biomasse-nicht-verholzt",
        "searchField": "bfs_nummer",
        "searchText": provider.municipality_sfso_number,
        "returnGeometry": "false",
        "sr": 2056,
    }
    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    attributes = data.get("results", [])[0].get("attributes", [])

                    woody_biomass = attributes.get("woody", 0) # TJ
                    non_woody_biomass = attributes.get("non_woody", 0) # TJ

                    # 1 kWh equiv. 3.6 MJ <=> 1 GWh equiv. 3.6 TJ
                    woody_biomass_GWh = woody_biomass / 3.6
                    non_woody_biomass_GWh = non_woody_biomass / 3.6

                    return (woody_biomass_GWh, non_woody_biomass_GWh)
                else:
                    print(f"Could not available biomass: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0, 0

async def _fetch_hydropower_infrastructure(municipality_name: str) -> float:
    """Asynchronously fetches the hydropower infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.

    Returns:
        float: The total hydropower electricity production in GWh/year.
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
        "layers": "all:ch.bfe.statistik-wasserkraftanlagen",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_production = 0 # GWh/year

                    # aggregate all features
                    for feature in data.get("results", []):
                        point = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if point.within(provider.geometry):
                            props = feature.get("properties", {})

                            production = float(props.get("productionexpected", 0))  # GWh/year
                            total_production += production

                    return total_production
                else:
                    print(f"Could not retrieve hydropower infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0

async def _fetch_wind_turbines_infrastructure(municipality_name: str) -> float:
    """Asynchronously fetches the wind turbines infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.

    Returns:
        float: The total wind turbines electricity production in GWh/year.
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
        "layers": "all:ch.bfe.windenergieanlagen",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_production = 0 # kWh/year
                    # regex pattern to retrieve
                    # information from HTML table
                    pattern = r"<production>(\d+)</production>\n</PROD></prods>$"

                    # aggregate all features
                    for feature in data.get("results", []):
                        point = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if point.within(provider.geometry):
                            props = feature.get("properties", {})
                            match = re.search(pattern, props.get("fac_xml_prod", ""))

                            production = float(match.group(1)) if match else 0.0 # kWh/year
                            total_production += production

                    total_production_GWh = total_production / 1e6

                    return total_production_GWh
                else:
                    print(f"Could not retrieve wind turbine infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0

async def _fetch_biogas_infrastructure(municipality_name: str) -> float:
    """Asynchronously fetches the biogas infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.

    Returns:
        float: The total biogas production in GWh/year.
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
        "layers": "all:ch.bfe.biogasanlagen",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_production = 0 # kWh/year

                    # aggregate all features
                    for feature in data.get("results", []):
                        point = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if point.within(provider.geometry):
                            props = feature.get("properties", {})

                            yearly_production = props.get("yearly_production", {})
                            last_production = max(yearly_production, key=lambda e: int(e.get("year", "0")))
                            production = float(last_production.get("electricity", 0)) # kWh

                            total_production += production

                    total_production_GWh = total_production / 1e6

                    return total_production_GWh
                else:
                    print(f"Could not retrieve biogas infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0

async def _fetch_incineration_infrastructure(municipality_name: str) -> tuple[float, float]:
    """Asynchronously fetches the incineration infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to fetch the data for.

    Returns:
        tuple[float, float]: The total electricity and heating production in GWh/year.
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
        "layers": "all:ch.bfe.kehrichtverbrennungsanlagen",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_electricity_production = 0 # MWh/year
                    total_heating_production = 0 # MWh/year

                    # aggregate all features
                    for feature in data.get("results", []):
                        point = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if point.within(provider.geometry):
                            props = feature.get("properties", {})

                            electricity_production = float(props.get("electricity", "0")) # MWh/year
                            heating_production = float(props.get("heat", "0")) # MWh/year

                            total_electricity_production += electricity_production
                            total_heating_production += heating_production

                    total_electricity_production_GWh = total_electricity_production / 1e3
                    total_heating_production_GWh = total_heating_production / 1e3

                    return total_electricity_production_GWh, total_heating_production_GWh
                else:
                    print(f"Could not retrieve incineration infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0, 0

async def _fetch_effective_infrastructure(municipality_name: str) -> tuple[float, float, float]:
    """Asynchronously fetches the effective photovoltaic (PV), biomass and geothermal energy from various infrastructure.

    Args:
        municipality_name (str): The name of the municipality to estimate the effective infrastructure for.

    Returns:
        tuple[float, float, float]: A tuple containing the effective photovoltaic, biomass and geothermal power in GWh/year.
    """
    # estimated effective power
    # per energy type to compute
    # yearly yield
    _effective_power_factor = {
        "Photovoltaic": 0.35,
        "Biomass" : 1.0,
        "Geothermal energy": 1.0
    }

    async def _fetch_infrastructure(session, tile) -> defaultdict:
        minx, miny, maxx, maxy = tile.bounds
        geometry_str = f"{minx},{miny},{maxx},{maxy}"

        identify_url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometry": geometry_str,
            "geometryType": "esriGeometryEnvelope",
            "layers": "all:ch.bfe.elektrizitaetsproduktionsanlagen",
            "returnGeometry": "false",
            "tolerance": "0",
            "sr": "2056",
            "geometryFormat": "geojson"
        }

        try:
            # retrieve and aggregate partial
            # results from every single tile
            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with session.get(identify_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    productions = defaultdict(float) # kWh/year
                    hours_per_year = 24 * 365

                    # sum up the potential for all features
                    for result in data.get("results", []):
                        props = result.get("properties", {})

                        # only consider those
                        # effective categories
                        sub_category = props.get("sub_category_en", "")
                        if sub_category in _effective_power_factor:
                            total_power = float(props.get("total_power", "0")[:-2]) # kW
                            effective_power = total_power * _effective_power_factor[sub_category] # kW

                            total_energy = effective_power * hours_per_year # kWh/year

                            productions[sub_category] += total_energy

                    return productions
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return defaultdict(float)

    # await needed GeoSession
    # for further computing
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=1.0)
    await provider.wait_until_ready()

    # create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # launch all tile energy
        # fetches concurrently
        tasks = [_fetch_infrastructure(session, tile) for tile in provider.sampled_tiles]
        sampled_energies = await asyncio.gather(*tasks)

    # aggregate all partial energies
    aggregated_energies = reduce(
        lambda res, d:
        {
            **res,
            **{
                k: res.get(k, 0) + v
                for k, v in d.items()
            }
        },
        sampled_energies,
        {}
    )

    # convert into GWh/year
    photovoltaic_energy_GWh = aggregated_energies.get("Photovoltaic", 0.0) / 1e6
    biomass_energy_GWh = aggregated_energies.get("Biomass", 0.0) / 1e6
    geothermal_energy_GWh = aggregated_energies.get("Geothermal energy", 0.0) / 1e6

    return (photovoltaic_energy_GWh, biomass_energy_GWh, geothermal_energy_GWh)

async def _fetch_thermal_networks_infrastructure(municipality_name: str) -> float:
    """Asynchronously fetches the thermal networks infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to fetch the data for.

    Returns:
        float: The total energy that can be delivered via thermal networks in GWh/year.
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
        "layers": "all:ch.bfe.thermische-netze",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_energy = 0 # MWh/year

                    # aggregate all features
                    for feature in data.get("results", []):
                        point = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if point.within(provider.geometry):
                            props = feature.get("properties", {})

                            energy_string = props.get("energy", "0")
                            energy = float(energy_string) if energy_string else 0 # MWh/a

                            total_energy += energy

                    total_energy_GWh = total_energy / 1e3

                    return total_energy_GWh
                else:
                    print(f"Could not retrieve thermal networks infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0

async def _fetch_sewage_treatment_potential(municipality_name: str) -> float:
    """Asynchronously fetches the sewage treatment (STEP) potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to fetch the data for.

    Returns:
        float: The potential heating energy from the sewage treatment infrastructure in GWh/year.
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
        "layers": "all:ch.bfe.fernwaerme-angebot",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_potential = 0 # MWh/year

                    # aggregate all features
                    for feature in data.get("results", []):
                        point = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if point.within(provider.geometry):
                            props = feature.get("properties", {})

                            potential = float(props.get("heatpotential", 0)) # MWh/year
                            total_potential += potential

                    total_potential_GWh = total_potential / 1e3

                    return total_potential_GWh
                else:
                    print(f"Could not retrieve sewage treatment infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0

async def _fetch_building_construction_periods(municipality_name: str) -> tuple[tuple[str, int], ...]:
    """Asynchronously fetches the building construction periods for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the effective infrastructure for.

    Returns:
        tuple[tuple[str, int], ...]: A tuple where each element is a tuple containing the construction period as a string and the number of buildings as an integer.
    """

    # RegBL building construction
    # period definition (from website)
    _gbaup_types = defaultdict(lambda: "Unknown", {
        "8011": "<1919",
        "8012": "1919-1945",
        "8013": "1946-1960",
        "8014": "1961-1970",
        "8015": "1971-1980",
        "8016": "1981-1985",
        "8017": "1986-1990",
        "8018": "1991-1995",
        "8019": "1996-2000",
        "8020": "2001-2005",
        "8021": "2006-2010",
        "8022": "2011-2015",
        "8023": "2016>"
    })

    async def _fetch_construction_periods(session, tile) -> defaultdict:
        minx, miny, maxx, maxy = tile.bounds
        geometry_str = f"{minx},{miny},{maxx},{maxy}"

        identify_url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometry": geometry_str,
            "geometryType": "esriGeometryEnvelope",
            "layers": "all:ch.bfs.gebaeude_wohnungs_register",
            "returnGeometry": "false",
            "tolerance": "0",
            "sr": "2056",
            "geometryFormat": "geojson"
        }

        try:
            # retrieve and aggregate partial
            # results from every single tile
            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with session.get(identify_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    construction_periods = defaultdict(int)

                    # sum up the potential for all features
                    for result in data.get("results", []):
                        props = result.get("properties", {})

                        gbaup_key = str(props.get("gbaup", "0"))
                        construction_period = _gbaup_types[gbaup_key]
                        construction_periods[construction_period] += 1

                    return construction_periods
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return defaultdict(int)

    # await needed GeoSession
    # for further computing
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=1.0)
    await provider.wait_until_ready()

    # create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # launch all tile construction
        # period fetches concurrently
        tasks = [_fetch_construction_periods(session, tile) for tile in provider.sampled_tiles]
        sampled_periods = await asyncio.gather(*tasks)

    # aggregate all partial results
    aggregated_periods = reduce(
        lambda res, e: {
            **res,
            e[0]: res.get(e[0], 0) +e[1]
        },
        reduce(lambda res, d: [*res, *d.items()], sampled_periods, []),
        defaultdict(int)
    )

    return tuple(aggregated_periods.items())

async def _fetch_heating_cooling_needs_industry(municipality_name: str) -> float:
    """Asynchronously fetches the heating/cooling energy needs for the industry in a given municipality.

    Args:
        municipality_name (str): The name of the municipality to fetch the data for.

    Returns:
        float: The heating/cooling energy needs for the industry in GWh/year.
    """
    # await GeoSession geometry
    # fetching for further computing ;
    # as no tiling is needed, await
    # most common GeoSession
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=0.3)
    await provider.wait_until_ready()

    minx, miny, maxx, maxy = provider.geometry.bounds
    geometry_str = f"{minx},{miny},{maxx},{maxy}"
    print(geometry_str)

    url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
    params = {
        "geometry": geometry_str,
        "geometryType": "esriGeometryEnvelope",
        "layers": "all:ch.bfe.fernwaerme-nachfrage_industrie",
        "returnGeometry": "true",
        "tolerance": "0",
        "sr": 2056,
        "geometryFormat": "geojson"
    }

    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_needs = 0 # MWh/year

                    # aggregate all features
                    for feature in data.get("results", []):
                        geometry = shape(feature.get("geometry"))

                        # intersect and clip features
                        # to municipality geometry
                        if geometry.intersects(provider.geometry):
                            clipped = geometry.intersection(provider.geometry)
                            # assume needs are proportional
                            # to the clipped area
                            factor = clipped.area / geometry.area

                            props = feature.get("properties", {})
                            needs = float(props.get("needindustry", 0)) * factor # MWh/year

                            total_needs += needs

                    total_needs_GWh = total_needs / 1e3

                    return total_needs_GWh
                else:
                    print(f"Could not retrieve heating/cooling needs: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0

async def _fetch_heating_cooling_needs_households(municipality_name: str) -> float:
    """Asynchronously fetches the heating/cooling needs for households in a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the effective infrastructure for.

    Returns:
        float: The heating/cooling energy needs for households in GWh/year.
    """
    async def _fetch_heating_cooling_needs(session, tile) -> float:
        minx, miny, maxx, maxy = tile.bounds
        geometry_str = f"{minx},{miny},{maxx},{maxy}"

        identify_url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometry": geometry_str,
            "geometryType": "esriGeometryEnvelope",
            "layers": "all:ch.bfe.fernwaerme-nachfrage_wohn_dienstleistungsgebaeude",
            "returnGeometry": "true",
            "tolerance": "0",
            "sr": "2056",
            "geometryFormat": "geojson"
        }

        try:
            # retrieve and aggregate partial
            # results from every single tile
            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with session.get(identify_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    total_needs = 0 # MWh/year

                    # sum up the needs for all features
                    for feature in data.get("results", []):
                        geometry = shape(feature.get("geometry"))
                        # consider clipped features only
                        if geometry.intersects(provider.geometry):
                            clipped = geometry.intersection(tile)
                            factor = clipped.area / geometry.area

                            props = feature.get("properties", {})
                            # only consider households needs
                            # and not commercial buildings
                            needs = float(props.get("needhome", 0)) * factor # MWh/year

                            total_needs += needs

                    total_needs_GWh = total_needs / 1e3

                    return total_needs_GWh
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return 0

    # await needed GeoSession
    # for further computing
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=1000, sampling_rate=1.0)
    await provider.wait_until_ready()

    # create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # launch all tile heating and
        # cooling needs fetches concurrently
        tasks = [_fetch_heating_cooling_needs(session, tile) for tile in provider.sampled_tiles]
        sampled_needs = await asyncio.gather(*tasks)

    # aggregate all partial results
    total_needs = sum(sampled_needs) # GWh/year

    return total_needs

async def _fetch_building_emissions_energy_source(municipality_name: str) -> tuple[dict[str, int], dict[str, int]]:
    """Asynchronously fetches the emissions and energy sources of buildings in a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the effective infrastructure for.

    Returns:
        tuple[dict[str, int], dict[str, int]]: The first dict maps from CO2 emissions range (in kg/m²) to the number of buildings, and the second dict maps from energy source to the number of buildings.
    """
    async def _fetch_emissions_energy_source(session, tile) -> tuple[defaultdict, defaultdict]:
        minx, miny, maxx, maxy = tile.bounds
        geometry_str = f"{minx},{miny},{maxx},{maxy}"

        identify_url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
        params = {
            "geometry": geometry_str,
            "geometryType": "esriGeometryEnvelope",
            "layers": "all:ch.bafu.klima-co2_ausstoss_gebaeude",
            "returnGeometry": "false",
            "tolerance": "0",
            "sr": "2056",
            "geometryFormat": "geojson"
        }

        try:
            # retrieve and aggregate partial
            # results from every single tile
            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with session.get(identify_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()

                    co2_ranges = defaultdict(int)
                    energy_sources = defaultdict(int)

                    # aggregate features
                    for result in data.get("results", []):
                        props = result.get("properties", {})

                        co2_ranges[props.get("co2_range", "")] += 1
                        # energy sources are only available
                        # in french, german and italian
                        energy_sources[props.get("genh1_fr", "")] += 1

                    return co2_ranges, energy_sources
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return defaultdict(int), defaultdict(int)

    # await needed GeoSession
    # for further computing
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=1.0)
    await provider.wait_until_ready()

    # create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # launch all tile heating and
        # cooling needs fetches concurrently
        tasks = [_fetch_emissions_energy_source(session, tile) for tile in provider.sampled_tiles]
        sampled_emissions_sources = await asyncio.gather(*tasks)

    # aggregate all partial results
    return reduce(
        lambda res, pair: (
            {
                **res[0],
                **{k: res[0].get(k, 0) + v for k, v in pair[0].items()}
            },
            {
                **res[1],
                **{k: res[1].get(k, 0) + v for k, v in pair[1].items()}
            }
        ),
        sampled_emissions_sources,
        ({}, {})
    )

async def _fetch_energy_needs(municipality_name: str, heuristic: Callable[[int], float] = lambda n: 3500 * n) -> float:
    """Asynchronously fetches the energy needs for a given municipality.

    The energy needs are estimated using an argument-given heuristic as it is not publicly available data.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.
        heuristic (Callable[[int], float]): The function used to estimate the energy needs based oon the number of primary households. By default, estimated to be 3500 kWh/year, per household.

    Returns:
        float: The estimated energy needs for the municipality in GWh/year.
    """
    # await GeoSession SFSO number
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=0.3)
    await provider.wait_until_sfso_ready()

    url = "https://api3.geo.admin.ch/rest/services/api/MapServer/find"
    params = {
        "layer": "ch.are.wohnungsinventar-zweitwohnungsanteil",
        "searchField": "gemeinde_nummer",
        "searchText": provider.municipality_sfso_number,
        "returnGeometry": "false",
        "sr": 2056,
    }
    try:
        headers = {"Referer": "dion.osmani@students.hevs.ch"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    attributes = data.get("results", [])[0].get("attributes", [])
                    # secondary households aren't considered
                    primary_households = attributes.get("zwg_3010", 0)
                    estimated_needs = heuristic(primary_households) # kWh/year

                    estimated_needs_GWh = estimated_needs / 1e6

                    return estimated_needs_GWh
                else:
                    print(f"Could not estimate electricity needs: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return 0

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
            async def wrapper(*args, **inner_kwargs) -> dict[str, tuple[str, Any]]:
                # write custom events
                # on custom tool call
                writer = get_stream_writer()
                clean_name = name.replace("_", " ")

                writer({"type": "tool_call", "name": clean_name, "isFinished": False})
                result = await func(*args, municipality_name=municipality_name, **inner_kwargs, **kwargs)
                writer({"type": "tool_call", "name": clean_name, "isFinished": True})

                return {
                    name: (layer_id, result)
                }

            super().__init__(
                coroutine=wrapper,
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

class LargeHydroPotentialTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_big_hydro_potential,
            name="large_hydro_potential",
            layer_id="ch.bfe.waermepotential-gewaesser",
            description="Returns the heating and cooling potential from large hydro sources (surface water) in the municipality in GWh/year. Useful for assessing renewable energy potential for heating and cooling at the municipal level.",
        )

class BiomassAvailabilityTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_available_biomass,
            name="available_biomass",
            # layer_id="ch.bfe.biomasse-nicht-verholzt",
            layer_id="",
            description="Returns the available woody and non-woody biomass in GWh for a given municipality. Useful for assessing renewable energy potential from biomass at the municipal level.",
        )

class HydropowerInfrastructureTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_hydropower_infrastructure,
            name="hydropower_infrastructure",
            layer_id="ch.bfe.statistik-wasserkraftanlagen",
            description="Returns the total hydropower electricity production in GWh/year for a given municipality. Useful for assessing the hydropower infrastructure and its contribution to renewable energy at the municipal level.",
        )

class WindTurbinesInfrastructureTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_wind_turbines_infrastructure,
            name="wind_turbines_infrastructure",
            layer_id="ch.bfe.windenergieanlagen",
            description="Returns the total wind turbines electricity production in GWh/year for a given municipality. Useful for assessing the wind energy infrastructure and its contribution to renewable energy at the municipal level.",
        )

class BiogasInfrastructureTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_biogas_infrastructure,
            name="biogas_infrastructure",
            layer_id="ch.bfe.biogasanlagen",
            description="Returns the total biogas production in GWh/year for a given municipality. Useful for assessing the biogas infrastructure and its contribution to renewable energy at the municipal level.",
        )

class IncinerationInfrastructureTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_incineration_infrastructure,
            name="incineration_infrastructure",
            layer_id="ch.bfe.kehrichtverbrennungsanlagen",
            description="Returns the total electricity and heating production in GWh/year from incineration infrastructure for a given municipality. Useful for assessing the contribution of waste incineration to renewable energy at the municipal level.",
        )

class ThermalNetworksInfrastructureTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_thermal_networks_infrastructure,
            name="thermal_networks_infrastructure",
            layer_id="ch.bfe.thermische-netze",
            description="Returns the total energy that can be delivered via thermal networks in GWh/year for a given municipality. Useful for assessing the thermal network infrastructure and its contribution to energy delivery at the municipal level.",
        )

class EffectiveInfrastructureTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_effective_infrastructure,
            name="effective_infrastructure",
            layer_id="ch.bfe.elektrizitaetsproduktionsanlagen",
            description="Returns the effective photovoltaic (PV), biomass, and geothermal energy production in GWh/year from various infrastructures for a given municipality. Useful for assessing the actual renewable energy production from these sources at the municipal level.",
        )

class SewageTreatmentPotentialTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_sewage_treatment_potential,
            name="sewage_treatment_potential",
            layer_id="ch.bfe.fernwaerme-angebot",
            description="Returns the potential heating energy from the sewage treatment infrastructure in GWh/year for a given municipality. Useful for assessing the sewage treatment (STEP) potential at the municipal level that may be injected into thermal networks.",
        )

class BuildingsConstructionPeriodsTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_building_construction_periods,
            name="building_construction_periods",
            layer_id="ch.bfs.gebaeude_wohnungs_register",
            description="Returns the building construction periods for a given municipality. Each result is a tuple of (construction period, number of buildings). Useful for understanding the age distribution of buildings in the municipality.",
        )

class HeatingCoolingNeedsIndustryTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_heating_cooling_needs_industry,
            name="heating_cooling_needs_industry",
            layer_id="ch.bfe.fernwaerme-nachfrage_industrie",
            description="Returns the heating/cooling energy needs for the industry in a given municipality in GWh/year. Useful for assessing the industrial energy demand at the municipal level.",
        )

class HeatingCoolingNeedsHouseholdsTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_heating_cooling_needs_households,
            name="heating_cooling_needs_households",
            layer_id="ch.bfe.fernwaerme-nachfrage_wohn_dienstleistungsgebaeude",
            description="Returns the heating/cooling energy needs for households in a given municipality in GWh/year. Useful for assessing the residential and service building energy demand at the municipal level.",
        )

class BuildingsEmissionEnergySourcesTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_building_emissions_energy_source,
            name="buildings_emission_energy_source",
            layer_id="ch.bafu.klima-co2_ausstoss_gebaeude",
            description="Returns the emissions and energy sources of buildings in a given municipality. The first dict maps from CO2 emissions range (in kg/m²) to the number of buildings, and the second dict maps from energy source to the number of buildings. Useful for assessing the climate impact and energy source distribution of buildings at the municipal level.",
        )

class EnergyNeedsTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_energy_needs, heuristic=lambda n_households: 3500.0 * n_households),
            name="energy_needs",
            layer_id="", # not defined as makes no sense to display the housing inventory layer
            description="Returns the estimated energy needs for a given municipality in GWh/year. The energy needs are estimated using a heuristic based on the number of primary households. Useful for assessing the total energy demand at the municipal level when detailed consumption data is not available.",
        )
