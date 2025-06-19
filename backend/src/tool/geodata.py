"""Exposes tools to query GeoContext-related data."""

import aiohttp
import asyncio

from shapely.geometry import shape

import re
import numpy as np
from scipy.stats import t

from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer

from typing import Callable, Any, Literal
from functools import partial, reduce
from collections import defaultdict

from provider.GeoSessionProvider import GeoSessionProvider

async def _fetch_solar_potential_roofing(municipality_name: str, confidence_level=0.8) -> tuple[str, str]:
    """Asynchronously estimates the roofing solar potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate solar potential for.
        confidence_level (float, optional): The confidence level for the margin of error (default is 0.8).

    Returns:
        tuple[str, str]: A tuple containing the estimated energy (electricity) in GWh/year and the estimated energy (heating) in GWh/year.
    """
    # DETEC states that 70%
    # of roofings' area is
    # considered as covered
    # for the computations
    area_coverage_factor = 0.7

    async def _fetch_solar_potential(session, tile) -> tuple[float, float]:
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
                    tile_energy_electricity = 0  # kWh/year
                    tile_energy_heating = 0  # kWh/year

                    # sum up the potential for all features
                    # if they are valid per DETEC rules :
                    # only consider areas >= 10m² which
                    # have a yield class >= 3 (good, very good, excellent)
                    for result in data.get("results", []):
                        attributes = result.get("attributes", {})

                        area = attributes.get("flaeche", 0) # m²
                        yield_class = attributes.get("klasse", 0)
                        if area >= 10.0 and int(yield_class) >= 3:
                            energy_electricity = attributes.get("stromertrag", 0) # kWh/year
                            energy_heating = attributes.get("waermeertrag", 0) # kWh/year
                            tile_energy_electricity += energy_electricity
                            tile_energy_heating += energy_heating

                    # include area coverage
                    return tile_energy_electricity * area_coverage_factor, tile_energy_heating * area_coverage_factor
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return 0, 0

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
    electricity, heating = reduce(lambda res, p: ([*res[0], p[0]], [*res[1], p[1]]), sampled_potentials, ([], []))

    n_electricity = len(electricity)
    n_heating = len(heating)
    mean_tile_value_electricity = np.mean(electricity)
    mean_tile_value_heating = np.mean(heating)
    std_dev_electricity = np.std(electricity, ddof=1)
    std_dev_heating = np.std(heating, ddof=1)

    # compute t-value for the
    # given confidence level
    t_value = t.ppf((1 + confidence_level) / 2, df=n_electricity - 1)
    total_estimate_electricity = mean_tile_value_electricity * provider.total_tiles # kWh/year
    total_estimate_heating = mean_tile_value_heating * provider.total_tiles # kWh/year
    margin_electricity = t_value * (std_dev_electricity / np.sqrt(n_electricity)) * provider.total_tiles # kWh/year
    margin_heating = t_value * (std_dev_heating / np.sqrt(n_heating)) * provider.total_tiles # kWh/year

    # convert to GWh/year
    total_estimate_electricity_gwh = total_estimate_electricity / 1e6
    total_estimate_heating_gwh = total_estimate_heating / 1e6
    margin_electricity_gwh = margin_electricity / 1e6
    margin_heating_gwh = margin_heating / 1e6
    # handle full aggregation
    if margin_electricity_gwh == float("+inf"):
        return f"{total_estimate_electricity_gwh:.2f}", f"{total_estimate_heating_gwh:.2f}"

    return f"{total_estimate_electricity_gwh:.2f} ± {margin_electricity_gwh:.2f}", f"{total_estimate_heating_gwh:.2f} ± {margin_heating_gwh:.2f}"

async def _fetch_solar_potential_facades(municipality_name: str, confidence_level=0.8) -> tuple[str, str]:
    """Asynchronously estimates the facades solar potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate solar potential for.
        confidence_level (float, optional): The confidence level for the margin of error (default is 0.8).

    Returns:
        tuple[str, str]: A tuple containing the estimated energy (electricity) in GWh/year and the estimated energy (heating) in GWh/year.
    """
    # DETEC states that 45-60%
    # of facades' area is
    # considered as covered
    # for the computations
    area_coverage_factor = 0.525

    async def _fetch_solar_potential(session, tile) -> tuple[float, float]:
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
                    tile_energy_electricity = 0  # kWh/year
                    tile_energy_heating = 0  # kWh/year

                    # sum up the potential for all features
                    # if they are valid per DETEC rules :
                    # only consider areas >= 20m² which
                    # have a yield class >= 2
                    for result in data.get("results", []):
                        attributes = result.get("attributes", {})

                        area = attributes.get("flaeche", 0) # m²
                        yield_class = attributes.get("klasse", 0)
                        if area >= 20.0 and int(yield_class) >= 2:
                            energy_electricity = attributes.get("stromertrag", 0) # kWh/year
                            energy_heating = attributes.get("waermeertrag", 0) # kWh/year
                            tile_energy_electricity += energy_electricity
                            tile_energy_heating += energy_heating

                    # include area coverage
                    return tile_energy_electricity * area_coverage_factor, tile_energy_heating * area_coverage_factor
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return 0, 0

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
    electricity, heating = reduce(lambda res, p: ([*res[0], p[0]], [*res[1], p[1]]), sampled_potentials, ([], []))

    n_electricity = len(electricity)
    n_heating = len(heating)
    mean_tile_value_electricity = np.mean(electricity)
    mean_tile_value_heating = np.mean(heating)
    std_dev_electricity = np.std(electricity, ddof=1)
    std_dev_heating = np.std(heating, ddof=1)

    # compute t-value for the
    # given confidence level
    t_value = t.ppf((1 + confidence_level) / 2, df=n_electricity - 1)
    total_estimate_electricity = mean_tile_value_electricity * provider.total_tiles # kWh/year
    total_estimate_heating = mean_tile_value_heating * provider.total_tiles # kWh/year
    margin_electricity = t_value * (std_dev_electricity / np.sqrt(n_electricity)) * provider.total_tiles # kWh/year
    margin_heating = t_value * (std_dev_heating / np.sqrt(n_heating)) * provider.total_tiles # kWh/year

    # convert to GWh/year
    total_estimate_electricity_gwh = total_estimate_electricity / 1e6
    total_estimate_heating_gwh = total_estimate_heating / 1e6
    margin_electricity_gwh = margin_electricity / 1e6
    margin_heating_gwh = margin_heating / 1e6
    # handle full aggregation
    if margin_electricity_gwh == float("+inf"):
        return f"{total_estimate_electricity_gwh:.2f}", f"{total_estimate_heating_gwh:.2f}"

    return f"{total_estimate_electricity_gwh:.2f} ± {margin_electricity_gwh:.2f}", f"{total_estimate_heating_gwh:.2f} ± {margin_heating_gwh:.2f}"

async def _fetch_small_hydro_potential(municipality_name: str, efficiency: float = 0.5) -> str:
    """Asynchronously fetches the small hydroelectricity potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate small hydroelectricity for.
        efficiency (float, optional): The hydroelectricity production efficiency used to compute the total potential (default is 0.5).

    Returns:
        str: The estimated small hydroelectricity potential in GWh/year.
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
                                length = props.get("laenge", 0) # m
                                kw_per_meter = props.get("kwprometer", 0)
                                # rescale based on clipped portion
                                factor = clipped.length / geometry.length
                                potential = kw_per_meter * length * factor # kW

                                total_potential += potential

                    # convert to GWh/year
                    hours_per_year = 24 * 365
                    total_potential_GWh = total_potential * hours_per_year * efficiency / 1e6

                    return f"{total_potential_GWh:.2f}"
                else:
                    print(f"Could not retrieve small hydroelectricity data: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0"

async def _fetch_big_hydro_potential(municipality_name: str) -> tuple[str, str]:
    """Asynchronously fetches the big hydro heating/cooling potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate big hydro for.

    Returns:
        tuple[str, str]: A tuple containing the heating and cooling potential in GWh/year.
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
                    # the absolute value of the heating
                    # potential is returned as it is represents
                    # the amount of energy we may be able to
                    # extract from the water body whereas the
                    # positive cooling potential represents
                    # the amount of energy we may reinject
                    # into the water body for cooling purposes
                    return f"{abs(heating_potential_GWh):.2f}", f"{cooling_potential_GWh:.2f}"
                else:
                    print(f"Could not retrieve big hydroelectricity data: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0", "0"

async def _fetch_available_biomass(municipality_name: str) -> tuple[str, str]:
    """Asynchronously fetches the available biomass for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.

    Returns:
        tuple[str, str]: A tuple containing the available woody and non-woody biomass in GWh.
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

                    return f"{woody_biomass_GWh:.2f}", f"{non_woody_biomass_GWh:.2f}"
                else:
                    print(f"Could not available biomass: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0", "0"

async def _fetch_hydropower_infrastructure(municipality_name: str) -> str:
    """Asynchronously fetches the hydropower infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.

    Returns:
        str: The total hydropower electricity production in GWh/year.
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

                    return f"{total_production:.2f}"
                else:
                    print(f"Could not retrieve hydropower infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0"

async def _fetch_wind_turbines_infrastructure(municipality_name: str) -> str:
    """Asynchronously fetches the wind turbines infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.

    Returns:
        str: The total wind turbines electricity production in GWh/year.
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

                    return f"{total_production_GWh:.2}"
                else:
                    print(f"Could not retrieve wind turbine infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0"

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

async def _fetch_effective_infrastructure(municipality_name: str, energy_agent: Literal["Photovoltaic", "Biomass", "Geothermal energy"], production_yield: float) -> tuple[str, str, str]:
    """Asynchronously fetches the effective photovoltaic (PV), biomass and geothermal energy from various infrastructure.

    Args:
        municipality_name (str): The name of the municipality to estimate the effective infrastructure for.$
        energy_agent (str): The type of energy agent to estimate (e.g., "Photovoltaic", "Biomass", "Geothermal energy").
        production_yield (float): The production yield factor to apply for the energy agent.

    Returns:
        tuple[str, str, str]: A tuple containing the effective photovoltaic, biomass and geothermal power in GWh/year.
    """

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
                        if sub_category == energy_agent:
                            total_power = float(props.get("total_power", "0")[:-2]) # kW
                            effective_power = total_power * production_yield # kW

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

    return (f"{photovoltaic_energy_GWh:.2f}", f"{biomass_energy_GWh:.2f}", f"{geothermal_energy_GWh:.2f}")

async def _fetch_thermal_networks_infrastructure(municipality_name: str) -> str:
    """Asynchronously fetches the thermal networks infrastructure for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to fetch the data for.

    Returns:
        str: The total energy that can be delivered via thermal networks in GWh/year.
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

                            energy_str = props.get("energy", None)
                            energy = float(energy_str if energy_str is not None else 0) # MWh/a

                            total_energy += energy

                    total_energy_GWh = total_energy / 1e3

                    return f"{total_energy_GWh:.2f}"
                else:
                    print(f"Could not retrieve thermal networks infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0"

async def _fetch_wastewater_treatment_potential(municipality_name: str) -> str:
    """Asynchronously fetches the wastewater treatment (STEP) potential for a given municipality.

    Args:
        municipality_name (str): The name of the municipality to fetch the data for.

    Returns:
        str: The potential heating energy from the wastewater treatment infrastructure in GWh/year.
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

                    return f"{total_potential_GWh:.2f}"
                else:
                    print(f"Could not retrieve sewage treatment infrastructure: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0"

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

async def _fetch_heating_cooling_needs_industry(municipality_name: str) -> str:
    """Asynchronously fetches the heating/cooling energy needs for the industry in a given municipality.

    Args:
        municipality_name (str): The name of the municipality to fetch the data for.

    Returns:
        str: The heating/cooling energy needs for the industry in GWh/year.
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

                    return f"{total_needs_GWh:.2f}"
                else:
                    print(f"Could not retrieve heating/cooling needs: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0"

async def _fetch_heating_cooling_needs(municipality_name: str) -> tuple[str, str]:
    """Asynchronously fetches the heating/cooling for households and services in a given municipality.

    Args:
        municipality_name (str): The name of the municipality to estimate the effective infrastructure for.

    Returns:
        tuple[str, str]: The heating/cooling energy needsfor households and services in GWh/year.
    """

    async def _fetch_needs(session, tile) -> tuple[float, float]:
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
                    total_needs_households = 0 # MWh/year
                    total_needs_services = 0 # MWh/year

                    # sum up the needs for all features
                    for feature in data.get("results", []):
                        geometry = shape(feature.get("geometry"))
                        # consider clipped features only
                        if geometry.intersects(provider.geometry):
                            clipped = geometry.intersection(tile)
                            factor = clipped.area / geometry.area

                            props = feature.get("properties", {})

                            needs_households = float(props.get("needhome", 0)) * factor # MWh/year
                            needs_services = float(props.get("needservice", 0)) * factor # MWh/year

                            total_needs_households += needs_households
                            total_needs_services += needs_services

                    total_needs_households_GWh = total_needs_households / 1e3
                    total_needs_services_GWh = total_needs_services / 1e3

                    return total_needs_households_GWh, total_needs_services_GWh
                else:
                    print(f"Failed request for tile, status: {response.status}")
        except Exception as e:
            print(f"Error for tile at {geometry_str}: {e}")

        return 0, 0

    # await needed GeoSession
    # for further computing
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=1000, sampling_rate=1.0)
    await provider.wait_until_ready()

    # create an aiohttp session for all requests
    async with aiohttp.ClientSession() as session:
        # launch all tile heating and
        # cooling needs fetches concurrently
        tasks = [_fetch_needs(session, tile) for tile in provider.sampled_tiles]
        sampled_needs = await asyncio.gather(*tasks)

    # aggregate all partial results
    total_needs_households, total_needs_services = reduce(
        lambda res, needs: (res[0] + needs[0], res[1] + needs[1]),
        sampled_needs,
        (0, 0)
    ) # GWh/year

    return f"{total_needs_households:.2f}", f"{total_needs_services:.2f}"

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

async def _fetch_energy_needs(municipality_name: str, heuristic: Callable[[int], float] = lambda n: 3500 * n) -> str:
    """Asynchronously fetches the energy needs for a given municipality.

    The energy needs are estimated using an argument-given heuristic as it is not publicly available data.

    Args:
        municipality_name (str): The name of the municipality to estimate the biomass for.
        heuristic (Callable[[int], float]): The function used to estimate the energy needs based oon the number of primary households. By default, estimated to be 3500 kWh/year, per household.

    Returns:
        str: The estimated energy needs for the municipality in GWh/year.
    """
    # await GeoSession SFSO number
    provider = GeoSessionProvider.get_or_create(municipality_name=municipality_name, tile_size=100, sampling_rate=0.3)
    await provider.wait_until_sfso_ready()

    url = "https://api3.geo.admin.ch/rest/services/api/MapServer/find"
    params = {
        "layer": "ch.are.wohnungsinventar-zweitwohnungsanteil",
        "searchField": "gemeinde_nummer",
        "searchText": provider.municipality_sfso_number,
        "contains": "false",
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

                    return f"{estimated_needs_GWh:.2f}"
                else:
                    print(f"Could not estimate electricity needs: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

    return "0"

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
                args_schema=None, # type: ignore
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
            description="Potential electrical/thermal source distributed: Solar energy generation from rooftop installations. Returns (photovoltaic_electricity, solar_thermal_heating) where photovoltaic_electricity is electricity generation potential in GWh/year and solar_thermal_heating is heating potential in GWh/year.",
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
            description="Potential electrical/thermal source distributed: Solar energy generation from facade installations. Returns (photovoltaic_electricity, solar_thermal_heating) where photovoltaic_electricity is electricity generation potential in GWh/year and solar_thermal_heating is heating potential in GWh/year.",
        )

class SmallHydroPotentialTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_small_hydro_potential, efficiency=0.5),
            name="small_hydro_potential",
            layer_id="ch.bfe.kleinwasserkraftpotentiale",
            description="Potential electrical source distributed: Hydroelectric generation from small water sources. Returns theoretical electricity generation potential in GWh/year. Assumes 50% efficiency factor to account for technical, ecological, economic and legal constraints.",
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
            # layer_id="ch.bfe.waermepotential-gewaesser", not supported as of now, WMS
            layer_id="",
            description="Potential thermal source centralized: Thermal energy extraction from large water bodies. Returns (heating_potential, cooling_potential) where heating_potential is heat extraction capacity in GWh/year and cooling_potential is heat discharge capacity in GWh/year. Includes lakes and rivers.",
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
            layer_id="",
            description="Potential chemical source distributed: Biomass energy availability from organic waste sources. Returns (woody_biomass, non_woody_biomass) where woody_biomass is available wood-based energy in GWh/year and non_woody_biomass is available organic waste energy in GWh/year.",
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
            description="Infrastructure electrical source centralized: Hydroelectric generation from operational plants. Returns actual electricity generation in GWh/year. Includes only plants with minimum 300 kW output capacity.",
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
            description="Infrastructure electrical source centralized: Wind energy generation from operational turbines. Returns actual electricity generation in GWh/year. Based on most recent annual production data from plant operators.",
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
            description="Infrastructure electrical/thermal source centralized: Biogas energy generation from operational plants. Returns total combined electricity and heat generation in GWh/year. Includes plants processing household, industrial and agricultural organic waste.",
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
            description="Infrastructure electrical/thermal source centralized: Waste incineration energy generation from operational plants. Returns (electricity_generation, heat_generation) where electricity_generation is electrical output in GWh/year and heat_generation is thermal output in GWh/year.",
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
            layer_id="",
            description="Infrastructure thermal network centralized: Thermal energy distribution through operational networks. Returns total deliverable energy through thermal networks in GWh/year. Includes district heating, local heating and district cooling systems.",
        )

class OperationalSolarInfrastructureTool(GeoDataTool):
    def __init__(self, municipality_name: str):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_effective_infrastructure, energy_agent="Photovoltaic", production_yield=0.35),
            name="operational_solar_infrastructure",
            layer_id="",
            description="Infrastructure electrical source distributed: Solar photovoltaic generation from operational installations. Returns actual electricity production from operational solar panels in GWh/year.",
        )

class OperationalBiomassInfrastructureTool(GeoDataTool):
    def __init__(self, municipality_name: str):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_effective_infrastructure, energy_agent="Biomass", production_yield=1.0),
            name="operational_biomass_infrastructure",
            layer_id="",
            description="Infrastructure electrical source centralized: Biomass electricity generation from operational plants. Returns actual electricity production from operational biomass facilities in GWh/year.",
        )

class OperationalGeothermalInfrastructureTool(GeoDataTool):
    def __init__(self, municipality_name: str):
        super().__init__(
            municipality_name=municipality_name,
            func=partial(_fetch_effective_infrastructure, energy_agent="Geothermal energy", production_yield=1.0),
            name="operational_geothermal_infrastructure",
            layer_id="",
            description="Infrastructure electrical source centralized: Deep geothermal electricity generation from operational plants. Returns actual electricity production from deep operational geothermal facilities in GWh/year.",
        )

class WastewaterTreatmentPotentialTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_wastewater_treatment_potential,
            name="wastewater_treatment_potential",
            layer_id="ch.bfe.fernwaerme-angebot",
            description="Potential thermal source centralized: Thermal energy recovery from wastewater treatment facilities. Returns recoverable heat energy in GWh/year. Heat extractable via heat pumps for district heating networks.",
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
            description="Buildings infrastructure analysis: Construction period distribution of building stock. Returns list of (construction_period, building_count) tuples showing temporal distribution of existing buildings.",
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
            description="Consumption thermal sink buildings: Heating and cooling energy demand for industrial facilities. Returns thermal energy requirements in GWh/year. Used for identifying areas suitable for thermal network development.",
        )

class HeatingCoolingNeedsHouseholdsServicesTool(GeoDataTool):
    def __init__(
        self,
        municipality_name: str,
    ):
        super().__init__(
            municipality_name=municipality_name,
            func=_fetch_heating_cooling_needs,
            name="heating_cooling_needs_households_services",
            layer_id="ch.bfe.fernwaerme-nachfrage_wohn_dienstleistungsgebaeude",
            description="Consumption thermal sink buildings: Heating and cooling energy demand for residential and service buildings. Returns (household_demand, service_building_demand) where household_demand is residential thermal needs in GWh/year and service_building_demand is administrative building thermal needs in GWh/year.",
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
            description="Buildings energy analysis: Energy source utilization and emission characteristics. Returns (emission_distribution, energy_source_distribution) where emission_distribution maps CO2 emission ranges in kg/m² to building counts and energy_source_distribution maps energy sources to building counts.",
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
            layer_id="",
            description="Consumption electrical sink buildings: Electricity energy demand for household operations. Returns estimated electrical energy requirements in GWh/year. Excludes heating and cooling energy, covers only everyday electricity usage.",
        )
