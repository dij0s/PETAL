import asyncio
import aiohttp

from shapely.geometry import shape, box
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union
import random
from numpy import floor, sqrt

from datetime import datetime
import json
import re

from typing import Optional, Any

class GeoSessionProvider:
    """A singleton class that manages geographical geometry data for a municipality.

    This class implements a singleton pattern and provides async methods for
    fetching and processing the geometry of a municipality with specific tiling
    resolution and sampling rate for different data processing requirements.

    Instances are uniquely identified by the combination of:
    - municipality_name
    - tiling_resolution
    - sampling_rate
    """

    _instances: dict[str, Any] = {}
    _lock = asyncio.Lock()

    def __new__(cls, municipality_name: str, tile_size: float, sampling_rate: float) -> "GeoSessionProvider":
        """Creates or returns a singleton instance based on the configuration parameters.

        Args:
            municipality_name (str): The municipality to create a session for.
            tile_size (float): The size of a single tile in width and height [m].
            sampling_rate (float): The sampling rate used to randomly select samples from the tiling pattern, which enables statistical estimation through aggregation [0.0-1.0].

        Returns:
            GeoSessionProvider: An instance of GeoSessionProvider with the specified configuration.
        """
        # unique key for configuration
        instance_key = f"{municipality_name}_{tile_size}_{sampling_rate}"

        if instance_key not in cls._instances:
            instance = super().__new__(cls)

            instance._initialized = False
            instance._ready_event = asyncio.Event()
            instance._sfso_ready_event = asyncio.Event()

            instance.municipality_name = municipality_name
            instance.tile_size = tile_size
            instance.sampling_rate = sampling_rate
            # create new instance from key
            cls._instances[instance_key] = instance

        return cls._instances[instance_key]

    @classmethod
    def get_or_create(
        cls,
        municipality_name: str,
        tile_size: float,
        sampling_rate: float
    ) -> "GeoSessionProvider":
        """
        Factory method to get or create a GeoSessionProvider instance and start initialization in the background.

        Args:
            municipality_name (str): The municipality to create a session for.
            tile_size (float): The size of a single tile in width and height [m].
            sampling_rate (float): The sampling rate used to randomly select samples from the tiling pattern, which enables statistical estimation through aggregation [0.0-1.0].

        Returns:
            GeoSessionProvider: An instance of GeoSessionProvider with the specified configuration.
        """
        instance = cls(municipality_name, tile_size, sampling_rate)
        # start initialisation in the background
        asyncio.create_task(instance.initialize())
        return instance

    async def initialize(self) -> Optional[bool]:
        """Initializes the session with geographical data based on configuration.

        Fetches and computes all necessary geographical data for the specified municipality with the given tiling resolution and
        sampling rate in a thread-safe way.

        Returns:
            Optional[bool]: True if the session was successfully initialized, False otherwise and None if session is already initialized.
        """
        async with self._lock:
            if not self._initialized:
                if not await self.fetch_geometry(self.municipality_name):
                    # prevent hanging by setting
                    # event to ready
                    self._ready_event.set()
                    return False
                # process geometry to remove unvalid areas
                if not await self.remove_unvalid_areas():
                    return False
                # compute tiles from the refined geometry
                self.total_tiles, self.sampled_tiles = await self.compute_tiles(self.tile_size, self.sampling_rate)
                self._initialized = True
                self._ready_event.set()

                return True
            else:
                return None

    async def wait_until_ready(self) -> None:
        """Waits until the session is fully initialized.

        Returns:
            None. Completes when the session is ready to use.
        """
        await self._ready_event.wait()

    async def wait_until_sfso_ready(self) -> None:
        """Wait until the SFSO municipality number is available.

        Returns:
            None. Completes when the SFSO number is ready to use.
        """
        await self._sfso_ready_event.wait()

    async def fetch_geometry(self, municipality_name: str) -> bool:
        """Fetches the geometry data for a municipality.

        Args:
            municipality_name (str): The municipality to fetch geometry for.

        Returns:
        bool: A boolean indicating if the geometry was successfully fetched for the municipality.
        """
        current_year = datetime.now().year

        try:
            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with aiohttp.ClientSession() as session:
                # find municipality feature
                search_url = "https://api3.geo.admin.ch/rest/services/api/SearchServer"
                search_params = {
                    "features": "ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill",
                    "type": "featuresearch",
                    "searchText": municipality_name,
                    "returnGeometry": "false",
                    "sr": "2056"
                }

                async with session.get(search_url, params=search_params, headers=headers) as response:
                    if response.status != 200:
                        print(f"SearchServer request failed: {response.status}")
                        return False

                    # request is successful
                    data = await response.json()
                    features = data.get("results", [])

                    if not features:
                        print(f"No features found for '{municipality_name}'")
                        return False

                    # match municipality name using regex pattern
                    target_name = municipality_name.lower().strip()
                    pattern = re.compile(rf"^{re.escape(target_name)}(?:\s|$)")

                    filtered = [
                        feature for feature in features
                        if pattern.match(feature.get("attrs", {}).get("label", "").lower())
                    ]
                    if not filtered:
                        print(f"No exact match found for '{municipality_name}'")
                        return False

                    # pick the most recent one from the
                    # previously matched features
                    matched_feature = max(
                        filtered,
                        key=lambda f: f.get("properties", {}).get("year", 0)
                    )

                    # request full geojson geometry
                    feature_id = matched_feature["id"]
                    detail_url = f"https://api3.geo.admin.ch/rest/services/api/MapServer/ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill/{feature_id}"
                    detail_params = {
                        "sr": 2056,
                        "geometryFormat": "geojson"
                    }

                    async with session.get(detail_url, params=detail_params, headers=headers) as detail_response:
                        if detail_response.status != 200:
                            print(f"Failed to retrieve detailed geometry: {detail_response.status}")
                            return False

                        detailed_feature = await detail_response.json()
                        geojson_feature = detailed_feature.get("feature", {}).get("geometry")

                        if not geojson_feature:
                            print("Geometry data is missing in the response")
                            return False

                        # store resulting feature in instance
                        self.geometry = shape(geojson_feature)
                        self.municipality_sfso_number = feature_id
                        self._sfso_ready_event.set()

                        return True

        except aiohttp.ClientError as e:
            print(f"HTTP error while fetching geometry for {municipality_name}: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Unexpected error while fetching geometry: {e}")

        return False

    async def remove_unvalid_areas(self) -> bool:
        """Removes unvalid areas from the municipality geometry.

        Processes the full municipality geometry to identify and remove areas that are not suitable
        for certain types of analysis (like forests, lakes, mountains).

        Returns:
            bool: a boolean indicating if valid areas were successfully processed.
        """
        try:
            # prepare bounding box string for api request
            bounding_box = self.geometry.bounds
            geometry_bounding_box = f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"

            headers = {"Referer": "dion.osmani@students.hevs.ch"}
            async with aiohttp.ClientSession() as session:
                # identify unvalid areas inside
                # the municipality bouding box
                url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
                params = {
                    "geometry": geometry_bounding_box,
                    "geometryType": "esriGeometryEnvelope",
                    "layers": "all:ch.swisstopo.vec200-landcover",
                    "tolerance": "0",
                    "geometryFormat": "geojson",
                    "sr": "2056"
                }

                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        print(f"identify api request failed: {response.status}")
                        return False

                    data = await response.json()

                    # filter out non-inhabited areas
                    # as we keep them for analysis
                    # and clip them to municipality
                    # shape
                    noninhabited_results = [
                        _res for _res in data.get("results", [])
                        if _res.get("properties", {}).get("objval") not in ["Siedl", "Stadtzentr"]
                    ]
                    # process non-inhabited areas
                    areas_to_remove = [
                        shape(result["geometry"]).intersection(self.geometry)
                        for result in noninhabited_results
                        if "geometry" in result and shape(result.get("geometry")).intersects(self.geometry)
                    ]
                    if areas_to_remove:
                        # union all the intersections
                        # of non-inhabited areas and
                        # subtract them from overall
                        # shape
                        # subtract from municipality shape
                        self.refined_geometry = self.geometry.difference(unary_union(areas_to_remove))
                    else:
                        self.refined_geometry = self.geometry

                    return True
        except aiohttp.ClientError as e:
            print(f"HTTP error while processing valid areas: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        except Exception as e:
            print(f"Unexpected error while processing valid areas: {e}")

        return False

    async def compute_tiles(self, tile_size: float, sampling_rate: float) -> tuple[int, list]:
        """Computes map tiles based on the provided geometry and tiling parameters.

        Applies a mesh grid to the municipality geometry and samples tiles based on the specified sampling rate. Only includes tiles that have a significant intersection with the valid/refined municipality area.

        Args:
            tile_size (float): The size of a single tile in width and height [m].
            sampling_rate (float): The sampling rate (0.0-1.0) for selecting tiles.

        Returns:
            tuple[int, list]: A tuple containing:
                - The total number of valid tiles
                - A list of sampled tiles based on the sampling rate
        """
        try:
            # minimum percentage of tile that
            # must intersect with valid area
            relative_valid_tile_area = max(0.3, 0.5 * sqrt(100 / (self.tile_size)))

            # refined bounding box
            r_minx, r_miny, r_maxx, r_maxy = self.refined_geometry.bounds

            # generate tiles intersecting
            # valid refined geometry
            tiles = []
            x = r_minx
            while x < r_maxx:
                y = r_miny
                while y < r_maxy:
                    # tile shape
                    tile = box(x, y, min(x + tile_size, r_maxx), min(y + tile_size, r_maxy))
                    if tile.intersects(self.refined_geometry):
                        # compute intersection and
                        # evaluate if valid
                        intersection = tile.intersection(self.refined_geometry)
                        area_ratio = intersection.area / tile.area
                        if area_ratio >= relative_valid_tile_area:
                            tiles.append(tile)
                    y += tile_size
                x += tile_size

            # ensure we have at least 1 tile if there are any tiles
            n = max(1, int(floor(len(tiles) * sampling_rate))) if sampling_rate > 0 else len(tiles)
            # randomly sample the tiles
            sampled_tiles = random.sample(tiles, min(n, len(tiles))) if tiles else []

            return len(tiles), sampled_tiles
        except Exception as e:
            print(f"Error computing tiles: {e}")
            return 0, []
