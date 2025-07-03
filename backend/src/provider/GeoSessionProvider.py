import asyncio
import aiohttp
import json
import re
from numpy import floor, sqrt
import random

from typing import Optional, Any

from shapely.geometry import shape, box
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union

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
    _initialized: bool
    _ready_event: asyncio.Event
    _sfso_ready_event: asyncio.Event
    _residents_count_event: asyncio.Event

    municipality_name: str
    tile_size: float
    sampling_rate: float
    _with_residents_count: bool

    geometry: Any
    municipality_sfso_number: Any
    refined_geometry: Any
    total_tiles: int
    exploitable_surface: float
    sampled_tiles: list
    residents_count: int

    def __new__(cls, municipality_name: str, tile_size: float, sampling_rate: float, with_residents_count: bool = False) -> "GeoSessionProvider":
        """Creates or returns a singleton instance based on the configuration parameters.

        Args:
            municipality_name (str): The municipality to create a session for.
            tile_size (float): The size of a single tile in width and height [m].
            sampling_rate (float): The sampling rate used to randomly select samples from the tiling pattern, which enables statistical estimation through aggregation [0.0-1.0].
            with_residents_count (bool, optional): Whether to initialize with residents count. Defaults to False.

        Returns:
            GeoSessionProvider: An instance of GeoSessionProvider with the specified configuration.
        """
        # unique key for configuration
        instance_key = f"{municipality_name}_{tile_size}_{sampling_rate}_{with_residents_count}"

        if instance_key not in cls._instances:
            instance = super().__new__(cls)

            instance._initialized = False
            instance._ready_event = asyncio.Event()
            instance._sfso_ready_event = asyncio.Event()
            instance._residents_count_event = asyncio.Event()

            instance.municipality_name = municipality_name
            instance.tile_size = tile_size
            instance.sampling_rate = sampling_rate
            instance._with_residents_count = with_residents_count
            # create new instance from key
            cls._instances[instance_key] = instance

        return cls._instances[instance_key]

    @classmethod
    def get_or_create(
        cls,
        municipality_name: str,
        tile_size: float,
        sampling_rate: float,
        with_residents_count: bool = False
    ) -> "GeoSessionProvider":
        """
        Factory method to get or create a GeoSessionProvider instance and start initialization in the background.

        Args:
            municipality_name (str): The municipality to create a session for.
            tile_size (float): The size of a single tile in width and height [m].
            sampling_rate (float): The sampling rate used to randomly select samples from the tiling pattern, which enables statistical estimation through aggregation [0.0-1.0].
            with_residents_count (bool, optional): Whether to initialize with residents count. Defaults to False.

        Returns:
            GeoSessionProvider: An instance of GeoSessionProvider with the specified configuration.
        """
        instance = cls(municipality_name, tile_size, sampling_rate, with_residents_count)
        # start initialisation in the background
        asyncio.create_task(instance.initialize())
        return instance

    def _set_all_events(self):
        """Helper method to set all events in case of failure."""
        self._ready_event.set()
        self._sfso_ready_event.set()
        self._residents_count_event.set()

    async def initialize(self) -> Optional[bool]:
        """
        Initializes the session with geographical data based on configuration.

        Fetches and computes all necessary geographical data for the specified municipality with the given tiling resolution and
        sampling rate in a thread-safe way.

        Returns:
            Optional[bool]: True if the session was successfully initialized, False otherwise and None if session is already initialized.
        """
        async with self._lock:
            if not self._initialized:
                # when a single method fails
                # during the initialization
                # the global flag is set to
                # False and then evaluated
                # inside the single events
                # getters to raise
                try:
                    # fetch geometry first and
                    # set SFSO event to ready if
                    # successful
                    if not await self.fetch_geometry(self.municipality_name):
                        self._set_all_events()
                        return False
                    self._sfso_ready_event.set()

                    # fetch and process geometry
                    if not await self.remove_unvalid_areas():
                        self._set_all_events()
                        return False

                    # compute tiling on top of
                    # processed municipal geometry
                    self.total_tiles, self.exploitable_surface, self.sampled_tiles = await self.compute_tiles(
                        self.tile_size, self.sampling_rate)

                    # fetch residents count
                    if self._with_residents_count:
                        await self.fetch_residents_count()
                    self._residents_count_event.set()

                    self._initialized = True
                    self._ready_event.set()
                    return True
                except Exception as e:
                    print(f"Exception: {e}")
                    self._set_all_events()
                    return False
            else:
                return None


    async def wait_until_ready(self) -> None:
        """Waits until the session is fully initialized.

        Returns:
            None. Completes when the session is ready to use.

        Raises:
            RuntimeError: If initialization failed.
        """
        await self._ready_event.wait()
        if not self._initialized:
            raise RuntimeError("Session initialization failed")

    async def wait_until_sfso_ready(self) -> None:
        """Wait until the SFSO municipality number is available.

        Returns:
            None. Completes when the SFSO number is ready to use.

        Raises:
            RuntimeError: If initialization failed.
        """
        await self._sfso_ready_event.wait()
        # ensure session is ready
        # to avoid race conditions
        await self.wait_until_ready()

    async def wait_until_residents_count_ready(self) -> None:
        """Wait until the residents count is available.

        Returns:
            None. Completes when the residents count is ready to use.

        Raises:
            RuntimeError: If initialization failed.
        """
        await self._residents_count_event.wait()
        # ensure session is ready
        # to avoid race conditions
        await self.wait_until_ready()

    async def fetch_geometry(self, municipality_name: str) -> bool:
        """Fetches the geometry data for a municipality.

        Args:
            municipality_name (str): The municipality to fetch geometry for.

        Returns:
        bool: A boolean indicating if the geometry was successfully fetched for the municipality.
        """

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

    async def compute_tiles(self, tile_size: float, sampling_rate: float) -> tuple[int, float, list]:
        """Computes map tiles based on the provided geometry and tiling parameters.

        Applies a mesh grid to the municipality geometry and samples tiles based on the specified sampling rate. Only includes tiles that have a significant intersection with the valid/refined municipality area.

        Args:
            tile_size (float): The size of a single tile in width and height [m].
            sampling_rate (float): The sampling rate (0.0-1.0) for selecting tiles.

        Returns:
            tuple[int, float, list]: A tuple containing:
                - The total number of valid tiles
                - The exploitable surface (from the original tiling)
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

            # ensure we have at least
            # a single sampled tile
            n = max(1, int(floor(len(tiles) * sampling_rate))) if sampling_rate > 0 else len(tiles)
            # randomly sample the tiles
            sampled_tiles = random.sample(tiles, min(n, len(tiles))) if tiles else []
            # compute total exploitable
            # surface from original tiling
            # unprocessed area is in m² as
            # the tile size as provided in
            # the signature is assumes to
            # be in meters
            exploitable_surface = sum(map(lambda t: t.area, tiles)) / 1e4 # ha

            return len(tiles), exploitable_surface, sampled_tiles
        except Exception as e:
            print(f"Error computing tiles: {e}")
            return 0, 0, []

    async def fetch_residents_count(self) -> bool:
        """Asynchronously fetches the residents count for the municipality.

        Returns:
            bool: True if the residents count was successfully fetched, False otherwise.
        """

        async def _fetch_residents(session, tile) -> float:
            minx, miny, maxx, maxy = tile.bounds
            geometry_str = f"{minx},{miny},{maxx},{maxy}"

            identify_url = "https://api3.geo.admin.ch/rest/services/api/MapServer/identify"
            params = {
                "geometry": geometry_str,
                "geometryType": "esriGeometryEnvelope",
                "layers": "all:ch.bfs.volkszaehlung-bevoelkerungsstatistik_einwohner",
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

                        residents_count = 0
                        # get most recent result
                        for result in data.get("results", []):
                            props = result.get("properties", {})

                            year = props.get("i_year", 0)
                            if year == 2023:
                                geometry = shape(result.get("geometry"))

                                # intersect and clip features
                                # to current tile
                                if geometry.intersects(tile):
                                    clipped = geometry.intersection(tile)
                                    factor = clipped.area / geometry.area

                                    residents = props.get("number", 0)
                                    residents_count += (residents * factor)

                        return residents_count
                    else:
                        print(f"Failed request for tile, status: {response.status}")
            except Exception as e:
                print(f"Error for tile at {geometry_str}: {e}")

            return 0

        # create an aiohttp session for all requests
        async with aiohttp.ClientSession() as session:
            # launch all tile construction
            # period fetches concurrently
            tasks = [_fetch_residents(session, tile) for tile in self.sampled_tiles]
            sampled_counts = await asyncio.gather(*tasks)

        # aggregate all partial results
        self.residents_count = int(sum(sampled_counts))
        return True
