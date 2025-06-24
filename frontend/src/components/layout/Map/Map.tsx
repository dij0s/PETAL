import Feature from "ol/Feature";
import Map from "ol/Map";
import View from "ol/View";
import { defaults as defaultControls, ScaleLine } from "ol/control";
import MultiPolygon from "ol/geom/MultiPolygon";
import Polygon from "ol/geom/Polygon";
import LayerGroup from "ol/layer/Group";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import Crop from "ol-ext/filter/Crop";
import "ol/ol.css";
import "ol-ext/dist/ol-ext.css";
import { register } from "ol/proj/proj4";
import VectorSource from "ol/source/Vector";
import WMTS from "ol/source/WMTS";
import Fill from "ol/style/Fill";
import Style from "ol/style/Style";
import TileGrid from "ol/tilegrid/WMTS";
import proj4 from "proj4";
import { useEffect, useRef, useState } from "react";
import MapControls from "../../ui/MapControls";
import "./Map.css";

// LV95 (EPSG:2056) definition
proj4.defs(
  "EPSG:2056",
  "+proj=somerc +lat_0=46.95240555555556 +lon_0=7.439583333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs",
);
register(proj4);

const LV95_RESOLUTIONS = [
  4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250, 2000, 1750, 1500, 1250, 1000,
  750, 650, 500, 250, 100, 50, 20, 10, 5, 2.5, 2, 1.5, 1, 0.5, 0.25, 0.1,
];
const LV95_ORIGIN = [2420000, 1350000];
const LV95_MATRIX_IDS = LV95_RESOLUTIONS.map((_, idx) => idx.toString());

const swissImageBaseLayer = new TileLayer({
  source: new WMTS({
    url: "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/current/2056/{TileMatrix}/{TileCol}/{TileRow}.jpeg",
    layer: "ch.swisstopo.swissimage-product",
    matrixSet: "2056",
    format: "image/jpeg",
    projection: "EPSG:2056",
    tileGrid: new TileGrid({
      origin: LV95_ORIGIN,
      resolutions: LV95_RESOLUTIONS,
      matrixIds: LV95_MATRIX_IDS,
      tileSize: 256,
    }),
    style: "default",
    requestEncoding: "REST",
    attributions: "© swisstopo",
  }),
});

const pixelkarteFarbeBaseLayer = new TileLayer({
  source: new WMTS({
    url: "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/2056/{TileMatrix}/{TileCol}/{TileRow}.jpeg",
    layer: "ch.swisstopo.pixelkarte-farbe",
    matrixSet: "2056",
    format: "image/jpeg",
    projection: "EPSG:2056",
    tileGrid: new TileGrid({
      origin: LV95_ORIGIN,
      resolutions: LV95_RESOLUTIONS,
      matrixIds: LV95_MATRIX_IDS,
      tileSize: 256,
    }),
    style: "default",
    requestEncoding: "REST",
    attributions: "© swisstopo",
  }),
});

const baseLayerGroup = new LayerGroup({
  layers: [swissImageBaseLayer, pixelkarteFarbeBaseLayer],
});

interface MapProps {
  mapLayers: string[];
  focusedMunicipalitySFSO: number | null;
}

const fetchMunicipalityGeoJSON = async (sfso: number) => {
  const url = `https://api3.geo.admin.ch/rest/services/api/MapServer/ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill/${sfso}?returnGeometry=true&sr=2056&geometryFormat=geojson`;
  const response = await fetch(url);
  const data = await response.json();
  return data.feature;
};

const MapComponent = ({ mapLayers, focusedMunicipalitySFSO }: MapProps) => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapObj = useRef<Map | null>(null);

  const [baseLayer, setBaseLayer] = useState<"swissimage" | "pixelkarte">(
    "swissimage",
  );

  const handleToggleBaseLayer = () => {
    if (!mapObj.current) return;
    setBaseLayer((prev) => {
      const next = prev === "swissimage" ? "pixelkarte" : "swissimage";
      return next;
    });
  };

  const handleToggleExtraLayer = (layerName: string) => {
    if (!mapObj.current) return;
    const layers = mapObj.current.getLayers();
    // find layer and toggle visibility
    const targetLayer = layers
      .getArray()
      .find((layer) => layer.get("name") === layerName);
    if (targetLayer) {
      targetLayer.setVisible(!targetLayer.getVisible());
    }
  };

  // create and dispose map
  useEffect(() => {
    const view = new View({
      projection: "EPSG:2056",
      center: [2600000, 1200000],
      zoom: 8,
      minZoom: 0,
      maxZoom: 28,
    });

    const map = new Map({
      target: mapRef.current as HTMLDivElement,
      controls: defaultControls().extend([new ScaleLine({ units: "metric" })]),
      layers: [baseLayerGroup],
      view: view,
    });

    mapObj.current = map;

    return () => {
      map.setTarget(undefined);
      mapObj.current = null;
    };
  }, []);

  useEffect(() => {
    if (!mapObj.current) return;
    const layers = mapObj.current.getLayers();
    // remove all non-base layers
    // that are not in mapLayers
    layers
      .getArray()
      .filter(
        (layer) =>
          layer !== baseLayerGroup &&
          layer.get("name") !== "municipality" &&
          layer.get("name") !== "mask" &&
          !mapLayers.includes(layer.get("name")),
      )
      .forEach((layer) => layers.remove(layer));

    // add new layers that
    // are not present
    mapLayers.forEach((layer_name, index) => {
      const alreadyPresent = layers
        .getArray()
        .some((layer) => layer.get("name") === layer_name);
      if (!alreadyPresent) {
        const tileLayer = new TileLayer({
          source: new WMTS({
            url: `https://wmts.geo.admin.ch/1.0.0/${layer_name}/default/current/2056/{TileMatrix}/{TileCol}/{TileRow}.png`,
            layer: layer_name,
            matrixSet: "2056",
            format: "image/png",
            projection: "EPSG:2056",
            tileGrid: new TileGrid({
              origin: LV95_ORIGIN,
              resolutions: LV95_RESOLUTIONS,
              matrixIds: LV95_MATRIX_IDS,
              tileSize: 256,
            }),
            style: "default",
            requestEncoding: "REST",
            attributions: "© swisstopo",
          }),
        });
        tileLayer.set("name", layer_name);
        tileLayer.setOpacity(0.7);
        tileLayer.setZIndex(10 + index);
        layers.push(tileLayer);
      }
    });
  }, [mapLayers]);

  // mask around the municipality
  useEffect(() => {
    if (!mapObj.current) return;
    const layers = mapObj.current.getLayers();

    // remove previous municipality
    // and mask layers
    layers
      .getArray()
      .filter((layer) => layer.get("name") === "mask")
      .forEach((layer) => layers.remove(layer));

    if (!focusedMunicipalitySFSO) return;

    fetchMunicipalityGeoJSON(focusedMunicipalitySFSO).then((feature) => {
      // mask layer with hole for municipality
      // world is assumed to be Switzerland
      const worldExtent = [2420000, 1030000, 2900000, 1350000];
      const maskCoords = [
        [
          [worldExtent[0], worldExtent[1]],
          [worldExtent[0], worldExtent[3]],
          [worldExtent[2], worldExtent[3]],
          [worldExtent[2], worldExtent[1]],
          [worldExtent[0], worldExtent[1]],
        ],
      ];
      // use all outer rings as holes
      const holes = feature.geometry.coordinates.map(
        (poly: number[][][]) => poly[0],
      );
      const maskPolygon = new Polygon([maskCoords[0], ...holes]);
      const maskFeature = new Feature(maskPolygon);

      const maskLayer = new VectorLayer({
        source: new VectorSource({ features: [maskFeature] }),
        style: new Style({
          fill: new Fill({ color: "rgba(0,0,0,0.6)" }),
        }),
      });
      maskLayer.set("name", "mask");
      layers.push(maskLayer);

      const geometryType = feature.geometry.type;
      const geometryCoords = feature.geometry.coordinates;
      const municipalityGeometry =
        geometryType === "Polygon"
          ? new Polygon(geometryCoords)
          : new MultiPolygon(geometryCoords);

      // apply crop filter to
      // all extra layers
      if (mapObj.current) {
        mapObj.current
          .getLayers()
          .getArray()
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .forEach((layer: any) => {
            if (
              layer instanceof TileLayer &&
              layer.getSource() instanceof WMTS
            ) {
              // Remove previous crop filter if present
              // @ts-expect-error: _cropFilter is used for cleanup, not typed
              if (layer._cropFilter) {
                // @ts-expect-error: removeFilter is provided by ol-ext
                layer.removeFilter(layer._cropFilter);
                // @ts-expect-error: _cropFilter is used for cleanup, not typed
                delete layer._cropFilter;
              }
              // Add new crop filter
              const crop = new Crop({
                feature: new Feature({ geometry: municipalityGeometry }),
                inner: false,
                wrapX: false,
              });
              layer.addFilter(crop);
              // @ts-expect-error: _cropFilter is used for cleanup, not typed
              layer._cropFilter = crop;
            }
          });
      }

      // restrict navigation to municipality bbox
      if (mapObj.current) {
        const view = mapObj.current.getView();
        view.setProperties({ extent: feature.bbox });
        view.fit(feature.bbox, {
          size: mapObj.current.getSize(),
          duration: 500,
        });
      }
    });
  }, [focusedMunicipalitySFSO]);

  // handle base layer control
  useEffect(() => {
    swissImageBaseLayer.setVisible(baseLayer === "swissimage");
    pixelkarteFarbeBaseLayer.setVisible(baseLayer === "pixelkarte");
  }, [baseLayer]);

  return (
    <div className="map-wrapper" ref={mapRef}>
      <MapControls
        currentBaseLayer={baseLayer}
        handleBaseLayerToggle={handleToggleBaseLayer}
        extraLayers={mapLayers}
        handleExtraLayerToggle={handleToggleExtraLayer}
      />
    </div>
  );
};

export default MapComponent;
