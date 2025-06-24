import Feature from "ol/Feature";
import Map from "ol/Map";
import View from "ol/View";
import { defaults as defaultControls, ScaleLine } from "ol/control";
import MultiPolygon from "ol/geom/MultiPolygon";
import Polygon from "ol/geom/Polygon";
import LayerGroup from "ol/layer/Group";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import "ol/ol.css";
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

const clipLayerToPolygonWithPrerender = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  layer: TileLayer<any> & {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _clipHandler?: (event: any) => void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _restoreHandler?: (event: any) => void;
  },
  map: Map,
  polygonFeature: Feature<Polygon | MultiPolygon> | null,
) => {
  // remove previous listeners if any
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  if (layer._clipHandler) layer.un("prerender" as any, layer._clipHandler);
  if (layer._restoreHandler)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    layer.un("postrender" as any, layer._restoreHandler);

  if (!polygonFeature) return;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  layer._clipHandler = function (event: any) {
    const ctx = event.context;
    ctx.save();

    const geom = polygonFeature.getGeometry();
    if (!geom) {
      ctx.restore();
      return;
    }

    ctx.beginPath();

    if (geom.getType() === "Polygon") {
      const rings = (geom as Polygon).getCoordinates();
      rings.forEach((ring: number[][]) => {
        ring.forEach((coord: number[], i: number) => {
          const pixel = map.getPixelFromCoordinate(coord);
          if (i === 0) ctx.moveTo(pixel[0], pixel[1]);
          else ctx.lineTo(pixel[0], pixel[1]);
        });
        ctx.closePath();
      });
    } else if (geom.getType() === "MultiPolygon") {
      const polygons = (geom as MultiPolygon).getCoordinates();
      polygons.forEach((rings: number[][][]) => {
        rings.forEach((ring: number[][]) => {
          ring.forEach((coord: number[], i: number) => {
            const pixel = map.getPixelFromCoordinate(coord);
            if (i === 0) ctx.moveTo(pixel[0], pixel[1]);
            else ctx.lineTo(pixel[0], pixel[1]);
          });
          ctx.closePath();
        });
      });
    }

    ctx.clip();
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  layer._restoreHandler = function (event: any) {
    event.context.restore();
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  layer.on("prerender" as any, layer._clipHandler);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  layer.on("postrender" as any, layer._restoreHandler);
};

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

    // remove any previous clip
    // handlers from layers
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    layers.getArray().forEach((layer: any) => {
      if (
        layer &&
        typeof layer === "object" &&
        typeof layer.un === "function"
      ) {
        if ("_clipHandler" in layer) {
          if (layer._clipHandler) {
            layer.un("prerender", layer._clipHandler);
          }
          delete layer._clipHandler;
        }
        if ("_restoreHandler" in layer) {
          if (layer._restoreHandler) {
            layer.un("postrender", layer._restoreHandler);
          }
          delete layer._restoreHandler;
        }
      }
    });

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
      const municipalityFeature =
        geometryType === "Polygon"
          ? new Feature({ geometry: new Polygon(geometryCoords) })
          : new Feature({ geometry: new MultiPolygon(geometryCoords) });

      // apply clipping to all WMTS
      // layers except base group
      if (mapObj.current) {
        mapObj.current
          .getLayers()
          .getArray()
          .forEach((layer) => {
            if (
              layer instanceof TileLayer &&
              layer.getSource() instanceof WMTS
            ) {
              clipLayerToPolygonWithPrerender(
                layer,
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                mapObj.current as any,
                municipalityFeature,
              );
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
