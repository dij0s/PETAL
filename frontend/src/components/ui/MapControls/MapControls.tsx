import { useState, useEffect, useCallback, useRef } from "react";
import { useTranslation } from "react-i18next";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleInfo } from "@fortawesome/free-solid-svg-icons";
import "./MapControls.css";

interface MapControlsProps {
  currentBaseLayer: string;
  handleBaseLayerToggle: () => void;
  extraLayers: string[];
  handleExtraLayerToggle: (layerName: string) => void;
}

interface LayerMetadata {
  id: string;
  name: string;
  isVisible: boolean;
  legendHtml?: string;
}

const MapControls = ({
  currentBaseLayer,
  handleBaseLayerToggle,
  extraLayers,
  handleExtraLayerToggle,
}: MapControlsProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [layersMetadata, setLayersMetadata] = useState<
    Record<string, LayerMetadata>
  >({});
  const [activeLegend, setActiveLegend] = useState<string | null>(null);
  const { i18n } = useTranslation();
  const currentLang = i18n.language;
  const lastLangRef = useRef(currentLang);

  useEffect(() => {
    let isMounted = true;
    const languageChanged = lastLangRef.current !== currentLang;
    lastLangRef.current = currentLang;

    // remove metadata for layers
    // longer exist
    setLayersMetadata((prev) => {
      const updated = { ...prev };
      Object.keys(updated).forEach((layerId) => {
        if (!extraLayers.includes(layerId)) {
          delete updated[layerId];
        }
      });
      return updated;
    });

    // fetch metadata and legend
    // for new layer or on lang
    // change
    extraLayers.forEach((layerId) => {
      const needsFetch = !layersMetadata[layerId] || languageChanged;

      if (needsFetch) {
        fetch(
          `https://api3.geo.admin.ch/rest/services/api/MapServer/${layerId}?lang=${currentLang}`,
        )
          .then((res) => res.json())
          .then((data) => {
            if (!isMounted) return;
            setLayersMetadata((prev) => ({
              ...prev,
              [layerId]: {
                ...prev[layerId],
                id: layerId,
                name: data.name || layerId,
                isVisible: prev[layerId]?.isVisible ?? true,
              },
            }));
          })
          .catch(() => {
            if (!isMounted) return;
            setLayersMetadata((prev) => ({
              ...prev,
              [layerId]: {
                ...prev[layerId],
                id: layerId,
                name: prev[layerId]?.name || layerId,
                isVisible: prev[layerId]?.isVisible ?? true,
              },
            }));
          });

        fetch(
          `https://api3.geo.admin.ch/rest/services/api/MapServer/${layerId}/legend?lang=${currentLang}`,
        )
          .then((res) => res.text())
          .then((html) => {
            if (!isMounted) return;
            setLayersMetadata((prev) => ({
              ...prev,
              [layerId]: {
                ...prev[layerId],
                legendHtml: html,
              },
            }));
          })
          .catch(() => {
            if (!isMounted) return;
            setLayersMetadata((prev) => ({
              ...prev,
              [layerId]: {
                ...prev[layerId],
                legendHtml: undefined,
              },
            }));
          });
      }
    });

    return () => {
      isMounted = false;
    };
  }, [extraLayers, currentLang]);

  const toggleLayerVisibility = useCallback(
    (layerId: string) => {
      setLayersMetadata((prev) => ({
        ...prev,
        [layerId]: {
          ...prev[layerId],
          isVisible: !(prev[layerId]?.isVisible ?? true),
        },
      }));
      handleExtraLayerToggle(layerId);
    },
    [handleExtraLayerToggle],
  );

  return (
    <div className="map-controls-wrapper">
      <div
        className="map-controls-preview"
        onClick={() => setIsExpanded((prev) => !prev)}
        data-base-layer={currentBaseLayer}
      ></div>
      <div className="map-controls-layers" data-visible={isExpanded}>
        <div
          className="alternative-base-layer-preview"
          onClick={() => {
            handleBaseLayerToggle();
            setIsExpanded(false);
          }}
          data-base-layer={
            currentBaseLayer === "swissimage" ? "pixelkarte" : "swissimage"
          }
        />
        <div className="extra-layers-wrapper">
          {extraLayers.map((layerId) => (
            <div
              key={layerId}
              className="extra-layer-wrapper"
              onClick={() => toggleLayerVisibility(layerId)}
            >
              <input
                type="checkbox"
                checked={layersMetadata[layerId]?.isVisible ?? true}
                onChange={() => {}}
                className="extra-layer-checkbox"
                readOnly
              />
              <span className="extra-layer-name">
                {layersMetadata[layerId]?.name || layerId}
              </span>
              <span
                className="extra-layer-info-button"
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveLegend((prev) =>
                    prev === layerId ? null : layerId,
                  );
                }}
                role="button"
                aria-label="Layer information"
              >
                <FontAwesomeIcon icon={faCircleInfo} />
              </span>
              {activeLegend === layerId &&
                layersMetadata[layerId]?.legendHtml && (
                  <div
                    className="extra-layer-legend-tooltip"
                    onClick={(e) => {
                      e.stopPropagation();
                      setActiveLegend(null);
                    }}
                  >
                    <div
                      dangerouslySetInnerHTML={{
                        __html: layersMetadata[layerId].legendHtml,
                      }}
                    />
                  </div>
                )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MapControls;
