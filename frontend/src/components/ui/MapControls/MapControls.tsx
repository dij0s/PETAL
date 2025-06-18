import { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faSquareCaretDown,
  faSquareCaretUp,
} from "@fortawesome/free-solid-svg-icons";
import "./MapControls.css";

interface MapControlsProps {
  mapLayers: string[];
  handleToggle: (layerName: string) => void;
}

const MapControls = ({ mapLayers, handleToggle }: MapControlsProps) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  return (
    <div className="map-controls-wrapper">
      <div
        className="map-controls-toggle"
        onClick={() => setIsExpanded((prev) => !prev)}
      >
        <FontAwesomeIcon
          icon={isExpanded ? faSquareCaretUp : faSquareCaretDown}
        />
      </div>
      {isExpanded && (
        <div className="map-layer-controls">
          {mapLayers.map((layer, index) => (
            <div key={index} className="map-single-control">
              <input
                type="checkbox"
                defaultChecked={true}
                onClick={() => {
                  handleToggle(layer);
                }}
              />
              <span className="map-layer-name">{layer}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MapControls;
