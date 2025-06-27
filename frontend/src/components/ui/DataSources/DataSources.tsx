import { useState } from "react";
import "./DataSources.css";

interface DataSourcesProps {
  dataSources: [string, string, any][];
}

const DataSources = ({ dataSources }: DataSourcesProps) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  console.log(dataSources);

  return (
    <>
      {dataSources && dataSources.length > 0 && (
        <div className="data-sources-wrapper" data-expanded={isExpanded}>
          <div
            className="data-sources-action"
            onClick={() => setIsExpanded((prev) => !prev)}
          >
            Datapoints
          </div>
          <div className="data-sources-table-wrapper" data-visible={isExpanded}>
            <table className="data-sources-table">
              <thead>
                <tr>
                  <th>Description</th>
                  <th className="th-centered">Value</th>
                </tr>
              </thead>
              <tbody>
                {dataSources.map(([description, source, data], index) => (
                  <tr key={index}>
                    <td
                      className="table-td-description"
                      onClick={() => {
                        if (source !== "") {
                          window.open(source as string, "_blank");
                        }
                      }}
                    >
                      {description}
                    </td>
                    <td className="table-td-value">
                      {Array.isArray(data) ? data.join(", ") : data}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
};

export default DataSources;
