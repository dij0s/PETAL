import { useState } from "react";
import "./DataSources.css";

interface DataSourcesProps {
  dataSources: [string, string, any][];
}

const DataSources = ({ dataSources }: DataSourcesProps) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  const temp = [["Heating and cooling needs", "somewebsite.com", 56]];

  return (
    <div className="data-sources-wrapper" data-expanded={isExpanded}>
      <div
        className="data-sources-action"
        onClick={() => setIsExpanded((prev) => !prev)}
      >
        Sources
      </div>
      <table className="data-sources-table" data-visible={isExpanded}>
        <thead>
          <tr>
            <th>Name</th>
            <th>Data</th>
            <th>Dataset URL</th>
          </tr>
        </thead>
        <tbody>
          {temp.map(([name, dataset_url, data]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{data}</td>
              <td>{dataset_url}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default DataSources;
