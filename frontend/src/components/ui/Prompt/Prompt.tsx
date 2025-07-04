import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowUp } from "@fortawesome/free-solid-svg-icons";
import { pelletConfig, type Indicator } from "../../../utils/feedbackPellet";
import { useTranslation } from "react-i18next";
import DataSources from "../DataSources";
import "./Prompt.css";

interface PromptProps {
  indicator?: Indicator;
  meanConsumption: string | null;
  lastConsumption: string | null;
  promptInput: string;
  setPromptInput: (value: string) => void;
  onSend: (prompt: string) => void;
  dataSources: [string, string, any][];
  disabled?: boolean;
}

const Prompt = ({
  indicator = "ok",
  meanConsumption = null,
  lastConsumption = null,
  promptInput,
  setPromptInput,
  onSend,
  dataSources,
  disabled = false,
}: PromptProps) => {
  const { color, translationKey } = pelletConfig[indicator];
  const { t } = useTranslation();

  return (
    <div className="prompt-wrapper">
      <DataSources dataSources={dataSources} />
      <div className="prompt-wrapper-inner">
        <textarea
          className="prompt-textarea"
          placeholder={t("prompt_placeholder")}
          value={promptInput}
          onChange={(e) => setPromptInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && promptInput.trim() !== "") {
              e.preventDefault();
              onSend(promptInput);
            }
          }}
          disabled={disabled}
        />
        <div className="prompt-actions-wrapper">
          <div className="prompt-feedback-wrapper">
            <div
              className="prompt-feedback-pastel"
              style={{ backgroundColor: color }}
            ></div>
            {meanConsumption && lastConsumption && (
              <span
                className="prompt-feedback-label"
                title={t("indicator_message_tooltip", {
                  mean: meanConsumption,
                })}
              >
                {t(translationKey, {
                  last: lastConsumption,
                })}
              </span>
            )}
          </div>
          <div
            className="prompt-action-wrapper"
            data-active={promptInput.replace(/\s+/g, "") != ""}
            onClick={() => {
              if (promptInput.trim() !== "") onSend(promptInput);
            }}
          >
            <FontAwesomeIcon icon={faArrowUp} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Prompt;
