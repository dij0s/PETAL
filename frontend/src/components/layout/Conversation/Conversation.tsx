import { useState, useRef, useEffect } from "react";
import Prompt from "../../ui/Prompt";
import { useTranslation } from "react-i18next";
import type { Message } from "../../../types/Message";
import Chat from "../../ui/Chat";
import loader from "../../../assets/loader.webp";
import "./Conversation.css";

interface ConversationProps {
  messages: Message[];
  onSendPrompt: (prompt: string) => void;
  isStreaming: boolean;
  processingStatus: string;
  toolCalls: string[];
  thinkingContent?: string;
  isThinking?: boolean;
  isInitialConversation?: boolean;
}

const Conversation = ({
  messages,
  onSendPrompt,
  isStreaming,
  processingStatus,
  thinkingContent = "",
  isThinking = false,
  isInitialConversation = false,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  toolCalls: _toolCalls,
}: ConversationProps) => {
  const [promptInput, setPromptInput] = useState<string>("");
  const { t } = useTranslation();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesWrapperRef = useRef<HTMLDivElement>(null);
  const autoScrollEnabledRef = useRef<boolean>(true);
  const lastScrollTopRef = useRef<number>(0);

  // check if user is near
  // the bottom of scroll
  const isNearBottom = () => {
    const wrapper = messagesWrapperRef.current;
    if (!wrapper) return true;
    // in px
    const threshold = 50;
    const { scrollTop, scrollHeight, clientHeight } = wrapper;
    return scrollHeight - scrollTop - clientHeight < threshold;
  };

  const scrollToBottom = () => {
    if (autoScrollEnabledRef.current && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  const handleScroll = () => {
    const wrapper = messagesWrapperRef.current;
    if (!wrapper) return;

    const currentScrollTop = wrapper.scrollTop;

    // user scrolled up manually
    // disable auto scroll
    if (currentScrollTop < lastScrollTopRef.current && !isNearBottom()) {
      autoScrollEnabledRef.current = false;
    }
    // user scrolled back to bottom is
    // snapped back into the ongoing scroll
    else if (isNearBottom()) {
      autoScrollEnabledRef.current = true;
    }

    lastScrollTopRef.current = currentScrollTop;
  };

  // scroll to bottom when messages
  // change or processing status changes
  useEffect(() => {
    scrollToBottom();
  }, [messages, processingStatus]);
  // scroll during streaming
  useEffect(() => {
    if (isStreaming) {
      scrollToBottom();
    }
  }, [isStreaming]);

  // (re)-enable auto scroll
  // on new prompt from user
  const handleSendPrompt = (prompt: string) => {
    if (prompt.trim() !== "") {
      autoScrollEnabledRef.current = true;
      onSendPrompt(prompt);
      setPromptInput("");
    }
  };

  return (
    <main>
      {messages.length === 0 && (
        <div
          className={"conversation-fallback-wrapper"}
          data-visible={
            promptInput.trim() === "" && isInitialConversation === true
          }
        >
          <div className="conversation-fallback-welcome">
            {t("welcome_message")}
          </div>
          <div className="conversation-fallback-prompts">
            {(Array.isArray(t("prompt_propositions", { returnObjects: true }))
              ? (t("prompt_propositions", { returnObjects: true }) as string[])
              : []
            ).map((proposition: string, index: number) => (
              <div key={index} className="conversation-fallback-prompt-wrapper">
                <div
                  className="conversation-fallback-prompt"
                  onClick={() => {
                    setPromptInput(proposition);
                    const input =
                      document.querySelector<HTMLInputElement>(
                        ".prompt-textarea",
                      );
                    if (input) input.focus();
                  }}
                >
                  {proposition}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {messages.length > 0 && (
        <div
          ref={messagesWrapperRef}
          className="conversation-messages-wrapper"
          onScroll={handleScroll}
        >
          {messages.map((msg, index) => (
            <Chat key={index} message={msg} />
          ))}
          {processingStatus !== "" && (
            <div className="conversation-status">
              <div className="conversation-status-loader-wrapper">
                <img
                  src={loader}
                  alt=""
                  className="conversation-status-loader"
                />
              </div>
              <div className="conversation-processing-status">
                {processingStatus}
              </div>
            </div>
          )}
          {isThinking && (
            <div className="conversation-thinking-wrapper">
              {thinkingContent}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}
      <Prompt
        promptInput={promptInput}
        setPromptInput={setPromptInput}
        onSend={handleSendPrompt}
        disabled={isStreaming}
      />
    </main>
  );
};

export default Conversation;
