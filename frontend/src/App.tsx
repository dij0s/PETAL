import { useState } from "react";
import Layout from "./components/layout";
import Conversation from "./components/layout/Conversation";
import { useTranslation } from "react-i18next";
import { useStreamingChat } from "./utils/useStreamingChat";
import { getOrCreateUserId } from "./utils/userId";
import "./i18n";
import "./App.css";

function App() {
  const [sidebarState, setSidebarState] = useState<"collapsed" | "expanded">(
    "collapsed",
  );

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [toolCalls, _setToolCalls] = useState<string[]>([]);

  const [currentThreadId, setCurrentThreadId] = useState<string>(() =>
    crypto.randomUUID(),
  );
  const [isInitialConversation, setIsInitialConversation] = useState(true);
  const { i18n } = useTranslation();

  const [streamingState, streamingActions] = useStreamingChat({
    userId: getOrCreateUserId(),
    threadId: currentThreadId,
    language: i18n.language,
  });

  const handleSelectConversation = async (threadId: string) => {
    setCurrentThreadId(threadId);
    setIsInitialConversation(false);
  };

  const handleNewConversation = () => {
    const newThreadId = crypto.randomUUID();
    setCurrentThreadId(newThreadId);
  };

  return (
    <Layout
      sidebarState={sidebarState}
      setSidebarState={setSidebarState}
      // mapLayers={streamingState.mapLayers}
      mapLayers={["ch.bfe.solarenergie-eignung-daecher"]}
      focusedMunicipalitySFSO={streamingState.mapFocusedMunicipality}
      onSelectConversation={handleSelectConversation}
      onNewConversation={handleNewConversation}
      currentThreadId={currentThreadId}
    >
      <Conversation
        messages={streamingState.messages}
        onSendPrompt={streamingActions.sendPrompt}
        isStreaming={streamingState.isStreaming}
        processingStatus={streamingState.processingStatus}
        toolCalls={toolCalls}
        thinkingContent={streamingState.thinkingContent}
        isThinking={streamingState.isThinking}
        isInitialConversation={isInitialConversation}
      />
    </Layout>
  );
}

export default App;
