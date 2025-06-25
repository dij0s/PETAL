import { useState } from "react";
import Layout from "./components/layout";
import Conversation from "./components/layout/Conversation";
import { useTranslation } from "react-i18next";
import { useStreamingChat } from "./utils/useStreamingChat";
import { getOrCreateUserId } from "./utils/userId";
import { checkpointStorage } from "./utils/checkpointStorage";
import type { Checkpoint } from "./types/Checkpoint";
import "./i18n";
import "./App.css";

function App() {
  const [sidebarState, setSidebarState] = useState<"collapsed" | "expanded">(
    "collapsed",
  );

  const [currentThreadId, setCurrentThreadId] = useState<string>(() =>
    crypto.randomUUID(),
  );
  const [isInitialConversation, setIsInitialConversation] =
    useState<boolean>(true);
  const [selectedCheckpoint, setSelectedCheckpoint] =
    useState<Checkpoint | null>(null);
  const { i18n } = useTranslation();

  const [streamingState, streamingActions] = useStreamingChat({
    userId: getOrCreateUserId(),
    threadId: currentThreadId,
    language: i18n.language,
  });

  const handleSelectConversation = async (threadId: string) => {
    setCurrentThreadId(threadId);
    setIsInitialConversation(false);

    try {
      const checkpoint = await checkpointStorage.getCheckpoint(threadId);
      setSelectedCheckpoint(checkpoint);
    } catch {
      setSelectedCheckpoint(null);
    }
  };

  const handleNewConversation = () => {
    const newThreadId = crypto.randomUUID();
    setCurrentThreadId(newThreadId);
  };

  const handleSendPrompt = (prompt: string) => {
    setSelectedCheckpoint(null);
    streamingActions.sendPrompt(prompt);
  };

  return (
    <Layout
      sidebarState={sidebarState}
      setSidebarState={setSidebarState}
      mapLayers={streamingState.mapLayers}
      focusedMunicipalitySFSO={streamingState.mapFocusedMunicipality}
      onSelectConversation={handleSelectConversation}
      onNewConversation={handleNewConversation}
      currentThreadId={currentThreadId}
    >
      <Conversation
        messages={streamingState.messages}
        onSendPrompt={handleSendPrompt}
        isStreaming={streamingState.isStreaming}
        processingStatus={streamingState.processingStatus}
        thinkingContent={streamingState.thinkingContent}
        isThinking={streamingState.isThinking}
        isInitialConversation={isInitialConversation}
        selectedCheckpoint={selectedCheckpoint}
      />
    </Layout>
  );
}

export default App;
