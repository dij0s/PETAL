import { useState, useRef, useCallback, useEffect } from "react";
import { checkpointStorage } from "./checkpointStorage";
import { conversationEvents } from "./conversationEvents";
import { getConversationType } from "./conversationUtils";
import type { Message } from "../types/Message";

interface UseStreamingChatOptions {
  userId: string;
  threadId: string;
  language: string;
  onCheckpointStored?: () => void;
}

interface StreamingState {
  isStreaming: boolean;
  messages: Message[];
  processingStatus: string;
  mapLayers: string[];
  mapFocusedMunicipality: number | null;
  thinkingContent: string;
  isThinking: boolean;
  conversationType: "active" | "checkpointed" | "new";
}

interface StreamingActions {
  sendPrompt: (prompt: string) => Promise<void>;
}

export const useStreamingChat = (
  options: UseStreamingChatOptions,
): [StreamingState, StreamingActions] => {
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [processingStatus, setProcessingStatus] = useState<string>("");
  const [mapLayers, setMapLayers] = useState<string[]>([]);
  const [mapFocusedMunicipality, setMapFocusedMunicipality] = useState<
    number | null
  >(null);
  const [thinkingContent, setThinkingContent] = useState<string>("");
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [conversationType, setConversationType] = useState<
    "active" | "checkpointed" | "new"
  >("new");

  const conversationCache = useRef<
    Map<
      string,
      {
        messages: Message[];
        mapLayers: string[];
        mapFocusedMunicipality: number | null;
      }
    >
  >(new Map());
  const previousThreadId = useRef<string | null>(null);

  const saveConversationState = useCallback(
    (threadId: string) => {
      conversationCache.current.set(threadId, {
        messages,
        mapLayers,
        mapFocusedMunicipality,
      });
    },
    [messages, mapLayers, mapFocusedMunicipality],
  );

  // restore conversation
  // state from cache
  const restoreConversationState = useCallback((threadId: string) => {
    const cachedState = conversationCache.current.get(threadId);
    if (cachedState) {
      setMessages(cachedState.messages);
      setMapLayers(cachedState.mapLayers);
      setMapFocusedMunicipality(cachedState.mapFocusedMunicipality);
      return true;
    }
    return false;
  }, []);

  // clear conversation state
  // for checkpointed ones
  const clearConversationState = useCallback(() => {
    setMessages([]);
    setMapLayers([]);
    setMapFocusedMunicipality(null);
  }, []);

  const isInitialThread = useRef<boolean>(true);
  // check conversation type
  // when threadId changes
  useEffect(() => {
    const handleConversationSwitch = async () => {
      if (isInitialThread.current) {
        // initialize previousThreadId
        // with the current threadId
        // on very first render
        previousThreadId.current = options.threadId;
        setConversationType("new");
        isInitialThread.current = false;
        return;
      }

      if (
        previousThreadId.current &&
        previousThreadId.current !== options.threadId
      ) {
        saveConversationState(previousThreadId.current);
      }

      const type = await getConversationType(options.userId, options.threadId);
      setConversationType(type);
      clearConversationState();
      // if conversation is
      // checkpointed, then
      // it might be because
      // the thread had been
      // cleared in the backend
      // before it was done in
      // the frontend itself
      if (type !== "new") {
        restoreConversationState(options.threadId);
      }

      previousThreadId.current = options.threadId;
    };

    handleConversationSwitch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options.threadId, options.userId]);

  const eventSourceRef = useRef<EventSource | null>(null);

  const tokenBufferRef = useRef<string>("");
  const displayedContentRef = useRef<string>("");
  const animationFrameRef = useRef<number | null>(null);
  const typewriterFrameRef = useRef<number | null>(null);
  const isFirstTokenRef = useRef<boolean>(true);

  const thinkingBufferRef = useRef<string>("");
  const responseBufferRef = useRef<string>("");
  const hasSeenThinkingEndRef = useRef<boolean>(false);

  const pendingLayersRef = useRef<string[]>([]);
  const pendingMunicipalityRef = useRef<number | null>(null);
  const layersAppliedRef = useRef<boolean>(false);

  const displayedWordsRef = useRef<number>(0);
  const targetWordsRef = useRef<string[]>([]);
  // animation settings
  // words per second
  const typewriterSpeedRef = useRef<number>(8);
  const lastUpdateTimeRef = useRef<number>(0);

  const pendingCheckpointRef = useRef<unknown>(null);
  // split text into words
  // while preserving spaces
  const splitIntoWords = useCallback((text: string): string[] => {
    if (!text) return [];
    // split by spaces but keep
    // spaces with preceding words
    const words: string[] = [];
    const parts = text.split(/(\s+)/);

    for (let i = 0; i < parts.length; i++) {
      if (parts[i].trim() !== "") {
        // add word with any
        // following whitespace
        const word = parts[i] + (parts[i + 1] || "");
        words.push(word);
        // skip whitespace part
        // since included
        if (parts[i + 1]) i++;
      }
    }

    return words;
  }, []);

  // parse thinking blocks
  // and response content
  const parseContent = useCallback((rawContent: string) => {
    // check if we have a
    // complete thinking block
    const thinkingMatch = rawContent.match(/<think>([\s\S]*?)<\/think>/);

    if (thinkingMatch) {
      const thinkingText = thinkingMatch[1].trim();
      const responseText = rawContent
        .replace(/<think>[\s\S]*?<\/think>/, "")
        .trim();
      return {
        thinking: thinkingText,
        response: responseText,
        hasCompleteThinking: true,
      };
    }

    // check if we are in the
    // middle of a thinking block
    const thinkingStartMatch = rawContent.match(/<think>([\s\S]*)$/);
    if (thinkingStartMatch) {
      return {
        thinking: thinkingStartMatch[1],
        response: "",
        hasCompleteThinking: false,
      };
    }

    // no thinking tags
    // treat as response
    return {
      thinking: "",
      response: rawContent,
      hasCompleteThinking: true,
    };
  }, []);

  const extractTitle = (responseContent: string) => {
    if (!responseContent || responseContent.trim() === "") {
      return "New Conversation";
    }

    // extract first line or
    // first sentence as title
    const firstLine = responseContent.trim().split("\n")[0];
    const firstSentence = firstLine.split(".")[0];

    // clean up any markdown or special characters for title
    const cleanTitle = firstSentence
      .replace(/[#*_`]/g, "")
      .replace(/\s+/g, " ")
      .trim();

    return cleanTitle || "New Conversation";
  };

  const animateTypewriter = useCallback((timestamp: number) => {
    const targetWords = targetWordsRef.current;
    const currentWordCount = displayedWordsRef.current;

    if (currentWordCount < targetWords.length) {
      // elapsed time since
      // last animated frame
      const deltaTime = timestamp - lastUpdateTimeRef.current;
      const wordsToAdd = Math.max(
        1,
        Math.floor((typewriterSpeedRef.current * deltaTime) / 1000),
      );

      // update displayed word count
      const newWordCount = Math.min(
        currentWordCount + wordsToAdd,
        targetWords.length,
      );
      displayedWordsRef.current = newWordCount;
      lastUpdateTimeRef.current = timestamp;

      // reconstruct text from words
      const newContent = targetWords.slice(0, newWordCount).join("");
      displayedContentRef.current = newContent;

      // update react state
      setMessages((msgs) => {
        if (msgs[msgs.length - 1]?.role === "assistant") {
          return [
            ...msgs.slice(0, -1),
            { role: "assistant", content: newContent },
          ];
        } else {
          return [...msgs, { role: "assistant", content: newContent }];
        }
      });

      // continue animation if
      // more words to show
      if (newWordCount < targetWords.length) {
        typewriterFrameRef.current = requestAnimationFrame(animateTypewriter);
      } else {
        typewriterFrameRef.current = null;
      }
    } else {
      typewriterFrameRef.current = null;
    }
  }, []);

  const startTypewriterAnimation = useCallback(() => {
    const newWords = splitIntoWords(responseBufferRef.current);
    const currentWords = targetWordsRef.current;

    // only start animation
    // if new words to show
    if (newWords.length > currentWords.length) {
      targetWordsRef.current = newWords;

      if (typewriterFrameRef.current === null) {
        lastUpdateTimeRef.current = performance.now();
        typewriterFrameRef.current = requestAnimationFrame(animateTypewriter);
      }
    }
  }, [animateTypewriter, splitIntoWords]);

  const applyPendingLayers = useCallback(() => {
    if (!layersAppliedRef.current && pendingLayersRef.current.length > 0) {
      setMapLayers(pendingLayersRef.current);
      setMapFocusedMunicipality(pendingMunicipalityRef.current);
      layersAppliedRef.current = true;
    }
  }, []);

  const flushTokenBuffer = useCallback(() => {
    const { thinking, response, hasCompleteThinking } = parseContent(
      tokenBufferRef.current,
    );

    if (thinking && thinking !== thinkingBufferRef.current) {
      // update thinking content
      // if we have thinking text
      thinkingBufferRef.current = thinking;
      setThinkingContent(thinking);

      if (!isThinking) {
        setIsThinking(true);
        setProcessingStatus("Thinking..");
      }
    }
    // if thinking block is complete
    // and we have response content
    if (hasCompleteThinking && response && !hasSeenThinkingEndRef.current) {
      hasSeenThinkingEndRef.current = true;
      setIsThinking(false);
      setProcessingStatus("");
      // apply layers when actual response starts
      applyPendingLayers();
    }
    // update response content if we
    // are past thinking mode or if
    // there is no thinking block
    if (
      (hasSeenThinkingEndRef.current || !thinking) &&
      response !== responseBufferRef.current
    ) {
      responseBufferRef.current = response;
      // also apply layers when response content starts
      // (for cases without thinking blocks)
      if (!thinking) {
        applyPendingLayers();
      }

      startTypewriterAnimation();
    }
    animationFrameRef.current = null;
  }, [parseContent, startTypewriterAnimation, isThinking, applyPendingLayers]);

  const scheduleTokenUpdate = useCallback(() => {
    if (animationFrameRef.current === null) {
      animationFrameRef.current = requestAnimationFrame(flushTokenBuffer);
    }
  }, [flushTokenBuffer]);

  const sendPrompt = useCallback(
    async (prompt: string) => {
      // save conversation state
      // in current frontend
      // session only
      saveConversationState(options.threadId);
      setMessages((msgs) => [...msgs, { role: "user", content: prompt }]);

      const requestBody: Record<string, unknown> = {
        user_id: options.userId,
        thread_id: options.threadId,
        prompt: prompt,
        lang: options.language,
      };

      // if this is a checkpointed
      // conversation we must then
      // include checkpoint data
      if (conversationType === "checkpointed") {
        try {
          const checkpoint = await checkpointStorage.getCheckpoint(
            options.threadId,
          );
          if (checkpoint) {
            requestBody.checkpoint_data = checkpoint.data;
          }
        } catch (error) {
          console.error("Failed to load checkpoint for rehydration:", error);
        }
      }
      setConversationType("active");

      // reset all state for new prompt
      tokenBufferRef.current = "";
      displayedContentRef.current = "";
      displayedWordsRef.current = 0;
      targetWordsRef.current = [];
      isFirstTokenRef.current = true;
      lastUpdateTimeRef.current = 0;
      thinkingBufferRef.current = "";
      responseBufferRef.current = "";
      hasSeenThinkingEndRef.current = false;
      pendingLayersRef.current = [];
      pendingMunicipalityRef.current = null;
      pendingCheckpointRef.current = null;
      layersAppliedRef.current = false;

      setThinkingContent("");
      setIsThinking(false);
      setProcessingStatus("");

      // cancel pending animations
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (typewriterFrameRef.current) {
        cancelAnimationFrame(typewriterFrameRef.current);
        typewriterFrameRef.current = null;
      }

      const response = await fetch("/api/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (response.status !== 200) {
        return;
      }

      // start streaming
      if (eventSourceRef.current) eventSourceRef.current.close();
      const es = new EventSource(
        `/api/stream?user_id=${options.userId}&thread_id=${options.threadId}`,
      );

      setIsStreaming(true);

      // handle incoming tokens
      es.addEventListener("token", (e) => {
        if (isFirstTokenRef.current) {
          isFirstTokenRef.current = false;
        }
        // accumulate tokens in buffer
        tokenBufferRef.current += e.data;
        // schedule animation update
        scheduleTokenUpdate();
      });

      es.addEventListener("checkpoint", async (e) => {
        try {
          // store checkpoint data
          // but defer its storage
          // until end of stream
          const checkpointData = JSON.parse(e.data);
          pendingCheckpointRef.current = checkpointData;
        } catch (error) {
          console.error("Failed to parse checkpoint data:", error);
        }
      });

      // handle status updates
      es.addEventListener("info", (e) => {
        const data = JSON.parse(e.data);
        // only update status if
        // not in thinking mode
        if (!isThinking) {
          setProcessingStatus(data.content);
        }
      });

      // handle focused geometry update
      es.addEventListener("sfso_number", (e) => {
        const data = JSON.parse(e.data);
        // always buffer municipality
        // until response starts
        pendingMunicipalityRef.current = Number(data.sfso_number);
      });

      // handle layers update
      es.addEventListener("layers", (e) => {
        const data = JSON.parse(e.data);
        // always buffer layers
        // until response starts
        pendingLayersRef.current = data.layers;
      });

      // handle end event
      es.addEventListener("end", async () => {
        // cancel batching frame
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }

        // complete animation
        // when stream ends
        if (typewriterFrameRef.current) {
          cancelAnimationFrame(typewriterFrameRef.current);
          typewriterFrameRef.current = null;
        }

        // apply any remaining
        // pending layers
        applyPendingLayers();

        // final parse and display
        const { response } = parseContent(tokenBufferRef.current);
        displayedContentRef.current = response;
        responseBufferRef.current = response;
        displayedWordsRef.current = splitIntoWords(response).length;
        targetWordsRef.current = splitIntoWords(response);

        // clear thinking state
        setIsThinking(false);
        setThinkingContent("");
        setProcessingStatus("");

        setMessages((msgs) => {
          if (msgs[msgs.length - 1]?.role === "assistant") {
            return [
              ...msgs.slice(0, -1),
              { role: "assistant", content: response },
            ];
          } else {
            return [...msgs, { role: "assistant", content: response }];
          }
        });

        // store conversation checkpoint
        if (pendingCheckpointRef.current) {
          try {
            const title = extractTitle(response);
            const conversation = checkpointStorage.createCheckpoint(
              options.threadId,
              options.userId,
              title,
              pendingCheckpointRef.current,
            );

            await checkpointStorage.storeCheckpoint(conversation);
            // clear the pending checkpoint
            pendingCheckpointRef.current = null;
            // notify conversations update
            conversationEvents.emit(conversation);
          } catch (error) {
            console.error("Failed to store checkpoint:", error);
          }
        }
        // save current state
        saveConversationState(options.threadId);

        es.close();
        setIsStreaming(false);
      });

      eventSourceRef.current = es;
    },
    [
      options,
      scheduleTokenUpdate,
      splitIntoWords,
      parseContent,
      isThinking,
      applyPendingLayers,
      conversationType,
      saveConversationState,
    ],
  );

  return [
    {
      isStreaming,
      messages,
      processingStatus,
      mapLayers,
      mapFocusedMunicipality,
      thinkingContent,
      isThinking,
      conversationType,
    },
    {
      sendPrompt,
    },
  ];
};
