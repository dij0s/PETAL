import { useState, useEffect, useCallback } from "react";
import { checkpointStorage } from "./checkpointStorage";
import { conversationEvents } from "./conversationEvents";
import type { Checkpoint } from "../types/Checkpoint";
import { getUserId } from "./userId";

interface UseConversations {
  conversations: Checkpoint[];
  loading: boolean;
  error: string | null;
}

export const useConversations = (): UseConversations => {
  const [conversations, setConversations] = useState<Checkpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const currentUserId = getUserId();
  // callback only defined
  // for effect dependency
  // reference
  const fetchConversations = useCallback(async () => {
    try {
      setError(null);
      const checkpoints = await checkpointStorage.getAllCheckpointsSorted(
        (checkpoint) => checkpoint.userId === currentUserId,
      );
      setConversations(checkpoints);
    } catch (err) {
      console.error("Failed to load conversations:", err);
      setError("Failed to load conversations");
    } finally {
      setLoading(false);
    }
  }, [currentUserId]);

  // fetch conversations immediately
  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  // listen for conversation updates
  useEffect(() => {
    const unsubscribe = conversationEvents.subscribe((newConversation) => {
      if (newConversation) {
        // add new conversation to
        // the beginning of the list
        setConversations((prev) => {
          // check if conversation
          // already exists
          const exists = prev.some(
            (c) => c.threadId === newConversation.threadId,
          );
          if (exists) {
            // update existing conversation
            return prev.map((c) =>
              c.threadId === newConversation.threadId ? newConversation : c,
            );
          } else {
            // add new conversation
            // at the beginning
            return [newConversation, ...prev];
          }
        });
      } else {
        // fallback to full refetch
        // if no data provided
        fetchConversations();
      }
    });

    return unsubscribe;
  }, [fetchConversations]);

  return {
    conversations,
    loading,
    error,
  };
};
