import { checkpointStorage } from "./checkpointStorage";

export async function getConversationType(
  userId: string,
  threadId: string,
): Promise<"active" | "checkpointed" | "new"> {
  // check if conversation
  // is active in backend
  try {
    const response = await fetch(
      `/api/conversation/status?user_id=${userId}&thread_id=${threadId}`,
    );
    if (response.ok) {
      const data = await response.json();
      if (data.active) {
        return "active";
      }
    }
  } catch (error) {
    console.error("Failed to check conversation status:", error);
  }

  // check if we have a
  // checkpoint for this
  // conversation
  const checkpoint = await checkpointStorage.getCheckpoint(threadId);
  return checkpoint ? "checkpointed" : "new";
}
