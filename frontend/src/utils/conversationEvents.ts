import type { Checkpoint } from "../types/Checkpoint";

type ConversationEventListener = (newConversation?: Checkpoint) => void;

class ConversationEvents {
  private listeners: ConversationEventListener[] = [];

  subscribe(listener: ConversationEventListener) {
    this.listeners.push(listener);
    // return unsubscribe function
    return () => {
      const index = this.listeners.indexOf(listener);
      if (index > -1) {
        this.listeners.splice(index, 1);
      }
    };
  }
  emit(newConversation?: Checkpoint) {
    this.listeners.forEach((listener) => {
      try {
        // send the conversation
        // to the listener as p.d.
        // any new event after first
        // fetch of the conversations
        // is the most recent one
        // which needs no binning
        // computation
        listener(newConversation);
      } catch (error) {
        console.error("Error in conversation event listener:", error);
      }
    });
  }
}

export const conversationEvents = new ConversationEvents();
