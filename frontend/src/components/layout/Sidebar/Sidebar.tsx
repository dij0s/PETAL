import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faTableColumns,
  faPlus,
  faTrash,
} from "@fortawesome/free-solid-svg-icons";
import { useTranslation } from "react-i18next";
import { useConversations } from "../../../utils/useConversation";
import type { Checkpoint } from "../../../types/Checkpoint";
import { checkpointStorage } from "../../../utils/checkpointStorage";
import { conversationEvents } from "../../../utils/conversationEvents";
import "./Sidebar.css";

interface SidebarProps {
  sidebarState: "collapsed" | "expanded";
  setSidebarState: (state: "collapsed" | "expanded") => void;
  onClickConversation: (threadId: string) => void;
  onNewConversation: () => void;
  currentThreadId: string;
}

interface GroupedConversations {
  today: Checkpoint[];
  lastWeek: Checkpoint[];
  older: Checkpoint[];
}

const Sidebar = ({
  sidebarState,
  setSidebarState,
  onClickConversation,
  onNewConversation,
  currentThreadId,
}: SidebarProps) => {
  const { t } = useTranslation();
  const { conversations, loading, error } = useConversations();
  const isCollapsed = sidebarState === "collapsed";

  // bin conversations
  // into time periods
  const groupConversationsByTime = (
    conversations: Checkpoint[],
  ): GroupedConversations => {
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

    return conversations.reduce(
      (groups, conversation) => {
        const conversationDate = new Date(conversation.timestamp);

        if (conversationDate >= today) {
          groups.today.push(conversation);
        } else if (conversationDate >= lastWeek) {
          groups.lastWeek.push(conversation);
        } else {
          groups.older.push(conversation);
        }

        return groups;
      },
      { today: [], lastWeek: [], older: [] } as GroupedConversations,
    );
  };
  const groupedConversations = groupConversationsByTime(conversations);

  const ConversationGroup = ({
    title,
    conversations,
  }: {
    title: string;
    conversations: Checkpoint[];
  }) => {
    if (conversations.length === 0) return null;

    return (
      <div className="panel-history-bin">
        <div className="panel-history-bin-title">{title}</div>
        <div className="panel-history-bin-items">
          {conversations.map((conversation) => (
            <div
              key={conversation.threadId}
              className="panel-history-item"
              data-active-conversation={
                currentThreadId === conversation.threadId
              }
              onClick={
                currentThreadId === conversation.threadId
                  ? undefined
                  : () => onClickConversation(conversation.threadId)
              }
              title={conversation.title}
            >
              <span>{conversation.title}</span>
              <FontAwesomeIcon
                icon={faTrash}
                onClick={async (e) => {
                  e.stopPropagation();
                  await checkpointStorage.deleteCheckpoint(
                    conversation.threadId,
                  );
                  conversationEvents.emit();
                }}
                className="panel-history-item-trash"
              />
            </div>
          ))}
        </div>
      </div>
    );
  };

  // expand panel container
  const expandPanel = (
    <div className="panel-actions-action">
      <div className="panel-actions-action-sidebar-wrapper panel-actions-action-inner">
        <FontAwesomeIcon
          icon={faTableColumns}
          onClick={() =>
            setSidebarState(isCollapsed ? "expanded" : "collapsed")
          }
        />
      </div>
      <span className="panel-actions-label">
        {isCollapsed ? "" : t("side_panel_close_label")}
      </span>
    </div>
  );

  // new conversation container
  const newConversation = (
    <div className="panel-actions-action">
      <div className="panel-actions-action-conversation-wrapper panel-actions-action-inner">
        <FontAwesomeIcon
          icon={faPlus}
          className="panel-actions-action-conversation"
          onClick={onNewConversation}
        />
      </div>
      <span className="panel-actions-label">
        {isCollapsed ? "" : t("side_panel_new_conversation_label")}
      </span>
    </div>
  );

  return (
    <aside>
      <div className="panel-actions-wrapper">
        {expandPanel}
        {newConversation}
      </div>
      {!isCollapsed && (
        <div className="panel-history-wrapper">
          {!loading &&
            !error &&
            (conversations.length === 0 ? (
              <div className="panel-history-empty">
                {t("side_panel_empty_label")}
              </div>
            ) : (
              <>
                <ConversationGroup
                  title={t("side_panel_today_label")}
                  conversations={groupedConversations.today}
                />
                <ConversationGroup
                  title={t("side_panel_last_week_label")}
                  conversations={groupedConversations.lastWeek}
                />
                <ConversationGroup
                  title={t("side_panel_older_label")}
                  conversations={groupedConversations.older}
                />
              </>
            ))}
        </div>
      )}
    </aside>
  );
};

export default Sidebar;
