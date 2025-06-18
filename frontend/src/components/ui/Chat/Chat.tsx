import type { Message } from "../../../types/Message";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./Chat.css";

interface ChatProps {
  message: Message;
}

const Chat = ({ message }: ChatProps) => {
  return (
    <div className="chat-wrapper" data-source={message.role}>
      <div className="chat">
        {message.role === "assistant" ? (
          <div className="chat-markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h3: ({ children }) => (
                  <h3 className="chat-markdown-h3">{children}</h3>
                ),
                h4: ({ children }) => (
                  <h4 className="chat-markdown-h4">{children}</h4>
                ),
                p: ({ children }) => (
                  <p className="chat-markdown-p">{children}</p>
                ),
                strong: ({ children }) => (
                  <strong className="chat-markdown-strong">{children}</strong>
                ),
                em: ({ children }) => (
                  <em className="chat-markdown-em">{children}</em>
                ),
                blockquote: ({ children }) => (
                  <strong className="chat-markdown-strong">{children}</strong>
                ),
                ul: ({ children }) => (
                  <ul className="chat-markdown-ul">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="chat-markdown-ol">{children}</ol>
                ),
                li: ({ children }) => (
                  <li className="chat-markdown-li">{children}</li>
                ),
                hr: ({ children }) => (
                  <hr className="chat-markdown-hr">{children}</hr>
                ),
                table: ({ children }) => (
                  <div className="chat-markdown-table-wrapper">
                    <table className="chat-markdown-table">{children}</table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="chat-markdown-thead">{children}</thead>
                ),
                tbody: ({ children }) => (
                  <tbody className="chat-markdown-tbody">{children}</tbody>
                ),
                tr: ({ children }) => (
                  <tr className="chat-markdown-tr">{children}</tr>
                ),
                th: ({ children }) => (
                  <th className="chat-markdown-th">{children}</th>
                ),
                td: ({ children }) => (
                  <td className="chat-markdown-td">{children}</td>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        ) : (
          message.content
        )}
      </div>
    </div>
  );
};

export default Chat;
