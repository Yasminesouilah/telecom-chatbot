
import React, { useEffect, useRef, useState } from "react";
import "./styles.css";

function BotIcon() {
  return (
    <div className="bot-icon">
      <span>✦</span>
    </div>
  );
}

function UserIcon() {
  return (
    <div className="user-icon">
      <span>U</span>
    </div>
  );
}

function Message({ m, showAvatar }) {
  const isUser = m.from === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "bot-row"}`}>
      {!isUser && showAvatar ? (
        <BotIcon />
      ) : (
        !isUser && <div className="avatar-placeholder" />
      )}

      <div className="message-content">
        <div className={`bubble ${isUser ? "user-bubble" : "bot-bubble"}`}>
          <div className="text">{m.text}</div>

          {m.ts && (
            <div className={`timestamp ${isUser ? "user-time" : ""}`}>
              {new Date(m.ts).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
          )}
        </div>
      </div>

      {isUser && showAvatar && <UserIcon />}
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message-row bot-row">
      <BotIcon />

      <div className="typing-bubble">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({
        behavior: "smooth",
      });
    }
  }, [messages, loading]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function send(customText = null) {
    const text = (customText ?? input).trim();

    if (!text || loading) return;

    const userMsg = {
      from: "user",
      text,
      ts: Date.now(),
    };

    // Save the history BEFORE adding the current user message
    const historyToSend = [...messages, userMsg];

    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      // ==========================================
      // 1. TRY TRUE STREAMING FIRST
      // ==========================================
      const res = await fetch("http://localhost:8000/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: text,
          history: historyToSend,
        }),
      });

      if (!res.ok) {
        throw new Error(`Streaming server error: ${res.status}`);
      }

      if (!res.body) {
        throw new Error("Streaming response has no body");
      }

      // Stream is available — create the optimistic bot message now
      setMessages((m) => [
        ...m,
        {
          from: "bot",
          text: "",
          ts: Date.now(),
        },
      ]);

      // ==========================================
      // 3. READ SSE (JSON-framed) STREAM PROGRESSIVELY
      // ==========================================
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines. Consume any complete events.
        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const raw = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 2);

          // Each event line starts with 'data: ' according to our server framing
          if (!raw.startsWith("data:")) continue;
          const jsonText = raw.replace(/^data:\s*/, "");
          try {
            const ev = JSON.parse(jsonText);
            if (ev.type === "token") {
              const chunk = ev.text;
              setMessages((currentMessages) => {
                const updated = [...currentMessages];
                for (let i = updated.length - 1; i >= 0; i--) {
                  if (updated[i].from === "bot") {
                    updated[i] = { ...updated[i], text: updated[i].text + chunk };
                    break;
                  }
                }
                return updated;
              });
            } else if (ev.type === "done") {
              // end-of-stream signal
              // nothing special to do — loop will finish when reader closes
            } else if (ev.type === "error") {
              throw new Error(ev.error || "Stream error");
            }
          } catch (e) {
            console.warn("Failed to parse SSE event", e, jsonText);
          }
        }
      }

    } catch (streamError) {
      // ==========================================
      // 4. STREAM FAILED → FALLBACK TO /chat
      // ==========================================
      console.warn("Streaming failed, falling back to normal /chat:", streamError);

      try {
        const fallbackRes = await fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: text,
            history: historyToSend,
          }),
        });

        if (!fallbackRes.ok) {
          throw new Error(`Fallback server error: ${fallbackRes.status}`);
        }

        const data = await fallbackRes.json();

        setMessages((currentMessages) => [
          ...currentMessages,
          {
            from: "bot",
            text: data.answer || "Sorry, I couldn't generate an answer.",
            ts: Date.now(),
          },
        ]);

      } catch (fallbackError) {
        console.error("Fallback failed:", fallbackError);

        setMessages((currentMessages) => [
          ...currentMessages,
          {
            from: "bot",
            text: "Sorry, something went wrong while connecting to the assistant.",
            ts: Date.now(),
          },
        ]);
      }

    } finally {
      setLoading(false);

      setTimeout(() => {
        inputRef.current?.focus();
      }, 50);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function startSuggestion(text) {
    send(text);
  }

  function clearConversation() {
    setMessages([]);
    setInput("");
    setTimeout(() => inputRef.current?.focus(), 50);
  }

  return (
    <div className="app">

      {/* Header */}
      <header className="header">
        <div className="brand-section">

          <div className="brand-logo">
            <BotIcon />
          </div>

          <div className="brand-info">
            <h1>Telecom Assistant</h1>

            <div className="status">
              <span className="status-dot"></span>
              Online · AI Support
            </div>
          </div>
        </div>

        <button
          className="new-chat-button"
          onClick={clearConversation}
          title="Start a new conversation"
        >
          <span>＋</span>
          <span className="new-chat-text">New chat</span>
        </button>
      </header>

      {/* Chat */}
      <main className="chat">

        {messages.length === 0 && (
          <div className="welcome">

            <div className="welcome-icon">
              <BotIcon />
            </div>

            <h2>How can we help you?</h2>

            <p>
              Ask me about your mobile services, SIM card,
              plans, billing, network issues, and more.
            </p>

            <div className="suggestions">

              <button
                onClick={() =>
                  startSuggestion("I want to change my SIM card")
                }
              >
                <span>📱</span>
                Change my SIM card
              </button>

              <button
                onClick={() =>
                  startSuggestion("I have a problem with my network")
                }
              >
                <span>📶</span>
                Network problem
              </button>

              <button
                onClick={() =>
                  startSuggestion("I want to know about my mobile plan")
                }
              >
                <span>💳</span>
                Mobile plans
              </button>

              <button
                onClick={() =>
                  startSuggestion("I have a question about my bill")
                }
              >
                <span>🧾</span>
                Billing
              </button>

            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <Message
            key={i}
            m={m}
            showAvatar={
              i === 0 ||
              messages[i - 1].from !== m.from
            }
          />
        ))}

        {loading && <TypingIndicator />}

        <div ref={bottomRef} />
      </main>

      {/* Composer */}
      <footer className="composer-container">

        <div className="composer">

          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Message your telecom assistant..."
            disabled={loading}
          />

          <button
            className="send-button"
            onClick={() => send()}
            disabled={!input.trim() || loading}
            aria-label="Send message"
          >
            <svg
              viewBox="0 0 24 24"
              width="20"
              height="20"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M22 2L11 13" />
              <path d="M22 2L15 22L11 13L2 9L22 2Z" />
            </svg>
          </button>

        </div>

        <div className="composer-hint">
          Press <strong>Enter</strong> to send · Your conversation stays private
        </div>

      </footer>

    </div>
  );
}

