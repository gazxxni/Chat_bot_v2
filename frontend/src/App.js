import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";
import logoImg from './chatbot_logo.svg';

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [showWelcome, setShowWelcome] = useState(true); // 추가

  const chatEndRef = useRef(null);
  const chatHistoryRef = useRef(null);

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages]);

  const askBot = async () => {
    if (!query.trim()) return;

    setShowWelcome(false); // 첫 메시지 전송 시 환영 메시지 숨김

    const userMessage = { sender: "user", text: query };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);
    setQuery("");

    try {
      const res = await axios.post("http://localhost:8000/ask", {
        query: userMessage.text,
      });
      const botMessage = { sender: "bot", text: res.data.answer };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      const errorMsg = {
        sender: "bot",
        text: "❗ 서버 응답 오류. 다시 시도해주세요.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    }

    setLoading(false);
  };

  return (
    <section className="container">
      <div className="container_box">
        <div className="title">
          <img src={logoImg} alt="logo" />
          <h1> 컴퓨터공학과 안내 챗봇 </h1>
        </div>

        <div className="chat-history" ref={chatHistoryRef}>
          {showWelcome && (
            <div className="welcome-message">
              <div className="welcome-icon">💬</div>
              <p>컴퓨터공학과에 관심있는 학생들을 위한 안내 챗봇입니다.</p>
              <p>궁금한 내용을 입력하고 대화해보세요!</p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="sender-label">
                {msg.sender === "user" ? "나" : "도우미봇"}
              </div>
              <div className="bubble">{msg.text}</div>
            </div>
          ))}

          {loading && (
            <div className="message bot">
              <div className="sender-label">도우미봇</div>
              <div className="bubble">
                답변 생성 중...
                <span className="loader"></span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <textarea
          placeholder="질문을 입력하세요 (예: 인공지능 관련 과목을 추천해줘)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              askBot();
            }
          }}
        ></textarea>

        <button onClick={askBot} disabled={loading}>
          {loading ? "답변 생성 중..." : "질문하기"}
        </button>
      </div>
    </section>
  );
}

export default App;