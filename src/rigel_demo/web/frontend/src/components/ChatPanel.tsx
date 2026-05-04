import { useState } from 'react';
import { ArrowUp, Bot, Database, User } from 'lucide-react';
import { sendChatMessage, type ApiGraphRAGQuery, type ChatMessage } from '../api/rigel';
import { formatGraphRAGQuery, formatQueryArgs } from './chatPresentation';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  queries?: ApiGraphRAGQuery[];
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isResponding, setIsResponding] = useState(false);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || isResponding) return;

    // 先把用户消息写入本地状态，再用同一份历史请求后端，保证界面与模型上下文一致。
    const nextMessages: Message[] = [...messages, { role: 'user', content: query }];
    setMessages(nextMessages);
    setInput('');
    setIsResponding(true);

    try {
      const response = await sendChatMessage(nextMessages.map(toChatMessage));
      setMessages((currentMessages) => [
        ...currentMessages,
        { role: 'assistant', content: response.message.content, queries: response.queries },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : '请求失败';
      setMessages((currentMessages) => [...currentMessages, { role: 'assistant', content: `未能完成模型调用：${message}` }]);
    } finally {
      setIsResponding(false);
    }
  };

  return (
    <aside className="relative flex h-full flex-col bg-white text-slate-800">
      <div className="flex items-center gap-2 border-b border-slate-100 p-4">
        <div className="h-2 w-2 animate-pulse rounded-full bg-black shadow-[0_0_8px_rgba(0,0,0,0.3)]"></div>
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-800">GraphRAG Chat</span>
      </div>

      <div className="flex flex-1 flex-col overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center p-6 text-center">
            <h2 className="text-xl font-medium tracking-tight text-slate-900">您想分析什么？</h2>
            <p className="mt-3 max-w-[260px] text-sm leading-relaxed text-slate-500">
              询问有关代码架构、依赖关系和复杂结构的问题。
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-6 p-5">
            {messages.map((message, index) => (
              <div key={index} className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${message.role === 'user' ? 'bg-slate-100' : 'bg-black text-white'}`}>
                  {message.role === 'user' ? <User className="h-4 w-4 text-slate-500" /> : <Bot className="h-4 w-4" />}
                </div>
                <div className={`flex flex-col ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                  {message.role === 'assistant' && message.queries && message.queries.length > 0 && (
                    <div className="mb-2 flex max-w-[280px] flex-wrap gap-1.5">
                      {message.queries.map((query, queryIndex) => (
                        <span
                          key={`${query.name}-${queryIndex}`}
                          className="inline-flex max-w-full items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] font-medium text-slate-600"
                          title={formatQueryArgs(query.args)}
                        >
                          <Database className="h-3 w-3 shrink-0" />
                          <span className="truncate">{formatGraphRAGQuery(query)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                  <div className={`max-w-[280px] whitespace-pre-wrap rounded-2xl px-4 py-3 text-[13px] leading-relaxed ${message.role === 'user' ? 'bg-slate-100 text-slate-900' : 'bg-transparent text-slate-800'}`}>
                    {message.content}
                  </div>
                </div>
              </div>
            ))}
            {isResponding && (
              <div className="flex gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-black text-white">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="rounded-2xl px-4 py-3 text-[13px] leading-relaxed text-slate-500">正在查询图谱...</div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="shrink-0 bg-white p-4 pb-6">
        <div className="flex flex-col rounded-[24px] border border-slate-200 bg-white p-2 shadow-sm shadow-slate-100 transition-all focus-within:border-slate-300 focus-within:ring-4 focus-within:ring-slate-50">
          <textarea
            className="min-h-[64px] w-full resize-none bg-transparent px-3 py-2 text-[14px] text-slate-900 outline-none placeholder:text-slate-400"
            placeholder="输入消息与AI聊天"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleSend();
              }
            }}
          ></textarea>

          <div className="flex items-center justify-end px-1 pb-1">
            <button
              onClick={handleSend}
              disabled={!input.trim() || isResponding}
              className={`flex h-8 w-8 items-center justify-center rounded-full transition-colors ${
                input.trim() && !isResponding ? 'bg-black text-white' : 'bg-[#f4f4f4] text-[#c4c4c4]'
              }`}
              aria-label="发送消息"
            >
              <ArrowUp className="h-[18px] w-[18px] stroke-[2.5]" />
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
}

function toChatMessage(message: Message): ChatMessage {
  return {
    role: message.role,
    content: message.content,
  };
}
