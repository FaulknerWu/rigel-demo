import { useState, useRef, useEffect } from 'react';
import { ArrowUp, Bot, Database, User } from 'lucide-react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { sendChatMessage, type ApiGraphRAGQuery, type ChatMessage } from '../api/rigel';
import { formatGraphRAGQuery, formatQueryArgs } from './chatPresentation';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  queries?: ApiGraphRAGQuery[];
}

const markdownComponents: Components = {
  h1: ({ children }) => <h1 className="mb-2 mt-3 text-base font-semibold leading-snug text-slate-900 first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-2 mt-3 text-[15px] font-semibold leading-snug text-slate-800 first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-1.5 mt-3 text-sm font-semibold leading-snug text-slate-700 first:mt-0">{children}</h3>,
  h4: ({ children }) => <h4 className="mb-1.5 mt-2.5 text-[13px] font-semibold leading-snug text-slate-600 first:mt-0">{children}</h4>,
  p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0 text-slate-700 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5 text-slate-700 first:mt-0 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5 text-slate-700 first:mt-0 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="pl-0.5">{children}</li>,
  blockquote: ({ children }) => <blockquote className="my-2 border-l-2 border-indigo-500/30 bg-indigo-50/50 pl-3 py-1 rounded-r-md text-slate-600 italic">{children}</blockquote>,
  a: ({ children, href }) => (
    <a className="font-medium text-indigo-600 underline decoration-indigo-600/30 underline-offset-2 hover:text-indigo-500 transition-colors" href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
  code: ({ children, className }) => {
    const isCodeBlock = className?.startsWith('language-');

    if (isCodeBlock) {
      return <code className={className}>{children}</code>;
    }

    return <code className="rounded-md bg-slate-100 px-1.5 py-0.5 text-[12px] font-mono text-indigo-600 break-words border border-slate-200">{children}</code>;
  },
  pre: ({ children }) => <pre className="my-3 max-w-full overflow-x-auto rounded-xl border border-slate-200 bg-slate-50 p-4 text-[13px] leading-relaxed text-slate-700 shadow-sm">{children}</pre>,
  table: ({ children }) => (
    <div className="my-3 max-w-full overflow-x-auto rounded-lg border border-slate-200 bg-white">
      <table className="min-w-max border-collapse text-left text-[13px] w-full">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="border-b border-slate-200 bg-slate-50 px-3 py-2 font-semibold text-slate-800">{children}</th>,
  td: ({ children }) => <td className="border-b border-slate-100 px-3 py-2 align-top text-slate-700">{children}</td>,
};

export default function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isResponding, setIsResponding] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isResponding]);

  const handleSend = async () => {
    const query = input.trim();
    if (!query || isResponding) return;

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
    <aside className="relative flex h-full min-w-0 flex-col overflow-hidden bg-white text-slate-700">
      <div className="flex shrink-0 items-center gap-3 border-b border-slate-200 bg-white/80 p-4 backdrop-blur-md">
        <div className="flex flex-col">
          <span className="text-[14px] font-bold tracking-wide text-slate-900">智能图谱助手</span>
        </div>
      </div>

      <div className="flex flex-1 flex-col overflow-y-auto overflow-x-hidden p-5 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-slate-200">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-slate-50 ring-1 ring-slate-200">
              <Bot className="h-10 w-10 text-slate-700" />
            </div>
            <h2 className="bg-gradient-to-br from-slate-900 to-slate-600 bg-clip-text text-2xl font-semibold tracking-tight text-transparent">
              您想探索什么？
            </h2>
            <p className="mt-4 max-w-[280px] text-[14px] leading-relaxed text-slate-500">
              询问有关代码架构、依赖关系和复杂模块结构的问题，我将通过分析代码图谱为您解答。
            </p>
          </div>
        ) : (
          <div className="flex min-w-0 flex-col gap-6">
            {messages.map((message, index) => (
              <div key={index} className={`flex w-full min-w-0 gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl shadow-sm ring-1 ring-slate-200 ${
                  message.role === 'user'
                    ? 'bg-slate-100 text-slate-600'
                    : 'bg-slate-800 text-white'
                }`}>
                  {message.role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                </div>

                <div className={`flex min-w-0 flex-1 flex-col ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                  {message.role === 'assistant' && message.queries && message.queries.length > 0 && (
                    <div className="mb-3 flex max-w-full flex-wrap gap-2">
                      {message.queries.map((query, queryIndex) => (
                        <span
                          key={`${query.name}-${queryIndex}`}
                          className="inline-flex max-w-full items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-100"
                          title={formatQueryArgs(query.args)}
                        >
                          <Database className="h-3 w-3 shrink-0" />
                          <span className="truncate">{formatGraphRAGQuery(query)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                  <MessageContent message={message} />
                </div>
              </div>
            ))}
            {isResponding && (
              <div className="flex gap-4">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-slate-800 text-white shadow-sm ring-1 ring-slate-200">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-5 py-3.5 text-[14px] shadow-sm">
                  <span className="flex h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400"></span>
                  <span className="flex h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: '0.15s' }}></span>
                  <span className="flex h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" style={{ animationDelay: '0.3s' }}></span>
                  <span className="ml-2 text-slate-600">正在分析图谱...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-slate-200 bg-slate-50/80 p-4 pb-6 backdrop-blur-xl">
        <div className="relative flex flex-col rounded-2xl border border-slate-200 bg-white p-1 shadow-sm ring-1 ring-transparent transition-all focus-within:border-slate-400 focus-within:ring-slate-400/20">
          <textarea
            className="min-h-[80px] w-full resize-none bg-transparent px-4 py-3 text-[14px] text-slate-900 outline-none placeholder:text-slate-400 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-slate-200"
            placeholder="输入您的问题..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                handleSend();
              }
            }}
          ></textarea>

          <div className="flex items-center justify-between px-2 pb-2">
            <div className="text-[11px] text-slate-500 pl-2 select-none">
              Press <kbd className="rounded border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-sans text-slate-500">Enter</kbd> to send
            </div>
            <button
              onClick={handleSend}
              disabled={!input.trim() || isResponding}
              className={`group flex h-9 w-9 items-center justify-center rounded-xl transition-all duration-300 ${
                input.trim() && !isResponding
                  ? 'bg-slate-800 text-white shadow-sm hover:bg-slate-700'
                  : 'bg-slate-100 text-slate-400 cursor-not-allowed'
              }`}
              aria-label="发送消息"
            >
              <ArrowUp className={`h-[18px] w-[18px] stroke-[2.5] transition-transform ${input.trim() && !isResponding ? 'group-hover:-translate-y-0.5' : ''}`} />
            </button>
          </div>
        </div>
      </div>
    </aside>
  );
}

function MessageContent({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  const messageClasses = `max-w-full overflow-hidden rounded-2xl px-5 py-3.5 text-[14px] leading-relaxed break-words shadow-sm ${
    isUser
      ? 'bg-slate-50 text-slate-800 rounded-tr-sm border border-slate-200'
      : 'bg-white text-slate-800 rounded-tl-sm border border-slate-200'
  }`;

  return (
    <div className={messageClasses}>
      <Markdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {message.content}
      </Markdown>
    </div>
  );
}

function toChatMessage(message: Message): ChatMessage {
  return {
    role: message.role,
    content: message.content,
  };
}
