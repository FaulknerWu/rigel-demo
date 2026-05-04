import ChatPanel from './components/ChatPanel';
import GraphPanel from './components/GraphPanel';

export default function App() {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#05070a] font-sans text-slate-300">
      <div className="flex flex-1 overflow-hidden">
        <div className="relative min-w-0 flex-1 bg-[#020406] lg:flex-[2.5]">
          <GraphPanel />
        </div>
        <div className="min-w-0 flex-1 overflow-hidden border-l border-white/10 bg-[#0a0d14] lg:flex-[1]">
          <ChatPanel />
        </div>
      </div>
    </div>
  );
}
