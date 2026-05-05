import { useState, useRef, useEffect } from 'react';
import ChatPanel from './components/ChatPanel';
import GraphPanel from './components/GraphPanel';

export default function App() {
  const [leftWidth, setLeftWidth] = useState(65);
  const isDragging = useRef(false);
  const appRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging.current || !appRef.current) return;
    const { left, width } = appRef.current.getBoundingClientRect();
    const newLeftWidth = ((e.clientX - left) / width) * 100;

    // 限制左右面板宽度，避免任一侧被拖拽到不可用。
    if (newLeftWidth > 20 && newLeftWidth < 80) {
      setLeftWidth(newLeftWidth);
    }
  };

  const handleMouseUp = () => {
    if (isDragging.current) {
      isDragging.current = false;
      document.body.style.cursor = 'default';
      document.body.classList.remove('select-none');
    }
  };

  useEffect(() => {
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <div ref={appRef} className="flex h-screen flex-col overflow-hidden bg-[#030509] font-sans text-slate-300 selection:bg-indigo-500/30">
      <div className="pointer-events-none absolute left-0 top-0 z-0 h-[500px] w-[500px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-indigo-500/10 blur-[120px]" />

      <div className="relative z-10 flex flex-1 overflow-hidden">
        <div
          className="relative min-w-0 bg-[#030509]"
          style={{ width: `${leftWidth}%` }}
        >
          <GraphPanel />
        </div>

        <div
          className="relative z-50 flex w-3 -ml-1.5 -mr-1.5 cursor-col-resize items-center justify-center group"
          onMouseDown={() => {
            isDragging.current = true;
            document.body.style.cursor = 'col-resize';
            document.body.classList.add('select-none');
          }}
        >
          <div className="h-full w-[1px] bg-white/10 transition-all duration-300 group-hover:bg-indigo-500/80 group-hover:w-[3px] group-hover:shadow-[0_0_10px_rgba(99,102,241,0.6)] group-active:bg-indigo-400 group-active:w-[3px]" />
        </div>

        <div className="min-w-0 flex-1 overflow-hidden border-l border-white/5 bg-[#0a0d14]/80 backdrop-blur-2xl">
          <ChatPanel />
        </div>
      </div>
    </div>
  );
}
