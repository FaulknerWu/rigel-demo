import { useEffect, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Maximize, RefreshCw, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { fetchGraph, fetchSummary, runIncrementalIndex, type GraphData, type GraphSummary } from '../api/rigel';
import { formatIndexResultMessage, graphNodeTooltipHtml } from './graphPresentation';

const EMPTY_GRAPH: GraphData = { nodes: [], links: [] };
const EMPTY_SUMMARY: GraphSummary = { nodeCount: 0, edgeCount: 0, nodeTypes: [] };

export default function GraphPanel() {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [graphData, setGraphData] = useState<GraphData>(EMPTY_GRAPH);
  const [summary, setSummary] = useState<GraphSummary>(EMPTY_SUMMARY);
  const [isLoading, setIsLoading] = useState(true);
  const [isIndexing, setIsIndexing] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    if (!containerRef.current) return;

    // ForceGraph 需要显式宽高；ResizeObserver 能跟随左右分栏尺寸变化重算画布。
    const observer = new ResizeObserver((entries) => {
      if (!entries[0]) return;
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });

    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    loadGraph();
  }, []);

  useEffect(() => {
    if (!graphRef.current) return;
    // 图谱节点来自代码层级关系，斥力和连线距离调大后更利于快速分辨模块边界。
    graphRef.current.d3Force('charge').strength(-600);
    graphRef.current.d3Force('link').distance(120);
    graphRef.current.d3ReheatSimulation();
  }, [dimensions.width, dimensions.height, graphData]);

  const loadGraph = async () => {
    setIsLoading(true);
    setErrorMessage('');

    try {
      // 图数据和摘要互不依赖，并行请求可以减少初次进入页面的空白时间。
      const [nextGraphData, nextSummary] = await Promise.all([fetchGraph(), fetchSummary()]);
      setGraphData(nextGraphData);
      setSummary(nextSummary);
    } catch (error) {
      setGraphData(EMPTY_GRAPH);
      setSummary(EMPTY_SUMMARY);
      setErrorMessage(error instanceof Error ? error.message : '图谱加载失败');
    } finally {
      setIsLoading(false);
    }
  };

  const zoomBy = (factor: number) => {
    if (!graphRef.current) return;
    graphRef.current.zoom(graphRef.current.zoom() * factor, 300);
  };

  const fitGraph = () => {
    if (!graphRef.current) return;
    graphRef.current.zoomToFit(500, 80);
  };

  const refreshIncrementally = async () => {
    if (isIndexing) return;
    setIsIndexing(true);
    setErrorMessage('');

    try {
      const result = await runIncrementalIndex();
      await loadGraph();
      window.alert(formatIndexResultMessage(result));
    } catch (error) {
      window.alert(`增量索引失败：${error instanceof Error ? error.message : '请求失败'}`);
    } finally {
      setIsIndexing(false);
    }
  };

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-transparent text-slate-300">


      <div ref={containerRef} className="flex-1">
        {dimensions.width > 0 && dimensions.height > 0 && !isLoading && !errorMessage && graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={graphRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphData}
            nodeRelSize={12}
            linkColor={() => 'rgba(255,255,255,0.2)'}
            linkWidth={1.5}
            backgroundColor="#030509"
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkCanvasObjectMode={() => 'after'}
            nodeLabel={(node: any) => graphNodeTooltipHtml(node)}
            linkCanvasObject={(link: any, canvasContext: CanvasRenderingContext2D, globalScale: number) => {
              if (globalScale < 2.5) return;

              const start = link.source;
              const end = link.target;
              if (typeof start !== 'object' || typeof end !== 'object') return;

              // 连线标签只在放大后绘制，避免全图视角下文字盖住节点和边。
              const textPosition = {
                x: start.x + (end.x - start.x) / 2,
                y: start.y + (end.y - start.y) / 2,
              };
              const relativeLink = { x: end.x - start.x, y: end.y - start.y };
              let textAngle = Math.atan2(relativeLink.y, relativeLink.x);
              if (textAngle > Math.PI / 2) textAngle = -(Math.PI - textAngle);
              if (textAngle < -Math.PI / 2) textAngle = -(Math.PI + textAngle);

              const label = link.type;
              const fontSize = 3.5;

              canvasContext.save();
              canvasContext.translate(textPosition.x, textPosition.y);
              canvasContext.rotate(textAngle);
              canvasContext.font = `${fontSize}px Sans-Serif`;

              const textWidth = canvasContext.measureText(label).width;
              const backgroundDimensions = [textWidth, fontSize].map((dimension) => dimension + fontSize * 0.2);

              canvasContext.fillStyle = '#030509';
              canvasContext.fillRect(-backgroundDimensions[0] / 2, -backgroundDimensions[1] / 2, backgroundDimensions[0], backgroundDimensions[1]);
              canvasContext.textAlign = 'center';
              canvasContext.textBaseline = 'middle';
              canvasContext.fillStyle = '#94a3b8';
              canvasContext.fillText(label, 0, 0);
              canvasContext.restore();
            }}
            nodeCanvasObject={(node: any, canvasContext: CanvasRenderingContext2D, globalScale: number) => {
              const radius = 12;

              canvasContext.beginPath();
              canvasContext.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
              canvasContext.fillStyle = node.color;
              canvasContext.fill();
              canvasContext.strokeStyle = 'rgba(255, 255, 255, 0.9)';
              canvasContext.lineWidth = 1.2;
              canvasContext.stroke();

              if (globalScale < 2.5) return;

              // 节点内部只放短标签，完整信息交给 hover tooltip，避免画布局部拥挤。
              const label = node.name;
              const fontSize = radius * 0.45;
              canvasContext.font = `600 ${fontSize}px Sans-Serif`;

              let displayLabel = label;
              const textWidth = canvasContext.measureText(displayLabel).width;
              if (textWidth > radius * 1.8 && displayLabel.length > 5) {
                displayLabel = `${displayLabel.substring(0, 5)}..`;
              }

              canvasContext.textAlign = 'center';
              canvasContext.textBaseline = 'middle';
              canvasContext.fillStyle = 'rgba(0, 0, 0, 0.8)';
              canvasContext.fillText(displayLabel, node.x, node.y);
            }}
          />
        )}
      </div>

      {(isLoading || errorMessage || graphData.nodes.length === 0) && (
        <div className="absolute inset-0 z-[5] flex items-center justify-center bg-[#030509]/80 backdrop-blur-sm">
          <div className="max-w-[320px] rounded-2xl border border-white/10 bg-black/60 p-6 text-center shadow-2xl backdrop-blur-xl ring-1 ring-white/5">
            {isLoading ? (
              <>
                <div className="mx-auto mb-5 relative flex h-12 w-12 items-center justify-center">
                  <div className="absolute inset-0 rounded-full border-2 border-indigo-500/30 border-t-indigo-500 animate-spin"></div>
                  <div className="h-6 w-6 rounded-full bg-indigo-500/20 animate-pulse"></div>
                </div>
                <div className="text-[15px] font-semibold text-white">正在加载代码图谱</div>
                <div className="mt-2 text-[13px] leading-relaxed text-slate-400">读取本地 Rigel 图数据库并生成可视化节点。</div>
              </>
            ) : errorMessage ? (
              <>
                <div className="mx-auto mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-red-500/10">
                  <div className="h-5 w-5 rounded-full bg-red-500/80"></div>
                </div>
                <div className="text-[15px] font-semibold text-white">图谱加载失败</div>
                <div className="mt-2 text-[13px] leading-relaxed text-red-400/80">{errorMessage}</div>
                <button onClick={loadGraph} className="mt-5 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-[13px] font-medium text-slate-200 transition-all hover:bg-white/10 hover:text-white active:scale-95">
                  重新加载
                </button>
              </>
            ) : (
              <>
                <div className="text-[15px] font-semibold text-white">暂无图谱数据</div>
                <div className="mt-2 text-[13px] leading-relaxed text-slate-400">当前数据库没有可展示的节点，请重新执行 rigel index。</div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="absolute bottom-6 left-6 z-10 flex flex-col gap-2 rounded-xl border border-white/10 bg-black/40 p-4 shadow-[0_4_20px_rgba(0,0,0,0.5)] backdrop-blur-xl ring-1 ring-white/5">
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-1.5 rounded-full bg-green-500"></div>
          <div className="text-[10px] uppercase tracking-widest text-slate-400 font-semibold">图谱统计</div>
        </div>
        <div className="flex gap-6 mt-1">
          <div>
            <div className="font-mono text-xl font-bold tracking-wide text-white">{summary.nodeCount}</div>
            <div className="text-[11px] text-slate-500 font-medium">节点数</div>
          </div>
          <div className="h-8 w-[1px] bg-white/10 self-center"></div>
          <div>
            <div className="font-mono text-xl font-bold tracking-wide text-white">{summary.edgeCount}</div>
            <div className="text-[11px] text-slate-500 font-medium">边数</div>
          </div>
        </div>
      </div>

      <div className="absolute bottom-6 right-6 z-10 flex items-center gap-2 rounded-xl border border-white/10 bg-black/40 p-1.5 shadow-[0_4_20px_rgba(0,0,0,0.5)] backdrop-blur-xl ring-1 ring-white/5">
        <button onClick={() => zoomBy(0.8)} className="group flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition-all hover:bg-white/10 hover:text-white active:scale-95" aria-label="缩小图谱">
          <ZoomOut className="h-[18px] w-[18px] transition-transform group-hover:scale-110" />
        </button>
        <button onClick={fitGraph} className="group flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition-all hover:bg-white/10 hover:text-white active:scale-95" aria-label="适配图谱">
          <Maximize className="h-[18px] w-[18px] transition-transform group-hover:scale-110" />
        </button>
        <button onClick={() => zoomBy(1.25)} className="group flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition-all hover:bg-white/10 hover:text-white active:scale-95" aria-label="放大图谱">
          <ZoomIn className="h-[18px] w-[18px] transition-transform group-hover:scale-110" />
        </button>

        <div className="w-[1px] h-5 bg-white/10 mx-1"></div>

        <button onClick={loadGraph} className="group flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition-all hover:bg-white/10 hover:text-white active:scale-95" aria-label="重新加载图谱">
          <RotateCcw className="h-[18px] w-[18px] transition-transform group-hover:-rotate-90" />
        </button>
        <button
          onClick={refreshIncrementally}
          disabled={isIndexing}
          className="group flex h-9 w-9 items-center justify-center rounded-lg text-slate-400 transition-all hover:bg-white/10 hover:text-white active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
          aria-label="执行增量索引"
        >
          <RefreshCw className={`h-[18px] w-[18px] ${isIndexing ? 'animate-spin text-indigo-400' : 'transition-transform group-hover:rotate-90'}`} />
        </button>
      </div>
    </div>
  );
}
