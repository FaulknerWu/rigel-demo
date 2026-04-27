import { useEffect, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Maximize, RotateCcw, ZoomIn, ZoomOut } from 'lucide-react';
import { fetchGraph, fetchSummary, type GraphData, type GraphSummary } from '../api/rigel';

const EMPTY_GRAPH: GraphData = { nodes: [], links: [] };
const EMPTY_SUMMARY: GraphSummary = { nodeCount: 0, edgeCount: 0, nodeTypes: [] };

export default function GraphPanel() {
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [graphData, setGraphData] = useState<GraphData>(EMPTY_GRAPH);
  const [summary, setSummary] = useState<GraphSummary>(EMPTY_SUMMARY);
  const [isLoading, setIsLoading] = useState(true);
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

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-[#020406] text-slate-300">
      <div className="pointer-events-none absolute left-6 top-6 z-10 flex flex-col gap-3">
        <div className="pointer-events-auto flex items-center gap-3 rounded-lg border border-white/5 bg-black/40 px-4 py-2 shadow-2xl backdrop-blur-xl">
          <div className="flex h-6 w-6 items-center justify-center rounded-md border border-white/20 bg-white/10">
            <div className="h-2 w-2 rotate-45 rounded-sm bg-white"></div>
          </div>
          <h1 className="text-base font-semibold uppercase tracking-wider text-white">Rigel</h1>
        </div>
      </div>

      <div ref={containerRef} className="flex-1">
        {dimensions.width > 0 && dimensions.height > 0 && !isLoading && !errorMessage && graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={graphRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphData}
            nodeRelSize={12}
            linkColor={() => 'rgba(255,255,255,0.4)'}
            linkWidth={1.5}
            backgroundColor="#020406"
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkCanvasObjectMode={() => 'after'}
            nodeLabel={(node: any) => `
              <div style="background-color: rgba(15, 23, 42, 0.95); color: #e2e8f0; padding: 10px 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); max-width: 250px; font-family: ui-sans-serif, system-ui, sans-serif; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); backdrop-filter: blur(8px);">
                <div style="font-weight: 600; color: #ffffff; margin-bottom: 6px; font-size: 13px;">${node.name || node.id}</div>
                <div style="font-size: 11px; color: #94a3b8; white-space: pre-wrap; line-height: 1.5;">${node.summary || ''}</div>
              </div>
            `}
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

              const label = link.type || '';
              const fontSize = 3.5;

              canvasContext.save();
              canvasContext.translate(textPosition.x, textPosition.y);
              canvasContext.rotate(textAngle);
              canvasContext.font = `${fontSize}px Sans-Serif`;

              const textWidth = canvasContext.measureText(label).width;
              const backgroundDimensions = [textWidth, fontSize].map((dimension) => dimension + fontSize * 0.2);

              canvasContext.fillStyle = '#020406';
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
              canvasContext.fillStyle = node.color || '#888888';
              canvasContext.fill();
              canvasContext.strokeStyle = 'rgba(255, 255, 255, 0.9)';
              canvasContext.lineWidth = 1.2;
              canvasContext.stroke();

              if (globalScale < 2.5) return;

              // 节点内部只放短标签，完整信息交给 hover tooltip，避免画布局部拥挤。
              const label = node.name || node.id;
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
        <div className="absolute inset-0 z-[5] flex items-center justify-center bg-[#020406]">
          <div className="max-w-[320px] rounded-xl border border-white/5 bg-black/40 p-5 text-center backdrop-blur-md">
            {isLoading ? (
              <>
                <div className="mx-auto mb-4 h-10 w-10 animate-pulse rounded-full border border-white/10 bg-white/5"></div>
                <div className="text-sm font-medium text-white">正在加载代码图谱</div>
                <div className="mt-2 text-xs leading-relaxed text-slate-500">读取本地 Rigel 图数据库并生成可视化节点。</div>
              </>
            ) : errorMessage ? (
              <>
                <div className="text-sm font-medium text-white">图谱加载失败</div>
                <div className="mt-2 text-xs leading-relaxed text-slate-500">{errorMessage}</div>
                <button onClick={loadGraph} className="mt-4 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-slate-300 transition-colors hover:bg-white/10 hover:text-white">
                  重新加载
                </button>
              </>
            ) : (
              <>
                <div className="text-sm font-medium text-white">暂无图谱数据</div>
                <div className="mt-2 text-xs leading-relaxed text-slate-500">当前数据库没有可展示的节点，请重新执行 rigel init。</div>
              </>
            )}
          </div>
        </div>
      )}

      <div className="absolute bottom-6 left-6 z-10 flex flex-col gap-1 rounded-lg border border-white/5 bg-black/40 p-4 backdrop-blur-md">
        <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">图谱统计</div>
        <div className="flex gap-6">
          <div>
            <div className="font-mono text-xl tracking-wide text-white">{summary.nodeCount}</div>
            <div className="text-[10px] text-slate-500">节点数</div>
          </div>
          <div className="h-8 w-[1px] bg-white/10"></div>
          <div>
            <div className="font-mono text-xl tracking-wide text-white">{summary.edgeCount}</div>
            <div className="text-[10px] text-slate-500">边数</div>
          </div>
        </div>
      </div>

      <div className="absolute bottom-6 right-6 z-10 flex items-center gap-2">
        <button onClick={() => zoomBy(0.8)} className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 backdrop-blur-md transition-colors hover:bg-white/10 hover:text-white" aria-label="缩小图谱">
          <ZoomOut className="h-4 w-4" />
        </button>
        <button onClick={fitGraph} className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 backdrop-blur-md transition-colors hover:bg-white/10 hover:text-white" aria-label="适配图谱">
          <Maximize className="h-4 w-4" />
        </button>
        <button onClick={() => zoomBy(1.25)} className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 backdrop-blur-md transition-colors hover:bg-white/10 hover:text-white" aria-label="放大图谱">
          <ZoomIn className="h-4 w-4" />
        </button>
        <button onClick={loadGraph} className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 backdrop-blur-md transition-colors hover:bg-white/10 hover:text-white" aria-label="重新加载图谱">
          <RotateCcw className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
