export interface ApiNode {
  id: string;
  type: string;
  label: string;
  properties: ApiNodeProperties;
}

export interface ApiEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
}

export interface ApiNodeProperties extends Record<string, unknown> {
  qualified_name?: string;
  relative_path?: string;
  start_line?: number;
  end_line?: number;
}

export interface ApiGraph {
  nodes: ApiNode[];
  edges: ApiEdge[];
}

export interface ApiGraphResponse {
  status: string;
  graph: ApiGraph;
}

export interface ApiSummary {
  node_count: number;
  edge_count: number;
  node_types: Array<{ type: string; count: number }>;
}

export interface ApiSummaryResponse {
  status: string;
  summary: ApiSummary;
}

export interface ApiChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ApiChatResponse {
  status: string;
  message: ApiChatMessage;
  model: string;
  provider: string;
}

export interface ApiIndexResult {
  mode: string;
  mode_label: string;
  added_files: string[];
  modified_files: string[];
  deleted_files: string[];
  skipped_file_count: number;
  indexed_file_count: number;
  deleted_node_count: number;
  graph_node_count: number;
  graph_edge_count: number;
  duration_ms: number;
}

export interface ApiIncrementalIndexResponse {
  status: string;
  result: ApiIndexResult;
}

export interface GraphNode {
  id: string;
  name: string;
  group: string;
  color: string;
  summary: string;
}

export interface GraphLink {
  id: string;
  source: string;
  target: string;
  type: string;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface GraphSummary {
  nodeCount: number;
  edgeCount: number;
  nodeTypes: Array<{ type: string; count: number }>;
}

export type ChatMessage = ApiChatMessage;

export interface IndexResult {
  mode: string;
  modeLabel: string;
  addedFiles: string[];
  modifiedFiles: string[];
  deletedFiles: string[];
  skippedFileCount: number;
  indexedFileCount: number;
  deletedNodeCount: number;
  graphNodeCount: number;
  graphEdgeCount: number;
  durationMs: number;
}

const NODE_TYPE_COLORS = [
  '#f87171',
  '#fb923c',
  '#fbbf24',
  '#a3e635',
  '#4ade80',
  '#2dd4bf',
  '#38bdf8',
  '#818cf8',
  '#a78bfa',
  '#e879f9',
  '#f472b6',
];

const GRAPH_LIMIT = 500;

export async function fetchGraph(): Promise<GraphData> {
  const response = await fetch(`/api/graph?limit=${GRAPH_LIMIT}`);
  const payload = await readJson<ApiGraphResponse>(response);
  return adaptGraph(payload.graph);
}

export async function fetchSummary(): Promise<GraphSummary> {
  const response = await fetch('/api/summary');
  const payload = await readJson<ApiSummaryResponse>(response);
  return adaptSummary(payload.summary);
}

export async function sendChatMessage(messages: ChatMessage[]): Promise<ChatMessage> {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ messages }),
  });
  const payload = await readJson<ApiChatResponse>(response);
  return payload.message;
}

export async function runIncrementalIndex(): Promise<IndexResult> {
  const response = await fetch('/api/index/incremental', { method: 'POST' });
  const payload = await readJson<ApiIncrementalIndexResponse>(response);
  return adaptIndexResult(payload.result);
}

export function adaptGraph(graph: ApiGraph): GraphData {
  return {
    nodes: graph.nodes.map(adaptNode),
    links: graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: edge.type,
    })),
  };
}

export function adaptSummary(summary: ApiSummary): GraphSummary {
  return {
    nodeCount: summary.node_count,
    edgeCount: summary.edge_count,
    nodeTypes: summary.node_types,
  };
}

export function adaptNode(node: ApiNode): GraphNode {
  return {
    id: node.id,
    name: node.label,
    group: node.type,
    color: colorForNodeType(node.type),
    summary: formatNodeSummary(node),
  };
}

export function adaptIndexResult(result: ApiIndexResult): IndexResult {
  return {
    mode: result.mode,
    modeLabel: result.mode_label,
    addedFiles: result.added_files,
    modifiedFiles: result.modified_files,
    deletedFiles: result.deleted_files,
    skippedFileCount: result.skipped_file_count,
    indexedFileCount: result.indexed_file_count,
    deletedNodeCount: result.deleted_node_count,
    graphNodeCount: result.graph_node_count,
    graphEdgeCount: result.graph_edge_count,
    durationMs: result.duration_ms,
  };
}

function colorForNodeType(nodeType: string): string {
  let hash = 0;
  for (const char of nodeType) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return NODE_TYPE_COLORS[hash % NODE_TYPE_COLORS.length];
}

function formatNodeSummary(node: ApiNode): string {
  const qualifiedName = node.properties.qualified_name;
  const relativePath = node.properties.relative_path;
  const lineRange = formatLineRange(node.properties.start_line, node.properties.end_line);
  const details = [qualifiedName, relativePath, lineRange].filter(Boolean);

  if (details.length === 0) {
    return `${node.type} · ${node.id}`;
  }

  return `${node.type}\n${details.join('\n')}`;
}

function formatLineRange(startLine: number | undefined, endLine: number | undefined): string {
  if (startLine === undefined) {
    return '';
  }
  if (endLine === undefined || endLine === startLine) {
    return `第 ${startLine} 行`;
  }
  return `第 ${startLine}-${endLine} 行`;
}

async function readJson<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json() as Promise<T>;
  }

  let message = `请求失败：${response.status}`;
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === 'string') {
      message = payload.detail;
    }
  } catch {
    message = response.statusText || message;
  }

  throw new Error(message);
}
