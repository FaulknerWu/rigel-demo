export interface ApiNode {
  id: string;
  type: string;
  label: string;
  properties: Record<string, unknown>;
}

export interface ApiEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  properties: Record<string, unknown>;
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

export interface ApiSearchResponse {
  status: string;
  nodes: ApiNode[];
}

export interface ApiRecallResult {
  score: number;
  summary: {
    id: string;
    text: string;
    summary_model: string;
    embedding_model: string;
    embedding_dimensions: number;
    source_hash: string;
  };
  node: ApiNode;
  related: Array<{
    direction: 'incoming' | 'outgoing';
    edge: ApiEdge;
    node: ApiNode;
  }>;
}

export interface ApiRecallResponse {
  status: string;
  results: ApiRecallResult[];
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

export interface RecallResult {
  score: number;
  summary: string;
  node: GraphNode;
  related: Array<{
    direction: 'incoming' | 'outgoing';
    edgeType: string;
    node: GraphNode;
  }>;
}

export type ChatMessage = ApiChatMessage;

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
const SEARCH_LIMIT = 8;

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

export async function searchNodes(query: string): Promise<GraphNode[]> {
  const params = new URLSearchParams({ q: query, limit: String(SEARCH_LIMIT) });
  const response = await fetch(`/api/search?${params.toString()}`);
  const payload = await readJson<ApiSearchResponse>(response);
  return payload.nodes.map(adaptNode);
}

export async function recallNodes(query: string): Promise<RecallResult[]> {
  const params = new URLSearchParams({ q: query, limit: String(SEARCH_LIMIT) });
  const response = await fetch(`/api/recall?${params.toString()}`);
  const payload = await readJson<ApiRecallResponse>(response);
  return payload.results.map(adaptRecallResult);
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
    name: node.label || node.id,
    group: node.type || 'Unknown',
    color: colorForNodeType(node.type || 'Unknown'),
    summary: formatNodeSummary(node),
  };
}

export function adaptRecallResult(result: ApiRecallResult): RecallResult {
  return {
    score: result.score,
    summary: result.summary.text,
    node: adaptNode(result.node),
    related: result.related.map((related) => ({
      direction: related.direction,
      edgeType: related.edge.type,
      node: adaptNode(related.node),
    })),
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
  const qualifiedName = readString(node.properties.qualified_name);
  const relativePath = readString(node.properties.relative_path);
  const lineRange = formatLineRange(node.properties.start_line, node.properties.end_line);
  const details = [qualifiedName, relativePath, lineRange].filter(Boolean);

  if (details.length === 0) {
    return `${node.type} · ${node.id}`;
  }

  return `${node.type}\n${details.join('\n')}`;
}

function formatLineRange(startLine: unknown, endLine: unknown): string {
  if (typeof startLine !== 'number') {
    return '';
  }
  if (typeof endLine !== 'number' || endLine === startLine) {
    return `第 ${startLine} 行`;
  }
  return `第 ${startLine}-${endLine} 行`;
}

function readString(value: unknown): string {
  return typeof value === 'string' ? value : '';
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
