import assert from 'node:assert/strict';
import {
  adaptGraph,
  adaptIndexResult,
  adaptNode,
  adaptSummary,
  type ApiChatResponse,
  type ApiGraph,
  type ApiIndexResult,
  type ApiNode,
  type ApiSummary,
} from './rigel';

const apiNode: ApiNode = {
  id: 'node-1',
  type: 'Entity',
  label: 'RepositoryIndexer',
  properties: {
    generated_summary: 'RepositoryIndexer 负责扫描仓库并构建图谱。',
    qualified_name: 'rigel_demo.project.RepositoryIndexer',
    relative_path: 'src/rigel_demo/project/repository_indexer.py',
    start_line: 12,
    end_line: 48,
  },
};

const apiGraph: ApiGraph = {
  nodes: [apiNode],
  edges: [
    {
      id: 'edge-1',
      source: 'node-1',
      target: 'node-2',
      type: 'DEPENDS_ON',
      properties: {},
    },
  ],
};

const apiSummary: ApiSummary = {
  node_count: 3,
  edge_count: 2,
  node_types: [{ type: 'Entity', count: 1 }],
};

const graphNode = adaptNode(apiNode);
assert.equal(graphNode.id, 'node-1');
assert.equal(graphNode.name, 'RepositoryIndexer');
assert.equal(graphNode.group, 'Entity');
assert.equal(graphNode.summary, 'RepositoryIndexer 负责扫描仓库并构建图谱。');

const fallbackGraphNode = adaptNode({
  ...apiNode,
  properties: {
    qualified_name: 'rigel_demo.project.RepositoryIndexer',
    relative_path: 'src/rigel_demo/project/repository_indexer.py',
    start_line: 12,
    end_line: 48,
  },
});
assert.match(fallbackGraphNode.summary, /RepositoryIndexer/);
assert.match(fallbackGraphNode.summary, /第 12-48 行/);

const graphData = adaptGraph(apiGraph);
assert.equal(graphData.nodes.length, 1);
assert.equal(graphData.links.length, 1);
assert.deepEqual(graphData.links[0], {
  id: 'edge-1',
  source: 'node-1',
  target: 'node-2',
  type: 'DEPENDS_ON',
});

const graphSummary = adaptSummary(apiSummary);
assert.deepEqual(graphSummary, {
  nodeCount: 3,
  edgeCount: 2,
  nodeTypes: [{ type: 'Entity', count: 1 }],
});

const apiIndexResult: ApiIndexResult = {
  mode: 'incremental',
  mode_label: '增量',
  added_files: ['src/main/java/demo/Added.java'],
  modified_files: ['src/main/java/demo/Changed.java'],
  deleted_files: [],
  skipped_file_count: 2,
  indexed_file_count: 2,
  deleted_node_count: 4,
  graph_node_count: 30,
  graph_edge_count: 24,
  duration_ms: 120,
};

const indexResult = adaptIndexResult(apiIndexResult);
assert.equal(indexResult.modeLabel, '增量');
assert.deepEqual(indexResult.addedFiles, ['src/main/java/demo/Added.java']);
assert.equal(indexResult.modifiedFiles.length, 1);
assert.equal(indexResult.skippedFileCount, 2);
assert.equal(indexResult.graphEdgeCount, 24);

const chatResponse: ApiChatResponse = {
  status: 'success',
  message: { role: 'assistant', content: 'PaymentService 处理付款流程' },
  queries: [{ name: 'vector_search_seeds', args: { query_text: 'PaymentService' } }],
  model: 'gpt-5.2',
  provider: 'openai',
};
assert.equal(chatResponse.queries[0].name, 'vector_search_seeds');
