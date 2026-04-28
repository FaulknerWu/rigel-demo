import assert from 'node:assert/strict';
import { adaptGraph, adaptNode, adaptRecallResult, adaptSummary, type ApiGraph, type ApiNode, type ApiRecallResult, type ApiSummary } from './rigel';

const apiNode: ApiNode = {
  id: 'node-1',
  type: 'Class',
  label: 'RepositoryIndexer',
  properties: {
    qualified_name: 'rigel_demo.indexing.RepositoryIndexer',
    relative_path: 'src/rigel_demo/indexing/repository_indexer.py',
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
      type: 'CALLS',
      properties: {},
    },
  ],
};

const apiSummary: ApiSummary = {
  node_count: 3,
  edge_count: 2,
  node_types: [{ type: 'Class', count: 1 }],
};

const graphNode = adaptNode(apiNode);
assert.equal(graphNode.id, 'node-1');
assert.equal(graphNode.name, 'RepositoryIndexer');
assert.equal(graphNode.group, 'Class');
assert.match(graphNode.summary, /RepositoryIndexer/);
assert.match(graphNode.summary, /第 12-48 行/);

const graphData = adaptGraph(apiGraph);
assert.equal(graphData.nodes.length, 1);
assert.equal(graphData.links.length, 1);
assert.deepEqual(graphData.links[0], {
  id: 'edge-1',
  source: 'node-1',
  target: 'node-2',
  type: 'CALLS',
});

const graphSummary = adaptSummary(apiSummary);
assert.deepEqual(graphSummary, {
  nodeCount: 3,
  edgeCount: 2,
  nodeTypes: [{ type: 'Class', count: 1 }],
});

const recallResult: ApiRecallResult = {
  score: 0.75,
  summary: {
    id: 'summary-1',
    text: 'Class RepositoryIndexer',
    summary_model: 'gpt-5.2',
    embedding_model: 'text-embedding-3-small',
    embedding_dimensions: 3,
    source_hash: 'sha256:test',
  },
  node: apiNode,
  related: [
    {
      direction: 'outgoing',
      edge: apiGraph.edges[0],
      node: {
        ...apiNode,
        id: 'node-2',
        label: 'GraphIR',
      },
    },
  ],
};

const adaptedRecallResult = adaptRecallResult(recallResult);
assert.equal(adaptedRecallResult.score, 0.75);
assert.equal(adaptedRecallResult.summary, 'Class RepositoryIndexer');
assert.equal(adaptedRecallResult.node.name, 'RepositoryIndexer');
assert.deepEqual(adaptedRecallResult.related.map((related) => [related.direction, related.edgeType, related.node.name]), [
  ['outgoing', 'CALLS', 'GraphIR'],
]);
