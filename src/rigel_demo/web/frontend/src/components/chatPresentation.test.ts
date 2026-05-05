import assert from 'node:assert/strict';
import { formatGraphRAGQuery, formatQueryArgs } from './chatPresentation';

assert.equal(
  formatGraphRAGQuery({ name: 'vector_search_seeds', args: { query_texts: ['PaymentService', '付款流程'] } }),
  'vector_search_seeds: PaymentService, 付款流程',
);

assert.equal(
  formatGraphRAGQuery({ name: 'expand_neighbors', args: { items: ['PaymentService', 'OrderRepository', 'CheckoutController'] } }),
  'expand_neighbors: PaymentService, OrderRepository, CheckoutController',
);

assert.equal(
  formatGraphRAGQuery({ name: 'expand_graph', args: { limit: 20 } }),
  'expand_graph',
);

assert.equal(
  formatQueryArgs({ query_texts: ['PaymentService'], items: [{ rerank_score: 0.93, vector_score: 0.72 }] }),
  '{"query_texts":["PaymentService"],"items":[{"rerank_score":0.93,"vector_score":0.72}]}',
);
