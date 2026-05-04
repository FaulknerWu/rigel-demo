import assert from 'node:assert/strict';
import { formatGraphRAGQuery, formatQueryArgs } from './chatPresentation';

assert.equal(
  formatGraphRAGQuery({ name: 'vector_search_seeds', args: { query_text: 'PaymentService' } }),
  'vector_search_seeds: PaymentService',
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
  formatQueryArgs({ query: 'MATCH (node)', limit: 20 }),
  '{"query":"MATCH (node)","limit":20}',
);
