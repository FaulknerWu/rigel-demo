import assert from 'node:assert/strict';
import { formatGraphRAGQuery, formatQueryArgs } from './chatPresentation';

assert.equal(
  formatGraphRAGQuery({ name: 'cypher', args: { query: 'MATCH (node) RETURN node' } }),
  'cypher: MATCH (node) RETURN node',
);

assert.equal(
  formatGraphRAGQuery({ name: 'recall', args: { items: ['PaymentService', 'OrderRepository', 'CheckoutController'] } }),
  'recall: PaymentService, OrderRepository, CheckoutController',
);

assert.equal(
  formatGraphRAGQuery({ name: 'expand_graph', args: { limit: 20 } }),
  'expand_graph',
);

assert.equal(
  formatQueryArgs({ query: 'MATCH (node)', limit: 20 }),
  '{"query":"MATCH (node)","limit":20}',
);
