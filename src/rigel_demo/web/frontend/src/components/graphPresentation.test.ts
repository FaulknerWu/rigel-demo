import assert from 'node:assert/strict';
import { formatIndexResultMessage, graphNodeTooltipHtml } from './graphPresentation';
import type { IndexResult } from '../api/rigel';

const tooltipHtml = graphNodeTooltipHtml({
  name: '<PaymentService & "Checkout">',
  summary: "src/App.java\nalert('x')",
});

assert.match(tooltipHtml, /&lt;PaymentService &amp; &quot;Checkout&quot;&gt;/);
assert.match(tooltipHtml, /alert\(&#39;x&#39;\)/);
assert.doesNotMatch(tooltipHtml, /<PaymentService/);
assert.match(tooltipHtml, /width: max-content/);
assert.match(tooltipHtml, /max-width: min\(720px, calc\(100vw - 32px\)\)/);
assert.match(tooltipHtml, /overflow-wrap: anywhere/);

const indexResult: IndexResult = {
  mode: 'incremental',
  modeLabel: '增量',
  addedFiles: [
    'src/main/java/demo/Added1.java',
    'src/main/java/demo/Added2.java',
    'src/main/java/demo/Added3.java',
    'src/main/java/demo/Added4.java',
    'src/main/java/demo/Added5.java',
    'src/main/java/demo/Added6.java',
  ],
  modifiedFiles: ['src/main/java/demo/Changed.java'],
  deletedFiles: [],
  skippedFileCount: 2,
  indexedFileCount: 7,
  deletedNodeCount: 4,
  graphNodeCount: 30,
  graphEdgeCount: 24,
  durationMs: 120,
};

const message = formatIndexResultMessage(indexResult);
assert.match(message, /增量索引完成/);
assert.match(message, /新增文件：6/);
assert.match(message, /新增：另有 1 个文件/);
assert.match(message, /修改：src\/main\/java\/demo\/Changed\.java/);
