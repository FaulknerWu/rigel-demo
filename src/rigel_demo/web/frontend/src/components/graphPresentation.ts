import type { GraphNode, IndexResult } from '../api/rigel';

const HTML_ESCAPE_REPLACEMENTS: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

export function graphNodeTooltipHtml(node: Pick<GraphNode, 'name' | 'summary'>): string {
  const safeName = escapeHtml(node.name);
  const safeSummary = escapeHtml(node.summary);

  return `
    <div style="background-color: rgba(15, 23, 42, 0.95); color: #e2e8f0; padding: 10px 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1); max-width: 250px; font-family: ui-sans-serif, system-ui, sans-serif; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); backdrop-filter: blur(8px);">
      <div style="font-weight: 600; color: #ffffff; margin-bottom: 6px; font-size: 13px;">${safeName}</div>
      <div style="font-size: 11px; color: #94a3b8; white-space: pre-wrap; line-height: 1.5;">${safeSummary}</div>
    </div>
  `;
}

export function formatIndexResultMessage(result: IndexResult): string {
  const lines = [
    '增量索引完成',
    `模式：${result.modeLabel}`,
    `新增文件：${result.addedFiles.length}`,
    `修改文件：${result.modifiedFiles.length}`,
    `删除文件：${result.deletedFiles.length}`,
    `跳过文件：${result.skippedFileCount}`,
    `删除旧节点：${result.deletedNodeCount}`,
    `图谱规模：${result.graphNodeCount} 节点 / ${result.graphEdgeCount} 边`,
    `耗时：${result.durationMs} ms`,
  ];

  for (const [label, files] of [
    ['新增', result.addedFiles],
    ['修改', result.modifiedFiles],
    ['删除', result.deletedFiles],
  ] as const) {
    for (const file of files.slice(0, 5)) {
      lines.push(`${label}：${file}`);
    }
    if (files.length > 5) {
      lines.push(`${label}：另有 ${files.length - 5} 个文件`);
    }
  }

  return lines.join('\n');
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => HTML_ESCAPE_REPLACEMENTS[character]);
}
