import type { ApiGraphRAGQuery } from '../api/rigel';

const QUERY_SUMMARY_KEYS = ['query_text', 'items', 'relations'] as const;

export function formatGraphRAGQuery(query: ApiGraphRAGQuery): string {
  const summary = QUERY_SUMMARY_KEYS
    .map((key) => query.args[key])
    .map(formatQueryArgumentPreview)
    .find((value) => value.length > 0);
  return summary ? `${query.name}: ${summary}` : query.name;
}

export function formatQueryArgs(args: Record<string, unknown>): string {
  return JSON.stringify(args);
}

function formatQueryArgumentPreview(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) {
    return value
      .filter((item): item is string => typeof item === 'string' && item.length > 0)
      .slice(0, 3)
      .join(', ');
  }
  return '';
}
