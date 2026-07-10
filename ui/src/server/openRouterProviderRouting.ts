export const DEFAULT_OPENROUTER_GROK_MODEL = 'x-ai/grok-4.3';
export const XAI_ZDR_PROVIDER = 'xai/zdr';

type OpenRouterProviderRouting = {
  order?: string[];
  require_parameters?: true;
};

export function openRouterProviderRouting(
  model: string,
  options: { requireParameters?: boolean } = {},
): OpenRouterProviderRouting | undefined {
  const routing: OpenRouterProviderRouting = {};

  if (model.trim() === DEFAULT_OPENROUTER_GROK_MODEL) {
    routing.order = [XAI_ZDR_PROVIDER];
  }
  if (options.requireParameters) {
    routing.require_parameters = true;
  }

  return Object.keys(routing).length > 0 ? routing : undefined;
}
