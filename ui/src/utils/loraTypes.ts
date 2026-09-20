export interface LoraPick {
  path: string;
  name: string;
  triggerWords?: string[];
  model?: Record<string, unknown>;
}
export type CloudLora = Pick<LoraPick, 'path' | 'name'>;
