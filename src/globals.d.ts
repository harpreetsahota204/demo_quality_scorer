// These packages are provided as globals by the FiftyOne App at runtime
// (externalized in vite.config.ts), so they aren't installed locally.
declare module "@fiftyone/plugins" {
  export const PluginComponentType: { Panel: unknown };
  export function registerComponent(config: Record<string, unknown>): void;
}

declare module "@fiftyone/operators" {
  export interface OperatorExecutor {
    execute(params: Record<string, unknown>): void;
    result?: Record<string, unknown> | null;
    error?: unknown;
    isLoading?: boolean;
  }
  export function useOperatorExecutor(uri: string): OperatorExecutor;
}

declare module "@fiftyone/state" {
  import type { RecoilValueReadOnly } from "recoil";
  export const view: RecoilValueReadOnly<unknown[]>;
  export const selectedSamples: RecoilValueReadOnly<Set<string>>;
  export const dataset: RecoilValueReadOnly<unknown>;
}
