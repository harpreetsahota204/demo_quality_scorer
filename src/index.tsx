import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import QualityPanel from "./QualityPanel";

// Signal/pulse line: motion smoothness is the panel's headline metric
// family. The <title> doubles as a hover tooltip in the panel tab.
function PanelIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ marginRight: 6, flexShrink: 0 }}
    >
      <title>
        Episode Quality — ranks episodes worst-first on motion smoothness,
        sensor health, and outlier metrics. Click charts to filter and triage.
      </title>
      <polyline points="2 13 6 13 9 5 13 19 16 10 18 13 22 13" />
    </svg>
  );
}

registerComponent({
  name: "quality_panel", // must match fiftyone.yml's panels entry
  label: "Episode Quality",
  component: QualityPanel,
  type: PluginComponentType.Panel,
  activator: ({ dataset }: { dataset: unknown }) => dataset !== null,
  surfaces: "grid",
  Icon: PanelIcon,
});
