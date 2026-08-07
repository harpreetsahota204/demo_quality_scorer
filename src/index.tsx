import { PluginComponentType, registerComponent } from "@fiftyone/plugins";
import QualityPanel from "./QualityPanel";

registerComponent({
  name: "quality_panel", // must match fiftyone.yml's panels entry
  label: "Episode Quality",
  component: QualityPanel,
  type: PluginComponentType.Panel,
  activator: ({ dataset }: { dataset: unknown }) => dataset !== null,
  surfaces: "grid",
  Icon: undefined,
});
