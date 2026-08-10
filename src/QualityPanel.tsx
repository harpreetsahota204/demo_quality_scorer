import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRecoilValue } from "recoil";
import * as fos from "@fiftyone/state";
import { useOperatorExecutor } from "@fiftyone/operators";
import { FamilyNotScored, HealthTab, MotionTab, OutliersTab } from "./tabs";
import { theme } from "./theme";
import { Banner, Button, Tabs } from "./ui";
import { PanelData, PLUGIN } from "./types";

const TABS = [
  { id: "motion", label: "Motion" },
  { id: "health", label: "Health" },
  { id: "outliers", label: "Outliers" },
];

// Which tabs the last run actually produced data for. A family can be
// deselected at scoring time, and its tab then has nothing to draw.
function scoredByTab(data: PanelData): Record<string, boolean> {
  return {
    motion: data.motion_scored,
    health: data.health_scored,
    outliers: data.outliers_scored,
  };
}

export default function QualityPanel() {
  const dataOp = useOperatorExecutor(`${PLUGIN}/get_quality_panel_data`);
  const openOp = useOperatorExecutor(`${PLUGIN}/open_quality_episode`);
  const tagOp = useOperatorExecutor(`${PLUGIN}/tag_quality_episodes`);
  const showOp = useOperatorExecutor(`${PLUGIN}/show_quality_episodes`);
  const promptOp = useOperatorExecutor(`${PLUGIN}/prompt_quality_scorer`);

  const view = useRecoilValue(fos.view);
  const selected = useRecoilValue(fos.selectedSamples);

  const [data, setData] = useState<PanelData | null>(null);
  const [activeTab, setActiveTab] = useState("motion");
  const tabPicked = useRef(false);

  // One backend call per refresh; re-fetch whenever the view changes.
  const viewKey = useMemo(() => JSON.stringify(view ?? []), [view]);
  useEffect(() => {
    dataOp.execute({});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewKey]);

  useEffect(() => {
    if (!dataOp.result) return;
    const next = dataOp.result as unknown as PanelData;
    setData(next);

    // Default to the first tab the last run actually scored, but never
    // fight the user once they've picked a tab themselves.
    if (!tabPicked.current && next.scored) {
      const scored = scoredByTab(next);
      setActiveTab(TABS.find((t) => scored[t.id])?.id ?? "motion");
    }
  }, [dataOp.result]);

  const openEpisode = useCallback(
    (sampleId: string) => openOp.execute({ sample_id: sampleId }),
    [openOp]
  );

  const showEpisodes = useCallback(
    (ids: string[], description: string) =>
      showOp.execute({ sample_ids: ids, description }),
    [showOp]
  );

  const tagScope = selected.size > 0 ? `${selected.size} selected` : `all ${data?.rows.length ?? 0} in view`;
  const tag = useCallback(
    (tagName: string) => tagOp.execute({ tag: tagName, sample_ids: Array.from(selected) }),
    [tagOp, selected]
  );

  if (!data) {
    return (
      <Center>
        <div style={{ color: theme.textDim, fontSize: 13 }}>Loading episode quality…</div>
      </Center>
    );
  }

  if (!data.scored) {
    return (
      <Center>
        <div style={{ textAlign: "center", maxWidth: 420 }}>
          <div style={{ fontSize: 16, fontWeight: 600, color: theme.text, marginBottom: 8 }}>
            No quality scores yet
          </div>
          <div style={{ fontSize: 13, color: theme.textDim, marginBottom: 16 }}>
            Run <b>Compute episode quality</b> to score the current view for motion smoothness,
            sensor health, and outliers.
          </div>
          <Button label="Compute episode quality" primary onClick={() => promptOp.execute({})} />
        </div>
      </Center>
    );
  }

  const scored = scoredByTab(data);
  const familyLabel: Record<string, string> = {
    motion: "Motion",
    health: "Sensor health",
    outliers: "Outliers",
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        color: theme.text,
        fontFamily: "inherit",
      }}
    >
      <div style={{ padding: "8px 16px 0" }}>
        {data.config_version_mismatch && (
          <div style={{ marginBottom: 8 }}>
            <Banner>
              This view mixes quality scores computed under different metric formulas
              (config_version). Rankings and histograms below blend incomparable runs — re-run{" "}
              <b>Compute episode quality</b> on the whole view to make scores comparable again.
            </Banner>
          </div>
        )}
        <Tabs
          tabs={TABS}
          active={activeTab}
          onChange={(id) => {
            tabPicked.current = true;
            setActiveTab(id);
          }}
        />
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
        {scored[activeTab] ? (
          activeTab === "motion" ? (
            <MotionTab data={data} onOpen={openEpisode} onShow={showEpisodes} />
          ) : activeTab === "health" ? (
            <HealthTab data={data} onOpen={openEpisode} onShow={showEpisodes} />
          ) : (
            <OutliersTab data={data} onOpen={openEpisode} />
          )
        ) : (
          <FamilyNotScored family={familyLabel[activeTab]} />
        )}
      </div>

      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          padding: "10px 16px",
          borderTop: `1px solid ${theme.cardBorder}`,
        }}
      >
        <Button label={`Tag ${tagScope}: review`} onClick={() => tag("review")} />
        <Button label={`Tag ${tagScope}: exclude-candidate`} onClick={() => tag("exclude-candidate")} />
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: theme.textDim }}>
          {data.rows.length} scored episode(s) in view
        </span>
        <Button label="Refresh" onClick={() => dataOp.execute({})} />
      </div>
    </div>
  );
}

function Center(props: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        minHeight: 240,
      }}
    >
      {props.children}
    </div>
  );
}
