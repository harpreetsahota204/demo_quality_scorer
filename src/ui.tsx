import React from "react";
import { theme } from "./theme";

const TIP_WIDTH = 280;

function InfoTip(props: { text: string }) {
  // position:fixed + viewport clamping so the tooltip never clips against
  // the panel's own scroll container or the window edge
  const [pos, setPos] = React.useState<{ top?: number; bottom?: number; left: number } | null>(null);
  const iconRef = React.useRef<HTMLSpanElement>(null);

  const open = () => {
    const rect = iconRef.current?.getBoundingClientRect();
    if (!rect) return;
    const left = Math.max(8, Math.min(rect.left - 8, window.innerWidth - TIP_WIDTH - 12));
    if (rect.bottom > window.innerHeight * 0.6) {
      setPos({ bottom: window.innerHeight - rect.top + 6, left });
    } else {
      setPos({ top: rect.bottom + 6, left });
    }
  };

  const active = pos !== null;
  return (
    <span
      style={{ position: "relative", display: "inline-flex", marginLeft: 6 }}
      onMouseEnter={open}
      onMouseLeave={() => setPos(null)}
    >
      <span
        ref={iconRef}
        style={{
          width: 14,
          height: 14,
          borderRadius: "50%",
          border: `1px solid ${active ? theme.accent : theme.textDim}`,
          color: active ? theme.accent : theme.textDim,
          fontSize: 10,
          fontWeight: 700,
          fontStyle: "italic",
          fontFamily: "Georgia, serif",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "help",
          lineHeight: 1,
          userSelect: "none",
          transition: "color 120ms, border-color 120ms",
        }}
      >
        i
      </span>
      {pos && (
        <div
          style={{
            position: "fixed",
            ...pos,
            zIndex: 1000,
            width: TIP_WIDTH,
            background: "#111",
            border: `1px solid ${theme.cardBorder}`,
            borderRadius: 6,
            padding: "8px 10px",
            fontSize: 11.5,
            lineHeight: 1.5,
            color: theme.text,
            fontWeight: 400,
            fontStyle: "normal",
            boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
            whiteSpace: "normal",
          }}
        >
          {props.text}
        </div>
      )}
    </span>
  );
}

export function Card(props: {
  title?: string;
  subtitle?: string;
  info?: string;
  /** Right-aligned element in the header row (e.g. an expand toggle) */
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        background: theme.card,
        border: `1px solid ${theme.cardBorder}`,
        borderRadius: 8,
        padding: "12px 14px",
        minWidth: 0,
      }}
    >
      {props.title && (
        <div
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: theme.text,
            display: "flex",
            alignItems: "center",
          }}
        >
          {props.title}
          {props.info && <InfoTip text={props.info} />}
          {props.action && <span style={{ marginLeft: "auto" }}>{props.action}</span>}
        </div>
      )}
      {props.subtitle && (
        <div style={{ fontSize: 11, color: theme.textDim, marginBottom: 6 }}>{props.subtitle}</div>
      )}
      {props.children}
    </div>
  );
}

export function Tabs(props: {
  tabs: { id: string; label: string }[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 4, borderBottom: `1px solid ${theme.cardBorder}` }}>
      {props.tabs.map((tab) => {
        const active = tab.id === props.active;
        return (
          <button
            key={tab.id}
            onClick={() => props.onChange(tab.id)}
            style={{
              background: "none",
              border: "none",
              borderBottom: `2px solid ${active ? theme.accent : "transparent"}`,
              color: active ? theme.accent : theme.textDim,
              padding: "8px 14px",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              transition: "color 120ms",
            }}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}

export function Button(props: {
  label: string;
  onClick: () => void;
  primary?: boolean;
}) {
  return (
    <button
      onClick={props.onClick}
      style={{
        background: props.primary ? theme.accent : "transparent",
        border: `1px solid ${props.primary ? theme.accent : theme.cardBorder}`,
        borderRadius: 6,
        color: props.primary ? "#fff" : theme.text,
        padding: "7px 14px",
        fontSize: 12,
        fontWeight: 600,
        cursor: "pointer",
        transition: "filter 120ms",
      }}
      onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.filter = "brightness(1.15)")}
      onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.filter = "none")}
    >
      {props.label}
    </button>
  );
}

export function Banner(props: { children: React.ReactNode }) {
  return (
    <div
      style={{
        background: "rgba(249, 168, 37, 0.12)",
        border: `1px solid ${theme.warn}`,
        borderRadius: 6,
        color: theme.warn,
        padding: "8px 12px",
        fontSize: 12,
        lineHeight: 1.5,
      }}
    >
      {props.children}
    </div>
  );
}

export function Chip(props: { label: string; color: string }) {
  return (
    <span
      style={{
        background: `${props.color}22`,
        border: `1px solid ${props.color}`,
        borderRadius: 10,
        color: props.color,
        padding: "1px 8px",
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      {props.label}
    </span>
  );
}

const cellStyle: React.CSSProperties = {
  padding: "6px 10px",
  fontSize: 12,
  whiteSpace: "nowrap",
  overflow: "hidden",
  textOverflow: "ellipsis",
  maxWidth: 260,
};

export function DataTable(props: {
  columns: { key: string; label: string; align?: "left" | "right" }[];
  rows: Record<string, React.ReactNode>[];
  rowKeys: string[];
  onRowClick: (key: string) => void;
}) {
  const [hovered, setHovered] = React.useState<string | null>(null);
  return (
    <div
      style={{
        border: `1px solid ${theme.cardBorder}`,
        borderRadius: 8,
        overflow: "auto",
        maxHeight: 420,
      }}
    >
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ position: "sticky", top: 0, background: theme.headerBg, zIndex: 1 }}>
            {props.columns.map((col) => (
              <th
                key={col.key}
                style={{
                  ...cellStyle,
                  textAlign: col.align ?? "left",
                  color: theme.textDim,
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.cardBorder}`,
                }}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.rows.map((row, i) => {
            const key = props.rowKeys[i];
            return (
              <tr
                key={key}
                onClick={() => props.onRowClick(key)}
                onMouseEnter={() => setHovered(key)}
                onMouseLeave={() => setHovered(null)}
                style={{
                  cursor: "pointer",
                  background: hovered === key ? theme.rowHover : "transparent",
                  transition: "background 100ms",
                }}
              >
                {props.columns.map((col) => (
                  <td
                    key={col.key}
                    style={{
                      ...cellStyle,
                      textAlign: col.align ?? "left",
                      color: theme.text,
                      borderBottom: `1px solid ${theme.cardBorder}44`,
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {row[col.key]}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
