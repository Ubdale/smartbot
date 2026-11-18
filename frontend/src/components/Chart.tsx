"use client";

import { useEffect, useRef } from "react";

import { formatValue, prettyColumn } from "@/lib/format";
import type { Chart as ChartSpec } from "@/lib/types";

type Option = Record<string, unknown>;

/** Build an ECharts option from the backend's chart spec and the computed rows. */
function buildOption(spec: ChartSpec, columns: string[], rows: unknown[][]): Option {
  const xi = columns.indexOf(spec.x);
  const yi = columns.indexOf(spec.y);
  const si = spec.series ? columns.indexOf(spec.series) : -1;
  const label = (v: unknown) => formatValue(v);
  const tooltip = { valueFormatter: (v: unknown) => formatValue(v) };

  if (spec.type === "pie") {
    return {
      tooltip: { trigger: "item", ...tooltip },
      legend: { bottom: 0, type: "scroll" },
      series: [
        {
          type: "pie",
          radius: ["40%", "70%"],
          itemStyle: { borderRadius: 4, borderWidth: 2, borderColor: "transparent" },
          data: rows.map((r) => ({ name: label(r[xi]), value: r[yi] })),
        },
      ],
    };
  }

  const categories = [...new Set(rows.map((r) => label(r[xi])))];
  const groups = si >= 0 ? [...new Set(rows.map((r) => label(r[si])))] : [prettyColumn(spec.y)];
  const series = groups.map((g) => {
    const byX = new Map<string, unknown>();
    for (const r of rows) if (si < 0 || label(r[si]) === g) byX.set(label(r[xi]), r[yi]);
    return {
      name: g,
      type: spec.type,
      smooth: spec.type === "line",
      showSymbol: categories.length <= 24,
      data: categories.map((c) => byX.get(c) ?? null),
    };
  });

  return {
    tooltip: { trigger: "axis", ...tooltip },
    legend: si >= 0 ? { bottom: 0, type: "scroll" } : undefined,
    grid: { left: 8, right: 16, top: 16, bottom: si >= 0 ? 40 : 8, containLabel: true },
    xAxis: {
      type: "category",
      data: categories,
      axisLabel: { hideOverlap: true },
    },
    yAxis: { type: "value", axisLabel: { formatter: (v: number) => formatValue(v) } },
    series,
  };
}

export function Chart({
  spec,
  columns,
  rows,
}: {
  spec: ChartSpec;
  columns: string[];
  rows: unknown[][];
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let disposed = false;
    let observer: ResizeObserver | undefined;
    let chart: { setOption(o: Option): void; resize(): void; dispose(): void } | undefined;

    // Loaded on demand so the chart library isn't in the initial page bundle.
    import("echarts").then((echarts) => {
      if (disposed || !ref.current) return;
      const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      chart = echarts.init(ref.current, dark ? "dark" : undefined, { renderer: "svg" });
      chart.setOption({ backgroundColor: "transparent", ...buildOption(spec, columns, rows) });
      observer = new ResizeObserver(() => chart?.resize());
      observer.observe(ref.current);
    });
    return () => {
      disposed = true;
      observer?.disconnect();
      chart?.dispose();
    };
  }, [spec, columns, rows]);

  return <div ref={ref} className="h-72 w-full" role="img" aria-label={`${spec.type} chart`} />;
}
