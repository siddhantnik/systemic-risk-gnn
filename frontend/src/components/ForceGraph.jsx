import { useRef, useEffect, useCallback, useState } from "react";
import * as d3 from "d3";

const SEVERE_THRESHOLD = 0.60;
const ELEVATED_THRESHOLD = 0.40;

export default function ForceGraph({
  nodes,
  edges,
  width = 800,
  height = 600,
  selectedNode,
  onNodeClick,
  shockScores,
  highlightEdges,
  criticalNodeCount,
  rescueNodes,
}) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [graphCriticalCount, setGraphCriticalCount] = useState(0);
  const [syncStatus, setSyncStatus] = useState(null);

  // --- Sync validation: count critical nodes using the backend is_critical flag ---
  useEffect(() => {
    if (!shockScores || criticalNodeCount === undefined) {
      setSyncStatus(null);
      return;
    }

    // Use the is_critical field set by the backend hybrid model (topology + p85 GNN).
    // This guarantees the graph badge always agrees with the summary chip count.
    const visualCritical = shockScores.filter(s => s.is_critical === true).length;
    setGraphCriticalCount(visualCritical);

    if (visualCritical === criticalNodeCount) {
      setSyncStatus("synced");
    } else {
      setSyncStatus("mismatch");
      console.warn(
        `[GRAPH SYNC WARNING] Visual critical nodes (${visualCritical}) != ` +
        `Backend critical_node_count (${criticalNodeCount}). Data may be stale.`
      );
    }
  }, [shockScores, criticalNodeCount]);

  const buildGraph = useCallback(() => {
    if (!nodes || !edges || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const w = width;
    const h = height;
    svg.attr("viewBox", `0 0 ${w} ${h}`);

    const g = svg.append("g");

    // Zoom
    const zoom = d3.zoom()
      .scaleExtent([0.3, 4])
      .on("zoom", (e) => g.attr("transform", e.transform));
    svg.call(zoom);

    // Build simulation data
    const simNodes = nodes.map((n) => ({ ...n }));
    const simLinks = edges.map((e) => ({
      source: e.from,
      target: e.to,
      weight: e.weight || 1,
    }));

    // Degree map for node sizing
    const degreeMap = {};
    simLinks.forEach(l => {
      degreeMap[l.source] = (degreeMap[l.source] || 0) + 1;
      degreeMap[l.target] = (degreeMap[l.target] || 0) + 1;
    });

    // Highlight set
    const hlSet = new Set();
    if (highlightEdges) {
      highlightEdges.forEach(e => {
        hlSet.add(`${e.source}->${e.target}`);
      });
    }

    // Score map
    const scoreMap = {};
    if (shockScores) {
      shockScores.forEach(s => { scoreMap[s.bank_id] = s.score; });
    }

    // Rescue set for bailout animation
    const rescueSet = new Set();
    if (rescueNodes) {
      rescueNodes.forEach(r => rescueSet.add(r.bank_id));
    }

    // Force simulation
    const sim = d3.forceSimulation(simNodes)
      .force("link", d3.forceLink(simLinks).id(d => d.id).distance(60).strength(0.4))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(w / 2, h / 2))
      .force("collision", d3.forceCollide().radius(18));

    simRef.current = sim;

    // --- SVG Defs for glow filter ---
    const defs = svg.append("defs");

    // Red glow for critical nodes
    const critGlow = defs.append("filter").attr("id", "glow-critical");
    critGlow.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "blur");
    critGlow.append("feMerge").selectAll("feMergeNode")
      .data(["blur", "SourceGraphic"]).join("feMergeNode").attr("in", d => d);

    // Orange glow for severe nodes
    const severeGlow = defs.append("filter").attr("id", "glow-severe");
    severeGlow.append("feGaussianBlur").attr("stdDeviation", "2").attr("result", "blur");
    severeGlow.append("feMerge").selectAll("feMergeNode")
      .data(["blur", "SourceGraphic"]).join("feMergeNode").attr("in", d => d);

    // Green/blue glow for rescue nodes
    const rescueGlow = defs.append("filter").attr("id", "glow-rescue");
    rescueGlow.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "blur");
    rescueGlow.append("feMerge").selectAll("feMergeNode")
      .data(["blur", "SourceGraphic"]).join("feMergeNode")
      .attr("in", d => d);

    // Edges
    const linkG = g.append("g").attr("class", "links");
    const link = linkG.selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", d => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}->${typeof d.target === 'object' ? d.target.id : d.target}`;
        return hlSet.has(key) ? "#ffa599" : "var(--outline-variant)";
      })
      .attr("stroke-opacity", d => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}->${typeof d.target === 'object' ? d.target.id : d.target}`;
        return hlSet.has(key) ? 0.9 : 0.2;
      })
      .attr("stroke-width", d => {
        const key = `${typeof d.source === 'object' ? d.source.id : d.source}->${typeof d.target === 'object' ? d.target.id : d.target}`;
        return hlSet.has(key) ? 3 : 1;
      });

    // Nodes
    const nodeG = g.append("g").attr("class", "nodes");
    const node = nodeG.selectAll("g")
      .data(simNodes)
      .join("g")
      .style("cursor", "pointer")
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      )
      .on("click", (e, d) => { if (onNodeClick) onNodeClick(d.id); });

    const getColor = (d) => {
      if (rescueSet.has(d.id)) return "#00e5ff";
      if (selectedNode === d.id) return "#ff3b30";
      if (scoreMap[d.id] !== undefined) {
        const entry = shockScores?.find(s => s.bank_id === d.id);
        // Critical: backend hybrid flag (topology + dynamic p85 GNN threshold)
        if (entry?.is_critical) return "#ff3b30";           // CRITICAL => Deep Red
        const s = scoreMap[d.id];
        if (s >= SEVERE_THRESHOLD)   return "#ff9800";      // SEVERE   => Orange
        if (s >= ELEVATED_THRESHOLD) return "#ffccc5";      // ELEVATED => Light Pink
        return "#55e16b";                                    // STABLE   => Green
      }
      return "var(--primary-container)";
    };

    node.append("circle")
      .attr("r", d => Math.min(25, 4 + Math.log((degreeMap[d.id] || 1) + 1) * 6))
      .attr("fill", getColor)
      .attr("stroke", d => {
        if (rescueSet.has(d.id)) return "#00e5ff";
        if (selectedNode === d.id) return "#ff3b30";
        return "transparent";
      })
      .attr("stroke-width", d => {
        if (rescueSet.has(d.id) || selectedNode === d.id) return 3;
        return 0;
      })
      .attr("filter", d => {
        if (rescueSet.has(d.id)) return "url(#glow-rescue)";
        const entry = shockScores?.find(s => s.bank_id === d.id);
        if (entry?.is_critical) return "url(#glow-critical)";
        if (scoreMap[d.id] >= SEVERE_THRESHOLD) return "url(#glow-severe)";
        return null;
      });

    // Rescue pulse ring animation
    node.filter(d => rescueSet.has(d.id))
      .append("circle")
      .attr("r", d => Math.min(30, 8 + Math.log((degreeMap[d.id] || 1) + 1) * 7.5))
      .attr("fill", "none")
      .attr("stroke", "#00e5ff")
      .attr("stroke-width", 2)
      .attr("opacity", 0.6)
      .each(function pulseLoop() {
        d3.select(this)
          .attr("r", d => Math.min(30, 8 + Math.log((degreeMap[d.id] || 1) + 1) * 7.5))
          .attr("opacity", 0.6)
          .transition().duration(1200).ease(d3.easeSinInOut)
          .attr("r", d => Math.min(45, 16 + Math.log((degreeMap[d.id] || 1) + 1) * 10))
          .attr("opacity", 0)
          .on("end", pulseLoop);
      });

    // Glow for selected
    if (selectedNode) {
      node.filter(d => d.id === selectedNode)
        .append("circle")
        .attr("r", d => Math.min(28, 10 + Math.log((degreeMap[d.id] || 1) + 1) * 6))
        .attr("fill", "none")
        .attr("stroke", "#ff3b30")
        .attr("stroke-width", 2)
        .attr("opacity", 0.4);
    }

    // Labels (only show on hover)
    node.append("text")
      .text(d => d.label || d.id)
      .attr("dx", 10)
      .attr("dy", 4)
      .attr("fill", "var(--on-surface-variant)")
      .attr("font-size", "9px")
      .attr("font-family", "var(--font-mono)")
      .attr("opacity", 0)
      .attr("class", "node-label");

    // Rescue labels always visible
    node.filter(d => rescueSet.has(d.id))
      .select(".node-label")
      .attr("opacity", 1)
      .attr("fill", "#00e5ff");

    node.on("mouseenter", function () {
      d3.select(this).select(".node-label").transition().duration(150).attr("opacity", 1);
      d3.select(this).select("circle").transition().duration(150).attr("r", d => Math.min(30, 6 + Math.log((degreeMap[d.id] || 1) + 1) * 7.5));
    }).on("mouseleave", function () {
      const d = d3.select(this).datum();
      if (!rescueSet.has(d.id)) {
        d3.select(this).select(".node-label").transition().duration(150).attr("opacity", 0);
      }
      d3.select(this).select("circle").transition().duration(150).attr("r", d => Math.min(25, 4 + Math.log((degreeMap[d.id] || 1) + 1) * 6));
    });

    sim.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    return () => sim.stop();
  }, [nodes, edges, width, height, selectedNode, onNodeClick, shockScores, highlightEdges, rescueNodes]);

  useEffect(() => {
    const cleanup = buildGraph();
    return () => { if (cleanup) cleanup(); };
  }, [buildGraph]);

  return (
    <div className="graph-container" style={{ position: "relative" }}>
      <svg ref={svgRef} style={{ width: "100%", height: "100%" }} />

      {/* Verification Badge */}
      {syncStatus && (
        <div style={{
          position: "absolute",
          top: 8,
          right: 8,
          padding: "4px 10px",
          borderRadius: 4,
          fontFamily: "var(--font-mono)",
          fontSize: "0.625rem",
          letterSpacing: "0.03em",
          background: syncStatus === "synced"
            ? "rgba(85, 225, 107, 0.15)"
            : "rgba(255, 59, 48, 0.15)",
          color: syncStatus === "synced"
            ? "var(--secondary)"
            : "var(--tertiary-container)",
          border: `1px solid ${syncStatus === "synced"
            ? "rgba(85, 225, 107, 0.3)"
            : "rgba(255, 59, 48, 0.3)"}`,
        }}>
          {syncStatus === "synced"
            ? `GRAPH SYNCED: ${graphCriticalCount} CRITICAL NODES`
            : `SYNC WARN: ${graphCriticalCount} vs ${criticalNodeCount}`
          }
        </div>
      )}
    </div>
  );
}
