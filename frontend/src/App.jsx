import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Navbar from "./components/Navbar";
import VaultScene from "./components/VaultScene";
import ForceGraph from "./components/ForceGraph";
import RiskTable from "./components/RiskTable";
import MetricsPanel from "./components/MetricsPanel";
import BailoutPanel from "./components/BailoutPanel";
import { generateNetwork, simulateShock, explainRisk } from "./services/api";
import "./theme.css";

export default function App() {
  const [activeSlide, setActiveSlide] = useState(0);

  // --- Data states ---
  const [network, setNetwork] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [shockScores, setShockScores] = useState(null);
  const [shockMeta, setShockMeta] = useState(null); // {critical_node_count, total_banks, ...}
  const [shockLoading, setShockLoading] = useState(false);
  const [explanations, setExplanations] = useState({});
  const [highlightEdges, setHighlightEdges] = useState(null);
  const [rescueNodes, setRescueNodes] = useState(null);

  // --- Target Selection Search ---
  const [searchQuery, setSearchQuery] = useState("");

  // --- Scroll tracking ---
  useEffect(() => {
    const onScroll = () => {
      const y = window.scrollY;
      const vh = window.innerHeight;
      if (y < vh * 0.5) setActiveSlide(0);
      else if (y < vh * 1.5) setActiveSlide(1);
      else if (y < vh * 2.5) setActiveSlide(2);
      else setActiveSlide(3);
    };
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // --- Scroll to dashboard after shock scores render ---
  // useEffect runs AFTER React re-renders with shockScores, so slide-dashboard
  // is already visible when scrollIntoView is called — unlike a raw setTimeout.
  useEffect(() => {
    if (shockScores) {
      const el = document.getElementById("slide-dashboard");
      if (el) el.scrollIntoView({ behavior: "smooth" });
    }
  }, [shockScores]);

  // --- Handlers ---
  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setShockScores(null);
    setShockMeta(null);
    setSelectedNode(null);
    setExplanations({});
    setHighlightEdges(null);
    setRescueNodes(null);
    try {
      const data = await generateNetwork();
      setNetwork(data);
      setTimeout(() => {
        document.getElementById("slide-target")?.scrollIntoView({ behavior: "smooth" });
      }, 600);
    } catch (err) {
      console.error("Network generation failed:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleNodeClick = useCallback((nodeId) => {
    setSelectedNode(nodeId);
  }, []);

  const handleShock = useCallback(async () => {
    if (!selectedNode) return;
    setShockLoading(true);
    setRescueNodes(null);
    try {
      const data = await simulateShock(selectedNode, 1.0);
      setShockScores(data.risk_scores);
      setShockMeta({
        critical_node_count: data.critical_node_count,
        total_banks: data.total_banks,
        safe_node_count: data.safe_node_count,
        threshold: data.critical_threshold,
      });
      // Scroll is handled by the useEffect watching shockScores
    } catch (err) {
      console.error("Shock simulation failed:", err);
      alert("Shock simulation failed: " + (err?.response?.data?.detail || err.message));
    } finally {
      setShockLoading(false);
    }
  }, [selectedNode]);

  const handleExplain = useCallback(async (bankId) => {
    try {
      const data = await explainRisk(bankId);
      setExplanations(prev => ({ ...prev, [bankId]: data }));
      if (data.toxic_edges) {
        setHighlightEdges(data.toxic_edges);
      }
    } catch (err) {
      setExplanations(prev => ({ ...prev, [bankId]: { error: String(err) } }));
    }
  }, []);

  const handleRescueNodes = useCallback((allocations) => {
    setRescueNodes(allocations);
  }, []);

  const ease = [0.16, 1, 0.3, 1];

  return (
    <div>
      <Navbar activeSlide={activeSlide} />

      {/* ====== SLIDE 1: INTRO ====== */}
      <section id="slide-intro" className="slide" style={{ background: "var(--surface)" }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1.2fr",
          width: "100%",
          maxWidth: 1400,
          margin: "0 auto",
          padding: "80px 40px 40px",
          gap: 40,
          alignItems: "center",
          minHeight: "100vh",
        }}>
          {/* Left: Text */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2, ease, delay: 0.2 }}
          >
            <div className="label-sm" style={{ marginBottom: 12, color: "var(--primary-container)" }}>
              GRASP // GRAPH-BASED RISK ASSESSMENT & SYSTEMIC PREDICTION
            </div>
            <h1 style={{
              fontSize: "clamp(2.5rem, 5vw, 4rem)",
              lineHeight: 1.05,
              marginBottom: 20,
              letterSpacing: "-0.03em",
            }}>
              GRASP<br />
              <span style={{ color: "var(--primary-container)" }}>SYSTEM</span>
            </h1>
            <p style={{
              color: "var(--on-surface-variant)",
              fontSize: "0.9375rem",
              maxWidth: 460,
              marginBottom: 32,
              lineHeight: 1.7,
            }}>
              Systemic risk is not a failure of the machine, but an inherent property of the
              network. GRASP maps the hidden corridors of interbank liquidity,
              predicts cascading failures, and prescribes optimal bailout strategies.
            </p>
            <button
              className="btn-primary"
              onClick={() => document.getElementById("slide-generate")?.scrollIntoView({ behavior: "smooth" })}
            >
              INITIALIZE SECURE UPLINK
            </button>
            <div style={{
              marginTop: 40,
              display: "flex",
              gap: 32,
              color: "var(--on-surface-variant)",
            }}>
              <div>
                <div className="mono" style={{ fontSize: "1.5rem", color: "var(--primary-container)" }}>
                  {network ? network.nodes.length.toLocaleString() : "4,548"}
                </div>
                <div className="label-sm">BANKS MONITORED</div>
              </div>
              <div>
                <div className="mono" style={{ fontSize: "1.5rem", color: "var(--secondary)" }}>
                  {network ? network.edges.length.toLocaleString() : "12,324"}
                </div>
                <div className="label-sm">INTERBANK LINKS</div>
              </div>
              <div>
                <div className="mono" style={{ fontSize: "1.5rem", color: "var(--tertiary)" }}>6</div>
                <div className="label-sm">CAMELS FEATURES</div>
              </div>
            </div>
          </motion.div>

          {/* Right: 3D Vault */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 40 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 1.5, ease, delay: 0.5 }}
            style={{ height: "70vh", minHeight: 500, background: "transparent" }}
          >
            <VaultScene />
          </motion.div>
        </div>
      </section>

      {/* ====== SLIDE 2: GENERATE ====== */}
      <section id="slide-generate" className="slide" style={{
        background: "var(--surface-container-low)",
        justifyContent: "center",
        flexDirection: "column",
        padding: "80px 40px",
      }}>
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1.2, ease }}
          viewport={{ once: true, margin: "-100px" }}
          style={{ textAlign: "center", maxWidth: 700, margin: "0 auto 40px" }}
        >
          <div className="label-sm" style={{ marginBottom: 12 }}>NETWORK TOPOLOGY ENGINE</div>
          <h2 style={{ fontSize: "2rem", marginBottom: 16 }}>
            Generate Interbank <span style={{ color: "var(--primary-container)" }}>Scenario</span>
          </h2>
          <p style={{ color: "var(--on-surface-variant)", marginBottom: 24, fontSize: "0.875rem" }}>
            Creates a Barabasi-Albert scale-free network simulating real-world interbank exposure
            distributions. Each node represents a financial institution. Edges are weighted loan exposures.
          </p>
          <button
            className="btn-primary"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? (
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span className="spinner" /> GENERATING...
              </span>
            ) : "GENERATE RANDOM BANK SCENARIO"}
          </button>
        </motion.div>

        {/* D3 Graph appears here */}
        <AnimatePresence>
          {network && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 40 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 1.0, ease }}
              style={{
                width: "100%",
                maxWidth: 1000,
                height: 500,
                margin: "0 auto",
                background: "var(--surface-container-lowest)",
                borderRadius: 8,
                border: "1px solid var(--glass-border)",
                overflow: "hidden",
              }}
            >
              <ForceGraph
                nodes={network.nodes}
                edges={network.edges}
                width={1000}
                height={500}
                selectedNode={selectedNode}
                onNodeClick={handleNodeClick}
                shockScores={null}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* ====== SLIDE 3: TARGET ACQUISITION ====== */}
      <section id="slide-target" className="slide" style={{
        background: "var(--surface)",
        padding: "80px 40px",
      }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: network ? "1.3fr 0.7fr" : "1fr",
          width: "100%",
          maxWidth: 1400,
          margin: "0 auto",
          gap: 32,
          alignItems: "start",
        }}>
          {/* Left: Shrunk graph */}
          {network && (
            <motion.div
              initial={{ scale: 1, y: 40, opacity: 0 }}
              whileInView={{ scale: 0.95, y: 0, opacity: 1 }}
              transition={{ duration: 1.2, ease }}
              viewport={{ once: true }}
              style={{
                background: "var(--surface-container-lowest)",
                borderRadius: 8,
                border: "1px solid var(--glass-border)",
                height: 500,
                overflow: "hidden",
                transformOrigin: "top left",
              }}
            >
              <ForceGraph
                nodes={network.nodes}
                edges={network.edges}
                width={850}
                height={500}
                selectedNode={selectedNode}
                onNodeClick={handleNodeClick}
                shockScores={shockScores}
                highlightEdges={highlightEdges}
                criticalNodeCount={shockMeta?.critical_node_count}
                rescueNodes={rescueNodes}
              />
            </motion.div>
          )}

          {/* Right: Target panel */}
          <motion.div
            initial={{ opacity: 0, x: 40, y: 20 }}
            whileInView={{ opacity: 1, x: 0, y: 0 }}
            transition={{ duration: 1.2, ease, delay: 0.2 }}
            viewport={{ once: true, margin: "-50px" }}
          >
            <div className="glass" style={{ position: "sticky", top: 100, maxHeight: "80vh", display: "flex", flexDirection: "column" }}>
              <div className="label-sm" style={{ marginBottom: 16, color: "var(--tertiary-container)" }}>
                TARGET ACQUISITION PROTOCOL
              </div>
              <h3 style={{ fontSize: "1.5rem", marginBottom: 12 }}>
                Bankrupt a Node
              </h3>
              <p style={{ color: "var(--on-surface-variant)", fontSize: "0.8125rem", marginBottom: 20 }}>
                Filter and select a node from the monitored network, then execute a bankruptcy
                shock simulation.
              </p>

              {/* Target Search / Selection Table Component */}
              {network ? (
                <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
                  <input
                    type="text"
                    placeholder="Search bank name or ID..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{
                      padding: "10px 14px",
                      background: "var(--surface-container-low)",
                      border: "1px solid var(--outline-variant)",
                      color: "var(--on-surface)",
                      borderRadius: 4,
                      marginBottom: 12,
                      width: "100%",
                      fontFamily: "var(--font-primary)"
                    }}
                  />

                  <div style={{
                    flex: 1,
                    overflowY: "auto",
                    background: "var(--surface-container-lowest)",
                    border: "1px solid var(--glass-border)",
                    borderRadius: 4,
                    marginBottom: 20,
                    maxHeight: "35vh"
                  }}>
                    {network.nodes
                      .filter(n => n.id.toLowerCase().includes(searchQuery.toLowerCase()))
                      .map(node => (
                        <div
                          key={node.id}
                          onClick={() => handleNodeClick(node.id)}
                          style={{
                            padding: "12px",
                            borderBottom: "1px solid var(--glass-border)",
                            cursor: "pointer",
                            background: selectedNode === node.id ? "rgba(255, 59, 48, 0.15)" : "transparent",
                            color: selectedNode === node.id ? "#ffccc5" : "inherit",
                            transition: "background 0.2s"
                          }}
                        >
                          <span className="mono" style={{ fontSize: "0.8125rem" }}>{node.id}</span>
                        </div>
                      ))}
                  </div>
                </div>
              ) : null}

              {selectedNode ? (
                <div style={{ flexShrink: 0 }}>
                  <div style={{
                    padding: "12px 16px",
                    background: "var(--surface-container-lowest)",
                    borderRadius: 4,
                    marginBottom: 16,
                    borderLeft: "3px solid var(--tertiary-container)",
                  }}>
                    <div className="label-sm">SELECTED TARGET</div>
                    <div className="mono" style={{ fontSize: "1.0rem", color: "var(--tertiary-container)" }}>
                      {selectedNode}
                    </div>
                  </div>
                  <button
                    className="btn-primary"
                    style={{
                      width: "100%",
                      background: "linear-gradient(135deg, var(--tertiary-container), #ff3b30)",
                    }}
                    onClick={handleShock}
                    disabled={shockLoading}
                  >
                    {shockLoading ? (
                      <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
                        <span className="spinner" /> SIMULATING SHOCK...
                      </span>
                    ) : "EXECUTE BANKRUPTCY SHOCK"}
                  </button>
                </div>
              ) : (
                <div style={{
                  flexShrink: 0,
                  padding: "20px",
                  textAlign: "center",
                  color: "var(--outline)",
                  border: "1px dashed var(--outline-variant)",
                  borderRadius: 4,
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.75rem",
                }}>
                  {network
                    ? "AWAITING TARGET SELECTION..."
                    : "GENERATE A NETWORK FIRST"
                  }
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </section>

      {/* ====== SLIDE 4: DASHBOARD ====== */}
      <section id="slide-dashboard" className="slide" style={{
        background: "var(--surface-container-low)",
        padding: "80px 40px",
        minHeight: "100vh",
        alignItems: "start",
      }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: shockScores ? "0.8fr 1.2fr" : "1fr",
          width: "100%",
          maxWidth: 1400,
          margin: "0 auto",
          gap: 32,
          alignItems: "start",
          paddingTop: 40,
        }}>
          {/* Left column: graph + bailout */}
          <div>
            {/* Smaller graph */}
            {network && shockScores && (
              <motion.div
                initial={{ scale: 0.95, opacity: 0, y: 30 }}
                whileInView={{ scale: 0.85, opacity: 1, y: 0 }}
                transition={{ duration: 1.2, ease }}
                viewport={{ once: true }}
                style={{
                  background: "var(--surface-container-lowest)",
                  borderRadius: 8,
                  border: "1px solid var(--glass-border)",
                  height: 400,
                  overflow: "hidden",
                  transformOrigin: "top left",
                  position: "sticky",
                  top: 100,
                  marginBottom: 24,
                }}
              >
                <ForceGraph
                  nodes={network.nodes}
                  edges={network.edges}
                  width={600}
                  height={400}
                  selectedNode={selectedNode}
                  shockScores={shockScores}
                  highlightEdges={highlightEdges}
                  criticalNodeCount={shockMeta?.critical_node_count}
                  rescueNodes={rescueNodes}
                />
              </motion.div>
            )}

            {/* Bailout Panel (below graph on left) */}
            {shockScores && (
              <BailoutPanel
                shockedNode={selectedNode}
                onRescueNodes={handleRescueNodes}
              />
            )}
          </div>

          {/* Right: Dashboard output */}
          <motion.div
            initial={{ opacity: 0, y: 60 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2, ease, delay: 0.1 }}
            viewport={{ once: true, margin: "-100px" }}
          >
            {shockScores ? (
              <div>
                <div className="label-sm" style={{ marginBottom: 8 }}>
                  POST-MORTEM ANALYSIS // SHOCK: {selectedNode}
                </div>
                <h2 style={{ fontSize: "1.75rem", marginBottom: 4 }}>
                  Contagion <span style={{ color: "var(--tertiary-container)" }}>Dashboard</span>
                </h2>
                <p style={{
                  color: "var(--on-surface-variant)",
                  fontSize: "0.8125rem",
                  marginBottom: 24,
                }}>
                  Risk propagation scores after simulated bankruptcy of {selectedNode}.
                  Highest-risk nodes have a WHY? button for GNNExplainer analysis.
                </p>

                {/* Summary chips */}
                <div style={{ display: "flex", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
                  <div className="glass" style={{ padding: "12px 16px" }}>
                    <div className="label-sm">ELEVATED NODES</div>
                    <div className="mono" style={{ fontSize: "1.25rem", color: "var(--tertiary)" }}>
                      {shockScores.filter(s => !s.is_critical && (s.delta ?? s.score) > 0).length}
                    </div>
                  </div>
                  <div className="glass" style={{ padding: "12px 16px" }}>
                    <div className="label-sm">STABLE NODES</div>
                    <div className="mono" style={{ fontSize: "1.25rem", color: "var(--secondary)" }}>
                      {shockScores.filter(s => !s.is_critical && (s.delta ?? s.score) <= 0).length}
                    </div>
                  </div>
                  <div className="glass" style={{ padding: "12px 16px" }}>
                    <div className="label-sm">CRITICAL NODES</div>
                    <div className="mono" style={{ fontSize: "1.25rem", color: "var(--primary-container)" }}>
                      {shockMeta?.critical_node_count ?? "—"}
                    </div>
                  </div>
                  <div className="glass" style={{ padding: "12px 16px" }}>
                    <div className="label-sm">TOTAL BANKS</div>
                    <div className="mono" style={{ fontSize: "1.25rem", color: "var(--on-surface-variant)" }}>
                      {shockMeta?.total_banks ?? shockScores.length}
                    </div>
                  </div>
                </div>

                {/* Metrics Panel */}
                <div style={{ marginBottom: 24 }}>
                  <MetricsPanel />
                </div>

                {/* Risk Table */}
                <div className="glass" style={{ padding: 0 }}>
                  <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--glass-border)" }}>
                    <div className="label-sm">RANKED RISK SCORES // ALL BANKS</div>
                  </div>
                  <div style={{ padding: "0 4px" }}>
                    <RiskTable
                      scores={shockScores}
                      onExplain={handleExplain}
                      explanations={explanations}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                minHeight: 400,
                color: "var(--outline)",
                fontFamily: "var(--font-mono)",
                fontSize: "0.8125rem",
              }}>
                AWAITING SHOCK SIMULATION DATA...
              </div>
            )}
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer style={{
        padding: "24px 40px",
        textAlign: "center",
        borderTop: "1px solid var(--glass-border)",
        background: "var(--surface-container-lowest)",
      }}>
        <div className="label-sm">
          GRASP // GRAPH-BASED RISK ASSESSMENT & SYSTEMIC PREDICTION // GNN-POWERED SYSTEMIC RISK ENGINE // PR-AUC OPTIMIZED
        </div>
      </footer>
    </div>
  );
}
