import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { optimizeBailout } from "../services/api";

export default function BailoutPanel({ shockedNode, onRescueNodes }) {
  const [budget, setBudget] = useState(500);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleOptimize = async () => {
    if (!shockedNode) return;
    setLoading(true);
    setError(null);
    try {
      const data = await optimizeBailout(shockedNode, budget);
      setResult(data);
      // Pass rescue nodes to parent for graph animation
      if (onRescueNodes && data.recommended_allocations) {
        onRescueNodes(data.recommended_allocations);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1], delay: 0.15 }}
    >
      <div className="glass" style={{ padding: "20px 24px" }}>
        <div className="label-sm" style={{ marginBottom: 4, color: "var(--secondary-container)" }}>
          PRESCRIPTIVE ENGINE
        </div>
        <h3 style={{ fontSize: "1.25rem", marginBottom: 16 }}>
          Bailout <span style={{ color: "var(--secondary-container)" }}>Optimizer</span>
        </h3>

        {!shockedNode ? (
          <div style={{
            padding: 16, textAlign: "center", color: "var(--outline)",
            border: "1px dashed var(--outline-variant)", borderRadius: 4,
            fontFamily: "var(--font-mono)", fontSize: "0.75rem",
          }}>
            EXECUTE A SHOCK FIRST TO ENABLE BAILOUT SIMULATION
          </div>
        ) : (
          <div>
            {/* Budget Slider */}
            <div style={{ marginBottom: 20 }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "baseline", marginBottom: 8,
              }}>
                <span className="label-sm">BAILOUT BUDGET</span>
                <span className="mono" style={{
                  fontSize: "1.25rem", color: "var(--secondary)",
                }}>
                  ${budget}M
                </span>
              </div>
              <input
                type="range"
                min={50}
                max={2000}
                step={50}
                value={budget}
                onChange={e => setBudget(Number(e.target.value))}
                style={{
                  width: "100%",
                  accentColor: "var(--secondary-container)",
                  height: 6,
                }}
              />
              <div style={{
                display: "flex", justifyContent: "space-between",
                fontSize: "0.625rem", color: "var(--outline)",
                fontFamily: "var(--font-mono)", marginTop: 4,
              }}>
                <span>$50M</span>
                <span>$2,000M</span>
              </div>
            </div>

            <button
              className="btn-primary"
              style={{
                width: "100%",
                background: "linear-gradient(135deg, var(--secondary-container), #00e5ff)",
              }}
              onClick={handleOptimize}
              disabled={loading}
            >
              {loading ? (
                <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}>
                  <span className="spinner" /> COMPUTING OPTIMAL ALLOCATION...
                </span>
              ) : "RUN BAILOUT OPTIMIZER"}
            </button>

            {error && (
              <div style={{
                marginTop: 12, padding: "8px 12px",
                background: "rgba(255, 59, 48, 0.1)", borderRadius: 4,
                color: "var(--tertiary)", fontSize: "0.75rem",
              }}>
                {error}
              </div>
            )}

            {/* Results */}
            <AnimatePresence>
              {result && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.5 }}
                  style={{ marginTop: 20 }}
                >
                  {/* Risk reduction summary */}
                  <div style={{
                    display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap",
                  }}>
                    <div style={{
                      flex: "1 1 120px", padding: "12px 14px",
                      background: "var(--surface-container-lowest)", borderRadius: 6,
                      borderLeft: "3px solid var(--tertiary-container)",
                    }}>
                      <div className="label-sm">ORIGINAL RISK</div>
                      <div className="mono" style={{ fontSize: "1.125rem", color: "var(--tertiary-container)" }}>
                        {Number(result.original_network_risk_sum).toFixed(4)}
                      </div>
                    </div>
                    <div style={{
                      flex: "1 1 120px", padding: "12px 14px",
                      background: "var(--surface-container-lowest)", borderRadius: 6,
                      borderLeft: "3px solid var(--secondary)",
                    }}>
                      <div className="label-sm">OPTIMIZED RISK</div>
                      <div className="mono" style={{ fontSize: "1.125rem", color: "var(--secondary)" }}>
                        {Number(result.optimized_network_risk_sum).toFixed(4)}
                      </div>
                    </div>
                    <div style={{
                      flex: "1 1 100px", padding: "12px 14px",
                      background: "var(--surface-container-lowest)", borderRadius: 6,
                      borderLeft: "3px solid var(--primary-container)",
                    }}>
                      <div className="label-sm">REDUCTION</div>
                      <div className="mono" style={{
                        fontSize: "1.125rem",
                        color: result.risk_reduction_pct >= 0 ? "var(--secondary)" : "var(--tertiary)",
                      }}>
                        {result.risk_reduction_pct >= 0 ? "▼ " : "▲ "}{Math.abs(result.risk_reduction_pct).toFixed(2)}%
                      </div>
                    </div>
                  </div>

                  {/* Allocation cards */}
                  <div className="label-sm" style={{ marginBottom: 8 }}>
                    RECOMMENDED CAPITAL INJECTIONS
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    {result.recommended_allocations.map((a, i) => (
                      <div
                        key={a.bank_id}
                        style={{
                          display: "flex", alignItems: "center", justifyContent: "space-between",
                          padding: "8px 12px",
                          background: "var(--surface-container-lowest)",
                          borderRadius: 4,
                          borderLeft: "3px solid #00e5ff",
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                          <span className="mono" style={{ fontSize: "0.6875rem", color: "var(--outline)" }}>
                            #{i + 1}
                          </span>
                          <span style={{ color: "#00e5ff", fontWeight: 500 }}>
                            {a.bank_id}
                          </span>
                        </div>
                        <span className="mono" style={{ fontSize: "1rem", color: "var(--secondary)" }}>
                          ${a.allocation_millions}M
                        </span>
                      </div>
                    ))}
                  </div>

                  {result.recommended_allocations.length === 0 && (
                    <div style={{
                      padding: 12, textAlign: "center",
                      color: "var(--outline)", fontFamily: "var(--font-mono)",
                      fontSize: "0.75rem",
                    }}>
                      No viable rescue targets found.
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
}
