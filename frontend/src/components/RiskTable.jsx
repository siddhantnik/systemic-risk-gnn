import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const PAGE_SIZE = 50;

export default function RiskTable({ scores, onExplain, explanations }) {
  const [expandedBank, setExpandedBank] = useState(null);
  const [page, setPage] = useState(0);

  if (!scores || scores.length === 0) return null;

  const totalPages = Math.ceil(scores.length / PAGE_SIZE);
  const pageScores = scores.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const handleWhy = (bankId) => {
    if (expandedBank === bankId) {
      setExpandedBank(null);
    } else {
      setExpandedBank(bankId);
      if (!explanations[bankId]) {
        onExplain(bankId);
      }
    }
  };

  // Use backend is_critical flag; fall back to score >= 0.5 for older responses
  const isCritical = (s) => s.is_critical ?? (s.score >= 0.5);

  // Delta: how much risk increased vs baseline (new field); fall back to score
  const getDelta = (s) => s.delta ?? s.score;

  const getChipClass = (s) => {
    if (isCritical(s)) return "chip chip-danger";
    if (getDelta(s) > 0) return "chip chip-neutral";
    return "chip chip-safe";
  };

  const getLabel = (s) => {
    if (isCritical(s)) return "CRITICAL";
    if (getDelta(s) > 0) return "ELEVATED";
    return "STABLE";
  };

  const getRowStyle = (s) => {
    if (isCritical(s)) {
      return { background: "rgba(255, 59, 48, 0.08)", borderLeft: "3px solid var(--tertiary-container)" };
    }
    if (getDelta(s) <= 0) {
      return { background: "rgba(85, 225, 107, 0.05)", borderLeft: "3px solid var(--secondary-container)", opacity: 0.7 };
    }
    return {};
  };

  return (
    <div>
      <div style={{ maxHeight: "55vh", overflowY: "auto" }}>
        <table className="risk-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Bank</th>
              <th>Risk Score</th>
              <th>Δ Risk</th>
              <th>Status</th>
              <th>XAI</th>
            </tr>
          </thead>
          <tbody>
            {pageScores.map((s, i) => {
              const globalRank = page * PAGE_SIZE + i + 1;
              return (
                <React.Fragment key={s.bank_id}>
                  <tr style={getRowStyle(s)}>
                    <td style={{ color: "var(--on-surface-variant)" }}>
                      {String(globalRank).padStart(3, "0")}
                    </td>
                    <td>{s.bank_id}</td>
                    <td>
                      <span className="mono">{s.score.toFixed(4)}</span>
                    </td>
                    <td>
                      <span className="mono" style={{
                        color: getDelta(s) > 0.05 ? "var(--tertiary-container)"
                          : getDelta(s) > 0 ? "var(--tertiary)"
                          : "var(--secondary)"
                      }}>
                        {getDelta(s) >= 0 ? "+" : ""}{getDelta(s).toFixed(4)}
                      </span>
                    </td>
                    <td>
                      <span className={getChipClass(s)}>{getLabel(s)}</span>
                    </td>
                    <td>
                      {isCritical(s) && (
                        <button className="btn-why" onClick={() => handleWhy(s.bank_id)}>
                          {expandedBank === s.bank_id ? "CLOSE" : "WHY?"}
                        </button>
                      )}
                    </td>
                  </tr>

                  <AnimatePresence>
                    {expandedBank === s.bank_id && (
                      <motion.tr
                        key={`explain-${s.bank_id}`}
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <td colSpan={5} style={{ padding: 0 }}>
                          <ExplainContent
                            data={explanations[s.bank_id]}
                            bankId={s.bank_id}
                          />
                        </td>
                      </motion.tr>
                    )}
                  </AnimatePresence>
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          gap: 12, padding: "12px 0", borderTop: "1px solid var(--glass-border)"
        }}>
          <button
            className="btn-ghost"
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            style={{ opacity: page === 0 ? 0.3 : 1 }}
          >
            PREV
          </button>
          <span className="mono" style={{ fontSize: "0.75rem", color: "var(--on-surface-variant)" }}>
            {page + 1} / {totalPages} ({scores.length} banks)
          </span>
          <button
            className="btn-ghost"
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            style={{ opacity: page >= totalPages - 1 ? 0.3 : 1 }}
          >
            NEXT
          </button>
        </div>
      )}
    </div>
  );
}

function ExplainContent({ data, bankId }) {
  if (!data) {
    return (
      <div className="explain-panel" style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span className="spinner" />
        <span className="label-sm">ANALYZING CONTAGION PATH FOR {bankId}...</span>
      </div>
    );
  }

  if (data.error) {
    return (
      <div className="explain-panel" style={{ color: "var(--tertiary-container)" }}>
        Error: {data.error}
      </div>
    );
  }

  return (
    <div className="explain-panel">
      <div className="label-sm" style={{ marginBottom: 12, color: "var(--primary-container)" }}>
        CONTAGION ANALYSIS // {data.target}
      </div>

      {data.toxic_edges && data.toxic_edges.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div className="label-sm" style={{ marginBottom: 8 }}>TOXIC INTERBANK EXPOSURES</div>
          {data.toxic_edges.map((e, i) => (
            <div
              key={i}
              style={{
                display: "flex", alignItems: "center", gap: 12,
                padding: "6px 0", borderBottom: "1px solid var(--glass-border)"
              }}
            >
              <span className="chip chip-danger" style={{ minWidth: 90 }}>
                {e.source}
              </span>
              <span style={{
                color: "var(--tertiary-container)",
                fontFamily: "var(--font-mono)", fontSize: "0.75rem"
              }}>
                {"--[" + e.toxicity_weight.toFixed(4) + "]-->"}
              </span>
              <span className="mono" style={{ fontSize: "0.8125rem" }}>
                {e.target}
              </span>
            </div>
          ))}
        </div>
      )}

      {data.weak_features && data.weak_features.length > 0 && (
        <div>
          <div className="label-sm" style={{ marginBottom: 8 }}>FEATURE VULNERABILITY INDICES</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {data.weak_features.map((f, i) => (
              <div key={i} className="glass" style={{ padding: "8px 12px", flex: "0 0 auto" }}>
                <div className="label-sm">ENCODED DIM #{f.encoded_feature_index}</div>
                <div className="mono" style={{ fontSize: "1.125rem", color: "var(--tertiary)" }}>
                  {f.importance_weight.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
