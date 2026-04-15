import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { getMetrics, PR_CURVE_URL } from "../services/api";

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMetrics()
      .then(data => { setMetrics(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="glass" style={{ padding: "16px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <span className="spinner" />
        <span className="label-sm">LOADING MODEL EVALUATION...</span>
      </div>
    );
  }

  if (!metrics) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="glass" style={{ padding: "20px 24px" }}>
        <div className="label-sm" style={{ marginBottom: 14, color: "var(--primary-container)" }}>
          MODEL EVALUATION // PR-AUC OPTIMIZED
        </div>

        {metrics.pr_curve_available ? (
          <div style={{
            borderRadius: 6,
            overflow: "hidden",
            border: "1px solid var(--glass-border)",
          }}>
            <img
              src={PR_CURVE_URL}
              alt="Precision-Recall Curve"
              style={{ width: "100%", display: "block" }}
            />
          </div>
        ) : (
          <div className="label-sm" style={{ color: "var(--outline)", padding: "12px 0" }}>
            PR CURVE NOT AVAILABLE — RUN TRAINING FIRST
          </div>
        )}

        {metrics.metrics_error && (
          <div className="label-sm" style={{ color: "var(--tertiary)", marginTop: 10 }}>
            Note: {metrics.metrics_error}
          </div>
        )}
      </div>
    </motion.div>
  );
}
