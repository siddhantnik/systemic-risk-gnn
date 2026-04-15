import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 120000, // Bailout optimizer can take time
});

export const getHealth = () => API.get("/health").then(r => r.data);

export const getBanks = () => API.get("/banks").then(r => r.data);

export const predictRisk = () => API.post("/predict").then(r => r.data);

export const generateNetwork = () => API.get("/generate_network").then(r => r.data);

export const simulateShock = (bankId, magnitude = 1.0) =>
  API.post("/simulate_shock", { bank_id: bankId, shock_magnitude: magnitude }).then(r => r.data);

export const explainRisk = (bankId) =>
  API.post("/explain_risk", { bank_id: bankId }).then(r => r.data);

export const optimizeBailout = (shockedBankId, budgetMillions = 500) =>
  API.post("/optimize_bailout", {
    shocked_bank_id: shockedBankId,
    budget_millions: budgetMillions,
  }).then(r => r.data);

export const getMetrics = () => API.get("/metrics").then(r => r.data);

export const PR_CURVE_URL = "http://127.0.0.1:8000/metrics/pr_curve";
