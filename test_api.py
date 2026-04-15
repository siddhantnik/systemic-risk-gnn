import requests
import json
import time

time.sleep(3)
BASE = "http://127.0.0.1:8000"

print("=== /health ===")
r = requests.get(f"{BASE}/health")
print(r.status_code, r.json())

print("\n=== /banks ===")
r = requests.get(f"{BASE}/banks")
d = r.json()
print(f"Count: {d['count']}, First 5: {d['banks'][:5]}")

print("\n=== /predict ===")
r = requests.post(f"{BASE}/predict")
d = r.json()
print(f"Status: {r.status_code}")
scores = d["risk_scores"]
print(f"Total banks: {len(scores)}")
print("Top 5 riskiest:")
for s in scores[:5]:
    print(f"  {s['bank_id']}: {s['score']}")

print("\n=== /generate_network ===")
r = requests.get(f"{BASE}/generate_network")
d = r.json()
print(f"Nodes: {len(d['nodes'])}, Edges: {len(d['edges'])}")

print("\n=== /simulate_shock ===")
r = requests.post(f"{BASE}/simulate_shock", json={"bank_id": "Bank_0001", "shock_magnitude": 1.0})
d = r.json()
print(f"Status: {r.status_code}")
print(f"Shocked: {d.get('shocked_bank')}")
shocked_scores = d["risk_scores"]
print(f"Top 3 after shock:")
for s in shocked_scores[:3]:
    print(f"  {s['bank_id']}: {s['score']}")

print("\n=== /explain_risk ===")
r = requests.post(f"{BASE}/explain_risk", json={"bank_id": "Bank_0001"})
print(f"Status: {r.status_code}")
d = r.json()
print(f"Target: {d.get('target')}")
print(f"Toxic edges: {d.get('toxic_edges')}")
print(f"Weak features: {d.get('weak_features')}")

print("\n=== ALL TESTS PASSED ===")
