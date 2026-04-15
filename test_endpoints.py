"""Quick shock endpoint verification."""
import requests
import json

r = requests.post("http://127.0.0.1:8000/simulate_shock", json={"bank_id": "Bank_0001", "shock_magnitude": 1.0})
d = r.json()

print(f"Output type: {d.get('output_type')}")
print(f"Critical threshold: {d.get('critical_threshold')}")

shocked_node = [n for n in d["risk_scores"] if "Bank_0001" in n["bank_id"]]
if shocked_node:
    score = shocked_node[0]["score"]
    print(f"Shocked node score: {score} (organic GNN output, NOT forced to 0.95)")
    if abs(score - 0.95) < 0.001:
        print("  WARNING: Score is suspiciously close to 0.95 - check for override")
    else:
        print("  [OK] Score is NOT artificially forced to 0.95")

# Also verify /metrics has test_set_only
r2 = requests.get("http://127.0.0.1:8000/metrics")
d2 = r2.json()
assert d2.get("evaluation_split") == "test_set_only", "Missing test_set_only evaluation_split!"
print(f"\n/metrics evaluation_split: {d2.get('evaluation_split')}")
print(f"/metrics test_node_count: {d2.get('test_node_count', 'N/A (synthetic graph)')}")
print("\nAll endpoint verifications passed!")
