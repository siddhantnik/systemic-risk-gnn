import sys, os, asyncio
sys.path.append(os.path.abspath(os.curdir))
from backend.api.routes import generate_network, simulate_shock, get_metrics, SimulateRequest

async def test():
    print("=== Generating network ===")
    await generate_network()

    print("\n=== Testing shock on a HUB bank (Bank_0000) ===")
    res_big = await simulate_shock(SimulateRequest(bank_id="Bank_0000", shock_magnitude=1.0))
    scores_big = res_big["risk_scores"]
    print(f"  Critical nodes: {res_big['critical_node_count']} / {res_big['total_banks']}")
    print(f"  Critical threshold: {res_big['critical_threshold']}")
    print(f"  Top 3 scores: {[s['score'] for s in scores_big[:3]]}")
    print(f"  Min score: {scores_big[-1]['score']}  Max score: {scores_big[0]['score']}")

    print("\n=== Re-generating for small bank test ===")
    await generate_network()
    print("\n=== Testing shock on a LEAF bank (Bank_0074) ===")
    res_small = await simulate_shock(SimulateRequest(bank_id="Bank_0074", shock_magnitude=1.0))
    print(f"  Critical nodes: {res_small['critical_node_count']} / {res_small['total_banks']}")
    print(f"  Critical threshold: {res_small['critical_threshold']}")

    print("\n=== Metrics endpoint ===")
    m = await get_metrics()
    print(f"  PR-AUC: {m.get('pr_auc')}")
    print(f"  n_critical: {m.get('n_critical')}")
    print(f"  n_total: {m.get('n_total')}")
    print(f"  metrics_warning: {m.get('metrics_warning')}")
    print(f"  metrics_error: {m.get('metrics_error')}")

asyncio.run(test())
