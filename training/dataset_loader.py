import pandas as pd
import pathlib

# Real dataset location — outside the project folder
NODES_DIR = pathlib.Path(r"C:\Users\prana\Desktop\ATML project\datasets\nodes")
EDGES_DIR = pathlib.Path(r"C:\Users\prana\Desktop\ATML project\datasets\edges")


def load_dataset(quarter: str):
    """
    Loads a single quarter. Format: '2016Q1', '2019Q3' (no underscore).
    Nodes file: 2016Q1.csv
    Edges file: edge_2016Q1.csv
    """
    nodes_path = NODES_DIR / f"{quarter}.csv"
    edges_path = EDGES_DIR / f"edge_{quarter}.csv"

    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes missing: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges missing: {edges_path}")

    nodes_df = pd.read_csv(nodes_path)
    edges_df  = pd.read_csv(edges_path)

    if nodes_df.empty:
        raise ValueError(f"{quarter}: nodes CSV is empty.")
    if edges_df.empty:
        raise ValueError(f"{quarter}: edges CSV is empty.")

    print(f"  [{quarter}] nodes={nodes_df.shape}, edges={edges_df.shape}")
    return nodes_df, edges_df


def get_all_quarters():
    """
    Returns sorted list of all quarters that have BOTH a nodes and edges file.
    """
    quarters = []
    for f in sorted(NODES_DIR.glob("*.csv")):
        quarter = f.stem                          # e.g. '2016Q1'
        edge_f  = EDGES_DIR / f"edge_{quarter}.csv"
        if edge_f.exists():
            quarters.append(quarter)
        else:
            print(f"  Skipping {quarter} — no matching edge file")
    return quarters


def load_all_quarters():
    """
    Loads every available quarter and concatenates into unified DataFrames.
    Offsets node IDs across quarters so they never collide.
    """
    quarters = get_all_quarters()
    if not quarters:
        raise FileNotFoundError(
            f"No valid quarters found.\n"
            f"  Nodes dir: {NODES_DIR}\n"
            f"  Edges dir: {EDGES_DIR}"
        )

    print(f"\nLoading {len(quarters)} quarters: {quarters[0]} → {quarters[-1]}")

    all_nodes, all_edges = [], []
    node_offset = 0

    for q in quarters:
        ndf, edf = load_dataset(q)
        ndf = ndf.copy()
        edf = edf.copy()

        ndf["quarter"]   = q
        edf["quarter"]   = q

        # Offset IDs for cross-quarter uniqueness
        ndf["index"]    = ndf["index"]    + node_offset
        edf["Sourceid"] = edf["Sourceid"] + node_offset
        edf["Targetid"] = edf["Targetid"] + node_offset
        node_offset += len(ndf)

        all_nodes.append(ndf)
        all_edges.append(edf)

    combined_nodes = pd.concat(all_nodes, ignore_index=True)
    combined_edges = pd.concat(all_edges, ignore_index=True)

    print(f"  Total nodes: {len(combined_nodes):,}, Total edges: {len(combined_edges):,}")
    return combined_nodes, combined_edges, quarters