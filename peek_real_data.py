import pandas as pd

NODES_DIR = r"C:\Users\prana\Desktop\ATML project\datasets\nodes"
EDGES_DIR = r"C:\Users\prana\Desktop\ATML project\datasets\edges"

nodes_df = pd.read_csv(f"{NODES_DIR}/2016Q1.csv")
edges_df  = pd.read_csv(f"{EDGES_DIR}/edge_2016Q1.csv")

print("=" * 60)
print(f"NODES — shape: {nodes_df.shape}")
print("=" * 60)
print("Columns:", list(nodes_df.columns))
print()
print(nodes_df.head(3).to_string())

print()
print("=" * 60)
print(f"EDGES — shape: {edges_df.shape}")
print("=" * 60)
print("Columns:", list(edges_df.columns))
print()
print(edges_df.head(3).to_string())