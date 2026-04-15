import os

PWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(PWD, "data", "processed", "graph_data.pt")
PARAM_JSON_PATH = os.path.join(PWD, "results", "automl_results.json")

GNN_MODEL_PATH = os.path.join(PWD, "models", "best_gnn_model.pt")
AUTOENCODER_MODEL_PATH = os.path.join(PWD, "models", "autoencoder_model.pt")
