from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router

app = FastAPI(
    title="Systemic Risk Prediction API",
    description="GNN-based systemic risk prediction engine for interbank networks. "
                "Features CAMELS-filtered features, AutoML-optimized models, "
                "GNNExplainer-based XAI, and synthetic network stress testing.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
