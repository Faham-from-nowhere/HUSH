# main.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel, ConfigDict # MODIFIED: Imported ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from sqlmodel import Field, Session, SQLModel, create_engine, select
from contextlib import asynccontextmanager # NEW: Required for lifespan
from fastapi.middleware.cors import CORSMiddleware

# --- Pydantic Models (The "Contract") ---

class ModelUpdatePayload(BaseModel):
    feature_attributions: Dict[str, float]
    user_id: str
    
    # MODIFIED: Replaced class Config with model_config
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature_attributions": {"text": 0.7, "typing": 0.2, "voice": 0.1},
                "user_id": "anon-12345"
            }
        }
    )

# MODIFIED: This is now our Database Table *and* our API Response Model
class DashboardDataPoint(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True) 
    timestamp: str = Field(index=True) 
    avg_text_importance: float
    avg_typing_importance: float
    avg_voice_importance: float

# --- Global State (For Simulation) ---
global_model_weights = {"text": 0.33, "typing": 0.33, "voice": 0.34}
global_update_count = 0

# --- Database Setup ---
sqlite_file_name = "hush.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# NEW: This "dependency" gives us a DB session in our endpoints
def get_session():
    with Session(engine) as session:
        yield session

# --- NEW: Lifespan Event Handler (replaces @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("Server starting up...")
    create_db_and_tables()
    # Pre-populate DB if it's empty
    with Session(engine) as session:
        if not session.exec(select(DashboardDataPoint)).first():
            print("Database is empty, pre-populating with mock data...")
            db_data = [
                DashboardDataPoint(timestamp="2025-11-01T10:00:00Z", avg_text_importance=0.4, avg_typing_importance=0.4, avg_voice_importance=0.2),
                DashboardDataPoint(timestamp="2025-11-01T11:00:00Z", avg_text_importance=0.42, avg_typing_importance=0.38, avg_voice_importance=0.2),
                DashboardDataPoint(timestamp="2025-11-01T12:00:00Z", avg_text_importance=0.38, avg_typing_importance=0.45, avg_voice_importance=0.17),
            ]
            for data in db_data:
                session.add(data)
            session.commit()
            print("Mock data populated.")
    
    yield # This is where the application runs
    
    # Code to run on shutdown (optional)
    print("Server shutting down...")

# --- FastAPI App Setup ---
app = FastAPI(
    title="HUSH Backend API",
    description="Backend for the HUSH adaptive wellness coach, supporting (simulated) FL/DP.",
    version="0.1.0",
    lifespan=lifespan  # MODIFIED: Replaced @app.on_event
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This is the key line
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/", tags=["Status"])
def read_root():
    """Server health check."""
    return {"status": "HUSH Backend is running."}


@app.post("/v1/submit-update", tags=["Federated Learning (Simulated)"])
def submit_model_update(payload: ModelUpdatePayload, session: Session = Depends(get_session)):
    """
    [Phase 1 Task - Now with DB Persistence]
    The on-device client sends its update here.
    """
    
    # 1. DP Sim: Add Laplace noise
    scale = 0.1 
    noisy_text = payload.feature_attributions.get("text", 0) + np.random.laplace(0, scale)
    noisy_typing = payload.feature_attributions.get("typing", 0) + np.random.laplace(0, scale)
    noisy_voice = payload.feature_attributions.get("voice", 0) + np.random.laplace(0, scale)
    
    # 2. FL Sim: Aggregate using Federated Averaging (FedAvg)
    global global_update_count, global_model_weights
    total = global_update_count + 1
    # Clamp to avoid divide-by-zero, though 'total' starts at 1
    global_update_count = max(1, global_update_count) 
    global_model_weights["text"] = (global_model_weights["text"] * (total - 1) + noisy_text) / total
    global_model_weights["typing"] = (global_model_weights["typing"] * (total - 1) + noisy_typing) / total
    global_model_weights["voice"] = (global_model_weights["voice"] * (total - 1) + noisy_voice) / total
    global_update_count = total
    
    # 3. DB Sim: Save the new global average to the *real database*
    new_data_point = DashboardDataPoint(
        timestamp=datetime.now().isoformat(),
        avg_text_importance=global_model_weights["text"],
        avg_typing_importance=global_model_weights["typing"],
        avg_voice_importance=global_model_weights["voice"]
    )
    
    session.add(new_data_point) 
    session.commit() 
    session.refresh(new_data_point) 

    print(f"Processed and saved update from {payload.user_id}")
    return {"status": "update received and saved", "new_data_point": new_data_point}


@app.get("/v1/dashboard-data", response_model=List[DashboardDataPoint], tags=["Dashboard"])
def get_dashboard_data(session: Session = Depends(get_session)):
    """
    [Phase 1 REAL - Now reads from SQLite DB]
    The admin dashboard hits this endpoint to get its chart data.
    """
    statement = select(DashboardDataPoint).order_by(DashboardDataPoint.timestamp) 
    results = session.exec(statement).all() 
    return results