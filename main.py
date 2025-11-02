# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone # MODIFIED
import numpy as np
from sqlmodel import Field, Session, SQLModel, create_engine, select
from contextlib import asynccontextmanager

# --- NEW: Security & JWT Imports ---
from passlib.context import CryptContext
from jose import jwt
from jose.exceptions import JWTError

# --- Pydantic Models (The "Contract") ---

class ModelUpdatePayload(BaseModel):
    feature_attributions: Dict[str, float]
    # MODIFIED: user_id is REMOVED. We get it from the token.
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature_attributions": {"text": 0.7, "typing": 0.2, "voice": 0.1}
            }
        }
    )

class DashboardDataPoint(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True) 
    timestamp: str = Field(index=True) 
    avg_text_importance: float
    avg_typing_importance: float
    avg_voice_importance: float
    owner_username: str = Field(index=True) # NEW: Track who submitted this

# --- NEW: User & Token Models ---

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- NEW: Security Configuration ---
SECRET_KEY = "a-very-secret-key-for-a-hackathon" # CHANGE THIS IN PRODUCTION
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login") # Tells FastAPI where the login endpoint is

# --- NEW: Security Utility Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Global State (For Simulation) ---
global_model_weights = {"text": 0.33, "typing": 0.33, "voice": 0.34}
global_update_count = 0

# --- Database Setup ---
sqlite_file_name = "hush.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine) # This will now create BOTH tables

def get_session():
    with Session(engine) as session:
        yield session

async def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = session.exec(select(User).where(User.username == token_data.username)).first()
    if user is None:
        raise credentials_exception
    return user


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    create_db_and_tables() # This will now create User and DashboardDataPoint
    
    # Pre-populate DB (no change)
    with Session(engine) as session:
        if not session.exec(select(DashboardDataPoint)).first():
            print("Database is empty, pre-populating with mock data...")
            # We add a default owner for the mock data
            db_data = [
                DashboardDataPoint(timestamp="2025-11-01T10:00:00Z", avg_text_importance=0.4, avg_typing_importance=0.4, avg_voice_importance=0.2, owner_username="mock_user"),
                DashboardDataPoint(timestamp="2025-11-01T11:00:00Z", avg_text_importance=0.42, avg_typing_importance=0.38, avg_voice_importance=0.2, owner_username="mock_user"),
                DashboardDataPoint(timestamp="2025-11-01T12:00:00Z", avg_text_importance=0.38, avg_typing_importance=0.45, avg_voice_importance=0.17, owner_username="mock_user"),
            ]
            for data in db_data:
                session.add(data)
            session.commit()
            print("Mock data populated.")
    
    yield 
    print("Server shutting down...")

# --- FastAPI App Setup ---
app = FastAPI(
    title="HUSH Backend API",
    description="Backend for the HUSH adaptive wellness coach, supporting (simulated) FL/DP.",
    version="0.1.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- NEW: Auth Endpoints ---

@app.post("/auth/signup", response_model=User, tags=["Auth"])
def signup(user: UserCreate, session: Session = Depends(get_session)):
    db_user = session.exec(select(User).where(User.username == user.username)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    return new_user

@app.post("/auth/login", response_model=Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

# --- Core API Endpoints ---

@app.get("/", tags=["Status"])
def read_root():
    return {"status": "HUSH Backend is running."}

# --- MODIFIED: Protected Endpoint ---
@app.post("/v1/submit-update", tags=["Federated Learning (Simulated)"])
def submit_model_update(
    payload: ModelUpdatePayload, 
    session: Session = Depends(get_session), 
    current_user: User = Depends(get_current_user) # NEW: This line "protects" the endpoint
):
    """
    [Phase 1 Task - Now Protected & User-Aware]
    The on-device client sends its update here.
    """
    
    # We now know *who* sent this data
    print(f"Processing update from user: {current_user.username}")
    
    # 1. DP Sim: Add Laplace noise
    scale = 0.1 
    noisy_text = payload.feature_attributions.get("text", 0) + np.random.laplace(0, scale)
    noisy_typing = payload.feature_attributions.get("typing", 0) + np.random.laplace(0, scale)
    noisy_voice = payload.feature_attributions.get("voice", 0) + np.random.laplace(0, scale)
    
    # 2. FL Sim: Aggregate using Federated Averaging (FedAvg)
    global global_update_count, global_model_weights
    total = global_update_count + 1
    global_update_count = max(1, global_update_count) 
    global_model_weights["text"] = (global_model_weights["text"] * (total - 1) + noisy_text) / total
    global_model_weights["typing"] = (global_model_weights["typing"] * (total - 1) + noisy_typing) / total
    global_model_weights["voice"] = (global_model_weights["voice"] * (total - 1) + noisy_voice) / total
    global_update_count = total
    
    # 3. DB Sim: Save new average, now with user info
    new_data_point = DashboardDataPoint(
        timestamp=datetime.now(timezone.utc).isoformat(), # MODIFIED: Use UTC
        avg_text_importance=global_model_weights["text"],
        avg_typing_importance=global_model_weights["typing"],
        avg_voice_importance=global_model_weights["voice"],
        owner_username=current_user.username # NEW: We tag the data point
    )
    
    session.add(new_data_point) 
    session.commit() 
    session.refresh(new_data_point) 

    return {"status": "update received and saved", "new_data_point": new_data_point}

# --- MODIFIED: Protected Endpoint ---
@app.get("/v1/dashboard-data", response_model=List[DashboardDataPoint], tags=["Dashboard"])
def get_dashboard_data(
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user) # NEW: Protects dashboard
):
    """
    [Phase 1 REAL - Now Protected]
    The admin dashboard hits this endpoint to get its chart data.
    
    We could also make this *filter* by user, e.g.:
    statement = select(DashboardDataPoint).where(DashboardDataPoint.owner_username == current_user.username)
    """
    # For now, we'll let any logged-in user see all data (for the admin dashboard)
    statement = select(DashboardDataPoint).order_by(DashboardDataPoint.timestamp) 
    results = session.exec(statement).all() 
    return results