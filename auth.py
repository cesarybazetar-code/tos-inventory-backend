# auth.py
import os, datetime
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from passlib.context import CryptContext

from main import SessionLocal, Item  # Item import forces models to load; SessionLocal for DB

# ---- Config ----
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_IN_RENDER")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---- DB User model (inline to avoid circular import) ----
from sqlalchemy import Column, Integer, String, Boolean
from main import Base  # main defines Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="staff")  # 'admin' | 'manager' | 'staff'
    active = Column(Boolean, default=True)

# ---- Pydantic ----
class UserOut(BaseModel):
    id: int
    email: str
    name: Optional[str] = None
    role: Literal["admin", "manager", "staff"]
    active: bool = True
    class Config:
        from_attributes = True

class RegisterIn(BaseModel):
    email: str
    name: Optional[str] = None
    password: str
    role: Literal["admin", "manager", "staff"] = "staff"

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

# ---- DB helpers ----
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_password_hash(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(p: str, hashed: str) -> bool:
    return pwd_context.verify(p, hashed)

def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ---- Current user / role deps ----
def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")
    user = db.query(User).get(int(uid))
    if not user or not user.active:
        raise HTTPException(401, "Inactive or missing user")
    return user

def require_roles(*roles: Literal["admin","manager","staff"]):
    def dep(user: User = Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(403, "Insufficient role")
        return user
    return dep

# ---- Router ----
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserOut)
def register(payload: RegisterIn, user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    # only admins can create users
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(400, "Email already exists")
    u = User(
        email=payload.email.lower(),
        name=payload.name,
        hashed_password=get_password_hash(payload.password),
        role=payload.role,
        active=True
    )
    db.add(u); db.commit(); db.refresh(u)
    return u

@router.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # form.username = email; form.password = password
    email = form.username.lower()
    u = db.query(User).filter(User.email == email).first()
    if not u or not verify_password(form.password, u.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    if not u.active:
        raise HTTPException(403, "Inactive user")
    token = create_access_token({"sub": str(u.id), "role": u.role})
    return TokenOut(access_token=token, user=u)