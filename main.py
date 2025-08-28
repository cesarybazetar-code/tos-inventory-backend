import os, re, csv, io
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict

import requests
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Header, Depends, Query, APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Date,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship

# -------------------- CONFIG --------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tos.db")
# Optional “master key” for quick admin calls (keeps old UI working):
ADMIN_KEY = os.getenv("ADMIN_KEY")
# OCR.space API key
OCR_API_KEY = os.getenv("OCR_API_KEY")

# JWT settings (change SECRET_KEY in Render env for prod)
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# -------------------- MODELS --------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="viewer")  # admin, manager, viewer
    active = Column(Boolean, default=True)

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    storage_area = Column(String, nullable=True)
    par = Column(Float, default=0.0)
    inv_unit_price = Column(Float, default=0.0)
    active = Column(Boolean, default=True)
    __table_args__ = (UniqueConstraint("name", "storage_area", name="uix_item_name_area"),)

class Count(Base):
    __tablename__ = "counts"
    id = Column(Integer, primary_key=True, index=True)
    count_date = Column(Date, nullable=False, default=date.today)
    storage_area = Column(String, nullable=True)

class CountLine(Base):
    __tablename__ = "count_lines"
    id = Column(Integer, primary_key=True, index=True)   # ✅ FIXED
    count_id = Column(Integer, ForeignKey("counts.id", ondelete="CASCADE"))
    item_id = Column(Integer, ForeignKey("items.id", ondelete="SET NULL"))
    qty = Column(Float, default=0.0)

    count = relationship("Count", backref="lines")
    item = relationship("Item")
    
class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(Integer, primary_key=True, index=True)
    received_at = Column(Date, default=date.today)
    receiver = Column(String, nullable=True)

class ReceiptLine(Base):
    __tablename__ = "receipt_lines"
    id = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id", ondelete="CASCADE"))
    item_id = Column(Integer, nullable=True)
    qty = Column(Float, default=0.0)
    unit_price = Column(Float, default=0.0)

def create_db():
    Base.metadata.create_all(bind=engine)

# -------------------- APP --------------------
app = FastAPI(title="TOS Inventory API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- AUTH HELPERS --------------------
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_err = HTTPException(401, "Invalid credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if uid is None:
            raise cred_err
    except JWTError:
        raise cred_err
    user = db.query(User).get(int(uid))
    if not user or not user.active:
        raise cred_err
    return user

def get_user_optional(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> Optional[User]:
    """Read Bearer token if present; otherwise return None (lets us support x-admin-key too)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid:
            return None
        return db.query(User).get(int(uid))
    except Exception:
        return None

def require_role_or_admin_key(allowed_roles: List[str]):
    """Dependency: allow if (Bearer user has allowed role) OR x-admin-key matches ADMIN_KEY."""
    def checker(
        user: Optional[User] = Depends(get_user_optional),
        x_admin_key: Optional[str] = Header(None)
    ):
        # allow admin key if set & matches
        if ADMIN_KEY and x_admin_key == ADMIN_KEY:
            return True
        # otherwise require user role
        if user and user.active and (user.role in allowed_roles):
            return True
        raise HTTPException(403, "Insufficient permissions")
    return checker

# -------------------- SCHEMAS --------------------
class UserOut(BaseModel):
    id: int
    email: str
    name: Optional[str] = None
    role: str
    active: bool
    class Config:
        from_attributes = True

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class RegisterIn(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    role: str = "viewer"  # admin, manager, viewer

class ItemOut(BaseModel):
    id: int
    name: str
    storage_area: Optional[str] = None
    par: float
    inv_unit_price: float
    active: bool
    class Config:
        from_attributes = True

class ItemCreate(BaseModel):
    name: str
    storage_area: Optional[str] = None
    par: Optional[float] = 0.0
    inv_unit_price: Optional[float] = 0.0
    active: Optional[bool] = True

class CountLineIn(BaseModel):
    item_id: int
    qty: float

class CountOut(BaseModel):
    id: int
    count_date: date
    storage_area: Optional[str] = None
    lines: List[CountLineIn] = []
    class Config:
        from_attributes = True

class CountCreate(BaseModel):
    storage_area: Optional[str] = None
    lines: List[CountLineIn]

# OCR schemas
class ReceiveOCRIn(BaseModel):
    receiver: Optional[str] = None
    lines: List[Dict]

# -------------------- AUTH ROUTES --------------------
auth_router = APIRouter(prefix="/auth", tags=["auth"])

@auth_router.post("/register", response_model=UserOut, dependencies=[Depends(require_role_or_admin_key(["admin"]))])
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == payload.email.lower()).first()
    if exists:
        raise HTTPException(400, "Email already exists")
    u = User(
        email=payload.email.lower(),
        name=payload.name,
        hashed_password=get_password_hash(payload.password),
        role=payload.role,
        active=True,
    )
    db.add(u); db.commit(); db.refresh(u)
    return u

@auth_router.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email = form.username.lower()
    u = db.query(User).filter(User.email == email).first()
    if not u or not verify_password(form.password, u.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    if not u.active:
        raise HTTPException(403, "Inactive user")
    token = create_access_token({"sub": str(u.id), "role": u.role})
    return TokenOut(access_token=token, user=u)

app.include_router(auth_router)

# -------------------- INVENTORY ROUTES --------------------
@app.on_event("startup")
def startup_event():
    create_db()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/items", response_model=List[ItemOut])
def list_items(
    q: Optional[str] = None,
    area: Optional[str] = Query(None, description="Filter by storage_area"),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_user_optional),  # optional read
):
    qry = db.query(Item).filter(Item.active == True)
    if q:
        qry = qry.filter(Item.name.ilike(f"%{q}%"))
    if area:
        qry = qry.filter(Item.storage_area == area)
    return qry.order_by(Item.storage_area.asc().nullsfirst(), Item.name.asc()).all()

@app.post("/items", response_model=ItemOut, status_code=201,
          dependencies=[Depends(require_role_or_admin_key(["admin", "manager"]))])
def create_item(payload: ItemCreate, db: Session = Depends(get_db)):
    name = payload.name.strip()
    area = payload.storage_area or None
    exists = db.query(Item).filter(Item.name == name, Item.storage_area == area).first()
    if exists:
        exists.par = payload.par or exists.par or 0.0
        exists.inv_unit_price = payload.inv_unit_price or exists.inv_unit_price or 0.0
        exists.active = bool(payload.active) if payload.active is not None else True
        db.commit(); db.refresh(exists)
        return exists
    rec = Item(
        name=name, storage_area=area,
        par=payload.par or 0.0,
        inv_unit_price=payload.inv_unit_price or 0.0,
        active=bool(payload.active) if payload.active is not None else True
    )
    db.add(rec); db.commit(); db.refresh(rec)
    return rec

@app.put("/items/{item_id}", response_model=ItemOut,
         dependencies=[Depends(require_role_or_admin_key(["admin", "manager"]))])
def update_item(item_id: int, payload: ItemCreate, db: Session = Depends(get_db)):
    rec = db.query(Item).get(item_id)
    if not rec:
        raise HTTPException(404, "Not found")
    if payload.name and payload.name.strip() != rec.name:
        rec.name = payload.name.strip()
    new_area = payload.storage_area if payload.storage_area is not None else rec.storage_area
    if new_area != rec.storage_area:
        conflict = db.query(Item).filter(
            Item.name == rec.name, Item.storage_area == new_area, Item.id != rec.id
        ).first()
        if conflict:
            raise HTTPException(400, "Item with same name exists in that area")
        rec.storage_area = new_area
    if payload.par is not None:
        rec.par = payload.par or 0.0
    if payload.inv_unit_price is not None:
        rec.inv_unit_price = payload.inv_unit_price or 0.0
    if payload.active is not None:
        rec.active = bool(payload.active)
    db.commit(); db.refresh(rec)
    return rec

@app.delete("/items/{item_id}", status_code=204,
            dependencies=[Depends(require_role_or_admin_key(["admin"]))])
def delete_item(item_id: int, db: Session = Depends(get_db)):
    rec = db.query(Item).get(item_id)
    if not rec:
        raise HTTPException(404, "Not found")
    db.delete(rec); db.commit()
    return None

@app.post("/import/catalog", status_code=201,
          dependencies=[Depends(require_role_or_admin_key(["admin", "manager"]))])
async def import_catalog(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    created = 0; updated = 0
    for row in reader:
        name = (row.get("name") or row.get("Item") or "").strip()
        if not name:
            continue
        storage = (row.get("storage_area") or row.get("Location") or None) or None
        def to_float(v, default=0.0):
            try: return float(v)
            except: return default
        par = to_float(row.get("par") or row.get("PAR"))
        price = to_float(row.get("inv_unit_price") or row.get("Inv Unit Price") or row.get("Order Unit Price"))
        exists = db.query(Item).filter(Item.name==name, Item.storage_area==storage).first()
        if exists:
            exists.par = par if par is not None else exists.par
            exists.inv_unit_price = price if price is not None else exists.inv_unit_price
            exists.active = True
            updated += 1
        else:
            db.add(Item(name=name, storage_area=storage, par=par or 0.0,
                        inv_unit_price=price or 0.0, active=True))
            created += 1
    db.commit()
    return {"created": created, "updated": updated}

@app.get("/counts", response_model=List[CountOut])
def list_counts(db: Session = Depends(get_db)):
    res: List[CountOut] = []
    counts = db.query(Count).order_by(Count.id.desc()).limit(100).all()
    for c in counts:
        lines = [CountLineIn(item_id=ln.item_id or 0, qty=ln.qty or 0.0) for ln in c.lines]
        res.append(CountOut(id=c.id, count_date=c.count_date, storage_area=c.storage_area, lines=lines))
    return res

@app.post("/counts", response_model=CountOut, status_code=201,
          dependencies=[Depends(require_role_or_admin_key(["admin", "manager"]))])
def create_count(payload: CountCreate, db: Session = Depends(get_db)):
    c = Count(storage_area=payload.storage_area or None)
    db.add(c); db.commit(); db.refresh(c)
    for ln in payload.lines:
        db.add(CountLine(count_id=c.id, item_id=ln.item_id, qty=ln.qty))
    db.commit(); db.refresh(c)
    return CountOut(id=c.id, count_date=c.count_date, storage_area=c.storage_area, lines=payload.lines)

@app.get("/auto-po")
def auto_po(storage_area: Optional[str] = None, db: Session = Depends(get_db)):
    latest = db.query(Count).filter(
        Count.storage_area == storage_area if storage_area else True
    ).order_by(Count.id.desc()).first()

    on_hand: Dict[int, float] = {}
    if latest:
        for ln in latest.lines:
            on_hand[ln.item_id] = ln.qty

    q = db.query(Item).filter(Item.active == True)
    if storage_area:
        q = q.filter(Item.storage_area == storage_area)

    out = []
    for i in q:
        oh = on_hand.get(i.id, 0.0)
        reorder = max((i.par or 0.0) - oh, 0.0)
        if reorder > 0:
            out.append({
                "item_id": i.id,
                "name": i.name,
                "storage_area": i.storage_area,
                "on_hand": oh,
                "par": i.par or 0.0,
                "suggested_order_qty": reorder
            })
    return {"storage_area": storage_area, "lines": out}

# -------------------- OCR ROUTES --------------------
PRICE_RE = re.compile(r"(?<!\d)(\d{1,4}(?:\.\d{2}))")
QTY_RE = re.compile(r"(?<!\d)(\d{1,4})(?![\d\.])")

def fuzzy_match(items, text):
    txt = (text or "").lower()
    best = None; score = 0
    for it in items:
        nm = (it.name or "").lower()
        if not nm: continue
        s = 0
        if nm in txt: s = len(nm)
        else:
            toks = [t for t in nm.split() if len(t) > 2]
            s = sum(1 for t in toks if t in txt)
        if s > score:
            best, score = it, s
    return best, score

@app.get("/ocr/health")
def ocr_health():
    return {"ok": True, "ocr": bool(OCR_API_KEY)}

@app.post("/invoice/ocr", dependencies=[Depends(require_role_or_admin_key(["admin","manager"]))])
async def ocr_invoice(
    file: UploadFile = File(...),
    receiver: Optional[str] = None,
    db: Session = Depends(get_db)
):
    if not OCR_API_KEY:
        raise HTTPException(400, "OCR_API_KEY not set on server")

    files = {'file': (file.filename, await file.read())}
    data = {'apikey': OCR_API_KEY, 'language': 'eng', 'scale': 'true', 'OCREngine': '2'}
    resp = requests.post("https://api.ocr.space/parse/image", files=files, data=data, timeout=90)
    if resp.status_code != 200:
        raise HTTPException(502, f"OCR provider error {resp.status_code}")
    payload = resp.json()
    if not payload.get("ParsedResults"):
        raise HTTPException(400, "No text detected")

    raw_text = "\n".join([p.get("ParsedText","") for p in payload["ParsedResults"]])
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    items = db.query(Item).filter(Item.active==True).all()
    parsed = []
    for ln in lines:
        if len(ln) < 3:
            continue
        price = None; qty = None
        m_price = list(PRICE_RE.finditer(ln))
        if m_price: price = float(m_price[-1].group(1))
        m_qty = QTY_RE.findall(ln.split()[0]) or QTY_RE.findall(ln)
        if m_qty:
            try: qty = float(m_qty[0])
            except: qty = None
        match, score = fuzzy_match(items, ln)
        if match and score >= 1:
            parsed.append({
                "text": ln,
                "item_id": match.id,
                "name": match.name,
                "storage_area": match.storage_area,
                "qty": qty or 0.0,
                "unit_price": (price if price is not None else (match.inv_unit_price or 0.0))
            })

    if not parsed:
        raise HTTPException(400, "No text detected")
    return {"lines": parsed}

@app.post("/receive/ocr", dependencies=[Depends(require_role_or_admin_key(["admin","manager"]))])
def receive_from_ocr(payload: ReceiveOCRIn, db: Session = Depends(get_db)):
    r = Receipt(receiver=payload.receiver or None)
    db.add(r); db.commit(); db.refresh(r)
    for ln in payload.lines:
        db.add(ReceiptLine(
            receipt_id=r.id,
            item_id=int(ln["item_id"]),
            qty=float(ln.get("qty") or 0),
            unit_price=float(ln.get("unit_price") or 0),
        ))
        # update last price on item
        it = db.query(Item).get(int(ln["item_id"]))
        if it and (ln.get("unit_price") is not None):
            it.inv_unit_price = float(ln["unit_price"])
    db.commit()
    return {"ok": True, "receipt_id": r.id}