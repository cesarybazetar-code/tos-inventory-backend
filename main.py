
def _parse_date(s: str|None):
    if not s: return None
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%m/%d/%y"):
        try: return datetime.strptime(s, fmt).date()
        except: pass
    return None
from sqlalchemy import func
# main.py
import os, re, csv, io, base64
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Query, APIRouter, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
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
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

ADMIN_KEY = os.getenv("ADMIN_KEY")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
GCV_API_KEY = os.getenv("GCV_API_KEY")
OCR_API_KEY = os.getenv("OCR_API_KEY")

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
    role = Column(String, default="viewer")  # admin, manager, counter, viewer
    active = Column(Boolean, default=True)

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    storage_area = Column(String, nullable=True)
    par = Column(Float, default=0.0)
    inv_unit_price = Column(Float, default=0.0)
    active = Column(Boolean, default=True)
    # NEW: purchase/packaging metadata
    order_unit = Column(String, nullable=True)           # e.g., 'case'
    inventory_unit = Column(String, nullable=True)       # e.g., 'lb', 'ea', 'gal'
    case_size = Column(Float, nullable=True)             # units per case
    conversion = Column(Float, nullable=True)            # units per order unit (generic)
    order_unit_price = Column(Float, nullable=True)      # price of the order unit
    price_basis = Column(String, nullable=True)          # 'per_unit' | 'per_case' | 'via_conversion'

class Count(Base):
    __tablename__ = "counts"
    id = Column(Integer, primary_key=True, index=True)
    count_date = Column(Date, default=date.today, index=True)
    storage_area = Column(String, nullable=True)
    lines = relationship("CountLine", backref="count", cascade="all, delete-orphan")

class CountLine(Base):
    __tablename__ = "count_lines"
    id = Column(Integer, primary_key=True, index=True)
    count_id = Column(Integer, ForeignKey("counts.id", ondelete="CASCADE"))
    item_id = Column(Integer, ForeignKey("items.id", ondelete="SET NULL"))
    qty = Column(Float, default=0.0)
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
app = FastAPI(title="TOS Inventory API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://tos-inventory-frontend.vercel.app",
    ],
    allow_origin_regex=r"^https://.*\.vercel\.app$",
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

# -------------------- PRICE HELPER --------------------
def compute_per_unit_price(
    inv_unit_price: float | None,
    price_basis: str | None,
    order_unit_price: float | None,
    case_size: float | None,
    conversion: float | None,
    order_unit: str | None,
) -> tuple[float | None, str]:
    def is_case_like(s: str | None) -> bool:
        if not s: return False
        s = s.lower()
        return any(k in s for k in ("case", "cs", "box", "pack"))
    if price_basis == "per_unit" and (inv_unit_price or 0) > 0:
        return inv_unit_price, "per_unit (direct)"
    if (order_unit_price or 0) > 0:
        if (conversion or 0) > 0:
            return order_unit_price / conversion, f"via_conversion ({conversion})"
        if (case_size or 0) > 0 and is_case_like(order_unit):
            return order_unit_price / case_size, f"via_case_size ({case_size})"
        return order_unit_price, "order_price_assumed_per_unit"
    if (inv_unit_price or 0) > 0:
        return inv_unit_price, "fallback_inv_unit_price"
    return None, "no_price"

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

def get_user_optional(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)) -> Optional[User]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = payload.get("sub")
        if not uid: return None
        return db.query(User).get(int(uid))
    except Exception:
        return None

def require_role_or_admin_key(allowed_roles: List[str]):
    def checker(
        user: Optional[User] = Depends(get_user_optional),
        x_admin_key: Optional[str] = Header(None)
    ):
        if ADMIN_KEY and x_admin_key == ADMIN_KEY:
            return True
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
    class Config: from_attributes = True

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class RegisterIn(BaseModel):
    email: str
    password: str
    name: Optional[str] = None
    role: str = "viewer"

class UpdateUserIn(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    active: Optional[bool] = None
    new_password: Optional[str] = None


class ItemOut(BaseModel):
    id: int
    name: str
    storage_area: Optional[str] = None
    par: float = 0.0
    inv_unit_price: float = 0.0
    active: bool = True
    # NEW fields
    order_unit: Optional[str] = None
    inventory_unit: Optional[str] = None
    case_size: Optional[float] = None
    conversion: Optional[float] = None
    order_unit_price: Optional[float] = None
    price_basis: Optional[str] = None
    class Config: from_attributes = True

class ItemCreate(BaseModel):
    name: str
    storage_area: Optional[str] = None
    par: Optional[float] = 0.0
    inv_unit_price: Optional[float] = 0.0
    active: Optional[bool] = True
    # NEW fields
    order_unit: Optional[str] = None
    inventory_unit: Optional[str] = None
    case_size: Optional[float] = None
    conversion: Optional[float] = None
    order_unit_price: Optional[float] = None
    price_basis: Optional[str] = None

class CountLineIn(BaseModel):
    item_id: int
    qty: float

class CountOut(BaseModel):
    id: int
    count_date: date
    storage_area: Optional[str] = None
    lines: List[CountLineIn] = []
    class Config: from_attributes = True

class CountCreate(BaseModel):
    storage_area: Optional[str] = None
    lines: List[CountLineIn]

class ReceiveOCRIn(BaseModel):
    receiver: Optional[str] = None
    lines: List[Dict]

# -------------------- AUTH ROUTES --------------------
auth_router = APIRouter(prefix="/auth", tags=["auth"])

@auth_router.post("/register", response_model=UserOut,
                  dependencies=[Depends(require_role_or_admin_key(["admin"]))])
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email.lower()).first():
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

# -------------------- ADMIN ROUTES --------------------
admin_router = APIRouter(prefix="/admin", tags=["admin"],
    dependencies=[Depends(require_role_or_admin_key(["admin"]))])

@admin_router.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).order_by(User.id.asc()).all()

@admin_router.put("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, payload: UpdateUserIn, db: Session = Depends(get_db)):
    u = db.query(User).get(user_id)
    if not u: raise HTTPException(404, "User not found")
    if payload.name is not None: u.name = payload.name
    if payload.role is not None:
        if payload.role not in ("admin", "manager", "counter", "viewer"):
            raise HTTPException(400, "Invalid role")
        u.role = payload.role
    if payload.active is not None: u.active = bool(payload.active)
    if payload.new_password: u.hashed_password = get_password_hash(payload.new_password)
    db.commit(); db.refresh(u); return u

class ResetIn(BaseModel):
    targets: List[str]

@admin_router.delete("/reset")
def reset_data(payload: ResetIn = Body(...), db: Session = Depends(get_db)):
    allowed = {"items", "counts", "users"}
    targets = set([t.lower() for t in (payload.targets or [])]) & allowed
    if not targets:
        raise HTTPException(400, "Specify at least one of: items, counts, users")
    if "counts" in targets:
        db.query(CountLine).delete(synchronize_session=False)
        db.query(Count).delete(synchronize_session=False)
    if "items" in targets:
        db.query(Item).delete(synchronize_session=False)
    if "users" in targets:
        db.query(User).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "message": f"Reset complete ({', '.join(sorted(list(targets)))})"}

# 🔹 MIGRATION ENDPOINT (for Postman)
@admin_router.post("/migrate-items")
def migrate_items(db: Session = Depends(get_db)):
    sql = """
    ALTER TABLE items ADD COLUMN IF NOT EXISTS order_unit TEXT;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS inventory_unit TEXT;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS case_size DOUBLE PRECISION;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS conversion DOUBLE PRECISION;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS order_unit_price DOUBLE PRECISION;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS price_basis TEXT;
    """
    try:
        db.execute(text(sql))
        db.commit()
        return {"status": "ok", "message": "Migration complete"}
    except Exception as e:
        db.rollback()
        return {"status": "error", "detail": str(e)}


# --- Migration endpoint to add new item columns (idempotent) ---
@admin_router.post("/migrate-items")
def migrate_items(db: Session = Depends(get_db)):
    sql = """
    ALTER TABLE items ADD COLUMN IF NOT EXISTS order_unit TEXT;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS inventory_unit TEXT;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS case_size DOUBLE PRECISION;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS conversion DOUBLE PRECISION;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS order_unit_price DOUBLE PRECISION;
    ALTER TABLE items ADD COLUMN IF NOT EXISTS price_basis TEXT;
    """
    try:
        db.execute(text(sql))
        db.commit()
        return {"status":"ok","message":"Migration complete"}
    except Exception as e:
        db.rollback()
        return {"status":"error","detail":str(e)}
app.include_router(admin_router)

# -------------------- STARTUP --------------------
@app.on_event("startup")
def startup_event():
    create_db()

# -------------------- CORE ROUTES --------------------
@app.get("/")
def root():
    return {"service": "tos-inventory-backend", "ok": True}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/items", response_model=List[ItemOut])
def list_items(
    q: Optional[str] = None,
    area: Optional[str] = Query(None, description="Filter by storage_area"),
    db: Session = Depends(get_db),
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
        exists.par = payload.par if payload.par is not None else (exists.par or 0.0)
        # compute / update price if info provided
        per_unit, _ = compute_per_unit_price(
            payload.inv_unit_price, payload.price_basis, payload.order_unit_price,
            payload.case_size, payload.conversion, payload.order_unit
        )
        if per_unit is not None:
            exists.inv_unit_price = per_unit
        if payload.active is not None:
            exists.active = bool(payload.active)
        # metadata
        for fld in ("order_unit","inventory_unit","case_size","conversion","order_unit_price","price_basis"):
            val = getattr(payload, fld)
            if val is not None:
                setattr(exists, fld, val)
        db.commit(); db.refresh(exists)
        return exists

    per_unit, _ = compute_per_unit_price(
        payload.inv_unit_price, payload.price_basis, payload.order_unit_price,
        payload.case_size, payload.conversion, payload.order_unit
    )
    rec = Item(
        name=name, storage_area=area,
        par=payload.par or 0.0,
        inv_unit_price=(per_unit or 0.0),
        active=bool(payload.active) if payload.active is not None else True,
        order_unit=payload.order_unit, inventory_unit=payload.inventory_unit,
        case_size=payload.case_size, conversion=payload.conversion,
        order_unit_price=payload.order_unit_price, price_basis=payload.price_basis,
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

    # update metadata first
    for fld in ("order_unit","inventory_unit","case_size","conversion","order_unit_price","price_basis"):
        val = getattr(payload, fld)
        if val is not None:
            setattr(rec, fld, val)

    # recompute per-unit if enough info was sent
    per_unit, _ = compute_per_unit_price(
        payload.inv_unit_price, payload.price_basis, payload.order_unit_price,
        payload.case_size, payload.conversion, payload.order_unit
    )
    if per_unit is not None:
        rec.inv_unit_price = per_unit
    elif payload.inv_unit_price is not None:
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
    def _f(v, default=0.0):
        try: 
            s = str(v).replace(',', '').replace('$','').strip()
            return float(s) if s else default
        except: 
            return default
    for row in reader:
        name = (row.get("name") or row.get("Item") or "").strip()
        if not name: continue
        storage = (row.get("storage_area") or row.get("Location") or None) or None
        par = _f(row.get("par") or row.get("PAR"))
        inv_unit_price = _f(row.get("inv_unit_price"))
        order_unit_price = _f(row.get("order_unit_price"))
        case_size = _f(row.get("case_size"), None)
        conversion = _f(row.get("conversion"), None)
        inventory_unit = (row.get("inventory_unit") or row.get("Unit") or None)
        order_unit = (row.get("order_unit") or None)
        price_basis = (row.get("price_basis") or None)

        per_unit, _ = compute_per_unit_price(
            inv_unit_price, price_basis, order_unit_price, case_size, conversion, order_unit
        )

        exists = db.query(Item).filter(Item.name==name, Item.storage_area==storage).first()
        if exists:
            exists.par = par if par is not None else exists.par
            if per_unit is not None: exists.inv_unit_price = per_unit
            exists.inventory_unit = inventory_unit or exists.inventory_unit
            exists.order_unit = order_unit or exists.order_unit
            if case_size is not None: exists.case_size = case_size
            if conversion is not None: exists.conversion = conversion
            if order_unit_price is not None: exists.order_unit_price = order_unit_price
            if price_basis is not None: exists.price_basis = price_basis
            exists.active = True
            updated += 1
        else:
            db.add(Item(
                name=name, storage_area=storage, par=par or 0.0,
                inv_unit_price=(per_unit or 0.0), active=True,
                inventory_unit=inventory_unit, order_unit=order_unit,
                case_size=case_size, conversion=conversion,
                order_unit_price=order_unit_price, price_basis=price_basis
            ))
            created += 1
    db.commit()
    return {"created": created, "updated": updated}

# -------------------- COUNTS & AUTO-PO --------------------
@app.get("/counts", response_model=List[CountOut])
def list_counts(db: Session = Depends(get_db)):
    res: List[CountOut] = []
    counts = db.query(Count).order_by(Count.id.desc()).limit(100).all()
    for c in counts:
        lines = [CountLineIn(item_id=ln.item_id or 0, qty=ln.qty or 0.0) for ln in c.lines]
        res.append(CountOut(id=c.id, count_date=c.count_date, storage_area=c.storage_area, lines=lines))
    return res

@app.post("/counts", response_model=CountOut, status_code=201,
          dependencies=[Depends(require_role_or_admin_key(["admin", "manager", "counter"]))])
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

# -------------------- OCR --------------------
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
    return {"ok": True, "gcv": bool(GCV_API_KEY), "ocrspace": bool(OCR_API_KEY)}

@app.post("/invoice/ocr", dependencies=[Depends(require_role_or_admin_key(["admin","manager"]))])
async def ocr_invoice(
    file: UploadFile = File(...),
    receiver: Optional[str] = None,
    db: Session = Depends(get_db)
):
    text = ""
    if GCV_API_KEY:
        img_bytes = await file.read()
        img_b64 = base64.b64encode(img_bytes).decode("ascii")
        gcv_payload = {
            "requests": [{
                "image": {"content": img_b64},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }
        resp = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={GCV_API_KEY}",
            json=gcv_payload, timeout=90
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"GCV error {resp.status_code}")
        data = resp.json()
        text = data["responses"][0].get("fullTextAnnotation", {}).get("text", "") or ""
    elif OCR_API_KEY:
        resp = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (file.filename, await file.read())},
            data={"apikey": OCR_API_KEY, "language": "eng", "scale": "true", "OCREngine": "2"},
            timeout=90
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"OCR.space error {resp.status_code}")
        payload = resp.json()
        text = "\n".join([p.get("ParsedText","") for p in payload.get("ParsedResults",[])])
    else:
        raise HTTPException(400, "No OCR key configured on server")

    if not text.strip():
        raise HTTPException(400, "No text detected")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
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
        return {"lines": [{"text": ln, "item_id": None, "name": "", "storage_area": None, "qty": 0.0, "unit_price": 0.0} for ln in lines[:50]]}

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
        it = db.query(Item).get(int(ln["item_id"]))
        if it and (ln.get("unit_price") is not None):
            it.inv_unit_price = float(ln["unit_price"])
    db.commit()
    return {"ok": True, "receipt_id": r.id}

# -------------------- SALES / LABOR / PMIX --------------------
class SalesRecord(Base):
    __tablename__ = "sales_records"
    id = Column(Integer, primary_key=True)
    sale_date = Column(Date, index=True)
    item_name = Column(String, index=True)
    qty = Column(Float, default=0.0)
    net_sales = Column(Float, default=0.0)
    gross_sales = Column(Float, default=0.0)

class LaborRecord(Base):
    __tablename__ = "labor_records"
    id = Column(Integer, primary_key=True)
    work_date = Column(Date, index=True)
    employee = Column(String, index=True)
    job_title = Column(String, nullable=True)
    hours = Column(Float, default=0.0)
    wages = Column(Float, default=0.0)

class PmixRecord(Base):
    __tablename__ = "pmix_records"
    id = Column(Integer, primary_key=True)
    sale_date = Column(Date, index=True)
    item_name = Column(String, index=True)
    qty = Column(Float, default=0.0)
    net_sales = Column(Float, default=0.0)

class ImportResult(BaseModel):
    created: int
    updated: int

def _to_date(v):
    if not v: return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try: return datetime.strptime(v.strip(), fmt).date()
        except: pass
    return None

def _to_float(v, default=0.0):
    try:
        if v is None: return default
        s = str(v).replace(',', '').replace('$','').strip()
        return float(s) if s else default
    except:
        return default

@app.post("/import/sales", response_model=ImportResult,
          dependencies=[Depends(require_role_or_admin_key(["admin", "manager"]))])
async def import_sales(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    created = updated = 0
    for row in reader:
        d = _to_date(row.get("Date") or row.get("Business Date") or row.get("sale_date"))
        item = (row.get("Menu Item") or row.get("Item") or row.get("item_name") or "").strip()
        if not d or not item: 
            continue
        qty = _to_float(row.get("Qty") or row.get("Quantity") or row.get("qty"), 0.0)
        net = _to_float(row.get("Net Sales") or row.get("Net") or row.get("net_sales"), 0.0)
        gross = _to_float(row.get("Gross Sales") or row.get("Gross") or row.get("gross_sales"), 0.0)

        rec = db.query(SalesRecord).filter(SalesRecord.sale_date==d, SalesRecord.item_name==item).first()
        if rec:
            rec.qty = qty; rec.net_sales = net; rec.gross_sales = gross; updated += 1
        else:
            rec = SalesRecord(sale_date=d, item_name=item, qty=qty, net_sales=net, gross_sales=gross)
            db.add(rec); created += 1
    db.commit(); Base.metadata.create_all(bind=engine)
    return ImportResult(created=created, updated=updated)

@app.post("/import/labor", response_model=ImportResult,
          dependencies=[Depends(require_role_or_admin_key(["admin", "manager"]))])
async def import_labor(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    created = updated = 0
    for row in reader:
        d = _to_date(row.get("Date") or row.get("Business Date") or row.get("work_date"))
        emp = (row.get("Employee") or row.get("Name") or row.get("employee") or "").strip()
        if not d or not emp: 
            continue
        job = (row.get("Job") or row.get("Job Title") or row.get("job_title") or None)
        hours = _to_float(row.get("Hours") or row.get("Total Hours") or row.get("hours"), 0.0)
        wages = _to_float(row.get("Wages") or row.get("Total Wages") or row.get("wages"), 0.0)

        rec = db.query(LaborRecord).filter(LaborRecord.work_date==d, LaborRecord.employee==emp, LaborRecord.job_title==job).first()
        if rec:
            rec.hours = hours; rec.wages = wages; updated += 1
        else:
            rec = LaborRecord(work_date=d, employee=emp, job_title=job, hours=hours, wages=wages)
            db.add(rec); created += 1
    db.commit(); Base.metadata.create_all(bind=engine)
    return ImportResult(created=created, updated=updated)

@app.post("/import/pmix", response_model=ImportResult,
          dependencies=[Depends(require_role_or_admin_key(["admin", "manager"]))])
async def import_pmix(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    created = updated = 0
    for row in reader:
        d = _to_date(row.get("Date") or row.get("Business Date") or row.get("sale_date"))
        item = (row.get("Menu Item") or row.get("Item") or row.get("item_name") or "").strip()
        if not d or not item: 
            continue
        qty = _to_float(row.get("Qty") or row.get("Quantity") or row.get("Sold Qty") or row.get("qty"), 0.0)
        net = _to_float(row.get("Net Sales") or row.get("Net") or row.get("net_sales"), 0.0)
        rec = db.query(PmixRecord).filter(PmixRecord.sale_date==d, PmixRecord.item_name==item).first()
        if rec:
            rec.qty = qty; rec.net_sales = net; updated += 1
        else:
            rec = PmixRecord(sale_date=d, item_name=item, qty=qty, net_sales=net)
            db.add(rec); created += 1
    db.commit(); Base.metadata.create_all(bind=engine)
    return ImportResult(created=created, updated=updated)


# -------------------- REPORTS (Sales/Labor/PMix) --------------------
from fastapi import Query

@app.get("/reports/summary")
def reports_summary(
    start: str | None = Query(None, description="YYYY-MM-DD"),
    end: str | None = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db)
):
    # default window: last 14 days
    today = date.today()
    d1 = _parse_date(start) or (today - timedelta(days=13))
    d2 = _parse_date(end) or today
    # inclusive range
    # sales
    sales_q = db.query(
        func.coalesce(func.sum(SalesRecord.net_sales), 0.0),
        func.coalesce(func.sum(SalesRecord.gross_sales), 0.0)
    ).filter(SalesRecord.sale_date >= d1, SalesRecord.sale_date <= d2).one()
    net_sales, gross_sales = float(sales_q[0]), float(sales_q[1])

    # labor
    labor_q = db.query(
        func.coalesce(func.sum(LaborRecord.hours), 0.0),
        func.coalesce(func.sum(LaborRecord.wages), 0.0)
    ).filter(LaborRecord.work_date >= d1, LaborRecord.work_date <= d2).one()
    labor_hours, labor_wages = float(labor_q[0]), float(labor_q[1])

    # pmix count of items sold
    pmix_qty = float(db.query(func.coalesce(func.sum(PmixRecord.qty), 0.0))
                     .filter(PmixRecord.sale_date >= d1, PmixRecord.sale_date <= d2).scalar() or 0.0)

    labor_pct = (labor_wages / net_sales) if net_sales > 0 else None

    return {
        "start": d1.isoformat(),
        "end": d2.isoformat(),
        "totals": {
            "net_sales": net_sales,
            "gross_sales": gross_sales,
            "labor_hours": labor_hours,
            "labor_wages": labor_wages,
            "labor_pct": labor_pct,
            "pmix_qty": pmix_qty
        }
    }

@app.get("/reports/sales")
def reports_sales(
    start: str | None = Query(None), end: str | None = Query(None),
    db: Session = Depends(get_db)
):
    today = date.today()
    d1 = _parse_date(start) or (today - timedelta(days=13))
    d2 = _parse_date(end) or today
    rows = (db.query(SalesRecord.sale_date, func.sum(SalesRecord.net_sales), func.sum(SalesRecord.gross_sales))
              .filter(SalesRecord.sale_date >= d1, SalesRecord.sale_date <= d2)
              .group_by(SalesRecord.sale_date).order_by(SalesRecord.sale_date.asc()).all())
    return [{"date": d.isoformat(), "net_sales": float(n), "gross_sales": float(g)} for d,n,g in rows]

@app.get("/reports/labor")
def reports_labor(
    start: str | None = Query(None), end: str | None = Query(None),
    db: Session = Depends(get_db)
):
    today = date.today()
    d1 = _parse_date(start) or (today - timedelta(days=13))
    d2 = _parse_date(end) or today
    rows = (db.query(LaborRecord.work_date, func.sum(LaborRecord.hours), func.sum(LaborRecord.wages))
              .filter(LaborRecord.work_date >= d1, LaborRecord.work_date <= d2)
              .group_by(LaborRecord.work_date).order_by(LaborRecord.work_date.asc()).all())
    out = []
    # need net sales per day to compute labor % per day
    sales_map = {d: float(n) for d,n,_ in
                 db.query(SalesRecord.sale_date, func.sum(SalesRecord.net_sales), func.sum(SalesRecord.gross_sales))
                   .filter(SalesRecord.sale_date >= d1, SalesRecord.sale_date <= d2)
                   .group_by(SalesRecord.sale_date).all()}
    for d, hrs, wages in rows:
        net = sales_map.get(d, 0.0)
        out.append({
            "date": d.isoformat(),
            "hours": float(hrs),
            "wages": float(wages),
            "labor_pct": (float(wages)/net) if net>0 else None
        })
    return out

@app.get("/reports/pmix")
def reports_pmix(
    start: str | None = Query(None), end: str | None = Query(None),
    limit: int = 100,
    db: Session = Depends(get_db)
):
    today = date.today()
    d1 = _parse_date(start) or (today - timedelta(days=13))
    d2 = _parse_date(end) or today
    rows = (db.query(PmixRecord.item_name, func.sum(PmixRecord.qty), func.sum(PmixRecord.net_sales))
              .filter(PmixRecord.sale_date >= d1, PmixRecord.sale_date <= d2)
              .group_by(PmixRecord.item_name)
              .order_by(func.sum(PmixRecord.net_sales).desc())
              .limit(limit).all())
    return [{"item_name": n, "qty": float(q), "net_sales": float(ns)} for n,q,ns in rows]
