
import os
from datetime import date
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
import csv, io

# ---- Config ----
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tos.db")
ADMIN_KEY = os.getenv("ADMIN_KEY")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ---- Models ----
class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    storage_area = Column(String, nullable=True)
    par = Column(Float, default=0.0)
    inv_unit_price = Column(Float, default=0.0)
    active = Column(Boolean, default=True)
    # allow same item name in multiple locations
    __table_args__ = (UniqueConstraint("name", "storage_area", name="uix_item_name_area"),)

class Count(Base):
    __tablename__ = "counts"
    id = Column(Integer, primary_key=True, index=True)
    count_date = Column(Date, nullable=False, default=date.today)
    storage_area = Column(String, nullable=True)

class CountLine(Base):
    __tablename__ = "count_lines"
    id = Column(Integer, primary_key=True, index=True)
    count_id = Column(Integer, ForeignKey("counts.id", ondelete="CASCADE"))
    item_id = Column(Integer, ForeignKey("items.id", ondelete="SET NULL"))
    qty = Column(Float, default=0.0)

    count = relationship("Count", backref="lines")
    item = relationship("Item")

def create_db():
    Base.metadata.create_all(bind=engine)

# ---- App ----
app = FastAPI(title="TOS Inventory API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def require_admin(x_admin_key: Optional[str] = Header(None)):
    # If ADMIN_KEY env var is set, require it; else allow
    if ADMIN_KEY and x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return True

# ---- Schemas ----
class ItemOut(BaseModel):
    id: int
    name: str
    storage_area: Optional[str] = None
    par: float = 0.0
    inv_unit_price: float = 0.0
    active: bool = True
    class Config:
        orm_mode = True

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
        orm_mode = True

class CountCreate(BaseModel):
    storage_area: Optional[str] = None
    lines: List[CountLineIn]

# ---- Routes ----
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
    db: Session = Depends(get_db)
):
    qry = db.query(Item).filter(Item.active == True)
    if q:
        qry = qry.filter(Item.name.ilike(f"%{q}%"))
    if area:
        qry = qry.filter(Item.storage_area == area)
    return qry.order_by(Item.storage_area.asc().nullsfirst(), Item.name.asc()).all()

@app.post("/items", response_model=ItemOut, status_code=201, dependencies=[Depends(require_admin)])
def create_item(payload: ItemCreate, db: Session = Depends(get_db)):
    # Upsert by (name, storage_area)
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
        name=name,
        storage_area=area,
        par=payload.par or 0.0,
        inv_unit_price=payload.inv_unit_price or 0.0,
        active=bool(payload.active) if payload.active is not None else True
    )
    db.add(rec); db.commit(); db.refresh(rec)
    return rec

@app.put("/items/{item_id}", response_model=ItemOut, dependencies=[Depends(require_admin)])
def update_item(item_id: int, payload: ItemCreate, db: Session = Depends(get_db)):
    rec = db.query(Item).get(item_id)
    if not rec:
        raise HTTPException(404, "Not found")
    if payload.name and payload.name.strip() != rec.name:
        rec.name = payload.name.strip()
    new_area = payload.storage_area or rec.storage_area
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

@app.delete("/items/{item_id}", status_code=204, dependencies=[Depends(require_admin)])
def delete_item(item_id: int, db: Session = Depends(get_db)):
    rec = db.query(Item).get(item_id)
    if not rec:
        raise HTTPException(404, "Not found")
    db.delete(rec); db.commit()
    return None

@app.post("/import/catalog", status_code=201, dependencies=[Depends(require_admin)])
async def import_catalog(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Upload a CSV exported from your Catalog sheet.")
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
            db.add(Item(name=name, storage_area=storage, par=par or 0.0, inv_unit_price=price or 0.0, active=True))
            created += 1
    db.commit()
    return {"created": created, "updated": updated}

@app.get("/counts", response_model=List[CountOut])
def list_counts(db: Session = Depends(get_db)):
    res = []
    counts = db.query(Count).order_by(Count.id.desc()).limit(100).all()
    for c in counts:
        lines = [CountLineIn(item_id=ln.item_id or 0, qty=ln.qty or 0.0) for ln in c.lines]
        res.append(CountOut(id=c.id, count_date=c.count_date, storage_area=c.storage_area, lines=lines))
    return res

@app.post("/counts", response_model=CountOut, status_code=201, dependencies=[Depends(require_admin)])
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

from main_ocr import app as ocr_app  # make sure file is committed
app.mount("/", ocr_app)  # merges OCR endpoints with your API

if __name__ == "__main__":
    create_db()
