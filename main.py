
import os
from datetime import date
from typing import Optional, List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
import csv, io

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tos.db")
ADMIN_KEY = os.getenv("ADMIN_KEY")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    storage_area = Column(String, nullable=True)
    par = Column(Float, default=0.0)
    inv_unit_price = Column(Float, default=0.0)
    active = Column(Boolean, default=True)

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

app = FastAPI(title="TOS Inventory API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def require_admin(x_admin_key: Optional[str] = Header(None)):
    if ADMIN_KEY and x_admin_key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    return True

class ItemOut(BaseModel):
    id:int; name:str; storage_area:Optional[str]=None; par:float=0.0; inv_unit_price:float=0.0; active:bool=True
    class Config: orm_mode=True

class ItemCreate(BaseModel):
    name:str; storage_area:Optional[str]=None; par:Optional[float]=0.0; inv_unit_price:Optional[float]=0.0; active:Optional[bool]=True

class CountLineIn(BaseModel):
    item_id:int; qty:float

class CountOut(BaseModel):
    id:int; count_date:date; storage_area:Optional[str]=None; lines:List[CountLineIn]=[]
    class Config: orm_mode=True

class CountCreate(BaseModel):
    storage_area:Optional[str]=None; lines:List[CountLineIn]

@app.on_event("startup")
def startup(): create_db()

@app.get("/health")
def health(): return {"ok": True}

@app.get("/items", response_model=List[ItemOut])
def list_items(q: Optional[str] = None, db: Session = Depends(get_db)):
    qry = db.query(Item)
    if q: qry = qry.filter(Item.name.ilike(f"%{q}%"))
    return qry.order_by(Item.name.asc()).all()

@app.post("/items", response_model=ItemOut, status_code=201, dependencies=[Depends(require_admin)])
def create_item(payload: ItemCreate, db: Session = Depends(get_db)):
    if db.query(Item).filter(Item.name == payload.name).first():
        raise HTTPException(400, "Item name exists")
    rec = Item(name=payload.name.strip(), storage_area=payload.storage_area, par=payload.par or 0.0, inv_unit_price=payload.inv_unit_price or 0.0, active=bool(payload.active))
    db.add(rec); db.commit(); db.refresh(rec); return rec

@app.post("/import/catalog", status_code=201, dependencies=[Depends(require_admin)])
async def import_catalog(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"): raise HTTPException(400, "Upload a CSV exported from your Catalog sheet.")
    content = await file.read(); reader = csv.DictReader(io.StringIO(content.decode("utf-8"))); created = 0
    for row in reader:
        name = (row.get("name") or row.get("Item") or row.get("NAME") or "").strip()
        if not name: continue
        storage = (row.get("storage_area") or row.get("Location") or "").strip()
        def f(v): 
            try: return float(v)
            except: return 0.0
        par = f(row.get("par") or row.get("PAR")); price = f(row.get("inv_unit_price") or row.get("Inv Unit Price"))
        exists = db.query(Item).filter(Item.name==name).first()
        if exists:
            exists.storage_area = storage; exists.par = par; exists.inv_unit_price = price; exists.active = True
        else:
            db.add(Item(name=name, storage_area=storage, par=par, inv_unit_price=price, active=True)); created += 1
    db.commit(); return {"created": created}

@app.get("/counts", response_model=List[CountOut])
def list_counts(db: Session = Depends(get_db)):
    res=[]; counts = db.query(Count).order_by(Count.id.desc()).limit(100).all()
    for c in counts:
        lines = [CountLineIn(item_id=ln.item_id or 0, qty=ln.qty or 0.0) for ln in c.lines]
        res.append(CountOut(id=c.id, count_date=c.count_date, storage_area=c.storage_area, lines=lines))
    return res

@app.post("/counts", response_model=CountOut, status_code=201, dependencies=[Depends(require_admin)])
def create_count(payload: CountCreate, db: Session = Depends(get_db)):
    c = Count(storage_area=payload.storage_area or None); db.add(c); db.commit(); db.refresh(c)
    for ln in payload.lines: db.add(CountLine(count_id=c.id, item_id=ln.item_id, qty=ln.qty))
    db.commit(); db.refresh(c); return CountOut(id=c.id, count_date=c.count_date, storage_area=c.storage_area, lines=payload.lines)

@app.get("/auto-po")
def auto_po(storage_area: Optional[str] = None, db: Session = Depends(get_db)):
    latest = db.query(Count).filter(Count.storage_area == storage_area if storage_area else True).order_by(Count.id.desc()).first()
    on_hand: Dict[int,float] = {}
    if latest: 
        for ln in latest.lines: on_hand[ln.item_id] = ln.qty
    q = db.query(Item); 
    if storage_area: q = q.filter(Item.storage_area == storage_area)
    out=[]
    for i in q:
        oh = on_hand.get(i.id, 0.0); reorder = max((i.par or 0.0) - oh, 0.0)
        if reorder > 0:
            out.append({"item_id":i.id,"name":i.name,"storage_area":i.storage_area,"on_hand":oh,"par":i.par or 0.0,"suggested_order_qty":reorder})
    return {"storage_area": storage_area, "lines": out}
