import os, re
from typing import Optional, List, Dict
from datetime import date

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Date,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# -------------------- Config --------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tos.db")
ADMIN_KEY = os.getenv("ADMIN_KEY")
OCR_API_KEY = os.getenv("OCR_API_KEY")  # <-- set this in Render (OCR.space key)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# -------------------- DB Models --------------------
class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    storage_area = Column(String, nullable=True)
    par = Column(Float, default=0.0)
    inv_unit_price = Column(Float, default=0.0)
    active = Column(Boolean, default=True)
    __table_args__ = (UniqueConstraint("name", "storage_area", name="uix_item_name_area"),)

class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id = Column(Integer, primary_key=True, index=True)
    vendor_id = Column(Integer, nullable=True)
    status = Column(String, default="received")
    created_at = Column(Date, default=date.today)   # ✅ fixed typo

class Receipt(Base):
    __tablename__ = "receipts"
    id = Column(Integer, primary_key=True, index=True)
    po_id = Column(Integer, nullable=True)
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

# -------------------- App --------------------
app = FastAPI(title="TOS Invoice OCR", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_admin(x_admin_key: Optional[str] = Header(None)):
    if ADMIN_KEY and x_admin_key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    return True

@app.on_event("startup")
def _startup():
    create_db()

# -------------------- Helpers --------------------
PRICE_RE = re.compile(r"(?<!\d)(\d{1,4}(?:\.\d{2}))")
QTY_RE   = re.compile(r"(?<!\d)(\d{1,4})(?![\d\.])")

def fuzzy_match(items: List[Item], text: str):
    """Naive substring/token overlap scoring against catalog items."""
    t = text.lower()
    best, score = None, 0
    for it in items:
        name = (it.name or "").lower()
        if not name:
            continue
        if name in t:
            s = len(name)
        else:
            tokens = [tok for tok in name.split() if len(tok) > 2]
            s = sum(1 for tok in tokens if tok in t)
        if s > score:
            best, score = it, s
    return best, score

# -------------------- Health --------------------
@app.get("/ocr/health")
def health():
    return {"ok": True, "ocr": bool(OCR_API_KEY)}

# -------------------- OCR Endpoint --------------------
@app.post("/invoice/ocr", dependencies=[Depends(require_admin)])
async def ocr_invoice(file: UploadFile = File(...), receiver: Optional[str] = None, db: Session = Depends(get_db)):
    if not OCR_API_KEY:
        raise HTTPException(400, "OCR_API_KEY not set on server (use OCR.space key)")

    content = await file.read()

    def call_ocr(engine: int):
        return requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": (file.filename, content)},
            data={
                "apikey": OCR_API_KEY,
                "language": "eng",
                "OCREngine": str(engine),
                "isTable": "true",
                "scale": "true",
                "detectOrientation": "true",
                "isOverlayRequired": "false",
            },
            timeout=90,
        )

    # Engine 2 first
    resp = call_ocr(2)
    if resp.status_code != 200:
        raise HTTPException(502, f"OCR provider error {resp.status_code}")
    data = resp.json()

    parsed_text = "".join(pr.get("ParsedText", "") for pr in data.get("ParsedResults", [])).strip()
    if not parsed_text:
        resp = call_ocr(1)
        if resp.status_code != 200:
            raise HTTPException(502, f"OCR provider error {resp.status_code}")
        data = resp.json()
        parsed_text = "".join(pr.get("ParsedText", "") for pr in data.get("ParsedResults", [])).strip()

    if not parsed_text:
        raise HTTPException(400, "No text detected")

    raw_lines = [ln.strip() for ln in parsed_text.splitlines() if ln.strip()]

    items = db.query(Item).filter(Item.active == True).all()
    parsed = []
    for ln in raw_lines:
        if len(ln) < 3:
            continue
        price, qty = None, None
        m_price = list(PRICE_RE.finditer(ln))
        if m_price:
            price = float(m_price[-1].group(1))
        m_qty = QTY_RE.findall(ln.split()[0]) or QTY_RE.findall(ln)
        if m_qty:
            try:
                qty = float(m_qty[0])
            except:
                qty = None
        match, score = fuzzy_match(items, ln)
        if match and score >= 1:
            parsed.append({
                "text": ln,
                "item_id": match.id,
                "name": match.name,
                "storage_area": match.storage_area,
                "qty": qty or 0.0,
                "unit_price": price or match.inv_unit_price or 0.0,
            })

    if not parsed:
        return {"lines": [{"text": ln, "item_id": None, "name": "", "storage_area": None, "qty": 0.0, "unit_price": 0.0} for ln in raw_lines]}

    return {"lines": parsed}

# -------------------- Receive Endpoint --------------------
class ReceiveOCRIn(BaseModel):
    receiver: Optional[str] = None
    lines: List[Dict]

@app.post("/receive/ocr", dependencies=[Depends(require_admin)])
def receive_from_ocr(payload: ReceiveOCRIn, db: Session = Depends(get_db)):
    r = Receipt(receiver=payload.receiver or None)
    db.add(r); db.commit(); db.refresh(r)

    for ln in payload.lines:
        db.add(
            ReceiptLine(
                receipt_id=r.id,
                item_id=int(ln["item_id"]) if ln.get("item_id") is not None else None,
                qty=float(ln.get("qty") or 0),
                unit_price=float(ln.get("unit_price") or 0.0),
            )
        )
        if ln.get("item_id") is not None:
            it = db.query(Item).get(int(ln["item_id"]))
            if it and ln.get("unit_price") is not None:
                it.inv_unit_price = float(ln["unit_price"])

    db.commit()
    return {"ok": True, "receipt_id": r.id}
