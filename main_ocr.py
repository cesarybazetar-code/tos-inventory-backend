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

# ------------------------------------------------
# Config
# ------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tos.db")
ADMIN_KEY    = os.getenv("ADMIN_KEY")
OCR_API_KEY  = os.getenv("OCR_API_KEY")  # <-- set on Render (your ocr.space key)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# ------------------------------------------------
# DB Models (minimal, compatible with your app)
# ------------------------------------------------
class Item(Base):
    __tablename__ = "items"
    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String, nullable=False)
    storage_area = Column(String, nullable=True)
    par          = Column(Float, default=0.0)
    inv_unit_price = Column(Float, default=0.0)
    active       = Column(Boolean, default=True)
    __table_args__ = (UniqueConstraint("name", "storage_area", name="uix_item_name_area"),)

class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id         = Column(Integer, primary_key=True, index=True)
    vendor_id  = Column(Integer, nullable=True)
    status     = Column(String, default="received")
    created_at = Column(Date, default=date.today)

class Receipt(Base):
    __tablename__ = "receipts"
    id          = Column(Integer, primary_key=True, index=True)
    po_id       = Column(Integer, nullable=True)
    received_at = Column(Date, default=date.today)
    receiver    = Column(String, nullable=True)

class ReceiptLine(Base):
    __tablename__ = "receipt_lines"
    id         = Column(Integer, primary_key=True, index=True)
    receipt_id = Column(Integer, ForeignKey("receipts.id", ondelete="CASCADE"))
    item_id    = Column(Integer, nullable=True)
    qty        = Column(Float, default=0.0)
    unit_price = Column(Float, default=0.0)

def create_db():
    Base.metadata.create_all(bind=engine)

# ------------------------------------------------
# FastAPI
# ------------------------------------------------
app = FastAPI(title="TOS Invoice OCR Patch", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_admin(x_admin_key: Optional[str] = Header(None)):
    # If ADMIN_KEY is set, enforce. If not set, allow (dev mode).
    if ADMIN_KEY and x_admin_key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    return True

# ------------------------------------------------
# Helpers
# ------------------------------------------------
PRICE_RE = re.compile(r"(?<!\d)(\d{1,4}(?:\.\d{2}))")
QTY_RE   = re.compile(r"(?<!\d)(\d{1,4})(?![\d\.])")

def fuzzy_match(items, text):
    """very simple substring/token overlap scoring"""
    tl = text.lower()
    best, score = None, 0
    for it in items:
        name = (it.name or "").lower()
        if not name:
            continue
        if name in tl:
            s = len(name)
        else:
            toks = [t for t in name.split() if len(t) > 2]
            s = sum(1 for t in toks if t in tl)
        if s > score:
            best, score = it, s
    return best, score

def _join_parsed_text(payload: dict) -> str:
    parts = []
    for pr in payload.get("ParsedResults", []):
        t = pr.get("ParsedText", "")
        if t:
            parts.append(t)
    return "\n".join(parts)

# ------------------------------------------------
# Routes
# ------------------------------------------------
@app.on_event("startup")
def _startup():
    create_db()

@app.get("/ocr/health")
def health():
    return {"ok": True, "ocr": bool(OCR_API_KEY)}

@app.post("/invoice/ocr", dependencies=[Depends(require_admin)])
async def ocr_invoice(
    file: UploadFile = File(...),
    receiver: Optional[str] = None,
    db: Session = Depends(get_db)
):
    if not OCR_API_KEY:
        raise HTTPException(400, "OCR_API_KEY not set on server (use OCR.space key)")

    # read bytes once
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "Empty file")

    # ---- OCR.space request (table-friendly settings) ----
    files = {"file": (file.filename, file_bytes)}
    data  = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "scale": "true",
        "isTable": "true",   # <— important for invoices
        "OCREngine": "3"     # engine 3 works better for structured docs
    }
    resp = requests.post("https://api.ocr.space/parse/image", files=files, data=data, timeout=90)

    # fallback: try engine 2 if engine 3 fails or returns no text
    try_engine2 = False
    if resp.status_code != 200:
        try_engine2 = True
    else:
        payload = resp.json()
        if not payload.get("ParsedResults"):
            try_engine2 = True

    if try_engine2:
        data_fallback = {
            "apikey": OCR_API_KEY,
            "language": "eng",
            "scale": "true",
            "isTable": "true",
            "OCREngine": "2"
        }
        resp = requests.post("https://api.ocr.space/parse/image", files=files, data=data_fallback, timeout=90)
        if resp.status_code != 200:
            raise HTTPException(502, f"OCR provider error {resp.status_code}")
        payload = resp.json()

    raw_text = _join_parsed_text(payload).strip()
    if not raw_text:
        raise HTTPException(400, "No text detected")

    # ---- Parse lines and match to catalog items ----
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    items = db.query(Item).filter(Item.active == True).all()

    parsed = []
    for ln in lines:
        if len(ln) < 3:
            continue

        price = None
        qty   = None

        # last price-like number on the line
        m_price = list(PRICE_RE.finditer(ln))
        if m_price:
            try:
                price = float(m_price[-1].group(1))
            except:
                price = None

        # naive qty guess: first integer-ish token
        tokens = ln.split()
        if tokens:
            m_qty = QTY_RE.findall(tokens[0]) or QTY_RE.findall(ln)
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
                "unit_price": (price if price is not None else (match.inv_unit_price or 0.0))
            })

    return {"lines": parsed}

class ReceiveOCRIn(BaseModel):
    receiver: Optional[str] = None
    lines: List[Dict]

@app.post("/receive/ocr", dependencies=[Depends(require_admin)])
def receive_from_ocr(payload: ReceiveOCRIn, db: Session = Depends(get_db)):
    # create a receipt and lines; update item last price
    r = Receipt(receiver=payload.receiver or None)
    db.add(r); db.commit(); db.refresh(r)

    for ln in payload.lines:
        item_id    = int(ln["item_id"])
        qty        = float(ln.get("qty") or 0)
        unit_price = float(ln.get("unit_price") or 0)

        db.add(ReceiptLine(receipt_id=r.id, item_id=item_id, qty=qty, unit_price=unit_price))

        it = db.query(Item).get(item_id)
        if it is not None:
            it.inv_unit_price = unit_price

    db.commit()
    return {"ok": True, "receipt_id": r.id}
