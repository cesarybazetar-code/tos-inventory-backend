import os, io, re
from typing import Optional, List, Dict
from datetime import date

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, ForeignKey, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from PIL import Image  # <— new

# -------------------- Config --------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tos.db")
ADMIN_KEY = os.getenv("ADMIN_KEY")
OCR_API_KEY = os.getenv("OCR_API_KEY")  # set on Render

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
    created_at = Column(Date, default=date.today)

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

app = FastAPI(title="TOS Invoice OCR", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_admin(x_admin_key: Optional[str] = Header(None)):
    # Only enforce when ADMIN_KEY is set
    if ADMIN_KEY and x_admin_key != ADMIN_KEY:
        raise HTTPException(401, "Invalid admin key")
    return True

# -------------------- Helpers --------------------
PRICE_RE = re.compile(r"(?<!\d)(\d{1,4}(?:\.\d{2}))")
QTY_RE = re.compile(r"(?<!\d)(\d{1,4})(?![\d\.])")

def fuzzy_match(items, text: str):
    tl = text.lower()
    best = None
    score = 0
    for it in items:
        name = (it.name or "").lower()
        if not name:
            continue
        s = 0
        if name in tl:
            s = len(name)
        else:
            toks = [t for t in name.split() if len(t) > 2]
            s = sum(1 for t in toks if t in tl)
        if s > score:
            score = s
            best = it
    return best, score

def read_and_normalize_image(upload: UploadFile) -> bytes:
    """
    Convert HEIC/huge images to a friendly JPEG under ~2500px max side.
    Returns JPEG bytes.
    """
    raw = upload.file.read()
    try:
        upload.file.seek(0)
    except Exception:
        pass

    try:
        # Try to open with PIL; if it fails, just send the raw bytes
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")  # ensure RGB
        # Downscale very large images (OCR.space free tier happier with smaller)
        max_side = 2500
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85, optimize=True)
        return out.getvalue()
    except Exception:
        # If PIL can’t read, fallback to original bytes
        return raw

def call_ocr_space(image_bytes: bytes, engine: int = 2) -> dict:
    """
    Call OCR.space with tuned params. engine=2 first; fallback to 1 if needed.
    """
    files = {'file': ('upload.jpg', image_bytes, 'image/jpeg')}
    data = {
        'apikey': OCR_API_KEY,
        'language': 'eng',
        'isTable': 'true',
        'scale': 'true',
        'detectOrientation': 'true',
        'OCREngine': str(engine),
    }
    r = requests.post("https://api.ocr.space/parse/image", files=files, data=data, timeout=90)
    if r.status_code != 200:
        raise HTTPException(502, f"OCR provider error {r.status_code}")
    return r.json()

# -------------------- Routes --------------------
@app.on_event("startup")
def startup():
    create_db()

@app.get("/ocr/health")
def health():
    return {"ok": True, "ocr": bool(OCR_API_KEY)}

class ReceiveOCRIn(BaseModel):
    receiver: Optional[str] = None
    lines: List[Dict]

@app.post("/invoice/ocr", dependencies=[Depends(require_admin)])
async def ocr_invoice(file: UploadFile = File(...), receiver: Optional[str] = None, db: Session = Depends(get_db)):
    if not OCR_API_KEY:
        raise HTTPException(400, "OCR_API_KEY not set on server (use OCR.space key)")

    # Normalize image for OCR
    image_bytes = read_and_normalize_image(file)

    # Call engine 2 first, then fallback to 1 if empty
    payload = call_ocr_space(image_bytes, engine=2)
    text_chunks = [p.get("ParsedText", "") for p in payload.get("ParsedResults", [])]
    raw_text = "\n".join([t for t in text_chunks if t])

    if not raw_text.strip():
        # Fallback to engine 1
        payload2 = call_ocr_space(image_bytes, engine=1)
        text_chunks2 = [p.get("ParsedText", "") for p in payload2.get("ParsedResults", [])]
        raw_text = "\n".join([t for t in text_chunks2 if t])

        if not raw_text.strip():
            # Surface provider error if present for easier debugging
            err = payload.get("ErrorMessage") or payload2.get("ErrorMessage") or "No text detected"
            raise HTTPException(400, str(err))

    # Parse lines
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    items = db.query(Item).filter(Item.active == True).all()

    parsed = []
    for ln in lines:
        if len(ln) < 3:
            continue
        price = None
        qty = None

        m_price = list(PRICE_RE.finditer(ln))
        if m_price:
            price = float(m_price[-1].group(1))

        # simple qty guess: first integer-ish token
        m_qty = QTY_RE.findall(ln.split()[0]) or QTY_RE.findall(ln)
        if m_qty:
            try:
                qty = float(m_qty[0])
            except Exception:
                qty = None

        match, score = fuzzy_match(items, ln)
        if match and score >= 1:
            parsed.append({
                "text": ln,
                "item_id": match.id,
                "name": match.name,
                "storage_area": match.storage_area,
                "qty": qty or 0.0,
                "unit_price": price or (match.inv_unit_price or 0.0),
            })

    return {"lines": parsed}

@app.post("/receive/ocr", dependencies=[Depends(require_admin)])
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
        if it and ln.get("unit_price") is not None:
            it.inv_unit_price = float(ln["unit_price"])
    db.commit()
    return {"ok": True, "receipt_id": r.id}