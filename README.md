
# TOS Inventory — Backend (FastAPI)

Deploy on Render.

Build:
```
pip install -r requirements.txt
```
Start:
```
uvicorn main:app --host 0.0.0.0 --port 10000
```
Env Vars:
- DATABASE_URL=sqlite:///./tos.db
- ADMIN_KEY=your-secret-key
