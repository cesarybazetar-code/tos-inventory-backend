
# TOS Inventory — Backend (FastAPI)

Deploy on Render.

Build:
```
pip install -r requirements.txt
```
Start:
```
uvicorn main:app --host 0.0.0.0 --port $PORT
```
Env Vars:
- DATABASE_URL=sqlite:///./tos.db
- SECRET_KEY=change-me
- DATABASE_URL=postgres://... (prod)
- (optional) ACCESS_TOKEN_EXPIRE_MINUTES=60
