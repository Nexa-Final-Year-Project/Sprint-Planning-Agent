# --- TEMP FIX for Python 3.13 + FastAPI + Pydantic bug ---
import typing

try:
    ForwardRef = typing.ForwardRef
except AttributeError:
    ForwardRef = None

if ForwardRef is not None:
    _orig_eval = getattr(ForwardRef, "_evaluate", None)

    def _evaluate_wrapper(self, globalns, localns, *args, **kwargs):
        try:
            fwd = getattr(self, "__forward_arg__", None) or getattr(self, "__forward_arg", None)
            if isinstance(fwd, str):
                return eval(fwd, globalns or {}, localns or {})
        except Exception:
            pass
        if _orig_eval is not None:
            return _orig_eval(self, globalns, localns, *args, **kwargs)
        raise TypeError("Unable to evaluate ForwardRef")

    ForwardRef._evaluate = _evaluate_wrapper

# ---------------------------------------------------------

from fastapi import FastAPI
from app.routes.sprint_routes import router as sprint_router
from app.mcp_server import mcp as _mcp_server

app = FastAPI(title="NEXA Sprint Planner Agent")
app.include_router(sprint_router, prefix="/api/sprint", tags=["Sprint Planner"])

# Mount MCP server alongside existing REST routes (Streamable HTTP at /mcp)
try:
    if hasattr(_mcp_server, 'get_asgi_app'):
        app.mount("/mcp", _mcp_server.get_asgi_app())
    elif hasattr(_mcp_server, 'asgi_app'):
        app.mount("/mcp", _mcp_server.asgi_app)
    else:
        print("[WARN] MCP server could not be mounted — REST endpoints will still work")
except Exception as e:
    print(f"[WARN] MCP mount failed ({e}) — REST endpoints will still work")

@app.get("/")
async def root():
    return {"message": "Sprint Planner Agent is active 🚀"}
