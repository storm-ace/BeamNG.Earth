"""The dashboard's HTTP API (FastAPI) and its page (static/)."""
import asyncio
import io
import json
import os

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import game, mesh, overrides, rebuild
from .data import Config, Data, _load

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PHOTO_MAX_PX = 4096


def create_app(cfg=None):
    cfg = cfg or Config()
    data = Data(cfg)
    runner = rebuild.Runner()
    app = FastAPI(title="Kunstwerk-dashboard")
    app.state.cfg, app.state.data, app.state.runner = cfg, data, runner
    photo_cache = {}

    @app.get("/api/info")
    def info():
        return {"name": cfg.name, "repo": cfg.repo, "design": cfg.design, "ref": cfg.ref, "mesh": cfg.mesh_path,
                "mesh_present": os.path.exists(cfg.mesh_path), "sv_root": cfg.sv_root, "place": cfg.place,
                "sv_places": data.sv_places(), "overrides": cfg.overrides_path, "mcp_url": cfg.mcp_url,
                "origin_rd": list(data.frame().origin)}

    @app.get("/api/objects")
    def objects():
        return data.objects()

    @app.get("/api/osm/ways")
    def osm_ways():
        return {"ways": data.osm_ways()}

    @app.get("/api/mesh.bin")
    def mesh_bin():
        m = mesh.load(cfg.mesh_path)
        if m is None:
            raise HTTPException(404, f"no {cfg.mesh_path} (tools/kunstwerk_export.py)")
        return Response(m["bin"], media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    @app.get("/api/ground")
    def ground():
        meta = _load(os.path.join(cfg.ref, "photo_hr.json"))
        photo = os.path.join(cfg.ref, "photo_hr.jpg")
        if meta is None or not os.path.exists(photo):
            return {"present": False}
        out = {"present": True, "box": meta["box"], "px": meta.get("px")}
        out.update(data.ground().grid(meta["box"], 2.0))
        return out

    @app.get("/api/ground/photo.jpg")
    def ground_photo():
        path = os.path.join(cfg.ref, "photo_hr.jpg")
        if not os.path.exists(path):
            raise HTTPException(404, "no photo")
        key = (path, os.stat(path).st_mtime_ns)
        if key not in photo_cache:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(path).convert("RGB")
            s = PHOTO_MAX_PX / max(im.size)
            if s < 1:
                im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88)
            photo_cache.clear()
            photo_cache[key] = buf.getvalue()
        return Response(photo_cache[key], media_type="image/jpeg")

    # -- Street View diagnosis
    @app.get("/api/sv")
    def sv(place: str = ""):
        try:
            return data.sv(place or cfg.place)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/sv/shot/{place}/{sid}.png")
    def sv_shot(place: str, sid: str, annotated: int = 0):
        try:
            path = data.shot_path(place, sid, bool(annotated))
        except ValueError as e:
            raise HTTPException(400, str(e))
        if not os.path.exists(path):
            raise HTTPException(404, "no such shot")
        return FileResponse(path, media_type="image/png")

    # -- overrides
    @app.get("/api/overrides")
    def get_overrides():
        return {"path": cfg.overrides_path, "items": overrides.load(cfg.overrides_path)}

    @app.post("/api/overrides")
    def post_override(o: dict = Body(...)):
        try:
            known = data.known_ids()
            if o.get("id") not in known:
                raise ValueError(f"unknown id {o.get('id')!r}")
            items = overrides.upsert(cfg.overrides_path, o)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"items": items}

    @app.post("/api/overrides/delete")
    def delete_override(o: dict = Body(...)):
        items = overrides.delete(cfg.overrides_path, o.get("id"))
        if items is None:
            raise HTTPException(404, "no override with that id")
        return {"items": items}

    # -- rebuild
    @app.post("/api/rebuild")
    def start_rebuild(o: dict = Body(default={})):
        steps = cfg.steps if cfg.steps is not None else rebuild.default_steps(cfg, bool(o.get("placement")))
        job = runner.start(steps, cfg.repo)
        if job is None:
            raise HTTPException(409, "a rebuild is running")
        return {"job": job.id, "steps": [s[0] for s in steps]}

    @app.get("/api/rebuild/{job_id}/events")
    async def rebuild_events(job_id: str, request: Request):
        job = runner.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")

        async def gen():
            i = 0
            while True:
                lines, done = await asyncio.to_thread(job.wait_lines, i, 0.5)
                for ln in lines:
                    yield f"data: {json.dumps({'line': ln})}\n\n"
                i += len(lines)
                if done and not lines:
                    yield f"event: done\ndata: {json.dumps({'ok': job.ok})}\n\n"
                    return
                if await request.is_disconnected():
                    return
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @app.get("/api/rebuild/status")
    def rebuild_status():
        j = runner.current
        return {"job": j and j.id, "done": j is None or j.done, "ok": j and j.ok}

    # -- live to game
    @app.get("/api/game/status")
    def game_status():
        st = game.status(cfg.mcp_url)
        st["reload_hint"] = "earthd met BNE_KUNSTWERK_RELOAD=1 starten, anders leest het de nieuwe export pas na een herstart"
        return st

    @app.post("/api/game/push")
    def game_push():
        try:
            return {"ok": True, "result": game.push(cfg.mcp_url)}
        except game.GameError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=502)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    # the page's own scripts change while it is in use (a reload must pick them up, not the browser's copy)
    @app.middleware("http")
    async def no_cache_static(request, call_next):
        resp = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path == "/":
            resp.headers["Cache-Control"] = "no-cache"
        return resp
    return app
