"""Live to game: the running BeamNG through its built-in MCP server (Options > Advanced > MCP; tools/mcp.py), the
control channel tools/benchmark.py and tools/autodrive_perf.py use (run_lua in the game engine's Lua).

The game gets a kunstwerk inside its road tiles: earthd's road workers cut it into each tile (server/kunstwerken.py,
blockworker.roads_tile). A push therefore
1. needs earthd to read the new export: started with BNE_KUNSTWERK_RELOAD=1 it re-reads a changed kunstwerk file
   (server/kunstwerken.py, the fork's hook), else only after a restart;
2. asks the mod to drop its road tiles and fetch them again: beamngEarth_core.setRoadsEnabled(false) then (true),
   the settings page's own path (core.lua: roads.clear(), then 'roads_reset' to the server, which sends them all again).
The terrain under the kunstwerk and the switchable signal heads' navgraph follow only on the next teleport.
"""
import json
import urllib.error
import urllib.request

PUSH_LUA = ("local c = beamngEarth_core "
            "if not c or not c.setRoadsEnabled then return 'no beamngEarth_core (BeamNG.Earth level not loaded?)' end "
            "c.setRoadsEnabled(false) c.setRoadsEnabled(true) return 'ok: road tiles requested again'")
PING_LUA = "return beamngEarth_core ~= nil and 'earth' or 'no-earth'"


class GameError(RuntimeError):
    pass


def rpc(url, method, params=None, timeout=5.0):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", "replace"), strict=False)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise GameError(f"game not reachable at {url}: {e}") from e
    if "error" in d:
        raise GameError(f"RPC error: {d['error']}")
    return d.get("result", {})


def run_lua(url, code, timeout=10.0):
    res = rpc(url, "tools/call", {"name": "run_lua", "arguments": {"code": code}}, timeout)
    text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
    if res.get("isError"):
        raise GameError(f"Lua error: {text}")
    return text


def status(url):
    try:
        return {"reachable": True, "earth": run_lua(url, PING_LUA, timeout=2.0).strip() == "earth"}
    except GameError as e:
        return {"reachable": False, "earth": False, "error": str(e)}


def push(url):
    return run_lua(url, PUSH_LUA)
