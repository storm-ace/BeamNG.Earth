#!/usr/bin/env python3
"""Kunstwerk-dashboard (fork only): the Droogleever Fortuynplein's kunstwerk in 3D in the browser, our signs, lights and
direction signs over it with NDW's, OSM's and the Street View scan's, and live fixes written to ref/overrides.json.

    python tools/dashboard/serve.py                    # http://localhost:8765
    python tools/dashboard/serve.py --name droogleever --design E:/BeamNG/design --port 8765

Reads design/kunstwerken/<name>/ref (placement.json, placement_fix.json, direction_signs.json, furniture.json,
approaches.json, ndw/signs.json, osm_signals.json, osm_map.xml, photo_hr.jpg/.json, ahn.npz), the export
<repo>/server/data/kunstwerken/<name>.kunstwerk.json and design/reference/sv/<place>/ (the Street View scan).
Rebuild runs the repo's tools/kunstwerk_*.py; Push to game talks to BeamNG's MCP server (game.py). Coordinates: geo.py.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # (tools/: the dashboard package)


def main(argv=None):
    from dashboard.data import Config
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=os.path.normpath(os.path.join(HERE, "..", "..")),
                    help="the repo whose tools run and whose server/data gets the export")
    ap.add_argument("--design", default="", help="the design folder (default: the one beside the repo)")
    ap.add_argument("--name", default="droogleever")
    ap.add_argument("--sv-root", default="", help="Street View scans (default: <design>/reference/sv)")
    ap.add_argument("--place", default="", help="the scan's place folder (default: --name)")
    ap.add_argument("--export-dir", default="", help="where the export goes and is read from "
                                                      "(default: <repo>/server/data/kunstwerken, earthd's)")
    ap.add_argument("--mcp",default="http://127.0.0.1:29292/mcp", help="BeamNG's MCP server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(argv)
    cfg = Config(repo=a.repo, design=a.design, name=a.name, sv_root=a.sv_root, place=a.place, mcp_url=a.mcp,
                 export_dir=a.export_dir)
    from dashboard.app import create_app
    import uvicorn
    print(f"kunstwerk-dashboard: {cfg.name}, ref {cfg.ref}, export {cfg.mesh_path}\n  http://localhost:{a.port}")
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
