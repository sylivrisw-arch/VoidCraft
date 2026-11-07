"""
Voidcraft Scanner server  (runs on the laptop)

Phone opens  http://<laptop-ip>:5055  -> uploads a video -> this server does the
heavy work -> the phone page shows the result, takes change requests, and
"Import" saves a .cells file into the export folder.

Needs:  pip install flask opencv-python numpy
Run:    python scanner_server.py            (optional: --port 5055 --export "C:\\path")
Pair phone: a QR code opens on this laptop at start (needs:  pip install segno)

NOTE: the 3D-shape step is still the FAKE one (one sharp frame, inflated).
Replace shape_from_frames() with the real photogrammetry step later; nothing
else has to change.
"""
import os, sys, io, json, time, uuid, shutil, socket, argparse, threading, traceback, webbrowser
from pathlib import Path
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, abort, Response

try:                      # optional, only for the pairing QR code:  pip install segno
    import segno
except ImportError:
    segno = None

BASE = Path(__file__).resolve().parent
JOBS_DIR = BASE / "scan_jobs"
DETAIL = 5          # cubes per block (match your engine's cube detail)
BLOCKS = 10         # default object height in blocks
NFRAMES = 12        # frames pulled from the video
FRAME_MAX = 384     # longest side of a stored video frame, px
SLOT_MAX = 640      # longest side of a guided photo (bigger = better feature matching)
BG_TOL = 55         # cut-out strictness
MATS = ["wood", "dirt", "stone", "leaf", "concrete"]   # concrete = white / ceramic stand-in
VERSION = "0.8"
PORT = 5055

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 800 * 1024 * 1024
JOBS = {}
LOCK = threading.Lock()
EXPORT_DIR = BASE / "scan_exports"


# ---------------------------------------------------------------- helpers
def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def set_job(job, **kw):
    with LOCK:
        job.update(kw)


def subject_kind(note):
    t = (note or "").lower()
    if any(w in t for w in ("rock", "stone", "boulder", "granite", "statue", "wall")):
        return "rock"
    if any(w in t for w in ("cup", "mug", "plate", "bowl", "vase", "ceramic", "porcelain", "white")):
        return "ceramic"
    if any(w in t for w in ("stump", "log", "post", "fence", "chair", "table", "shed", "wood", "plank")):
        return "wood"
    return "auto"


def default_edits():
    return {"thick": 1.0, "trim": 0.0, "loose": False, "flat": False, "hgt": 1.0}


def write_jpg(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("could not encode frame")
    buf.tofile(str(path))


def read_img(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------- frames
def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the video (try an H.264 .mp4)")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in np.linspace(total * 0.04, total * 0.96, NFRAMES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    else:
        allf = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            allf.append(f)
        if allf:
            frames = [allf[i] for i in np.linspace(0, len(allf) - 1, NFRAMES).astype(int)]
    cap.release()
    if not frames:
        raise RuntimeError("No frames could be read from that video")
    out = []
    for f in frames:
        s = FRAME_MAX / max(f.shape[:2])
        out.append(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return out


def demo_frames():
    out = []
    for i in range(NFRAMES):
        c = np.zeros((256, 256, 3), np.uint8)
        cv2.fillPoly(c, [np.array([[116, 230], [124, 130], [134, 130], [144, 230]])], (35, 68, 107))
        cv2.rectangle(c, (124, 150), (174, 159), (35, 68, 107), -1)
        for (x, y, r) in [(128, 90, 52), (88, 112, 34), (168, 112, 36), (110, 60, 30), (150, 62, 30), (128, 125, 30)]:
            cv2.circle(c, (x, y), r, (44, 122, 47), -1)
        sx = 0.85 + 0.15 * abs(np.cos(i / NFRAMES * np.pi * 2))
        M = np.float32([[sx, 0, 128 * (1 - sx)], [0, 1, 0]])
        out.append(cv2.warpAffine(c, M, (256, 256)))
    return out


def sharpest(frames):
    return max(frames, key=lambda f: cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


# ---------------------------------------------------------------- 3D shape (FAKE for now)
def shape_from_frames(frames, note, edits, primary=None):
    """Replace this with real photogrammetry. It must return a 'model' dict:
       {W, H, cells:[[x,y,t,m,sh],...], solid, pieces}.
       Today: takes the sharpest frame and inflates its cut-out into a slab."""
    src = frames[primary] if primary is not None and 0 <= primary < len(frames) else sharpest(frames)
    return build_model(src, edits, subject_kind(note))


_CUT_CACHE = {}


def cutout_mask(frame):
    """Foreground mask for a frame (bool, same size). GrabCut from a centred box, with the old
    row-edge method as a fallback if GrabCut returns nothing sensible."""
    key = id(frame)
    if key in _CUT_CACHE:
        return _CUT_CACHE[key]
    h, w = frame.shape[:2]
    mask = None
    try:
        gc = np.zeros((h, w), np.uint8)
        mx, my = int(w * 0.06), int(h * 0.06)
        rect = (mx, my, w - 2 * mx, h - 2 * my)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(frame, gc, rect, bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        # keep the biggest blob, fill holes
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fg = (lab == big).astype(np.uint8)
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fg = np.zeros_like(fg)
            cv2.drawContours(fg, cnts, -1, 1, thickness=cv2.FILLED)
            frac = fg.mean()
            if 0.02 < frac < 0.9:
                mask = fg.astype(bool)
    except Exception:
        traceback.print_exc()
    if mask is None:                       # fallback: compare each row with its own edges
        f = frame[:, :, ::-1].astype(np.float32)
        edge = np.concatenate([f[:, :2], f[:, -2:]], axis=1).mean(axis=1)
        mask = np.linalg.norm(f - edge[:, None, :], axis=2) > BG_TOL
    if len(_CUT_CACHE) > 64:
        _CUT_CACHE.clear()
    _CUT_CACHE[key] = mask
    return mask


def build_model(src_bgr, edits, kind):
    edits = dict(edits, hgt=min(3.0, max(0.2, edits["hgt"])), thick=min(3.0, max(0.3, edits["thick"])))
    H = int(round(BLOCKS * edits["hgt"] * DETAIL))
    h0, w0 = src_bgr.shape[:2]
    W = max(4, int(round(H * w0 / h0)))
    img = cv2.resize(src_bgr, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)  # RGB

    mask = cv2.resize(cutout_mask(src_bgr).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    if edits["flat"]:
        mask[-4:] = False
    tr = int(round(H * edits["trim"]))
    if tr > 0:
        mask[:tr] = False
    if not mask.any():
        return {"W": W, "H": H, "cells": [], "solid": 0, "pieces": 0}

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    pieces = n - 1
    if edits["loose"] and pieces > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = lab == big
        pieces = 1

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L1, 3)
    tmax = max(1, int(round(8 * edits["thick"])))
    t = np.clip(np.round(np.sqrt(dist) * 1.1 * edits["thick"]), 1, tmax).astype(int)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx, mn = img.max(axis=2), img.min(axis=2)
    if kind == "rock":
        m = np.full((H, W), 2)
    elif kind == "wood":
        m = np.full((H, W), 0)
    elif kind == "ceramic":
        m = np.full((H, W), 4)
    else:
        m = np.select([(g > r * 1.08) & (g > b * 1.05), ((mx - mn) < 28) & (mx > 175), (mx - mn) < 28, (r > g) & (r < 135)],
                      [3, 4, 2, 0], default=1)
    sh = 0.82 + 0.3 * ((r + g + b) / 765.0 - 0.4)

    ys, xs = np.nonzero(mask)
    cells = [[int(x), int(H - 1 - y), int(t[y, x]), int(m[y, x]), round(float(sh[y, x]), 3)]
             for y, x in zip(ys, xs)]
    return {"W": W, "H": H, "cells": cells, "solid": int(2 * t[mask].sum()), "pieces": int(pieces)}


# ---------------------------------------------------------------- edit requests
def interpret_edit(text, edits):
    """Keyword version. Swap this for a call to your local LLM (LM Studio) later."""
    t = text.lower()
    e = dict(edits)
    done = []
    has = lambda *w: any(x in t for x in w)
    if has("thicker", "fatter", "bulk", "wider trunk", "thick trunk"):
        e["thick"] *= 1.35; done.append("thicker")
    if has("thinner", "slimmer", "skinny"):
        e["thick"] /= 1.35; done.append("thinner")
    if has("trim", "top", "canopy", "less leaves", "fewer leaves"):
        e["trim"] = min(0.5, e["trim"] + 0.12); done.append("trim top")
    if has("loose", "floating", "stray", "bits"):
        e["loose"] = True; done.append("remove loose bits")
    if has("flat", "base", "grass"):
        e["flat"] = True; done.append("flat base")
    if has("taller", "bigger", "larger"):
        e["hgt"] *= 1.25; done.append("taller")
    if has("smaller", "shorter"):
        e["hgt"] /= 1.25; done.append("smaller")
    e["hgt"] = min(3.0, max(0.2, e["hgt"]))
    e["thick"] = min(3.0, max(0.3, e["thick"]))
    return e, done


# ---------------------------------------------------------------- guided capture
# Four sides round the object (front = where you start) plus an optional top shot.
SLOTS = [("front", "Front"), ("right", "Right"), ("back", "Back"), ("left", "Left"), ("top", "Top"),
         ("whole", "Whole object")]   # whole = one photo of the entire thing, for when the sides are hard
SLOT_IDS = [x[0] for x in SLOTS]
EMPTY_SLOT = {"state": "empty", "advice": "", "blur": None, "matches": None}
RING = SLOT_IDS[:4]
BLUR_MIN = 25.0         # Laplacian variance below this = too blurry / too plain
DARK_MIN = 30           # mean brightness below this = too dark
MATCH_MIN = 25          # agreeing feature matches that count as good overlap with a neighbouring side (tuned on made-up images only)
_ORB = None


def orb():
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(1200)
    return _ORB


def feature_matches(a, b):
    """Number of feature matches between two views that agree on one camera geometry (RANSAC)."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = orb().detectAndCompute(ga, None)
    kb, db = orb().detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 8:
        return len(good) // 2
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 3.0, 0.99)
    return int(inl.sum()) if inl is not None else 0


def slot_path(jid, sid):
    return JOBS_DIR / jid / "slots" / f"{sid}.jpg"


def neighbours(sid):
    if sid == "whole":
        return []
    if sid in RING:
        i = RING.index(sid)
        return [RING[(i - 1) % 4], RING[(i + 1) % 4]]
    return list(RING)        # the top shot should overlap the sides


def check_slot(job, sid, img):
    """Returns (state, advice, detail). state is 'ok', 'warn' (kept, but little overlap) or 'redo'."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    detail = {"blur": round(blur, 1), "bright": round(bright, 1), "matches": None}
    if bright < DARK_MIN:
        return "redo", "Too dark. Add light and take it again.", detail
    if blur < BLUR_MIN:
        return "redo", "Blurry or too plain. Hold steady, let it focus, and include some texture or edges.", detail
    best, checked = 0, False
    names = dict(SLOTS)
    seen = []
    for nb in neighbours(sid):
        p = slot_path(job["id"], nb)
        if job["slots"].get(nb, {}).get("state") in ("ok", "warn", "redo") and p.exists():
            checked = True
            seen.append(names[nb])
            best = max(best, feature_matches(img, read_img(p)))
    detail["matches"] = best
    if checked and best < MATCH_MIN and sid == "top":
        return "warn", f"Saved, but it shares little with the side shots ({best} matches). Fine for a rough top.", detail
    if checked and best < MATCH_MIN:
        return "warn", (f"Saved, but little overlap with the {' / '.join(seen)} shot ({best} matches, {MATCH_MIN}+ is good). "
                        "Sides this far apart share little, so expect a rougher model. Tilt a bit toward the corner if you want it tighter."), detail
    if sid == "whole":
        return "ok", "Good. This will be the main view for the model.", detail
    return "ok", "Good." if checked else "Good. Next, go to another side.", detail


def frame_from_upload(f, jid, sid):
    """Image or short video -> one sharp BGR frame."""
    raw = f.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:                                  # not an image: try as a video
        tmp = JOBS_DIR / jid / "slots" / f"_{sid}.vid"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(raw)
        try:
            cap = cv2.VideoCapture(str(tmp))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cands = []
            for i in np.linspace(total * 0.2, total * 0.8, 8).astype(int) if total > 0 else []:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if ok:
                    cands.append(fr)
            cap.release()
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        if not cands:
            raise RuntimeError("could not read that photo or clip")
        img = sharpest(cands)
    s = SLOT_MAX / max(img.shape[:2])
    return cv2.resize(img, None, fx=min(1.0, s), fy=min(1.0, s), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- job pipeline
def run_job(job, source, path=None):
    try:
        set_job(job, step=1, msg="Collecting your angles..." if source == "frames" else "Pulling sharp frames from the video...")
        frames = demo_frames() if source == "demo" else (path if source == "frames" else extract_frames(path))
        fdir = JOBS_DIR / job["id"] / "frames"
        fdir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            write_jpg(fdir / f"{i}.jpg", f)
        set_job(job, nframes=len(frames))
        job["_frames"] = frames
        time.sleep(0.4)

        set_job(job, step=2, msg="Working out camera positions... (placeholder)")
        time.sleep(0.8)

        set_job(job, step=3, msg="Building 3D shape... (placeholder: one frame, inflated)")
        edits = default_edits()
        model = shape_from_frames(frames, job["note"], edits, job.get("primary"))
        with LOCK:
            job["versions"].append({"label": "Initial scan", "edits": edits, "model": model})
        set_job(job, step=4, msg="v1 ready. Spin it, then suggest changes or import.", ready=True)
        save_job(job)
    except Exception as e:
        traceback.print_exc()
        set_job(job, error=str(e), msg="Failed: " + str(e))
        save_job(job)


def new_job(note):
    job = {"id": uuid.uuid4().hex[:10], "step": 0, "msg": "Uploaded.", "note": note or "",
           "nframes": 0, "versions": [], "ready": False, "error": None,
           "created": time.time(), "exports": []}
    with LOCK:
        JOBS[job["id"]] = job
    return job


def save_job(job):
    """Write the job (without the in-memory frames) so History survives a restart."""
    d = JOBS_DIR / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    with LOCK:
        data = {k: v for k, v in job.items() if not k.startswith("_")}
    (d / "job.json").write_text(json.dumps(data))


def load_jobs():
    for d in JOBS_DIR.glob("*/job.json"):
        try:
            j = json.loads(d.read_text())
            j.setdefault("created", d.stat().st_mtime)
            j.setdefault("exports", [])
            capturing = j.get("mode") == "guided" and j.get("step", 0) == 0
            if not j.get("ready") and not j.get("error") and not capturing:
                j["error"] = "interrupted (server restarted)"
                j["msg"] = "Interrupted when the server restarted."
            JOBS[j["id"]] = j
        except Exception:
            traceback.print_exc()


def job_frames(j):
    if "_frames" not in j:
        fdir = JOBS_DIR / j["id"] / "frames"
        j["_frames"] = [read_img(p) for p in sorted(fdir.glob("*.jpg"), key=lambda p: int(p.stem))]
    if not j["_frames"]:
        raise RuntimeError("frames for this scan are missing")
    return j["_frames"]


def get_job(jid):
    j = JOBS.get(jid)
    if not j:
        abort(404)
    return j


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return send_file(BASE / "scanner.html")


def phone_url():
    return f"http://{lan_ip()}:{PORT}"


def qr_svg():
    buf = io.BytesIO()
    segno.make(phone_url(), error="m").save(buf, kind="svg", scale=8, border=2, xmldecl=False,
                                            dark="#000000", light="#ffffff")
    return buf.getvalue().decode()


@app.route("/qr.svg")
def qr_route():
    if segno is None:
        abort(404)
    return Response(qr_svg(), mimetype="image/svg+xml")


@app.route("/connect")
def connect_page():
    url = phone_url()
    if segno is None:
        body = ("<p>The QR code needs one small package. In a console run:</p>"
                "<pre>pip install segno</pre><p>then restart the server. Or just type this on your phone:</p>")
        qr = ""
    else:
        body = "<p>Scan with your phone camera.<br>Phone and laptop must be on the same WiFi.</p>"
        qr = '<img src="/qr.svg" style="width:min(80vw,380px);background:#fff;padding:8px;border-radius:10px">'
    return ('<!DOCTYPE html><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<body style="background:#0d1015;color:#dfe5ee;font:16px system-ui;text-align:center;padding:24px">'
            '<h2>Voidcraft Scanner</h2>' + body + qr +
            f'<p style="font:18px ui-monospace,monospace;color:#8fb4ff">{url}</p>'
            '<p style="color:#7d8896;font-size:13px">Server is running.</p></body>')


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/ping")
def ping():
    return jsonify(ok=True, ip=lan_ip())


@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify(error="no video"), 400
    job = new_job(request.form.get("note", ""))
    d = JOBS_DIR / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    ext = Path(f.filename or "v.mp4").suffix.lower()
    if not ext.lstrip(".").isalnum() or len(ext) > 6:
        ext = ".mp4"
    path = d / ("video" + ext)
    f.save(path)
    threading.Thread(target=run_job, args=(job, "video", path), daemon=True).start()
    return jsonify(job=job["id"])


@app.route("/api/demo", methods=["POST"])
def demo():
    job = new_job(request.form.get("note", "") or "a tree")
    threading.Thread(target=run_job, args=(job, "demo"), daemon=True).start()
    return jsonify(job=job["id"])


@app.route("/api/job/<jid>")
def job_status(jid):
    j = get_job(jid)
    with LOCK:
        vers = [{"label": v["label"], "W": v["model"]["W"], "H": v["model"]["H"],
                 "solid": v["model"]["solid"], "pieces": v["model"]["pieces"]} for v in j["versions"]]
        return jsonify(id=j["id"], note=j["note"], created=j["created"], step=j["step"], msg=j["msg"],
                       nframes=j["nframes"], ready=j["ready"], error=j["error"], versions=vers,
                       exports=j["exports"], mode=j.get("mode", "video"),
                       slots=[dict(id=sid, label=lab, **j["slots"].get(sid, EMPTY_SLOT)) for sid, lab in SLOTS] if j.get("slots") else [])


@app.route("/api/jobs")
def jobs_list():
    with LOCK:
        rows = []
        for j in JOBS.values():
            last = j["versions"][-1]["model"] if j["versions"] else None
            rows.append({"id": j["id"], "note": j["note"], "created": j["created"], "nframes": j["nframes"],
                         "versions": len(j["versions"]), "cubes": last["solid"] if last else 0,
                         "ready": j["ready"], "error": j["error"], "exports": len(j["exports"]),
                         "mode": j.get("mode", "video"), "step": j["step"],
                         "captured": sum(1 for sid in SLOT_IDS if j.get("slots", {}).get(sid, EMPTY_SLOT)["state"] != "empty")})
    rows.sort(key=lambda r: r["created"], reverse=True)
    return jsonify(jobs=rows)


@app.route("/api/job/<jid>", methods=["DELETE"])
def job_delete(jid):
    j = get_job(jid)
    with LOCK:
        JOBS.pop(jid, None)
    shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
    return jsonify(ok=True)


@app.route("/api/about")
def about():
    return jsonify(version=VERSION, ip=lan_ip(), export_dir=str(EXPORT_DIR), jobs=len(JOBS),
                   cubes_per_block=DETAIL, blocks=BLOCKS, frames=NFRAMES, shape="placeholder (inflated single frame)",
                   qr=segno is not None, url=phone_url())


@app.route("/api/guided/start", methods=["POST"])
def guided_start():
    job = new_job(request.form.get("note", ""))
    job["mode"] = "guided"
    job["msg"] = "Capture each angle round the object."
    job["slots"] = {sid: dict(EMPTY_SLOT) for sid in SLOT_IDS}
    save_job(job)
    return jsonify(job=job["id"])


@app.route("/api/job/<jid>/slot/<sid>", methods=["POST"])
def slot_upload(jid, sid):
    j = get_job(jid)
    if j.get("mode") != "guided" or j["step"] != 0:
        return jsonify(error="this scan is not taking photos any more"), 400
    if sid not in SLOT_IDS:
        return jsonify(error="unknown angle"), 400
    f = request.files.get("file")
    if not f:
        return jsonify(error="no photo or clip"), 400
    try:
        img = frame_from_upload(f, jid, sid)
    except Exception as e:
        return jsonify(error=str(e)), 400
    slot_path(jid, sid).parent.mkdir(parents=True, exist_ok=True)
    write_jpg(slot_path(jid, sid), img)
    j["slots"].setdefault(sid, dict(EMPTY_SLOT))["state"] = "pending"   # never its own neighbour
    state, advice, detail = check_slot(j, sid, img)
    with LOCK:
        j["slots"][sid] = {"state": state, "advice": advice, "blur": detail["blur"], "matches": detail["matches"]}
    save_job(j)
    return jsonify(state=state, advice=advice, slot=sid)


@app.route("/api/job/<jid>/build", methods=["POST"])
def guided_build(jid):
    j = get_job(jid)
    if j.get("mode") != "guided" or j["step"] != 0 or j["ready"]:
        return jsonify(error="nothing to build"), 400
    force = bool((request.get_json(silent=True) or {}).get("force"))
    st = lambda sid: j["slots"].get(sid, EMPTY_SLOT)["state"]
    got = [sid for sid in SLOT_IDS if st(sid) in ("ok", "warn", "redo")]
    ok = [sid for sid in SLOT_IDS if st(sid) in ("ok", "warn")]
    has_whole = "whole" in ok
    missing = [dict(SLOTS)[sid] for sid in RING if sid not in ok]
    if not force and not has_whole and missing:
        return jsonify(error="Still needed: " + ", ".join(missing) + " (or add a whole-object photo)"), 400
    use = got if force else ok
    if "whole" in use:                                   # the whole-object photo leads, and is the main view
        use = ["whole"] + [x for x in use if x != "whole"]
    need = 1 if "whole" in use else 3
    if len(use) < need:
        return jsonify(error="Need at least 3 sides, or one whole-object photo."), 400
    j["primary"] = 0 if use[0] == "whole" else None
    frames = [read_img(slot_path(jid, sid)) for sid in use]
    threading.Thread(target=run_job, args=(j, "frames", frames), daemon=True).start()
    return jsonify(ok=True, used=len(use))


@app.route("/api/job/<jid>/frame/<int:i>")
def job_frame(jid, i):
    p = JOBS_DIR / jid / "frames" / f"{i}.jpg"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="image/jpeg")


@app.route("/api/job/<jid>/model/<int:v>")
def job_model(jid, v):
    j = get_job(jid)
    if v < 0 or v >= len(j["versions"]):
        abort(404)
    return jsonify(j["versions"][v]["model"])


@app.route("/api/job/<jid>/edit", methods=["POST"])
def job_edit(jid):
    j = get_job(jid)
    body = request.get_json(force=True)
    text = (body.get("text") or "").strip()
    if not text or not j["versions"]:
        return jsonify(error="nothing to apply"), 400
    try:
        base = int(body.get("base", len(j["versions"]) - 1))
    except (TypeError, ValueError):
        base = -1
    if not 0 <= base < len(j["versions"]):
        return jsonify(error="unknown version"), 400
    edits, done = interpret_edit(text, j["versions"][base]["edits"])
    if not done:
        return jsonify(error="I could not map that to a change yet. Try: thicker, thinner, trim the top, "
                             "remove loose bits, flat base, taller, smaller."), 400
    model = shape_from_frames(job_frames(j), j["note"], edits, j.get("primary"))
    with LOCK:
        j["versions"].append({"label": ", ".join(done), "edits": edits, "model": model})
        n = len(j["versions"])
    save_job(j)
    return jsonify(version=n - 1, label=", ".join(done))


@app.route("/api/job/<jid>/import", methods=["POST"])
def job_import(jid):
    j = get_job(jid)
    try:
        v = int(request.get_json(force=True).get("version", len(j["versions"]) - 1))
    except (TypeError, ValueError):
        v = -1
    if not 0 <= v < len(j["versions"]):
        return jsonify(error="unknown version"), 400
    m = j["versions"][v]["model"]
    out = {"format": "voidcraft.cells.v1", "cubes_per_block": DETAIL, "size": [m["W"], m["H"]],
           "materials": MATS, "note": j["note"],
           "cells": [[c[0], c[1], c[2], c[3]] for c in m["cells"]]}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "".join(ch if ch.isalnum() else "_" for ch in (j["note"] or "scan"))[:24] or "scan"
    path = EXPORT_DIR / f"{stem}_{j['id']}_v{v + 1}.cells"
    path.write_text(json.dumps(out))
    with LOCK:
        j["exports"].append({"version": v + 1, "path": str(path), "cubes": m["solid"], "time": time.time()})
    save_job(j)
    return jsonify(path=str(path), cubes=m["solid"])


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--export", default=None, help="folder for imported .cells files")
    ap.add_argument("--no-browser", action="store_true", help="do not open the pairing page on this laptop")
    ap.add_argument("--terminal-qr", action="store_true", help="also print the QR code in the console")
    a = ap.parse_args()
    PORT = a.port
    if a.export:
        EXPORT_DIR = Path(a.export)
    JOBS_DIR.mkdir(exist_ok=True)
    load_jobs()
    print(f"\n  Voidcraft Scanner running.\n  Phone (same WiFi):  {phone_url()}\n"
          f"  Exports go to:      {EXPORT_DIR}")
    if segno is None:
        print("  QR pairing:         off  (pip install segno  to turn it on)\n")
    else:
        print(f"  QR pairing:         http://127.0.0.1:{PORT}/connect\n")
        if a.terminal_qr:
            segno.make(phone_url()).terminal(compact=True)
        if not a.no_browser:
            threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/connect")).start()
    app.run(host="0.0.0.0", port=PORT, threaded=True)
