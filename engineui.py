# engineui.py
# A launcher with the engine built in. Open this file, press "Open game...", pick a game script: it runs.
#
# The engine source lives at the bottom of this file between the ENGINE BEGIN / ENGINE END markers.
# To edit it (command line):
#   python engineui.py --export-engine engine.py     write the built-in engine to a file
#   python engineui.py --replace-engine engine.py    put an edited engine back (keeps a .bak of this file)
#
# A game is a .py file with a main() function. It gets the engine with `import voidcraft_engine as _E`
# (this file provides that module while the game runs). See game.py.
#
# Loading screen: a game can report its loading progress by printing lines to stdout:
#   @@LOAD|<0..1>|<stage name>|<detail>     a stage started (the previous one is finished)
#   @@LOAD|done                             loading is finished
# This window then shows the progress bar and the list of finished stages while the game loads. A game that
# prints nothing just shows as running. The launcher sets the environment variable VOIDENGINE_UI=1 for the game.
import sys, os, re, types, importlib.util, subprocess, threading, time
import linecache, shutil, multiprocessing, collections

sys.dont_write_bytecode = True     # no __pycache__ folders next to this file or the games

FROZEN = bool(getattr(sys, "frozen", False))
HUB_FILE = os.path.abspath(sys.executable if FROZEN else __file__)
HERE = os.path.dirname(HUB_FILE)
ENGINE_NAME = "voidcraft_engine"            # the module name games import from; mesh workers import it by this name too
ENGINE_FILENAME = "<voidcraft_engine>"      # what tracebacks call the built-in engine
BEGIN_MARK = "# =====" + " ENGINE BEGIN"
END_MARK = "# =====" + " ENGINE END"


# ── the built-in engine ────────────────────────────────────────────────────────────────────────────
def engine_source():
    return _ENGINE_SRC


def install_engine():
    """Make the built-in engine importable as `voidcraft_engine` (games and mesh workers import it by that name)."""
    mod = sys.modules.get(ENGINE_NAME)
    if mod is not None:
        return mod
    src = engine_source()
    mod = types.ModuleType(ENGINE_NAME)
    mod.__file__ = ENGINE_FILENAME
    linecache.cache[ENGINE_FILENAME] = (len(src), None, src.splitlines(True), ENGINE_FILENAME)   # tracebacks show real lines
    sys.modules[ENGINE_NAME] = mod
    try:
        exec(compile(src, ENGINE_FILENAME, "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(ENGINE_NAME, None)
        raise
    return mod


# ── running a game ─────────────────────────────────────────────────────────────────────────────────
def run_game(game_path, args=()):
    """Run a game in this process, on the engine built into this file. Used by the launcher's own subprocess."""
    multiprocessing.freeze_support()
    install_engine()
    game_path = os.path.abspath(game_path)
    sys.argv = [game_path, *args]
    folder = os.path.dirname(game_path)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    name = "game_" + re.sub(r"\W", "_", os.path.splitext(os.path.basename(game_path))[0])
    spec = importlib.util.spec_from_file_location(name, game_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "main"):
        print("this game has no main() function")
        return 2
    mod.main()
    return 0


_RUN_TAIL = collections.deque(maxlen=200)     # the running game's output, kept in memory (no log file)
_EVENTS = collections.deque()                 # loading progress reported by the game: ("step", frac, stage, detail) / ("done",)


def launch(game_path, extra=()):
    """Start the game in its own process: this same file, running the game on the engine inside it."""
    cmd = [sys.executable] + ([] if FROZEN else [HUB_FILE]) + ["--run-game", os.path.abspath(game_path), *extra]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VOIDENGINE_UI"] = "1"                 # tells the game this window shows its loading screen
    _RUN_TAIL.clear()
    _EVENTS.clear()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0     # no console window for the game
    p = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(game_path)), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
                         creationflags=flags)

    def pump():
        for line in p.stdout:
            line = line.rstrip("\n")
            if line.startswith("@@LOAD|"):
                parts = line.split("|", 3)
                try:
                    if parts[1] == "done":
                        _EVENTS.append(("done",))
                    else:
                        _EVENTS.append(("step", float(parts[1]), parts[2], parts[3] if len(parts) > 3 else ""))
                    continue
                except (ValueError, IndexError):
                    pass
            _RUN_TAIL.append(line)
    p.pump_thread = threading.Thread(target=pump, daemon=True)
    p.pump_thread.start()
    return p


def tail_log(n=40):
    return "\n".join(list(_RUN_TAIL)[-n:])


# ── editing the built-in engine (command line) ─────────────────────────────────────────────────────
def export_engine(dest):
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(engine_source())
    return dest


def replace_engine(src_path):
    """Put an edited engine file into this file (the old copy is kept as .bak). Takes effect on the next start."""
    with open(src_path, "r", encoding="utf-8") as f:
        new = f.read().replace("\r\n", "\n")
    compile(new, src_path, "exec")                           # refuse a broken engine
    if "'''" in new or new.endswith("\\"):
        raise ValueError("the engine contains ''' or ends in a backslash, which the built-in copy cannot hold")
    with open(HUB_FILE, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    a = text.rfind(BEGIN_MARK)
    b = text.rfind(END_MARK)
    if a < 0 or b < a:
        raise ValueError("engine markers not found in this file")
    head = text.index("\n", a) + 1
    opener = "_ENGINE_SRC = r'''"
    if not text.startswith(opener, head):
        raise ValueError("engine block is not in the expected form")
    body_start = head + len(opener)
    body_end = text.rindex("'''", head, b)
    shutil.copyfile(HUB_FILE, HUB_FILE + ".bak")
    with open(HUB_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(text[:body_start] + new + text[body_end:])


# ── the window ─────────────────────────────────────────────────────────────────────────────────────
def gui():
    import tkinter as tk
    from tkinter import filedialog

    BG, FG, DIM, BAD, ACCENT, TRACK = "#12141c", "#e2e6ee", "#8890a0", "#e04848", "#46bee6", "#28303c"
    READY = "Engine running..."
    TITLE = "VoidEngine Launcher"
    proc = {"p": None, "name": ""}
    CONFIG = os.path.join(HERE, "engineui_config.json")

    def load_config():
        try:
            import json
            with open(CONFIG, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_config(**kw):
        try:
            import json
            data = load_config()
            data.update(kw)
            with open(CONFIG, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass

    load = {"active": False, "stage": None, "t": 0.0, "t0": 0.0, "frac": 0.0, "done": [], "detail": ""}

    root = tk.Tk()
    root.title("Engine UI")
    root.configure(bg=BG)
    W, H = 380, 720                                   # a tall panel
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    H = min(H, sh - 120)
    root.geometry("%dx%d+%d+%d" % (W, H, 60, max(0, (sh - H) // 2 - 20)))
    root.minsize(300, 420)

    PAD = 20                                   # one margin and one gap, used everywhere, so the spacing is even
    wrap = tk.Frame(root, bg=BG)
    wrap.pack(fill="both", expand=True, padx=PAD, pady=PAD)
    status = tk.Label(wrap, text=TITLE, font=("Segoe UI", 16, "bold"), bg=BG, fg=FG, anchor="w", padx=0, pady=0)
    status.pack(fill="x", pady=(0, PAD))
    row = tk.Frame(wrap, bg=BG)
    row.pack(fill="x", pady=(0, 8))
    btn = tk.Button(row, text="Open game...", bg="#232838", fg=FG, activebackground="#2e3550", activeforeground=FG,
                    disabledforeground="#555b6b", relief="flat", bd=0, highlightthickness=0, padx=18, pady=10,
                    font=("Segoe UI", 12))
    btn.pack(side="left")
    eng_btn = tk.Button(row, text="View engine", bg="#232838", fg=FG, activebackground="#2e3550", activeforeground=FG,
                        relief="flat", bd=0, highlightthickness=0, padx=18, pady=10, font=("Segoe UI", 12))
    eng_btn.pack(side="left", padx=(8, 0))
    sub = tk.Label(wrap, text="", font=("Segoe UI", 10), bg=BG, fg=DIM, anchor="w", justify="left", wraplength=W - 2 * PAD,
                   padx=0, pady=0)
    sub.pack(fill="x", pady=(0, PAD), after=row)
    # quick launch: one button per saved game, plus "+ Add game..." to save another
    BTN = dict(bg="#232838", fg=FG, activebackground="#2e3550", activeforeground=FG, disabledforeground="#555b6b",
               relief="flat", bd=0, highlightthickness=0, font=("Segoe UI", 10))
    ql = tk.Frame(wrap, bg=BG)
    ql.pack(fill="x", pady=(0, PAD))
    head = tk.Frame(ql, bg=BG)
    head.pack(fill="x", pady=(0, 6))
    tk.Label(head, text="Quick launch", font=("Segoe UI", 11, "bold"), bg=BG, fg=FG, anchor="w").pack(side="left")
    ql_add = tk.Button(head, text="+ Add game...", padx=10, pady=3, **BTN)
    ql_add.pack(side="right")
    ql_list = tk.Frame(ql, bg=BG)
    ql_list.pack(fill="x")
    bar = tk.Canvas(wrap, height=10, bg=TRACK, bd=0, highlightthickness=0)      # the loading bar: shown only while loading
    content = tk.Frame(wrap, bg=BG)                 # the big lower area: shows the log, or the engine source
    content.pack(fill="both", expand=True)
    out = tk.Text(content, bg="#0c0e14", fg=FG, font=("Consolas", 10), relief="flat", bd=0, highlightthickness=0,
                  wrap="word", padx=12, pady=10, state="disabled")
    out.pack(fill="both", expand=True)
    engine_view = tk.Frame(content, bg=BG)          # filled in by view_engine() the first time it is opened

    def on_resize(event):
        if event.widget is root:                                   # ignore the Configure events of child widgets
            root.title("Engine UI  -  %d x %d" % (event.width, event.height))

    root.bind("<Configure>", on_resize)

    def show(text):
        out.config(state="normal")
        out.delete("1.0", "end")
        out.insert("end", text)
        out.config(state="disabled")

    def draw_bar():
        w = int(bar.winfo_width() or 1)
        bar.delete("all")
        bar.create_rectangle(0, 0, int(w * load["frac"]), 10, fill=ACCENT, width=0)

    def finished_list():
        lines = []
        for t, secs, detail in load["done"][-10:]:
            lines.append("done  %-46s %5.2fs" % (t[:46], secs))
            if detail.startswith(">"):                 # a summary the game sent for the stage (what was cached, what was built)
                lines.append("      " + detail[1:].strip())
        return "\n".join(lines)

    def reset_load():
        load.update(active=False, stage=None, frac=0.0, done=[], detail="")
        bar.pack_forget()

    def on_event(ev):
        now = time.perf_counter()
        if ev[0] == "step":
            _, frac, text, detail = ev
            if not load["active"]:                 # the first report: switch the window to its loading screen
                load.update(active=True, stage=None, t=now, t0=now, done=[], detail="")
                status.config(text="Loading " + proc["name"], fg=FG)
                bar.pack(fill="x", pady=(0, PAD), before=content)
            if load["stage"] is not None and text != load["stage"]:      # the previous stage is finished: note how long it took
                load["done"].append((load["stage"], now - load["t"], load["detail"]))
                load["t"] = now
            load["stage"], load["frac"], load["detail"] = text, max(0.0, min(1.0, frac)), detail
            sub.config(text="%d%%   %s" % (round(load["frac"] * 100), text))
            draw_bar()
            show(finished_list())
        elif load["active"]:                       # "done": the game window is open
            if load["stage"] is not None:
                load["done"].append((load["stage"], now - load["t"], load["detail"]))
            status.config(text="Running " + proc["name"], fg=FG)
            sub.config(text="loaded in %.1fs - close the game window to come back here" % (now - load["t0"]))
            show(finished_list())
            load.update(active=False, stage=None)
            bar.pack_forget()

    def poll():
        p = proc["p"]
        if p is None:
            return
        while _EVENTS:
            on_event(_EVENTS.popleft())
        code = p.poll()
        if code is None:
            root.after(100, poll)
            return
        pump = getattr(p, "pump_thread", None)
        if pump is not None:
            pump.join(1.0)                         # let the last of the game's output arrive
        while _EVENTS:
            on_event(_EVENTS.popleft())
        proc["p"] = None
        set_buttons("normal")
        if code == 0:                              # a normal close resets everything: no log is left over
            _RUN_TAIL.clear()
            reset_load()
            status.config(text=TITLE, fg=FG)
            sub.config(text="")
            show(READY)
        else:
            stage = load["stage"] if load["active"] else None
            if eng["open"]:
                view_engine()
            status.config(text="The game stopped with an error", fg=BAD)
            sub.config(text="exit code %d%s - the game's output is below" % (code, (" while loading: " + stage) if stage else ""))
            reset_load()
            show(tail_log())

    def games_saved():
        cfg = load_config()
        games = [g for g in cfg.get("games", []) if isinstance(g, str)]
        old = cfg.get("quick_launch")                      # older single-path config: carry it over
        if isinstance(old, str) and old and old not in games:
            games.append(old)
        return games

    def args_for(path):
        """The launch commands saved for a game (a list of words), passed to the game when it starts."""
        saved = load_config().get("launch_args")
        words = saved.get(path, []) if isinstance(saved, dict) else []
        return [w for w in words if isinstance(w, str) and w]

    def save_args(path, words):
        saved = load_config().get("launch_args")
        saved = dict(saved) if isinstance(saved, dict) else {}
        if words:
            saved[path] = list(words)
        else:
            saved.pop(path, None)
        save_config(launch_args=saved)

    def set_buttons(state):
        for b in [btn, ql_add] + [w for w in ql_list.winfo_children() if isinstance(w, tk.Button)]:
            b.config(state=state)
        for row_ in ql_list.winfo_children():
            for w in row_.winfo_children():
                if isinstance(w, tk.Button):
                    w.config(state=state)

    def start_game(path):
        if proc["p"] is not None:
            return
        path = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
        if not os.path.isfile(path):
            status.config(text="Could not start", fg=BAD)
            sub.config(text="file not found: " + path)
            return
        proc["name"] = os.path.basename(path)
        reset_load()
        try:
            proc["p"] = launch(path, args_for(path))
        except Exception as err:
            status.config(text="Could not start", fg=BAD)
            sub.config(text=str(err))
            return
        if eng["open"]:
            view_engine()                          # switch back to the log so loading progress is visible
        status.config(text="Running " + proc["name"], fg=FG)
        sub.config(text="close the game window to come back here")
        show("")
        set_buttons("disabled")
        root.after(100, poll)

    def ask_path():
        return filedialog.askopenfilename(title="Open a game script", filetypes=[("Python", "*.py"), ("All", "*.*")],
                                          initialdir=HERE)

    def open_game():
        if proc["p"] is not None:
            return
        path = ask_path()
        if path:
            start_game(path)

    def refresh_games():
        for w in ql_list.winfo_children():
            w.destroy()
        games = games_saved()
        if not games:
            tk.Label(ql_list, text="No saved games yet - use + Add game...", font=("Segoe UI", 9), bg=BG, fg=DIM,
                     anchor="w").pack(fill="x")
        for g in games:
            r = tk.Frame(ql_list, bg=BG)
            r.pack(fill="x", pady=(0, 6))
            name = os.path.splitext(os.path.basename(g))[0]
            words = args_for(g)
            b1 = tk.Button(r, text=name + ("   " + " ".join(words) if words else ""), anchor="w", padx=12, pady=6,
                           command=lambda g=g: start_game(g), **BTN)
            b1.pack(side="left", fill="x", expand=True)
            b1.bind("<Button-3>", lambda e, g=g: edit_args(g))                 # right-click: launch commands
            b2 = tk.Button(r, text="x", padx=9, pady=6, command=lambda g=g: remove_game(g), **BTN)
            b2.pack(side="left", padx=(6, 0))
        if games:
            tk.Label(ql_list, text="Right-click a game to set its launch commands", font=("Segoe UI", 8), bg=BG, fg=DIM,
                     anchor="w").pack(fill="x")
        if proc["p"] is not None:
            set_buttons("disabled")

    def edit_args(path):
        """A small window to type the launch commands for one saved game, e.g. cubedetail-max or cubedetail-4."""
        if proc["p"] is not None:
            return
        win = tk.Toplevel(root)
        win.title("Launch commands")
        win.configure(bg=BG)
        win.transient(root)
        win.resizable(False, False)
        win.geometry("+%d+%d" % (root.winfo_rootx() + 30, root.winfo_rooty() + 120))
        body = tk.Frame(win, bg=BG)
        body.pack(fill="both", expand=True, padx=PAD, pady=PAD)
        tk.Label(body, text="Launch commands", font=("Segoe UI", 13, "bold"), bg=BG, fg=FG, anchor="w").pack(fill="x")
        tk.Label(body, text=os.path.splitext(os.path.basename(path))[0], font=("Segoe UI", 10), bg=BG, fg=DIM,
                 anchor="w").pack(fill="x", pady=(0, 10))
        box = tk.Text(body, width=34, height=5, bg="#0c0e14", fg=FG, insertbackground=FG, font=("Consolas", 11), relief="flat",
                      bd=0, highlightthickness=1, highlightbackground=TRACK, highlightcolor=ACCENT, padx=8, pady=6)
        box.pack(fill="x")
        box.insert("1.0", "\n".join(args_for(path)))
        tk.Label(body, text="One per line, or separated by spaces. They are sent to the game every time it starts from "
                            "Quick launch.\n\nExamples:\n  cubedetail-max     the finest ground cubes\n  cubedetail-4       "
                            "4 cubes per block (any number)",
                 font=("Segoe UI", 9), bg=BG, fg=DIM, anchor="w", justify="left", wraplength=330).pack(fill="x", pady=(10, 14))
        bar_ = tk.Frame(body, bg=BG)
        bar_.pack(fill="x")

        def done(save=True, clear=False):
            if save:
                save_args(path, [] if clear else box.get("1.0", "end").split())
                refresh_games()
            win.destroy()

        tk.Button(bar_, text="Save", padx=16, pady=6, command=done, **BTN).pack(side="left")
        tk.Button(bar_, text="Clear", padx=12, pady=6, command=lambda: done(True, True), **BTN).pack(side="left", padx=(8, 0))
        tk.Button(bar_, text="Cancel", padx=12, pady=6, command=lambda: done(False), **BTN).pack(side="right")
        win.bind("<Escape>", lambda e: done(False))
        win.bind("<Control-Return>", lambda e: done())
        try:
            win.wait_visibility()
            win.grab_set()                                   # modal: the launcher waits for this window
        except Exception:
            pass
        box.focus_set()

    def add_game():
        path = ask_path()
        if not path:
            return
        path = os.path.abspath(path)
        games = games_saved()
        if path not in games:
            games.append(path)
        save_config(games=games, quick_launch=None)
        refresh_games()

    def remove_game(path):
        save_config(games=[g for g in games_saved() if g != path], quick_launch=None)
        save_args(path, [])
        refresh_games()

    eng = {"built": False, "open": False}

    def build_engine_view():
        top = tk.Frame(engine_view, bg=BG)
        top.pack(fill="x", pady=(0, 6))
        info = tk.Label(top, text="", font=("Segoe UI", 10), bg=BG, fg=DIM, anchor="w")
        info.pack(side="left")
        find_var = tk.StringVar()
        find_entry = tk.Entry(top, textvariable=find_var, width=18, bg="#0c0e14", fg=FG, insertbackground=FG, relief="flat",
                              bd=0, highlightthickness=1, highlightbackground=TRACK, highlightcolor=ACCENT,
                              font=("Consolas", 10))
        find_entry.pack(side="right", ipady=4)
        tk.Label(top, text="Find", font=("Segoe UI", 10), bg=BG, fg=DIM).pack(side="right", padx=(0, 6))
        body = tk.Frame(engine_view, bg=BG)
        body.pack(fill="both", expand=True)
        sb = tk.Scrollbar(body)
        sb.pack(side="right", fill="y")
        hs = tk.Scrollbar(body, orient="horizontal")
        hs.pack(side="bottom", fill="x")
        txt = tk.Text(body, bg="#0c0e14", fg=FG, font=("Consolas", 10), relief="flat", bd=0, highlightthickness=0,
                      wrap="none", padx=12, pady=10, yscrollcommand=sb.set, xscrollcommand=hs.set, tabs=("1c",))
        txt.pack(side="left", fill="both", expand=True)
        sb.config(command=txt.yview)
        hs.config(command=txt.xview)
        src = _ENGINE_SRC
        total = "%d lines" % (src.count("\n") + 1)
        txt.insert("1.0", src)
        txt.tag_configure("cur", background=ACCENT, foreground="#000000")
        txt.config(state="disabled")
        info.config(text=total)
        state = {"last": "1.0"}

        def find(again=False):
            q = find_var.get()
            txt.tag_remove("cur", "1.0", "end")
            if not q:
                info.config(text=total)
                return
            start = state["last"] if again else "1.0"
            pos = txt.search(q, start, nocase=True, stopindex="end") or txt.search(q, "1.0", nocase=True, stopindex="end")
            if not pos:
                info.config(text="no match")
                return
            end = "%s+%dc" % (pos, len(q))
            txt.tag_add("cur", pos, end)
            txt.see(pos)
            state["last"] = end
            info.config(text="line " + pos.split(".")[0])

        find_entry.bind("<Return>", lambda e: find(True))
        find_entry.bind("<KeyRelease>", lambda e: (state.update(last="1.0"), find()) if e.keysym != "Return" else None)
        root.bind("<Control-f>", lambda e: find_entry.focus_set() if eng["open"] else None)

    def view_engine():
        if not eng["built"]:
            build_engine_view()
            eng["built"] = True
        if eng["open"]:                                           # back to the log
            engine_view.pack_forget()
            out.pack(fill="both", expand=True)
            eng_btn.config(text="View engine")
        else:
            out.pack_forget()
            engine_view.pack(fill="both", expand=True)
            eng_btn.config(text="Back")
        eng["open"] = not eng["open"]

    eng_btn.config(command=view_engine)
    ql_add.config(command=add_game)
    refresh_games()

    btn.config(command=open_game)
    show(READY)
    root.mainloop()


def main():
    multiprocessing.freeze_support()
    argv = sys.argv[1:]
    if "--run-game" in argv:                       # this file's own subprocess: run a game on the built-in engine
        i = argv.index("--run-game")
        sys.exit(run_game(argv[i + 1], argv[i + 2:]))
    if "--export-engine" in argv:
        print("wrote", export_engine(argv[argv.index("--export-engine") + 1]))
        return
    if "--replace-engine" in argv:
        replace_engine(argv[argv.index("--replace-engine") + 1])
        print("engine replaced (previous copy kept as .bak)")
        return
    if os.name == "nt" and not FROZEN and os.path.basename(sys.executable).lower() == "python.exe":
        pyw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")      # same Python, but without a console window
        if os.path.exists(pyw):
            subprocess.Popen([pyw, HUB_FILE], cwd=HERE, close_fds=True,
                             creationflags=getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return
    gui()


# ===== ENGINE BEGIN (plain engine source; Export engine... / Replace engine... edit this block) =====
_ENGINE_SRC = r'''# voidcraft_engine.py
# Engine + renderer: world, chunk meshing, physics, structures, sound, and the OpenGL graphics layer.
# The UI (menus, HUD, main loop) is in voidcraft_ui.py, which imports this module.
import math, os, sys, time, ctypes, functools, subprocess, threading, heapq, collections, bisect
import logging, multiprocessing
import numpy as np
import pygame
from pygame.locals import *
from concurrent.futures import ProcessPoolExecutor
from OpenGL import GL as gl

try:
    from scipy import ndimage as _ndi
    from scipy.sparse import coo_matrix as _coo_matrix
    from scipy.sparse.csgraph import connected_components as _connected_components
    HAS_SCIPY = True            # labels connected cells in compiled code: about 50x faster than the numpy fallback
except Exception:
    HAS_SCIPY = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ═══════════════════════════════════════════════
#  GPU COMPUTE (optional, CuPy). Started lazily from run(), never at import, so the mesh
#  worker processes do not each open a CUDA context. Everything falls back to numpy.
# ═══════════════════════════════════════════════
cp = None
GPU_OK = False
GPU_NAME = "not started"
GPU_MIN_CELLS = 20000      # smaller grids are faster on the CPU than the copy to the GPU and back

def _gpu_init():
    """Try to start CuPy on the first CUDA device. Returns a one-line status for the console."""
    global cp, GPU_OK, GPU_NAME
    try:
        import cupy as _cp
        if _cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device")
        probe = _cp.arange(8, dtype=_cp.int64)
        if int(probe.sum()) != 28:                              # one real kernel, so a broken install shows up now
            raise RuntimeError("CUDA self-test failed")
        cp = _cp
        GPU_NAME = _cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        GPU_OK = True
        return f"GPU physics: CuPy on {GPU_NAME}"
    except Exception as err:
        cp, GPU_OK = None, False
        GPU_NAME = f"unavailable ({type(err).__name__}: {err})"
        return f"GPU physics: off, using CPU. {GPU_NAME}"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ═══════════════════════════════════════════════
#  >>> GLOBAL SHIMS & CONFIGURATION <<<
# ═══════════════════════════════════════════════
ENGINE_VERSION   = 1      # API level games are checked against (see gamelink.py); raise it when a game-facing name changes
TARGET_FPS        = 60
RENDER_DIST       = 15
MESH_WORKERS      = 2     # background processes that build chunk meshes (separate from the GIL)
MESH_MAX_PENDING  = 6     # chunks queued for meshing at once; the rest wait for a later frame
VERT_STRIDE       = 9 * 4  # bytes per vertex: position (3) + colour (3) + texture coords and flag (3)
ARENA_VERTS       = 1_000_000   # starting size of the shared chunk-mesh buffer (grows as needed)
FOV               = 110
MOVE_SPEED        = 3.0    # blocks per second walking (was 5)
SPRINT_MULT       = 2.5
FLY_SPEED         = 5.0    # blocks per second flying (was 10)


# Missing Globals required by the original script
GFX = {'ambient_floor': 0.35, 'physics_device': 'cpu'}

# ═══════════════════════════════════════════════
#  BLOCK IDs & DATA
# ═══════════════════════════════════════════════
(AIR, GRASS, DIRT, STONE, SAND, WOOD, LEAVES,
 WATER, SNOW, GRAVEL, PLANKS, GLASS, TORCH, BRICK, SLAB) = range(15)

(MARBLE, GRANITE, SLATE, LIMESTONE, BASALT,
 OBSIDIAN, QUARTZ, SANDSTONE, MOSSY_STONE, DIAMOND_BLOCK, GOLD_BLOCK) = range(15, 26)

(CLAY, CONCRETE_WHITE, CONCRETE_BLACK, CONCRETE_RED, CONCRETE_BLUE,
 WOOD_OAK, WOOD_BIRCH, WOOD_SPRUCE, WOOD_DARK, WOOD_JUNGLE, WOOD_ACACIA) = range(26, 37)

(FLOWER_RED, FLOWER_YELLOW, FLOWER_BLUE, FLOWER_PURPLE, MUSHROOM_RED,
 MUSHROOM_BROWN, TALL_GRASS, FERN, VINE, CRYSTAL_AMETHYST, CRYSTAL_EMERALD) = range(37, 48)

(COPPER_ORE, COPPER_BLOCK, IRON_ORE, IRON_BLOCK, COAL_ORE, COAL_BLOCK,
 ALUMINUM_ORE, ALUMINUM_BLOCK, SILVER_ORE, SILVER_BLOCK, GOLD_ORE,
 DIAMOND_ORE, EMERALD_ORE, GLASS_DARK, GLASS_LIGHT, GLASS_TINTED, GLASS_COLORED) = range(48, 65)

DIAMOND, GOLD, COPPER, IRON, COAL, EMERALD = DIAMOND_ORE, GOLD_ORE, COPPER_ORE, IRON_ORE, COAL_ORE, EMERALD_ORE
STEEL = 65
APPLE = 66                      # a fruit cube hanging in a tree canopy


COLORS = {
    GRASS:  (0.34, 0.58, 0.20), DIRT:   (0.65, 0.40, 0.15), STONE:  (0.60, 0.60, 0.60),
    SAND:   (0.95, 0.90, 0.50), WOOD:   (0.55, 0.35, 0.12), LEAVES: (0.24, 0.55, 0.17),
    WATER:  (0.20, 0.50, 0.95), SNOW:   (0.98, 1.00, 1.00), GRAVEL: (0.70, 0.68, 0.65),
    PLANKS: (0.80, 0.65, 0.30), GLASS:  (0.75, 0.90, 0.98), TORCH:  (1.00, 0.85, 0.25),
    BRICK:  (0.85, 0.40, 0.30), SLAB:   (0.65, 0.65, 0.65), STEEL:  (0.44, 0.52, 0.64)
}

WIDTH, HEIGHT  = 1920, 1080
NEAR, FAR      = 0.05, 300.0
GRAVITY        = -28.0
JUMP_VEL       = 9.0
CHUNK_SIZE     = 15
CELL = 0.2; GS = 5; NC = 75; NH = 57     # the ground cube grid: set by set_ground_res() below
PLAYER_HEIGHT  = 1.8
PLAYER_RADIUS  = 0.3
STEP_HEIGHT    = 0.6       # the player climbs slopes and small steps up to this height

# ═══════════════════════════════════════════════
#  FLAT WORLD GENERATION
# ═══════════════════════════════════════════════
FLAT_SURFACE_Y  = 10   # y of the grass layer (keep it below the water line)

# ═══════════════════════════════════════════════
#  GROUND CUBES
# ═══════════════════════════════════════════════
# The ground is a grid of cubes CELL (half a block) wide. Each chunk keeps one array of them, +1 where a cube is solid and -1
# where it is air (world.dens). Digging takes cubes out; ground cut off from the rest falls (World.settle).

FLAT_GROUND_H = FLAT_SURFACE_Y + 1.0    # height at the start point; the rest of the world is hills, mountains and rivers
TERRAIN_GEN_MAX = 48                    # the generated land never rises above this many blocks
TERRAIN_MAX_H = TERRAIN_GEN_MAX              # tallest column the ground grid has room for, in blocks
WATER_Y = 6.0                           # rivers: surfaces at or below this height are drawn as water
DIRT_ALT = 19.0                         # above this altitude the hills show bare dirt cliffs
STONE_ALT = 27.0                        # above this altitude it is stone

MAX_GROUND_RES = 50                                         # most cubes per block the ground accepts (memory grows with its cube: see dense())


def set_ground_res(res):
    """Choose how big the ground cubes are: `res` cubes per block along each side (the same number the structures use, so
    ground cubes and tower cubes are the same size). The world has to be regenerated after this changes."""
    global CELL, GS, NC, NH
    GS = int(max(1, min(MAX_GROUND_RES, res)))
    CELL = 1.0 / GS
    NC = CHUNK_SIZE * GS                                    # ground cubes along a chunk's x and z
    NH = int(math.ceil((TERRAIN_MAX_H * GS + 2) / 15.0)) * 15   # along y: multiple of 15 so every LOD step divides it

set_ground_res(5)

def _hash01(ix, iz, seed):
    n = (ix * 374761393 + iz * 668265263 + seed * 982451653) & 0x7fffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0x7fffffff
    n = n ^ (n >> 16)
    return (n & 0x7fffffff) / 2147483647.0

def _vnoise(x, z, seed):
    fx0, fz0 = np.floor(x), np.floor(z)
    ix, iz = fx0.astype(np.int64), fz0.astype(np.int64)
    fx, fz = x - fx0, z - fz0
    ux, uz = fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz)
    a, b = _hash01(ix, iz, seed), _hash01(ix + 1, iz, seed)
    c, d = _hash01(ix, iz + 1, seed), _hash01(ix + 1, iz + 1, seed)
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uz

def _fbm(x, z, seed, octaves=3):
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * _vnoise(x + 11.7 * o, z - 5.3 * o, seed + o)
        norm += amp
        x, z, amp = x * 2.03, z * 2.03, amp * 0.5
    return tot / norm

def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def terrain_height(gx, gz):
    """Ground height (whole blocks) at world column (gx, gz): terraced hills, jagged stone mountains and rivers.
    Constant over each 1x1 block. Works on numpy arrays or scalars."""
    x = np.floor(np.asarray(gx, dtype=np.float64)) + 0.5
    z = np.floor(np.asarray(gz, dtype=np.float64)) + 0.5
    r = np.hypot(x, z)
    h = 13.0 + 12.0 * (_fbm(x / 60.0, z / 60.0, 1, 3) - 0.5)
    t = h / 3.0
    h = 3.0 * (np.floor(t) + _smooth(0.3, 0.7, t - np.floor(t)))      # terraces, one-block risers
    region = _smooth(0.48, 0.64, _fbm(x / 160.0, z / 160.0, 2, 2)) * _smooth(45.0, 95.0, r)
    ridge = 1.0 - np.abs(2.0 * _fbm(x / 34.0, z / 34.0, 3, 3) - 1.0)
    r2 = np.clip((ridge - 0.72) / 0.28, 0.0, 1.0)
    h = h + region * (8.0 + 30.0 * r2 ** 1.5 + 4.0 * _fbm(x / 9.0, z / 9.0, 4, 2))
    h = 11.0 + (h - 11.0) * _smooth(26.0, 56.0, r)                      # flat ground round the start point
    xa = 82.0 + 24.0 * np.sin(z / 33.0) + 9.0 * np.sin(z / 14.0 + 1.0)
    da = np.abs(x - xa)
    bed = WATER_Y - 1.0
    h = np.minimum(h, np.where(da < 3.5, bed, bed + 1.0 + (da - 3.5) * 0.7))
    zc = -88.0 + 14.0 * np.sin(x / 37.0) + 6.0 * np.sin(x / 15.0)
    h = np.where(np.abs(z - zc) < 3.0, WATER_Y - 2.0, h)                # narrow gorge, steep walls
    return np.clip(np.floor(h + 1e-6), 2, TERRAIN_GEN_MAX - 1)

class ChunkData:
    """One chunk's ground cubes. A chunk nobody has dug is stored as just a height per cube column (`top`: how many cubes from
    the bottom are solid); the full (NC, NH, NC) array of +1 / -1 is built the first time something is dug out of it."""
    __slots__ = ('top', 'arr')

    def __init__(self, top):
        self.top = top
        self.arr = None

    def dense(self):
        if self.arr is None:
            self.arr = np.where(np.arange(NH)[None, :, None] < self.top[:, None, :], np.int8(1), np.int8(-1))   # int8 straight away: no int64 copy first
        return self.arr

    def sample(self, lx, ly, lz):
        """+1 / -1 at local cube indices (arrays)."""
        if self.arr is not None:
            return self.arr[lx, ly, lz]
        return np.where(ly < self.top[lx, lz], 1, -1)

    def box(self, x0, x1, y0, y1, z0, z1):
        """+1 / -1 for the local cube range (ends inclusive). A view when the chunk has been dug, a fresh array otherwise."""
        if self.arr is not None:
            return self.arr[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
        ys = np.arange(y0, y1 + 1)[None, :, None]
        return np.where(ys < self.top[x0:x1 + 1, None, z0:z1 + 1], np.int8(1), np.int8(-1))

    def tops(self):
        """(NC, NC): the number of cubes up to and including the highest solid one in each column (0 if none)."""
        if self.arr is None:
            return self.top
        solid = self.arr > 0
        return np.where(solid.any(axis=1), NH - solid[:, ::-1, :].argmax(axis=1), 0)

# ---- disk cache: terrain generation and chunk meshes ------------------------------------------------------------------
# Two caches in a voidcraft_cache folder next to the running game (or VOIDCRAFT_CACHE_DIR), each switched on and off by a game
# through DISK_CACHE.terrain_on / DISK_CACHE.mesh_on:
#   terrain  the height of every ground column of a chunk (what generate_chunk makes), one small file per chunk
#   meshes   the finished mesh of a chunk, keyed by a hash of the cubes it was built from (and its neighbours' edge cubes), so a
#            chunk that has been dug, or whose neighbour has, misses and is built again
import hashlib, shutil, inspect, concurrent.futures


def _cache_root():
    env = os.environ.get('VOIDCRAFT_CACHE_DIR')
    if env:
        return env
    arg0 = sys.argv[0] if sys.argv and sys.argv[0] else ''
    base = os.path.dirname(os.path.abspath(arg0)) if arg0.lower().endswith('.py') else os.getcwd()
    return os.path.join(base, 'voidcraft_cache')


class DiskCache:
    def __init__(self):
        self.terrain_on = True
        self.mesh_on = True
        self.t_hit = self.t_new = self.m_hit = self.m_new = 0       # this run
        self.n = dict(terrain=0, mesh=0)                           # files on disk for the current cube detail
        self.size = dict(terrain=0, mesh=0)
        self._tsig = {}
        self._msig = None

    def root(self):
        return _cache_root()

    def _terrain_sig(self):
        gs = int(GS)
        if gs not in self._tsig:
            try:
                xs = np.linspace(-220.0, 220.0, 17)
                bx, bz = np.meshgrid(xs, xs, indexing='ij')
                h = np.asarray(terrain_height(bx, bz), np.float64)
                raw = h.tobytes() + repr((gs, CHUNK_SIZE, NH)).encode()
            except Exception:
                raw = repr(gs).encode()
            self._tsig[gs] = hashlib.blake2b(raw, digest_size=5).hexdigest()
        return self._tsig[gs]

    def _mesh_sig(self):
        if self._msig is None:
            try:
                self._msig = hashlib.blake2b(inspect.getsource(_mesh_cubes).encode(), digest_size=5).hexdigest()
            except Exception:
                self._msig = 'x'
        return self._msig

    def terrain_dir(self):
        return os.path.join(self.root(), 'terrain_gs%d_%s' % (int(GS), self._terrain_sig()))

    def mesh_dir(self):
        return os.path.join(self.root(), 'meshes_gs%d' % int(GS))

    @staticmethod
    def _write(path, saver):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + '.tmp'
            with open(tmp, 'wb') as f:
                saver(f)
            os.replace(tmp, path)
            return os.path.getsize(path)
        except Exception:
            return 0

    def load_terrain(self, cx, cz):
        path = os.path.join(self.terrain_dir(), '%d_%d.npy' % (cx, cz))
        try:
            top = np.load(path, allow_pickle=False)
            if top.shape == (NC, NC):
                self.t_hit += 1
                return top.astype(np.int32)
        except Exception:
            pass
        return None

    def save_terrain(self, cx, cz, top):
        path = os.path.join(self.terrain_dir(), '%d_%d.npy' % (cx, cz))
        nb = self._write(path, lambda f: np.save(f, top))
        if nb:
            self.t_new += 1
            self.n['terrain'] += 1
            self.size['terrain'] += nb

    def mesh_key(self, snap, cx, cz, step):
        h = hashlib.blake2b(digest_size=16)
        h.update(repr((snap.shape, str(snap.dtype), cx, cz, step, int(GS), self._mesh_sig())).encode())
        h.update(np.ascontiguousarray(snap))
        return os.path.join(self.mesh_dir(), h.hexdigest() + '.npz')

    def submit_mesh(self, worker, snap, cx, cz, step):
        if not self.mesh_on:
            return worker.pool.submit(_mesh_cubes, snap, cx, cz, step, GS)
        path = self.mesh_key(snap, cx, cz, step)
        try:
            with np.load(path, allow_pickle=False) as z:
                payload = (z['pos'], z['col'], z['uvf'])
            fut = concurrent.futures.Future()
            fut.set_result(payload)
            self.m_hit += 1
            return fut
        except Exception:
            pass
        fut = worker.pool.submit(_mesh_cubes, snap, cx, cz, step, GS)

        def done(f, path=path):
            try:
                pos, col, uvf = f.result()
            except Exception:
                return
            nb = self._write(path, lambda fh: np.savez(fh, pos=pos, col=col, uvf=uvf))
            if nb:
                self.m_new += 1
                self.n['mesh'] += 1
                self.size['mesh'] += nb
        fut.add_done_callback(done)
        return fut

    def refresh(self):
        """Count what is on disk for the current cube detail (call when a menu opens and after a delete)."""
        for kind, d in (('terrain', self.terrain_dir()), ('mesh', self.mesh_dir())):
            n = s = 0
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.name.endswith(('.npy', '.npz')):
                            n += 1
                            s += e.stat().st_size
            except OSError:
                pass
            self.n[kind], self.size[kind] = n, s

    @staticmethod
    def _mb(b):
        return '%.1f MB' % (b / 1048576.0) if b >= 1048576 else '%d KB' % max(1, b // 1024) if b else '0 KB'

    def tick(self, kind):
        return self.n[kind] > 0

    def text(self, kind):
        hit, new = (self.t_hit, self.t_new) if kind == 'terrain' else (self.m_hit, self.m_new)
        if not self.n[kind]:
            return 'Nothing cached yet for this cube detail.'
        return '%d files, %s on disk. This run: %d reused, %d new.' % (self.n[kind], self._mb(self.size[kind]), hit, new)

    def clear(self):
        """Delete every cached file (only ever a folder named voidcraft_cache)."""
        try:
            d = self.root()
            if os.path.basename(d) == 'voidcraft_cache' and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
        self.t_hit = self.t_new = self.m_hit = self.m_new = 0
        self.refresh()


DISK_CACHE = DiskCache()


def _generate_chunk_raw(cx, cz):
    """Build one chunk of ground as a grid of cubes. Returns (a placeholder for block types, ChunkData). Heights come from
    terrain_height, one value per block, so the land is made of whole-block steps."""
    bx = cx * CHUNK_SIZE + np.arange(CHUNK_SIZE) + 0.5
    bz = cz * CHUNK_SIZE + np.arange(CHUNK_SIZE) + 0.5
    bxx, bzz = np.meshgrid(bx, bz, indexing='ij')
    hb = terrain_height(bxx, bzz)                             # (15, 15) blocks
    h = np.repeat(np.repeat(hb, GS, 0), GS, 1)                # (NC, NC)
    top = np.clip(np.rint(h * GS), 0, NH).astype(np.int32)
    return np.zeros((1, 1, 1), np.uint8), ChunkData(top)


def generate_chunk(cx, cz):
    """One chunk of ground, from the disk cache when it has it (see DiskCache), otherwise generated (and cached)."""
    if DISK_CACHE.terrain_on:
        top = DISK_CACHE.load_terrain(cx, cz)
        if top is not None:
            return np.zeros((1, 1, 1), np.uint8), ChunkData(top)
    res = _generate_chunk_raw(cx, cz)
    if DISK_CACHE.terrain_on:
        DISK_CACHE.save_terrain(cx, cz, res[1].top)
    return res

# ═══════════════════════════════════════════════
#  CORE CLASSES & GENERATION
# ═══════════════════════════════════════════════
class _DirtySet(set):
    def __init__(self, *a):
        set.__init__(self, *a); self.ver = {}
    def add(self, k):
        self.ver[k] = self.ver.get(k, 0) + 1; set.add(self, k)
    def update(self, it):
        for k in it: self.add(k)
    def clear(self):
        set.clear(self); self.ver.clear()

class ChunkCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.mesh = collections.OrderedDict()
        self.mesh_bytes = 0
        self.hit_m = self.miss_m = 0

    def put_mesh(self, k, sig, ver, payload):
        n = sum(int(a.nbytes) for a in payload if a is not None)
        with self.lock:
            old = self.mesh.pop(k, None)
            if old: self.mesh_bytes -= old[3]
            self.mesh[k] = (sig, ver, payload, n); self.mesh_bytes += n

    def get_mesh(self, k, sig, ver):
        with self.lock:
            v = self.mesh.get(k)
            if v is None or v[0] != sig or v[1] != ver:
                self.miss_m += 1
                if v is not None:
                    self.mesh.pop(k); self.mesh_bytes -= v[3]
                return None
            self.mesh.move_to_end(k); self.hit_m += 1
            return v[2]

CACHE = ChunkCache()

DMG_AMOUNT = 1.6      # ground damage at the centre of a blast; it falls in a straight line to 0 at twice the crater radius
DMG_SOFT = 0.35       # ground damaged past this no longer holds anything up (counts as a broken bond)
NECK_THIN = 0.4       # blocks: ground thinner than this is not counted as holding anything up
NECK_REACH = 4.0      # blocks of ground next to a hanging group that are checked for the weakest cross-section
NECK_LIMIT = 20.0     # load a damaged neck carries: weight (blocks^3) x lever arm (blocks) / neck area (blocks^2) ^ 1.5. Lower = ledges fall sooner


class World:
    def __init__(self):
        self.chunks = {}
        self.dirty = _DirtySet()
        self._building = set()
        self.modified_chunks = set()
        self.touched = set()          # chunk keys whose draw-list entry may need a fresh look
        self.saved = {}         # edited chunks kept after being unloaded
        self.dens = {}          # density per chunk: above zero is solid
        self.saved_dens = {}
        self.dug_cells = []     # (n,3) arrays of cells dug out since the last take_dug_cells()
        self.gdmg = {}          # ground damage: chunk key -> [lowest y, float16 array (NC, rows, NC)], 0 = sound, 1 = broken
        self.edit_ver = 0       # bumps whenever the ground or its damage changes (the damage overlay redraws from it)

    def cell(self, ix, iy, iz):
        """+1 if the ground cube (ix, iy, iz) is solid, -1 if it is air, None if its chunk is not loaded. Below the world is
        solid and above it is air."""
        if iy < 0:
            return 1
        if iy >= NH:
            return -1
        cx = ix // NC
        cz = iz // NC
        d = self.dens.get((cx, cz))
        if d is None:
            return None
        lx, lz = ix - cx * NC, iz - cz * NC
        if d.arr is None:
            return 1 if iy < d.top[lx, lz] else -1
        return int(d.arr[lx, iy, lz])

    def density(self, x, y, z):
        """Solid (+1) or air (-1) at a world position (the cube holding it), or None if its chunk is not loaded."""
        return self.cell(math.floor(x / CELL), math.floor(y / CELL), math.floor(z / CELL))

    def dig(self, center, radius, strength=0.0):
        """Remove every ground cube the sphere (centre, radius) touches. The centres of the cubes taken are queued in
        dug_cells (see take_dug_cells), and the chunks that changed are remeshed along with the neighbours that read
        across their edge. Returns how many cubes were removed. `strength` is unused: cubes are either there or not."""
        cx, cy, cz = (float(v) for v in center)
        r = float(radius)
        lo = (math.floor((cx - r) / CELL), max(0, math.floor((cy - r) / CELL)), math.floor((cz - r) / CELL))
        hi = (math.floor((cx + r) / CELL), min(NH - 1, math.floor((cy + r) / CELL)), math.floor((cz + r) / CELL))
        if lo[1] > hi[1]:
            return 0
        removed = 0
        carved = set()
        taken = []
        for kx in range(lo[0] // NC, hi[0] // NC + 1):
            for kz in range(lo[2] // NC, hi[2] // NC + 1):
                d = self.dens.get((kx, kz))
                if d is None:
                    continue
                x0, x1 = max(lo[0], kx * NC), min(hi[0], kx * NC + NC - 1)
                z0, z1 = max(lo[2], kz * NC), min(hi[2], kz * NC + NC - 1)
                if x0 > x1 or z0 > z1:
                    continue
                ix = np.arange(x0, x1 + 1)[:, None, None]
                iy = np.arange(lo[1], hi[1] + 1)[None, :, None]
                iz = np.arange(z0, z1 + 1)[None, None, :]
                # distance from the centre to the cube's box, per axis (0 when the centre is inside the slab)
                ex = np.maximum(np.maximum(ix * CELL - cx, cx - (ix + 1) * CELL), 0.0)
                ey = np.maximum(np.maximum(iy * CELL - cy, cy - (iy + 1) * CELL), 0.0)
                ez = np.maximum(np.maximum(iz * CELL - cz, cz - (iz + 1) * CELL), 0.0)
                near = (ex * ex + ey * ey + ez * ez) <= r * r
                sl = (slice(x0 - kx * NC, x1 - kx * NC + 1), slice(lo[1], hi[1] + 1), slice(z0 - kz * NC, z1 - kz * NC + 1))
                hit = near & (d.box(sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1, sl[2].start, sl[2].stop - 1) > 0)
                if not hit.any():
                    continue
                view = d.dense()[sl]
                ii, jj, kk = np.nonzero(hit)
                taken.append(np.stack([(ii + x0 + 0.5) * CELL, (jj + lo[1] + 0.5) * CELL, (kk + z0 + 0.5) * CELL], axis=1))
                view[hit] = -1
                removed += int(hit.sum())
                carved.add((kx, kz))
        if not carved:
            return 0
        self.dug_cells.append(np.concatenate(taken))
        self._changed_chunks(carved, lo, hi)
        return removed

    # The proportions of a shovel's bite: half-width across, length along the swing, and depth (blocks, before scaling to the amount).
    SCOOP_ACROSS, SCOOP_ALONG, SCOOP_DEEP = 0.35, 0.45, 0.30

    def dig_n(self, center, n, facing=None, window=1.6):
        """Dig a rounded, shovel-shaped bowl of exactly n cubes. The bowl is the half of an oval below `center` (put it on the ground
        surface): the n solid cubes nearest it. Every dig is a different bowl: the oval is turned a random amount round the way you
        face, its width, length and depth change a little, and the edge is made ragged, but the amount is always exactly n.
        Queued in dug_cells like dig(). Returns how many cubes were removed (fewer if there are not that many)."""
        cx, cy, cz = (float(v) for v in center)
        base = math.atan2(float(facing[0]), -float(facing[2])) if facing is not None else float(_rng.uniform(0.0, 2.0 * math.pi))
        th = base + float(_rng.uniform(-1.0, 1.0)) * 1.1                         # turned up to about 60 degrees either way
        a = self.SCOOP_ACROSS * float(_rng.uniform(0.8, 1.15))                  # half-width across the swing
        b = self.SCOOP_ALONG * float(_rng.uniform(0.8, 1.2))                    # half-length along it
        c = self.SCOOP_DEEP * float(_rng.uniform(0.75, 1.3))                    # depth
        ct, st = math.cos(th), math.sin(th)                                     # (st, -ct) is the swing direction on the ground
        lo = np.array([math.floor((cx - window) / CELL), max(0, math.floor((cy - window) / CELL)), math.floor((cz - window) / CELL)])
        hi = np.array([math.floor((cx + window) / CELL), min(NH - 1, math.floor((cy + window) / CELL)), math.floor((cz + window) / CELL)])
        if lo[1] > hi[1]:
            return 0
        idx = np.argwhere(self.cells_box(lo, hi)) + lo
        if len(idx) == 0:
            return 0
        q = (idx + 0.5) * CELL
        dx, dy, dz = q[:, 0] - cx, q[:, 1] - cy, q[:, 2] - cz
        along = dx * st - dz * ct
        across = dx * ct + dz * st
        d = (along / b) ** 2 + (across / a) ** 2 + (dy / c) ** 2
        d = d * (1.0 + 0.22 * (_rng.random(len(d)) - 0.5))                      # a ragged edge: no two bites are cut the same
        pick = idx[np.argsort(d, kind='stable')[:int(n)]]
        self.clear_cells(pick)
        self.dug_cells.append((pick + 0.5) * CELL)
        return len(pick)

    def _changed_chunks(self, carved, lo, hi):
        """Record that cubes changed in these chunks, and remesh the neighbours that read across a shared edge (the
        coarse level-of-detail meshes read up to 5 cubes across it)."""
        self.modified_chunks |= carved
        self.edit_ver += 1
        m = 5
        for kx, kz in carved:
            near_x = (lo[0] < kx * NC + m, hi[0] > kx * NC + NC - 1 - m)
            near_z = (lo[2] < kz * NC + m, hi[2] > kz * NC + NC - 1 - m)
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == -1 and not near_x[0] or dx == 1 and not near_x[1]:
                        continue
                    if dz == -1 and not near_z[0] or dz == 1 and not near_z[1]:
                        continue
                    nb = (kx + dx, kz + dz)
                    if nb in self.dens:
                        self.dirty.add(nb)
                        self.touched.add(nb)

    def cells_box(self, lo, hi):
        """The solid cubes in the box of cube indices lo..hi (inclusive) as a bool array. Unloaded chunks read as air."""
        lo = [int(v) for v in lo]
        hi = [int(v) for v in hi]
        out = np.zeros((hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1), bool)
        for kx in range(lo[0] // NC, hi[0] // NC + 1):
            for kz in range(lo[2] // NC, hi[2] // NC + 1):
                d = self.dens.get((kx, kz))
                if d is None:
                    continue
                x0, x1 = max(lo[0], kx * NC), min(hi[0], kx * NC + NC - 1)
                z0, z1 = max(lo[2], kz * NC), min(hi[2], kz * NC + NC - 1)
                y0, y1 = max(lo[1], 0), min(hi[1], NH - 1)
                if x0 > x1 or z0 > z1 or y0 > y1:
                    continue
                out[x0 - lo[0]:x1 - lo[0] + 1, y0 - lo[1]:y1 - lo[1] + 1, z0 - lo[2]:z1 - lo[2] + 1] = \
                    d.box(x0 - kx * NC, x1 - kx * NC, y0, y1, z0 - kz * NC, z1 - kz * NC) > 0
        return out

    def clear_cells(self, idx):
        """Remove the cubes at these (n, 3) cube indices."""
        carved = set()
        idx = np.asarray(idx)
        if len(idx) == 0:
            return
        ckey = (idx[:, 0] // NC + 100000).astype(np.int64) * 200000 + (idx[:, 2] // NC + 100000)     # one 1-D key per chunk: no Python loop over the cubes
        for kk in np.unique(ckey).tolist():
            kx, kz = kk // 200000 - 100000, kk % 200000 - 100000
            d = self.dens.get((kx, kz))
            if d is None:
                continue
            sel = ckey == kk
            q = idx[sel]
            d.dense()[q[:, 0] - kx * NC, q[:, 1], q[:, 2] - kz * NC] = -1
            carved.add((kx, kz))
            g = self.gdmg.get((kx, kz))
            if g is not None:                                  # a cube that is gone takes its damage with it
                yy = q[:, 1] - g[0]
                ok = (yy >= 0) & (yy < g[1].shape[1])
                g[1][(q[ok, 0] - kx * NC), yy[ok], (q[ok, 2] - kz * NC)] = 0
        if carved:
            self._changed_chunks(carved, idx.min(axis=0), idx.max(axis=0))

    def settle(self, center, radius, limit=3000):
        """Ground that digging cut loose falls. Looks at the cubes around (centre, radius), and any solid group that no longer
        joins the rest of the ground (or the rock below) is removed. Returns the centres of the cubes that fall, (n, 3)."""
        try:
            from scipy import ndimage
        except Exception:
            return np.zeros((0, 3))
        m = int(radius / CELL) + 8
        c0 = np.floor(np.asarray(center, np.float64) / CELL).astype(np.int64)
        lo = np.array([c0[0] - m, max(0, c0[1] - m), c0[2] - m])
        hi = np.array([c0[0] + m, min(NH - 1, c0[1] + m), c0[2] + m])
        if lo[1] > hi[1]:                                      # the hit is above the top of the ground grid: nothing to settle
            return np.zeros((0, 3))
        box = self.cells_box(lo, hi)
        if not box.any():
            return np.zeros((0, 3))
        lab, n = ndimage.label(box)
        keep = np.zeros(n + 1, bool)
        for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
            keep[np.unique(face)] = True
        if lo[1] == 0:
            keep[np.unique(lab[:, 0, :])] = True               # the lowest layer is bedrock
        keep[0] = True
        floating = box & ~keep[lab]
        count = int(floating.sum())
        if count == 0 or count > limit:
            return np.zeros((0, 3))
        idx = np.argwhere(floating) + lo
        self.clear_cells(idx)
        return (idx + 0.5) * CELL

    def settle_islands(self, center, radius, margin=8.0, max_scan=1_500_000):
        """Like settle(), but it only looks: nothing is removed. Scans the ground out to `radius + margin` blocks around `center`
        and returns each solid group that no longer joins the rest of the ground (or the rock below) as its own (n, 3) array of
        cube indices. A group that touches the edge of the scan counts as still attached (it may continue past the edge), so the
        margin has to be big enough to contain the whole ledge. The scan volume is capped by `max_scan` cubes of work."""
        try:
            from scipy import ndimage
        except Exception:
            return []
        m = int((radius + margin) / CELL) + 2
        c0 = np.floor(np.asarray(center, np.float64) / CELL).astype(np.int64)
        while True:
            lo = np.array([c0[0] - m, max(0, c0[1] - m), c0[2] - m])
            hi = np.array([c0[0] + m, min(NH - 1, c0[1] + m), c0[2] + m])
            if lo[1] > hi[1]:
                return []
            vol = int(np.prod(hi - lo + 1))
            if vol <= max_scan or m <= int(radius / CELL) + 8:
                break
            m = max(int(radius / CELL) + 8, int(m * 0.9))
        box = self.cells_box(lo, hi)
        if not box.any():
            return []
        lab, n = ndimage.label(box)
        keep = np.zeros(n + 1, bool)
        for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
            keep[np.unique(face)] = True
        if lo[1] == 0:
            keep[np.unique(lab[:, 0, :])] = True               # the lowest layer is bedrock
        keep[0] = True
        floating = box & ~keep[lab]
        if not floating.any():
            return []
        idx = np.argwhere(floating)
        labs = lab[floating]                                     # argwhere and boolean indexing both walk the array in the same order
        order = np.argsort(labs, kind='stable')
        idx, labs = idx[order], labs[order]
        cuts = np.nonzero(np.diff(labs))[0] + 1
        return [g + lo for g in np.split(idx, cuts)]

    # ---- ground damage ------------------------------------------------------------------------------
    def _dmg_rows(self, key, y0, y1):
        """The damage array of chunk `key`, grown so it covers rows y0..y1 (inclusive). Returns (first row, array)."""
        ent = self.gdmg.get(key)
        if ent is None:
            ent = self.gdmg[key] = [y0, np.zeros((NC, y1 - y0 + 1, NC), np.float16)]
            return ent
        a0, arr = ent
        n0, n1 = min(a0, y0), max(a0 + arr.shape[1] - 1, y1)
        if n0 != a0 or n1 != a0 + arr.shape[1] - 1:
            new = np.zeros((NC, n1 - n0 + 1, NC), np.float16)
            new[:, a0 - n0:a0 - n0 + arr.shape[1], :] = arr
            ent[0], ent[1] = n0, new
        return ent

    def add_damage(self, center, radius, amount=DMG_AMOUNT):
        """A blast weakens the ground around it, out to twice its radius: `amount` at the centre, falling in a straight line to 0.
        Damage stacks (capped at 1) and stays until reset_damage / reset_all."""
        R2 = float(radius) * 2.0
        c = np.asarray(center, np.float64)
        m = int(R2 / CELL) + 1
        c0 = np.floor(c / CELL).astype(np.int64)
        lo = np.array([c0[0] - m, max(0, c0[1] - m), c0[2] - m])
        hi = np.array([c0[0] + m, min(NH - 1, c0[1] + m), c0[2] + m])
        if lo[1] > hi[1]:
            return
        gx = (np.arange(lo[0], hi[0] + 1) + 0.5) * CELL - c[0]
        gy = (np.arange(lo[1], hi[1] + 1) + 0.5) * CELL - c[1]
        gz = (np.arange(lo[2], hi[2] + 1) + 0.5) * CELL - c[2]
        r = np.sqrt(gx[:, None, None] ** 2 + gy[None, :, None] ** 2 + gz[None, None, :] ** 2)
        d = np.clip(1.0 - r / R2, 0.0, 1.0) * float(amount)
        for kx in range(lo[0] // NC, hi[0] // NC + 1):
            for kz in range(lo[2] // NC, hi[2] // NC + 1):
                x0, x1 = max(lo[0], kx * NC), min(hi[0], kx * NC + NC - 1)
                z0, z1 = max(lo[2], kz * NC), min(hi[2], kz * NC + NC - 1)
                sub = d[x0 - lo[0]:x1 - lo[0] + 1, :, z0 - lo[2]:z1 - lo[2] + 1]
                if not (sub > 0.0).any():
                    continue
                a0, arr = self._dmg_rows(( kx, kz), int(lo[1]), int(hi[1]))
                v = arr[x0 - kx * NC:x1 - kx * NC + 1, lo[1] - a0:hi[1] - a0 + 1, z0 - kz * NC:z1 - kz * NC + 1]
                v[...] = np.minimum(1.0, v.astype(np.float32) + sub).astype(np.float16)
        self.edit_ver += 1

    def damage_box(self, lo, hi):
        """The ground damage in the box of cube indices lo..hi (inclusive) as a float32 array (0 where there is none)."""
        lo = [int(v) for v in lo]
        hi = [int(v) for v in hi]
        out = np.zeros((hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1), np.float32)
        if not self.gdmg:
            return out
        for kx in range(lo[0] // NC, hi[0] // NC + 1):
            for kz in range(lo[2] // NC, hi[2] // NC + 1):
                g = self.gdmg.get((kx, kz))
                if g is None:
                    continue
                a0, arr = g
                x0, x1 = max(lo[0], kx * NC), min(hi[0], kx * NC + NC - 1)
                z0, z1 = max(lo[2], kz * NC), min(hi[2], kz * NC + NC - 1)
                y0, y1 = max(lo[1], a0), min(hi[1], a0 + arr.shape[1] - 1)
                if x0 > x1 or z0 > z1 or y0 > y1:
                    continue
                out[x0 - lo[0]:x1 - lo[0] + 1, y0 - lo[1]:y1 - lo[1] + 1, z0 - lo[2]:z1 - lo[2] + 1] = \
                    arr[x0 - kx * NC:x1 - kx * NC + 1, y0 - a0:y1 - a0 + 1, z0 - kz * NC:z1 - kz * NC + 1]
        return out

    def weak_islands(self, center, radius, margin=8.0, max_scan=1_500_000, soft=DMG_SOFT, min_cells=300):
        """Ground that damage alone has cut off. Cubes damaged past `soft` count as broken bonds; each group of sound cubes
        that no longer joins the rest of the ground (or the rock below) through sound cubes is judged by its neck: the damaged
        cubes it still touches. Load = weight x lever arm / (neck area left)^1.5, against NECK_LIMIT. Returns one dict per group
        that is over its limit: idx (cube indices, int32), ratio (load / limit; 1000 = no neck at all), com (centre, blocks).
        Nothing is removed. Needs SciPy, like settle_islands."""
        if not self.gdmg:
            return []
        try:
            from scipy import ndimage
        except Exception:
            return []
        m = int((radius + margin) / CELL) + 2
        c0 = np.floor(np.asarray(center, np.float64) / CELL).astype(np.int64)
        while True:
            lo = np.array([c0[0] - m, max(0, c0[1] - m), c0[2] - m])
            hi = np.array([c0[0] + m, min(NH - 1, c0[1] + m), c0[2] + m])
            if lo[1] > hi[1]:
                return []
            if int(np.prod(hi - lo + 1)) <= max_scan or m <= int(radius / CELL) + 8:
                break
            m = max(int(radius / CELL) + 8, int(m * 0.9))
        dm = self.damage_box(lo, hi)
        if float(dm.max()) < soft:
            return []
        box = self.cells_box(lo, hi)
        intact = box & (dm < soft)
        if not intact.any():
            return []
        thin = max(1, int(NECK_THIN / CELL))
        lab, n = ndimage.label(ndimage.binary_erosion(intact, iterations=thin, border_value=1))   # a neck thinner than this holds nothing: ledges joined only by one are separate groups
        keep = np.zeros(n + 1, bool)
        for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
            keep[np.unique(face)] = True
        if lo[1] == 0:
            keep[np.unique(lab[:, 0, :])] = True
        keep[0] = True
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        cand = [i for i in range(1, n + 1) if not keep[i] and sizes[i] >= min_cells]
        if not cand:
            return []
        cand.sort(key=lambda i: -sizes[i])
        objs = ndimage.find_objects(lab)
        out = []
        K = max(4, int(NECK_REACH / CELL))
        for i in cand[:4]:
            sl0 = objs[i - 1]
            sl = tuple(slice(max(s0.start - K - 1, 0), min(s0.stop + K + 1, lab.shape[k])) for k, s0 in enumerate(sl0))
            solid = box[sl]
            sub = ndimage.binary_dilation(lab[sl] == i, structure=np.ones((3, 3, 3), bool), iterations=thin) & intact[sl]   # the group again, with the rim the erosion took: a cube-shaped step, so edges and corners come back too (a diamond step left them behind as a second, overlapping slab)
            if int(sub.sum()) < min_cells:
                continue
            neck = ndimage.binary_dilation(sub) & solid & ~sub
            nn = int(neck.sum())
            off = np.array([s0.start for s0 in sl]) + lo
            gi = np.argwhere(sub) + off
            com = (gi.mean(0) + 0.5) * CELL
            if nn == 0:
                ratio = 1000.0                                    # no neck at all: this was cut off outright
            else:
                # The neck is as strong as its weakest cross-section: slice the damaged ground next to the group across the
                # direction it hangs by, and take the slice with the least sound ground left (carved cubes and damage both weaken it).
                reach = ndimage.binary_dilation(sub, iterations=K) & solid & ~sub
                ni = np.argwhere(neck)
                ncen = (ni.mean(0) + off + 0.5) * CELL
                ax = int(np.argmax(np.abs(ncen - com)))
                sgn = 1 if ncen[ax] >= com[ax] else -1
                cap = np.where(reach, np.maximum(1.0 - dm[sl], 0.02), 0.0)
                other = tuple(a for a in range(3) if a != ax)
                layer = cap.sum(axis=other)
                gl_ = np.argwhere(sub)[:, ax]
                edge = int(gl_.max() if sgn > 0 else gl_.min())
                idx_l = np.arange(len(layer))
                ok = (layer > 0) & ((idx_l - edge) * sgn >= -1) & ((idx_l - edge) * sgn <= K // 2)
                if not ok.any():
                    ok = layer > 0
                area = float(layer[ok].min()) * CELL * CELL
                arm = max(0.5, float(np.hypot(com[0] - ncen[0], com[2] - ncen[2])))
                ratio = (len(gi) * CELL ** 3) * arm / (area ** 1.5) / NECK_LIMIT
            if ratio >= 1.0:
                out.append(dict(idx=gi.astype(np.int32), ratio=float(ratio), com=com))
        return out

    def reset_damage(self):
        """Put back every dug block: edited chunks are regenerated and saved edits are dropped."""
        for key in list(self.modified_chunks):
            self.saved.pop(key, None)
            self.saved_dens.pop(key, None)
            if key in self.chunks:
                self.chunks[key], self.dens[key] = generate_chunk(*key)
            for dx, dz in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (key[0] + dx, key[1] + dz)
                if nb in self.chunks:
                    self.dirty.add(nb)
                    self.touched.add(nb)
        self.modified_chunks.clear()
        self.dug_cells = []
        self.gdmg.clear()
        self.edit_ver += 1

    def reset_all(self):
        """Forget every chunk and every edit, so the whole map is generated again (used when the cube size changes)."""
        self.chunks.clear()
        self.dens.clear()
        self.saved.clear()
        self.saved_dens.clear()
        self.modified_chunks.clear()
        self.touched.clear()
        self._building.clear()
        self.dirty.clear()
        self.dug_cells = []
        self.gdmg.clear()
        self.edit_ver += 1

    def take_dug_cells(self):
        out = self.dug_cells
        self.dug_cells = []
        return out

    def ensure_chunk(self, cx, cz):
        """Load the chunk at (cx, cz): restore its saved edits if it has any, otherwise generate it."""
        if (cx, cz) in self.chunks:
            return
        saved = self.saved.get((cx, cz))
        if saved is not None:
            self.chunks[(cx, cz)] = saved
            self.dens[(cx, cz)] = self.saved_dens[(cx, cz)]
        else:
            self.chunks[(cx, cz)], self.dens[(cx, cz)] = generate_chunk(cx, cz)
        # Mark the new chunk and its four side neighbours for remeshing,
        # since their shared edge faces may have changed.
        for dx, dz in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            self.dirty.add((cx + dx, cz + dz))
        # Diagonal neighbours can become ready to mesh too, so they are looked at as well.
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                self.touched.add((cx + dx, cz + dz))

    def unload_chunk(self, key):
        """Drop a chunk from memory. Edited chunks are kept in self.saved so
        the player's changes survive; untouched chunks are simply regenerated."""
        arr = self.chunks.pop(key, None)
        if arr is None:
            return
        dens = self.dens.pop(key, None)
        if key in self.modified_chunks:
            self.saved[key] = arr
            self.saved_dens[key] = dens
        self.dirty.discard(key)

    def neighbours_ready(self, cx, cz):
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0: continue
                if (cx + dx, cz + dz) not in self.chunks or (cx + dx, cz + dz) in self._building: return False
        return True

def load_around(world, px, pz, radius=RENDER_DIST, budget=None):
    """Make sure every chunk within `radius` of world position (px, pz) exists, nearest first.
    Call this each frame with the player's position. With a `budget`, at most that many chunks are
    generated per call, so raising the render distance spreads the work over a few frames."""
    pcx = math.floor(px / CHUNK_SIZE)
    pcz = math.floor(pz / CHUNK_SIZE)
    made = 0
    for dx, dz in _ring_offsets(radius):
        k = (pcx + dx, pcz + dz)
        if k in world.chunks:
            continue
        world.ensure_chunk(*k)
        made += 1
        if budget is not None and made >= budget:
            break

# ═══════════════════════════════════════════════
#  CHUNK MESHER & CACHE LOOKUP
# ═══════════════════════════════════════════════
# Colour per block ID. Unknown IDs render magenta so gaps are easy to spot.
_COLOR_LUT = np.tile(np.array([1.0, 0.0, 1.0], dtype=np.float32), (256, 1))
for _blk, _rgb in COLORS.items():
    _COLOR_LUT[_blk] = _rgb

def _padded_occ(world, cx, cz, step=1):
    """The solid cubes of one chunk plus `step` cubes of border from the neighbouring chunks on each side, as a bool array
    (NC + 2*step, H, NC + 2*step), H cropped to just above the tallest column. Cubes of unloaded chunks read as air.
    The chunk's own cube (i, y, k) is at [i + step, y, k + step]."""
    s_ = step
    srcs = {}
    hmax = 0
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            src = world.dens.get((cx + dx, cz + dz))
            if src is not None:
                srcs[(dx, dz)] = src
                hmax = max(hmax, NH if src.arr is not None else int(src.top.max()))
    H = min(NH, (hmax // 15 + 1) * 15)
    big = np.zeros((NC + 2 * s_, H, NC + 2 * s_), bool)
    for (dx, dz), src in srcs.items():
        gx0, gz0 = dx * NC + s_, dz * NC + s_              # where the neighbour's cube 0 lands in `big`
        x0, x1 = max(gx0, 0), min(gx0 + NC, NC + 2 * s_)
        z0, z1 = max(gz0, 0), min(gz0 + NC, NC + 2 * s_)
        if x0 < x1 and z0 < z1:
            big[x0:x1, :, z0:z1] = src.box(x0 - gx0, x1 - gx0 - 1, 0, H - 1, z0 - gz0, z1 - gz0 - 1) > 0
    return big


def _greedy_rects(m):
    """Cover the True cells of a 2-D bool array with as few rectangles as a simple greedy pass finds.
    Returns (i, j, height, width) tuples."""
    A, B = m.shape
    used = np.zeros_like(m)
    out = []
    for i in range(A):
        if not m[i].any():
            continue
        j = 0
        row = m[i]
        while j < B:
            if row[j] and not used[i, j]:
                w = 1
                while j + w < B and row[j + w] and not used[i, j + w]:
                    w += 1
                h = 1
                while i + h < A and m[i + h, j:j + w].all() and not used[i + h, j:j + w].any():
                    h += 1
                used[i:i + h, j:j + w] = True
                out.append((i, j, h, w))
                j += w
            else:
                j += 1
    return out

_FACE_SHADE = {'+y': 1.0, '-y': 0.55, '+x': 0.80, '-x': 0.80, '+z': 0.70, '-z': 0.70}

def _mesh_cubes(big, cx, cz, step=1, res=2):
    """Build the ground mesh of one chunk from its padded cube grid (see _padded_occ). Only faces between a solid cube and air
    are made, and neighbouring faces on one plane are merged into larger rectangles.
    With step > 1 groups of step**3 cubes are merged into one bigger cube (solid when most of them are), for far chunks.
    Reads only its arguments, so it can run in a worker process.
    Returns (positions, colours, uv_flags) float32 arrays of shape (N, 3) in world space. uv_flags.z is a material code:
    0 land top (the shader picks grass / dirt / stone by height), 1 water, 2 sand, 3 dirt side, 4 stone side."""
    s_ = step
    cell = 1.0 / res
    if s_ > 1:
        ny = big.shape[1] // s_
        v = big[:, :ny * s_, :]
        v = v.reshape(big.shape[0] // s_, s_, ny, s_, big.shape[2] // s_, s_).mean(axis=(1, 3, 5)) >= 0.5
        big = v
    sz = s_ * cell
    own = big[1:-1, :, 1:-1]
    NX, NY, NZ = own.shape
    up = np.zeros_like(own)
    up[:, :-1, :] = big[1:-1, 1:, 1:-1]
    dn = np.ones_like(own)
    dn[:, 1:, :] = big[1:-1, :-1, 1:-1]
    faces = {
        '+y': own & ~up, '-y': own & ~dn,
        '+x': own & ~big[2:, :, 1:-1], '-x': own & ~big[:-2, :, 1:-1],
        '+z': own & ~big[1:-1, :, 2:], '-z': own & ~big[1:-1, :, :-2],
    }
    top = np.where(own.any(axis=1), NY - own[:, ::-1, :].argmax(axis=1), 0)     # (NX, NZ) cubes up to the highest solid one
    ox, oz = cx * CHUNK_SIZE, cz * CHUNK_SIZE
    DIRT_A, STONE_A, WATER = 19.0, 27.0, 6.0
    ys_col = np.arange(NY)[:, None]

    def side_dirt(top_line):
        ta = top_line[None, :] * sz
        dp = (top_line[None, :] - 1 - ys_col) * sz
        return (ta < STONE_A) & ((ta >= DIRT_A) | (dp < 1.5))

    quads = []
    shades = []
    codes = []
    for name, M in faces.items():
        if not M.any():
            continue
        ax = name[1]
        pos_side = name[0] == '+'
        corners = []
        cds = []
        if ax == 'y':
            for j in np.nonzero(M.any(axis=(0, 2)))[0]:
                y = (j + (1 if pos_side else 0)) * sz
                if not pos_side:
                    code = 3
                elif y <= WATER + 0.05:
                    code = 1
                elif y <= WATER + 1.05:
                    code = 2
                else:
                    code = 0
                for i, k, h, w in _greedy_rects(M[:, j, :]):
                    x0, x1 = ox + i * sz, ox + (i + h) * sz
                    z0, z1 = oz + k * sz, oz + (k + w) * sz
                    c = [(x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0)]
                    corners.append(c if pos_side else c[::-1])
                    cds.append(code)
        else:
            if ax == 'x':
                planes = [(i, M[i, :, :], side_dirt(top[i, :])) for i in np.nonzero(M.any(axis=(1, 2)))[0]]
            else:
                planes = [(k, M[:, :, k], side_dirt(top[:, k]).T) for k in np.nonzero(M.any(axis=(0, 1)))[0]]
            for p, M2, dirtm in planes:
                if ax == 'x':
                    x = ox + (p + (1 if pos_side else 0)) * sz
                else:
                    z = oz + (p + (1 if pos_side else 0)) * sz
                for mask, code in ((M2 & dirtm, 3), (M2 & ~dirtm, 4)):
                    if not mask.any():
                        continue
                    for a, b, h, w in _greedy_rects(mask):
                        if ax == 'x':
                            y0, y1 = a * sz, (a + h) * sz
                            z0, z1 = oz + b * sz, oz + (b + w) * sz
                            c = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
                        else:
                            x0, x1 = ox + a * sz, ox + (a + h) * sz
                            y0, y1 = b * sz, (b + w) * sz
                            c = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
                        corners.append(c if pos_side else c[::-1])
                        cds.append(code)
        if corners:
            quads.append(np.array(corners, np.float32))
            shades.append(np.full(len(corners), _FACE_SHADE[name], np.float32))
            codes.append(np.array(cds, np.float32))
    if not quads:
        empty = np.zeros((0, 3), np.float32)
        return empty, empty.copy(), empty.copy()
    Q = np.concatenate(quads)                                  # (n, 4, 3)
    sh = np.concatenate(shades)
    cd = np.concatenate(codes)
    idx = [0, 1, 2, 0, 2, 3]
    pos = Q[:, idx, :].reshape(-1, 3)
    col = (_COLOR_LUT[DIRT][None, None, :] * np.repeat(sh, 6)[:, None]).reshape(-1, 3).astype(np.float32)
    uvf = np.stack([pos[:, 0], pos[:, 2], np.repeat(cd, 6)], axis=1).astype(np.float32)
    return pos.astype(np.float32), col, uvf

class MeshWorker:
    """Builds chunk meshes in background processes, so they never hold up the main thread's
    Python lock. The main thread copies each chunk's density when it queues the job, so the
    workers never read live world data. It also collects finished meshes, which the renderer uploads."""
    def __init__(self):
        self.pool = ProcessPoolExecutor(max_workers=MESH_WORKERS)
        self.pending = {}     # chunk key -> (version it was built from, future, level-of-detail step)

    def submit(self, world, key, step=1):
        cx, cz = key
        ver = world.dirty.ver.get(key, 0)
        snap = _padded_occ(world, cx, cz, step)
        self.pending[key] = (ver, DISK_CACHE.submit_mesh(self, snap, cx, cz, step), step)

    def collect(self, world):
        """Finished meshes as (key, version, payload, step). Jobs for chunks that have unloaded are dropped."""
        done = []
        for key, (ver, fut, step) in list(self.pending.items()):
            if key not in world.chunks:
                fut.cancel()
                del self.pending[key]
            elif fut.done():
                del self.pending[key]
                done.append((key, ver, fut.result(), step))
        return done

@functools.lru_cache(maxsize=None)
def _ring_offsets(radius):
    """(dx, dz) offsets for a square of chunks around the player, nearest first."""
    offs = [(dx, dz) for dx in range(-radius, radius + 1) for dz in range(-radius, radius + 1)]
    offs.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])
    return tuple(offs)

# ═══════════════════════════════════════════════
#  GRAPHICS QUALITY: presets, terrain level of detail, shadow levels
# ═══════════════════════════════════════════════
QUALITY_NAMES = ('Low', 'Medium', 'High', 'Custom')
# The settings a preset drives, and the value each preset gives them. High is the game's original look.
QUALITY_KEYS = ('lod', 'render_distance', 'shadow_quality', 'shadow_update')
QUALITY_PRESETS = ((3, 8, 1, 3),      # Low
                   (2, 12, 2, 2),     # Medium
                   (0, 15, 2, 1))     # High
LOD_NAMES = ('Off', 'Gentle', 'Balanced', 'Aggressive', 'Maximum')
# Level of detail -> (chunk distance where the 3-block mesh starts, where the 5-block mesh starts).
LOD_RINGS = {0: (None, None), 1: (8, 12), 2: (5, 9), 3: (3, 6), 4: (2, 4)}
SHADOW_NAMES = ('Off', 'Low 1024', 'Medium 2048', 'High 4096')
SHADOW_LEVELS = (0, 1024, 2048, 4096)     # shadow map size in texels; 0 = no shadows
LOD_SIG = {1: CHUNK_SIZE * 10 + 1, 3: CHUNK_SIZE * 10 + 3, 5: CHUNK_SIZE * 10 + 5}   # mesh-cache signature per step

def lod_step(lod, dist):
    """Mesh step for a chunk `dist` chunks (Chebyshev) from the player at level of detail `lod`."""
    near, far = LOD_RINGS[int(lod)]
    if near is None:
        return 1
    return 5 if dist >= far else (3 if dist >= near else 1)

def quality_snapshot(settings):
    return {k: settings[k] for k in ('quality',) + QUALITY_KEYS}

def sync_quality(settings, last):
    """Keep the Quality preset and the settings it drives in step. Picking Low / Medium / High copies the
    preset's values over them; changing any one of them by hand switches the preset to the one that matches
    (or Custom). `last` is the snapshot from the previous call; returns the new snapshot."""
    q = int(settings['quality'])
    if q != int(last['quality']) and q < 3:
        for key, val in zip(QUALITY_KEYS, QUALITY_PRESETS[q]):
            settings[key] = val
    else:
        cur = tuple(settings[k] for k in QUALITY_KEYS)
        if cur != tuple(last[k] for k in QUALITY_KEYS):
            settings['quality'] = next((i for i, p in enumerate(QUALITY_PRESETS) if cur == p), 3)
    return quality_snapshot(settings)

# ═══════════════════════════════════════════════
#  RENDERER (pygame window + OpenGL 3.3 core)
# ═══════════════════════════════════════════════
SKY_RGB = (0.76, 0.83, 0.92)   # horizon colour: the clear colour under the sky pass (the sky itself is drawn by SKY_FRAG)

# Sun and shadows. A low sun in front and to the left of the spawn view. The shaders use the same direction (normalize(-0.70, 0.62, -0.40)).
SUN_DIR = np.array([-0.70, 0.62, -0.40])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)
SHADOW_RES = 2048          # shadow map is SHADOW_RES x SHADOW_RES texels
SHADOW_HALF = 48.0         # blocks covered by the shadow map, half-width, centred on the player

def _look_at(eye, target, up):
    """Row-major view matrix looking from eye towards target."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -(m[:3, :3] @ eye)
    return m

def _ortho(l, r, b, t, n, f):
    """Row-major orthographic projection."""
    m = np.eye(4)
    m[0, 0] = 2.0 / (r - l)
    m[1, 1] = 2.0 / (t - b)
    m[2, 2] = -2.0 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m

def light_matrix(centre, res=SHADOW_RES):
    """Sun's view-projection, centred on the player. The centre is snapped to shadow-map texels so the shadows
    do not shimmer as the player walks."""
    texel = 2.0 * SHADOW_HALF / res
    c = np.floor(np.asarray(centre, np.float64) / texel) * texel
    eye = c + SUN_DIR * SHADOW_HALF * 2.0
    view = _look_at(eye, c, np.array([0.0, 1.0, 0.0]))
    proj = _ortho(-SHADOW_HALF, SHADOW_HALF, -SHADOW_HALF, SHADOW_HALF, 0.1, SHADOW_HALF * 4.0)
    return (proj @ view).astype(np.float32)

# Depth-only pass: the scene drawn from the sun into the shadow map.
_DEPTH_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uLVP;
uniform mat4 uModel;
void main() {
    gl_Position = uLVP * uModel * vec4(aPos, 1.0);
}
"""

_DEPTH_FRAG = """
#version 330 core
void main() {
}
"""

# Shadow lookup shared by the terrain and structure shaders: soft, 16 taps spread over a disc (Vogel spiral) and
# turned a little differently at every pixel, so the edge fades out over about a quarter of a block instead of stepping.
_SHADOW_GLSL = """
uniform mat4 uLVP;
uniform sampler2D uShadow;
uniform float uShadowTexel;                          // 1 / shadow map size
float sunShadow(vec3 wp, vec3 nrm) {
    float q = max(1.0, 2048.0 * uShadowTexel);       // coarser maps need a bigger nudge and bias
    vec4 ls = uLVP * vec4(wp + nrm * (0.08 * q), 1.0);     // nudged along the normal to avoid self-shadowing
    vec3 s = ls.xyz / ls.w * 0.5 + 0.5;
    if (s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0 || s.z > 1.0) return 1.0;
    float rad = max(0.0019, 1.5 * uShadowTexel);     // blur radius in map units: the softness of the shadow edge
    float rot = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float lit = 0.0;
    for (int i = 0; i < 16; i++) {
        float r = sqrt((float(i) + 0.5) / 16.0);
        float a = float(i) * 2.3999632 + rot;
        float d = texture(uShadow, s.xy + vec2(cos(a), sin(a)) * (r * rad)).r;
        lit += (s.z - 0.0018 * q > d) ? 0.0 : 1.0;
    }
    return lit / 16.0;
}
"""

# One daylight model for terrain, structures, debris and sky: the same sun, sky fill and haze colour everywhere.
_LIGHT_GLSL = """
const vec3 SUN_DIR  = vec3(-0.6882, 0.6096, -0.3933);  // normalize(-0.70, 0.62, -0.40): low sun, front left
const vec3 SUN_COL  = vec3(1.00, 0.82, 0.56);          // warm golden sun
const vec3 SKY_TINT = vec3(0.72, 0.85, 1.08);          // sky fill: cool, so shadows lean blue
const vec3 BOUNCE_COL = vec3(0.95, 0.78, 0.55);        // warm light bounced off the ground onto upright faces
const vec3 HORIZON_COL = vec3(0.76, 0.83, 0.92);       // pale blue haze away from the sun
const vec3 SUN_HAZE = vec3(0.98, 0.90, 0.78);          // cream haze toward the sun
const vec3 FOG_COL  = HORIZON_COL;
// Haze colour looking along dir: pale blue, warming to cream toward the sun, so distance and sky blend.
vec3 fogColor(vec3 dir) {
    vec2 h = dir.xz, s = SUN_DIR.xz;
    float toward = max(dot(h, s) / max(length(h) * length(s), 1e-4), 0.0);
    return mix(HORIZON_COL, SUN_HAZE, 0.75 * toward * toward * toward);
}
// Light reaching a surface with normal n; sh is the sun visibility (0 in shadow, 1 in sun).
vec3 dayLight(vec3 n, float sh) {
    float sky = 0.5 + 0.5 * n.y;
    float diff = max(dot(n, SUN_DIR), 0.0);
    float side = 1.0 - abs(n.y);
    return SKY_TINT * (0.36 + 0.18 * sky) + BOUNCE_COL * (0.17 * side) + SUN_COL * (1.1 * diff * sh);
}
"""

# The sky: a full-screen pass drawn first. Blue gradient, warm glow around the sun, thin stretched clouds that drift.
_SKY_VERT = """
#version 330 core
out vec2 vNdc;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vNdc = p * 2.0 - 1.0;
    gl_Position = vec4(vNdc, 1.0, 1.0);
}
"""

_SKY_FRAG = """
#version 330 core
in vec2 vNdc;
out vec4 fragColor;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform float uTime;
LIGHT_GLSL_PLACEHOLDER
float hash21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p) {
    float a = 0.5, t = 0.0;
    for (int i = 0; i < 4; i++) { t += a * vnoise(p); p = p * 2.03 + 11.7; a *= 0.5; }
    return t;
}
float cloudDensity(vec2 uv) {
    float f = fbm(uv);
    return smoothstep(0.40, 0.66, f);
}
void main() {
    vec4 q = uInvVP * vec4(vNdc, 1.0, 1.0);
    vec3 dir = normalize(q.xyz / q.w - uEye);
    float e = dir.y;
    float t = clamp(e, 0.0, 1.0);
    vec3 hor = fogColor(dir);
    vec3 mid = vec3(0.62, 0.77, 0.95);
    vec3 zen = vec3(0.40, 0.60, 0.91);
    vec3 sky = mix(hor, mid, smoothstep(0.0, 0.20, t));
    sky = mix(sky, zen, smoothstep(0.12, 0.90, t));
    float sd = max(dot(dir, SUN_DIR), 0.0);
    sky += vec3(1.0, 0.80, 0.54) * (0.13 * pow(sd, 3.0) + 0.10 * pow(sd, 20.0));      // warm glow around the sun
    if (e > 0.01) {
        vec2 uv = dir.xz / (e + 0.14);
        uv = vec2(uv.x * 0.60, uv.y * 0.90) * 0.46 + vec2(uTime * 0.008, 17.0);
        float d = cloudDensity(uv);
        vec2 toSun = normalize(SUN_DIR.xz) * vec2(0.55, 0.95) * 0.08;
        float d2 = cloudDensity(uv + toSun);
        float lit = clamp(0.62 + (d - d2) * 5.0, 0.0, 1.0);             // the side facing the sun is bright, the far side shaded
        vec3 shade = vec3(0.66, 0.74, 0.88);
        vec3 bright = vec3(1.00, 0.95, 0.88);
        vec3 cc = mix(shade, bright, lit);
        cc = mix(cc, hor, 0.35 * (1.0 - smoothstep(0.0, 0.35, t)));     // thinner and bluer toward the horizon
        float fade = smoothstep(0.03, 0.30, e);
        sky = mix(sky, cc, d * fade * 0.92);
    }
    fragColor = vec4(min(sky, vec3(1.0)), 1.0);
}
"""
_SKY_FRAG = _SKY_FRAG.replace("LIGHT_GLSL_PLACEHOLDER", _LIGHT_GLSL)

def _eye_from_vp(vp):
    """Camera position from a view-projection matrix: the point that lands on clip x = y = w = 0."""
    m = np.asarray(vp, np.float64)
    rows = [0, 1, 3]
    return np.linalg.solve(m[rows, :3], -m[rows, 3]).astype(np.float32)

# Colour is baked into the mesh (face shade * block colour); the grass texture
# is multiplied in on top faces only.
_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aCol;
layout(location = 2) in vec3 aUVF;   // xy = texture coords, z = 1 if textured (top faces)
uniform mat4 uVP;
out vec3 vCol;
out vec3 vUVF;
out vec3 vWorld;     // terrain vertices are already in world space
out float vDepth;    // view distance, for the haze
void main() {
    vCol = aCol;
    vUVF = aUVF;
    vWorld = aPos;
    gl_Position = uVP * vec4(aPos, 1.0);
    vDepth = gl_Position.w;
}
"""

_FRAG_SRC = """
#version 330 core
in vec3 vCol;
in vec3 vUVF;        // z = material code: 0 land top, 1 water, 2 sand, 3 dirt side, 4 stone side
in vec3 vWorld;
in float vDepth;
out vec4 fragColor;
uniform sampler2DArray uTiles;   // layer 0 is dirt, layer 1 is stone
uniform float uLight;
uniform vec3 uEye;
uniform float uGrid;
uniform float uCell;
SHADOW_GLSL_PLACEHOLDER
LIGHT_GLSL_PLACEHOLDER
float hash21(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), f.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), f.x), f.y);
}
void main() {
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (dot(n, uEye - vWorld) < 0.0) n = -n;
    float sh = sunShadow(vWorld, n);
    vec3 light = dayLight(n, sh);
    vec2 uv = abs(n.y) > 0.5 ? vWorld.xz : (abs(n.x) > 0.5 ? vWorld.zy : vWorld.xy);
    vec3 dirtT = texture(uTiles, vec3(uv, 0.0)).rgb;
    vec3 stoneT = texture(uTiles, vec3(uv * 0.5, 1.0)).rgb;
    float lum = dot(dirtT, vec3(0.33));
    float shade = vCol.r / 0.65;
    int code = int(vUVF.z + 0.5);
    vec3 base;
    if (code == 1) {
        base = vec3(0.16, 0.42, 0.88) * (0.95 + 0.1 * hash21(floor(vWorld.xz * 2.0)));
    } else if (code == 2) {
        base = vec3(0.93, 0.87, 0.58) * (0.8 + 0.5 * lum);
    } else if (code == 3) {
        base = dirtT * vec3(1.02, 0.86, 0.70);
    } else if (code == 4) {
        base = stoneT;
    } else {
        float hb = hash21(floor(vWorld.xz));
        float ya = vWorld.y + (hb - 0.5) * 5.0;
        if (ya >= STONE_ALT_PLACEHOLDER) base = stoneT;
        else if (ya >= DIRT_ALT_PLACEHOLDER) base = dirtT * vec3(1.02, 0.86, 0.70);
        else {
            // muted grass in large patches: lush green drifting into dry olive, darker in the low patches
            float pn = vnoise(vWorld.xz * 0.18);
            float pn2 = vnoise(vWorld.xz * 0.05 + 17.0);
            float dry = smoothstep(0.50, 0.85, 0.6 * pn + 0.4 * pn2);
            vec3 gcol = mix(vec3(0.40, 0.50, 0.17), vec3(0.62, 0.55, 0.20), dry) * (0.86 + 0.28 * pn2);
            float grain = vnoise(vWorld.xz * 9.0) * 0.5 + vnoise(vWorld.xz * 23.0) * 0.5;   // fine grass grain
            base = gcol * (0.70 + 0.80 * lum) * (0.97 + 0.06 * hb) * (0.90 + 0.20 * grain);
        }
    }
    if (code == 4) {
        float seamHeight = 30.0 + 1.5 * vnoise(vWorld.xz * 0.035);
        float seam = 1.0 - smoothstep(0.20, 0.75, abs(vWorld.y - seamHeight));
        base *= mix(vec3(1.0), vec3(1.14, 1.02, 0.86), seam * 0.65);
        base *= 0.91 + 0.18 * vnoise(vWorld.xz * 0.18 + vWorld.y * vec2(0.19, 0.07));
    }
    vec3 c = min(base * shade, vec3(1.0));
    c *= light * uLight;
    if (uGrid > 0.5) {
        vec3 g = vWorld / uCell;
        vec2 gv = abs(n.y) > 0.5 ? g.xz : (abs(n.x) > 0.5 ? g.zy : g.xy);
        vec2 fw = max(fwidth(gv), vec2(1e-5));
        vec2 e = abs(fract(gv + 0.5) - 0.5) / fw;
        float edge = 1.0 - smoothstep(0.4, 1.4, min(e.x, e.y));
        edge *= 1.0 - smoothstep(0.25, 0.5, max(fw.x, fw.y));
        c = mix(c, c * 0.4, edge);
    }
    float fog = 1.0 - exp(-vDepth * 0.0105);
    fragColor = vec4(mix(c, fogColor(normalize(vWorld - uEye)), fog), 1.0);
}
"""
_FRAG_SRC = (_FRAG_SRC.replace("SHADOW_GLSL_PLACEHOLDER", _SHADOW_GLSL)
             .replace("LIGHT_GLSL_PLACEHOLDER", _LIGHT_GLSL)
             .replace("STONE_ALT_PLACEHOLDER", repr(float(STONE_ALT)))
             .replace("DIRT_ALT_PLACEHOLDER", repr(float(DIRT_ALT))))

_OVL_VERT = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_OVL_FRAG = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = texture(uTex, vUV);
}
"""

def _compile_shader(src, kind):
    sh = gl.glCreateShader(kind)
    gl.glShaderSource(sh, src)
    gl.glCompileShader(sh)
    if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
        raise RuntimeError(gl.glGetShaderInfoLog(sh).decode())
    return sh

def _link_program(vs, fs):
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog

def _perspective(fovy_deg, aspect, near, far):
    """Row-major perspective matrix (upload with transpose=GL_TRUE)."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m

def _view(eye, yaw, pitch):
    """Row-major view matrix. yaw=0, pitch=0 looks down -Z."""
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = np.array([cp * math.sin(yaw), sp, -cp * math.cos(yaw)], dtype=np.float32)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = right
    m[1, :3] = up
    m[2, :3] = -fwd
    m[:3, 3] = -(m[:3, :3] @ eye)
    return m

class Renderer:
    def __init__(self):
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(SKY_RGB[0], SKY_RGB[1], SKY_RGB[2], 1.0)
        self.program = self._build()
        self.sky_prog = _link_program(_compile_shader(_SKY_VERT, gl.GL_VERTEX_SHADER), _compile_shader(_SKY_FRAG, gl.GL_FRAGMENT_SHADER))
        self.sky_vao = gl.glGenVertexArrays(1)           # an empty one: the triangle is made in the vertex shader
        self.u_sky_inv = gl.glGetUniformLocation(self.sky_prog, "uInvVP")
        self.u_sky_eye = gl.glGetUniformLocation(self.sky_prog, "uEye")
        self.u_sky_time = gl.glGetUniformLocation(self.sky_prog, "uTime")
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_tiles = gl.glGetUniformLocation(self.program, "uTiles")
        self.u_light = gl.glGetUniformLocation(self.program, "uLight")
        self.u_eye = gl.glGetUniformLocation(self.program, "uEye")
        self.u_grid = gl.glGetUniformLocation(self.program, "uGrid")
        self.u_cell = gl.glGetUniformLocation(self.program, "uCell")
        self.grid = False                    # X: cube grid lines, set by the game loop
        self.u_lvp = gl.glGetUniformLocation(self.program, "uLVP")
        self.u_shadow = gl.glGetUniformLocation(self.program, "uShadow")
        self.u_stexel = gl.glGetUniformLocation(self.program, "uShadowTexel")
        self.light_vp = np.eye(4, dtype=np.float32)   # set each frame by the main loop
        self.shadow_tex = 0                            # the shadow map's depth texture, set by the main loop
        self.shadow_texel = 1.0 / SHADOW_RES           # 1 / shadow map size, set by the main loop
        self.lod = 0                                   # terrain level of detail (index into LOD_NAMES), set by the main loop
        self._lod_key = None                           # (player chunk x, z, radius, lod) the per-chunk steps were last checked for
        self.step_of = {}                              # chunk key -> mesh step of the mesh on the GPU
        self.mesh_s = self.upload_s = 0.0      # seconds spent meshing / uploading in the last draw
        self.list_s = self.call_s = self.collect_s = 0.0   # visible-list rebuild, draw call, mesh collection
        self._touched = set()         # chunk keys to look at this frame
        self._ring_key = None         # (player chunk x, z, radius) the draw list was built for
        self._ring_set = set()       # chunk keys inside that square
        self._center = (0, 0)        # player chunk, used to order waiting chunks
        self._waiting = []           # heap of (distance, cx, cz): chunks waiting for a free mesh slot
        self._waiting_set = set()
        self._slot_of = {}           # chunk key -> its index in the draw list
        self._key_at = []            # index -> chunk key
        self._n = 0                  # entries in the draw list
        self._first = np.zeros(1, np.int32)   # draw list: first vertex of each range
        self._count = np.zeros(1, np.int32)   # draw list: vertex count of each range
        self.mesher = MeshWorker()
        self.gpu = {}   # chunk key -> {'src': payload, 'start': first vertex, 'count': vertex count}
        # Every chunk mesh lives in one shared buffer. Each chunk owns a range of it, and the
        # visible ranges are drawn with one glMultiDrawArrays call per frame.
        self.capacity = ARENA_VERTS
        self.free = [(0, self.capacity)]          # sorted (start, length) free ranges
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.capacity * VERT_STRIDE, None, gl.GL_DYNAMIC_DRAW)
        self._set_attrib_pointers()
        gl.glBindVertexArray(0)

    @staticmethod
    def _set_attrib_pointers():
        """Describe the vertex layout. Call with the shared VAO and buffer bound."""
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, VERT_STRIDE, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, VERT_STRIDE, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, VERT_STRIDE, ctypes.c_void_p(24))

    @staticmethod
    def _build():
        vs = _compile_shader(_VERT_SRC, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_FRAG_SRC, gl.GL_FRAGMENT_SHADER)
        return _link_program(vs, fs)

    def _alloc(self, n):
        """Find n vertices of free space in the shared buffer, growing it if needed. Returns the start."""
        while True:
            for i, (s, length) in enumerate(self.free):
                if length >= n:
                    if length == n:
                        del self.free[i]
                    else:
                        self.free[i] = (s + n, length - n)
                    return s
            self._grow(n)

    def _release(self, start, n):
        """Return n vertices from start to the free list, merging with neighbouring free ranges."""
        if n <= 0:
            return
        i = bisect.bisect_left(self.free, (start, 0))
        if i < len(self.free) and start + n == self.free[i][0]:
            n += self.free[i][1]
            del self.free[i]
        if i > 0 and self.free[i - 1][0] + self.free[i - 1][1] == start:
            s, length = self.free[i - 1]
            self.free[i - 1] = (s, length + n)
        else:
            self.free.insert(i, (start, n))

    def _grow(self, need):
        """Double the shared buffer, copying existing meshes across, so at least `need` more vertices fit."""
        old_vbo, old_cap = self.vbo, self.capacity
        new_cap = max(old_cap * 2, old_cap + need)
        new_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, new_vbo)
        gl.glBufferData(gl.GL_COPY_WRITE_BUFFER, new_cap * VERT_STRIDE, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_COPY_READ_BUFFER, old_vbo)
        gl.glCopyBufferSubData(gl.GL_COPY_READ_BUFFER, gl.GL_COPY_WRITE_BUFFER, 0, 0, old_cap * VERT_STRIDE)
        gl.glBindBuffer(gl.GL_COPY_READ_BUFFER, 0)
        gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, 0)
        gl.glDeleteBuffers(1, [old_vbo])
        self.vbo, self.capacity = new_vbo, new_cap
        # The VAO remembers which buffer its pointers use, so point it at the new one.
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        self._set_attrib_pointers()
        gl.glBindVertexArray(0)
        self._release(old_cap, new_cap - old_cap)

    def _want_step(self, k):
        """Mesh step this chunk should have now: coarser the further it is from the player."""
        if not self.lod:
            return 1
        pcx, pcz = self._center
        return lod_step(self.lod, max(abs(k[0] - pcx), abs(k[1] - pcz)))

    def _upload(self, key, payload, step=1):
        self._touched.add(key)
        self.step_of[key] = step
        old = self.gpu.pop(key, None)
        if old is not None:
            self._release(old['start'], old['count'])
        pos, col, uvf = payload
        count = pos.shape[0]
        if count == 0:
            self.gpu[key] = {'src': payload, 'start': 0, 'count': 0}
            return
        data = np.ascontiguousarray(np.hstack([pos, col, uvf]), dtype=np.float32)
        start = self._alloc(count)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, start * VERT_STRIDE, data.nbytes, data)
        self.gpu[key] = {'src': payload, 'start': start, 'count': count}

    def evict(self, key):
        """Free the GPU mesh for a chunk that has been unloaded."""
        self._touched.add(key)
        self.step_of.pop(key, None)
        entry = self.gpu.pop(key, None)
        if entry is not None:
            self._release(entry['start'], entry['count'])

    def _change_ring(self, pcx, pcz, radius):
        """The player moved into another chunk, or the radius changed. Only the chunks that entered
        or left the square around the player are marked, so nothing else is looked at."""
        offs = _ring_offsets(radius)
        if self._ring_key is None or radius != self._ring_key[2]:
            # The table's size depends on the radius, so start it again.
            n = len(offs)
            self._first = np.zeros(n, np.int32)
            self._count = np.zeros(n, np.int32)
            self._key_at = [None] * n
            self._slot_of = {}
            self._n = 0
            self._ring_set = set()
        new_set = {(pcx + dx, pcz + dz) for dx, dz in offs}
        self._touched |= new_set ^ self._ring_set
        self._ring_set = new_set
        self._ring_key = (pcx, pcz, radius)
        self._center = (pcx, pcz)

    def _set_slot(self, k, entry):
        """Keep one chunk's range in the draw list in step with its GPU entry."""
        want = entry is not None and entry['count'] > 0
        idx = self._slot_of.get(k)
        if want:
            if idx is None:
                idx = self._n
                self._n += 1
                self._slot_of[k] = idx
                self._key_at[idx] = k
            self._first[idx] = entry['start']
            self._count[idx] = entry['count']
        elif idx is not None:
            last = self._n - 1
            if idx != last:               # move the last range into the gap
                moved = self._key_at[last]
                self._key_at[idx] = moved
                self._slot_of[moved] = idx
                self._first[idx] = self._first[last]
                self._count[idx] = self._count[last]
            self._key_at[last] = None
            del self._slot_of[k]
            self._n -= 1

    def _refresh(self, world, k):
        """Bring one chunk up to date: queue a new mesh if it needs one, upload a cached mesh it
        doesn't have on the GPU yet, and set its entry in the draw list."""
        in_ring = k in self._ring_set
        if in_ring and world.neighbours_ready(*k):
            ver = world.dirty.ver.get(k, 0)
            step = self._want_step(k)
            payload = CACHE.get_mesh(k, LOD_SIG[step], ver)
            if payload is None or k in world.dirty:
                pend = self.mesher.pending.get(k)
                if pend is None or pend[0] != ver or pend[2] != step:
                    if len(self.mesher.pending) < MESH_MAX_PENDING:
                        t_s = time.perf_counter()
                        self.mesher.submit(world, k, step)
                        self.mesh_s += time.perf_counter() - t_s
                    else:
                        self._wait(k)
            else:
                entry = self.gpu.get(k)
                if entry is None or entry['src'] is not payload:
                    t_c = time.perf_counter()
                    self._upload(k, payload, step)
                    self.upload_s += time.perf_counter() - t_c
        self._set_slot(k, self.gpu.get(k) if in_ring else None)

    def _wait(self, k):
        if k not in self._waiting_set:
            self._waiting_set.add(k)
            cx, cz = self._center
            heapq.heappush(self._waiting, ((k[0] - cx) ** 2 + (k[1] - cz) ** 2, k[0], k[1]))

    def _promote_waiting(self, world):
        """When mesh slots free up, queue the nearest waiting chunks."""
        while self._waiting and len(self.mesher.pending) < MESH_MAX_PENDING:
            _, cx, cz = heapq.heappop(self._waiting)
            k = (cx, cz)
            self._waiting_set.discard(k)
            if k in self._ring_set:
                self._refresh(world, k)

    def _draw_sky(self, vp, eye):
        """The sky: one full-screen triangle behind everything, with gradient, sun glow and clouds."""
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(self.sky_prog)
        gl.glUniformMatrix4fv(self.u_sky_inv, 1, gl.GL_TRUE, np.linalg.inv(vp.astype(np.float64)).astype(np.float32))
        gl.glUniform3f(self.u_sky_eye, float(eye[0]), float(eye[1]), float(eye[2]))
        gl.glUniform1f(self.u_sky_time, float(time.time() % 100000.0))
        gl.glBindVertexArray(self.sky_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def draw(self, world, eye, yaw, pitch, radius, fov, ambient=1.0):
        eye = np.asarray(eye, dtype=np.float32)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        vp = _perspective(fov, WIDTH / HEIGHT, NEAR, FAR) @ _view(eye, yaw, pitch)
        self.last_vp = vp     # kept for the debug chunk grid, which is drawn over this frame
        self._draw_sky(vp, eye)
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniformMatrix4fv(self.u_lvp, 1, gl.GL_TRUE, self.light_vp.astype(np.float32))
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.shadow_tex)
        gl.glUniform1i(self.u_shadow, 1)
        gl.glUniform1f(self.u_stexel, self.shadow_texel)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tile_texture())
        gl.glUniform1i(self.u_tiles, 2)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1f(self.u_light, max(float(ambient), GFX['ambient_floor']))
        gl.glUniform3f(self.u_eye, float(eye[0]), float(eye[1]), float(eye[2]))
        gl.glUniform1f(self.u_grid, 1.0 if self.grid else 0.0)
        gl.glUniform1f(self.u_cell, float(CELL))

        pcx = math.floor(eye[0] / CHUNK_SIZE)
        pcz = math.floor(eye[2] / CHUNK_SIZE)
        # Finished meshes from the worker threads go up first. Then each chunk is checked: a chunk
        # that needs a new mesh is queued for a worker, and its old mesh keeps drawing until the new
        # one arrives. At most MESH_MAX_PENDING chunks are queued at once.
        self.mesh_s = self.upload_s = self.list_s = self.call_s = 0.0
        t_col = time.perf_counter()
        completed = self.mesher.collect(world)
        self.collect_s = time.perf_counter() - t_col
        for k, ver, payload, step in completed:
            t_c = time.perf_counter()
            self._upload(k, payload, step)
            self.upload_s += time.perf_counter() - t_c
            if world.dirty.ver.get(k, 0) == ver:      # no edit happened while it was being built
                CACHE.put_mesh(k, LOD_SIG[step], ver, payload)
                world.dirty.discard(k)
            if step != self._want_step(k):            # the player moved on while it was building: build it at the right detail
                self._touched.add(k)

        # Update the draw list incrementally: only chunks that changed since last frame are looked at.
        t_list = time.perf_counter()
        mesh0, up0 = self.mesh_s, self.upload_s
        if (pcx, pcz, radius) != self._ring_key:
            self._change_ring(pcx, pcz, radius)
        if (pcx, pcz, radius, self.lod) != self._lod_key:
            # The player moved to another chunk, or the detail level changed: chunks whose mesh is at the wrong
            # step are looked at again (their old mesh keeps drawing until the new one arrives).
            self._lod_key = (pcx, pcz, radius, self.lod)
            for k in self._ring_set:
                st = self.step_of.get(k)
                if st is not None and st != self._want_step(k):
                    self._touched.add(k)
        self._touched |= world.touched
        world.touched.clear()
        # Removals come first: a chunk that left the view frees its slot before any chunk
        # entering the view can take one, so the table never needs more slots than the view has.
        for k in [k for k in self._touched if k not in self._ring_set]:
            self._touched.discard(k)
            self._refresh(world, k)
        if completed:
            self._promote_waiting(world)
        while self._touched:
            self._refresh(world, self._touched.pop())
        # The list time leaves out the meshing and uploading it triggered, which have their own timers.
        self.list_s = (time.perf_counter() - t_list) - (self.mesh_s - mesh0) - (self.upload_s - up0)

        # Every visible chunk is one range of the shared buffer, so one call draws them all.
        n = self._n
        t_call = time.perf_counter()
        if n:
            gl.glBindVertexArray(self.vao)
            gl.glMultiDrawArrays(gl.GL_TRIANGLES, self._first[:n], self._count[:n], n)
        gl.glBindVertexArray(0)
        self.call_s = time.perf_counter() - t_call


# ═══════════════════════════════════════════════
#  GPU & FRAME STATS
# ═══════════════════════════════════════════════
class GpuPoller:
    """Polls nvidia-smi on a background thread so it never stalls a frame.
    `stats` is None until the first reading, or if nvidia-smi isn't available."""
    QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem,fan.speed"

    def __init__(self, interval=0.5):
        self.interval = interval
        self.stats = None
        self.available = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _num(s):
        try:
            return float(s)
        except ValueError:
            return None   # nvidia-smi prints [N/A] for unsupported fields

    def _run(self):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # no console flash on Windows
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, creationflags=flags)
                if out.returncode != 0 or not out.stdout.strip():
                    raise RuntimeError("nvidia-smi returned no data")
                f = [self._num(v.strip()) for v in out.stdout.strip().splitlines()[0].split(",")]
                f = (f + [None] * 8)[:8]
                self.stats = {'util': f[0], 'mem_used': f[1], 'mem_total': f[2],
                              'temp': f[3], 'power': f[4],
                              'clock_sm': f[5], 'clock_mem': f[6], 'fan': f[7]}
            except Exception:
                self.available = False
                self.stats = None
                return
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

DEBRIS_AMOUNT = 100   # percent of debris made and kept: fixed, no longer a menu setting
WOBBLE_PCT = 50      # loose-piece wobble: fixed at the original 50%, no longer a menu setting
STRENGTH_PCT = 100   # material strength: fixed at 100%, no longer a menu setting
STRUCT_RES = 2       # cells per block along each side of a test cube: fixed at 2, no longer a menu setting
STRUCT_SHARDS = True   # broken cells crumble off as loose cubes (capped per hit, see _throw_chunks)
STRUCT_DEBRIS_SPLIT = 1       # 1: a cell destroyed by a blast is thrown as exactly one cube of its own size. 2: the old way, eight half-size cubes per cell
def unload_far_chunks(world, renderer, pcx, pcz, keep):
    """Unload chunks more than `keep` chunks from the player (used when caching is off)."""
    far = [k for k in world.chunks if max(abs(k[0] - pcx), abs(k[1] - pcz)) > keep]
    for key in far:
        world.unload_chunk(key)
        renderer.evict(key)
        with CACHE.lock:
            old = CACHE.mesh.pop(key, None)
            if old:
                CACHE.mesh_bytes -= old[3]

# ═══════════════════════════════════════════════
#  PLAYER PHYSICS (AABB vs voxels, gravity, jump, fly toggle)
# ═══════════════════════════════════════════════
def _density_at(world, x, y, z):
    """+1 where the ground has a cube at this position, -1 where it is air (unloaded counts as air)."""
    v = world.cell(math.floor(x / CELL), math.floor(y / CELL), math.floor(z / CELL))
    return -1.0 if v is None else float(v)

def _box_hits_solid(world, x, y, z):
    """True if the player's body at feet position (x, y, z) is inside the solid surface.
    Samples a ring of points at the feet and at the head, so slopes and crater edges are felt."""
    r, h = PLAYER_RADIUS, PLAYER_HEIGHT
    ring = ((-r, -r), (-r, 0.0), (-r, r), (0.0, -r), (0.0, 0.0), (0.0, r), (r, -r), (r, 0.0), (r, r))
    for y_s in (y + 0.002, y + 0.55, y + 1.05, y + h - 0.05):
        for dx, dz in ring:
            if _density_at(world, x + dx, y_s, z + dz) > 0.0:
                return True
    return STRUCTS.hits_box(x, y, z, r, h)          # structures and loose bodies are solid too

# ═══════════════════════════════════════════════
#  SOUND EFFECTS. Every sound is built in numpy from a tone-and-noise recipe (the same recipes as the Block Pad
#  sound board), rendered once at start-up into pygame Sounds, and played by name from the places in the game where
#  things happen. No sound files. With no audio device everything here quietly does nothing.
#  Sounds that happen somewhere in the world are placed: quieter with distance, panned left / right.
# ═══════════════════════════════════════════════
import random as _random
try:
    from scipy.signal import lfilter as _sp_lfilter
except Exception:
    _sp_lfilter = None

SFX_VOLUME = 0.6            # master level, 0..1
SFX_HEARD = 70.0            # a placed sound further than this from the player is not played
SFX_FALLOFF = 9.0           # distance at which a placed sound is down to half volume


def _blep(t, dt):
    """PolyBLEP correction that smooths the jump in a square or saw wave, so high notes do not alias."""
    out = np.zeros_like(t)
    lo = t < dt
    x = t[lo] / dt[lo]
    out[lo] = x + x - x * x - 1.0
    hi = t > 1.0 - dt
    x = (t[hi] - 1.0) / dt[hi]
    out[hi] = x * x + x + x + 1.0
    return out


def _biquad(kind, f, q, sr):
    """Coefficients of a Web Audio style biquad. Low / high pass take Q in dB, band pass takes it linear."""
    f = min(max(float(f), 10.0), sr * 0.45)
    w0 = 2.0 * math.pi * f / sr
    cw = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * (10.0 ** (q / 20.0) if kind != 'bandpass' else q))
    if kind == 'lowpass':
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
    elif kind == 'highpass':
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    else:
        b = [alpha, 0.0, -alpha]
    a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def _run_biquad(b, a, x, z):
    if _sp_lfilter is not None:
        return _sp_lfilter(b, a, x, zi=z)
    b0, b1, b2, a1, a2 = b[0], b[1], b[2], a[1], a[2]
    z1, z2 = z
    y = []
    for xv in x.tolist():
        yv = b0 * xv + z1
        z1 = b1 * xv - a1 * yv + z2
        z2 = b2 * xv - a2 * yv
        y.append(yv)
    return np.array(y), (z1, z2)


def _filtered(x, kind, f, f2, dur, q, sr):
    """Filter x. With f2 the cutoff sweeps exponentially from f to f2 over dur seconds, then holds."""
    if not f2 or abs(f2 - f) < 1e-6:
        b, a = _biquad(kind, f, q, sr)
        return _run_biquad(b, a, x, np.zeros(2))[0]
    out = np.empty_like(x)
    z = np.zeros(2)
    block = 256
    for i0 in range(0, len(x), block):
        frac = min(((i0 + block / 2) / sr) / dur, 1.0)
        b, a = _biquad(kind, f * (f2 / f) ** frac, q, sr)
        out[i0:i0 + block], z = _run_biquad(b, a, x[i0:i0 + block], z)
    return out


class _Synth:
    """Collects the tones and noise bursts of one sound. T and N take the same options as the Block Pad's."""

    def __init__(self, sr, rng):
        self.sr, self.rng = sr, rng
        self.layers = []                 # (start sample, samples)
        self.vs, self.ds = 1.0, 0.0      # volume scale and delay of the sound being layered in
        self.random = False              # True once the recipe used rnd(): that sound is worth several variants

    def rnd(self, a, b):
        self.random = True
        return a + (b - a) * self.rng.random()

    def layer(self, name, vol=1.0, delay=0.0):
        """Mix another recipe into this one."""
        old = (self.vs, self.ds)
        self.vs *= vol
        self.ds += delay
        SFX_RECIPES[name](self)
        self.vs, self.ds = old

    def T(self, type='square', f=440.0, f2=None, dur=0.2, vol=0.3, a=0.005, delay=0.0, lp=None, vib=None, vibDepth=20.0):
        sr = self.sr
        n = int((dur + 0.05) * sr)
        t = np.arange(n) / sr
        freq = f * (f2 / f) ** (np.minimum(t, dur) / dur) if f2 else np.full(n, float(f))
        if vib:
            freq = freq + vibDepth * np.sin(2 * math.pi * vib * t)
        freq = np.maximum(freq, 1.0)
        ph = (np.cumsum(freq) / sr) % 1.0
        dt = np.minimum(freq / sr, 0.49)
        if type == 'sine':
            w = np.sin(2 * math.pi * ph)
        elif type == 'triangle':
            w = 2.0 * np.abs(2.0 * ph - 1.0) - 1.0
        elif type == 'sawtooth':
            w = 2.0 * ph - 1.0 - _blep(ph, dt)
        else:
            w = np.where(ph < 0.5, 1.0, -1.0) + _blep(ph, dt) - _blep((ph + 0.5) % 1.0, dt)
        env = np.zeros(n)
        m1 = t < a
        env[m1] = 0.0001 + (vol - 0.0001) * t[m1] / a
        m2 = (t >= a) & (t < dur)
        env[m2] = vol * (0.0001 / vol) ** ((t[m2] - a) / max(dur - a, 1e-4))
        y = w * env
        if lp:
            y = _filtered(y, 'lowpass', lp, None, dur, 1.0, sr)
        self.layers.append((int((delay + self.ds) * sr), y * self.vs))

    def N(self, type='lowpass', f=1000.0, f2=None, dur=0.2, vol=0.3, a=0.003, delay=0.0, q=1.0, swell=False):
        sr = self.sr
        n = int((dur + 0.05) * sr)
        t = np.arange(n) / sr
        x = _filtered(self.noise(n), type, f, f2, dur, q, sr)
        env = np.zeros(n)
        m1 = t < a
        env[m1] = 0.0001 + (vol - 0.0001) * t[m1] / a
        if swell:
            m2 = (t >= a) & (t < dur * 0.8)
            env[m2] = vol
            m3 = (t >= dur * 0.8) & (t < dur)
            env[m3] = vol * (1.0 - (t[m3] - dur * 0.8) / (dur * 0.2))
        else:
            m2 = (t >= a) & (t < dur)
            env[m2] = vol * (0.0001 / vol) ** ((t[m2] - a) / max(dur - a, 1e-4))
        self.layers.append((int((delay + self.ds) * sr), x * env * self.vs))

    def noise(self, n):
        return self.np_rng.uniform(-1.0, 1.0, n)

    def mix(self):
        end = max((s + len(y) for s, y in self.layers), default=1)
        buf = np.zeros(end)
        for s, y in self.layers:
            buf[s:s + len(y)] += y
        return buf


def _note_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# Force 1.0 .. 5.0 in half steps is nine pling notes climbing a major scale from middle C.
PLING_SCALE = (0, 2, 4, 5, 7, 9, 11, 12, 14)


def _pling(deg):
    def r(s):
        f = _note_hz(60 + deg)
        s.T(type='square', f=f, dur=.35, vol=.15, lp=2500)
        s.T(type='sine', f=f * 2, dur=.3, vol=.2)
    return r


SFX_RECIPES = {
    # blocks
    'break_stone': lambda s: (s.N(f=2500, f2=400, dur=.28, vol=.5), s.N(type='highpass', f=3000, dur=.08, vol=.3),
                              s.T(type='square', f=140, f2=50, dur=.15, vol=.2)),
    'break_wood': lambda s: (s.T(type='triangle', f=260, f2=90, dur=.14, vol=.5), s.N(type='bandpass', f=900, dur=.15, vol=.4, q=2),
                             s.T(type='triangle', f=180, f2=70, dur=.12, vol=.3, delay=.07)),
    'break_gravel': lambda s: ([s.N(type='highpass', f=s.rnd(1200, 2800), dur=.06, vol=.3, delay=i * s.rnd(.02, .04)) for i in range(9)],
                               s.N(f=900, dur=.3, vol=.3)),
    'break_grass': lambda s: (s.N(type='bandpass', f=1800, f2=900, dur=.18, vol=.5, q=.8),
                              s.N(type='highpass', f=4000, dur=.1, vol=.2, delay=.03)),
    'place_block': lambda s: (s.T(type='sine', f=160, f2=60, dur=.12, vol=.7), s.N(f=700, f2=200, dur=.1, vol=.4)),
    'pickaxe_tink': lambda s: (s.T(type='square', f=1900, f2=1400, dur=.05, vol=.2), s.T(type='triangle', f=900, f2=700, dur=.08, vol=.25),
                               s.N(type='highpass', f=4000, dur=.04, vol=.3)),
    'glass_break': lambda s: (s.N(type='highpass', f=3500, dur=.35, vol=.5),
                              [s.T(type='sine', f=s.rnd(2500, 6000), dur=.12, vol=.1, delay=s.rnd(0, .3)) for _ in range(10)]),
    'anvil': lambda s: (s.T(type='square', f=700, dur=.5, vol=.25), s.T(type='square', f=1130, dur=.4, vol=.2),
                        s.T(type='triangle', f=1870, dur=.3, vol=.15), s.N(type='highpass', f=3000, dur=.05, vol=.4)),
    # steps and the player
    'step_grass': lambda s: s.N(type='bandpass', f=s.rnd(1500, 2200), dur=.1, vol=.45, q=.7),
    'step_sand': lambda s: s.N(type='bandpass', f=3000, f2=1500, dur=.14, vol=.35, q=.5),
    'step_stone': lambda s: (s.N(f=1400, f2=300, dur=.09, vol=.5), s.T(type='sine', f=110, f2=60, dur=.08, vol=.3)),
    'jump': lambda s: (s.T(type='sine', f=180, f2=320, dur=.08, vol=.2), s.N(type='bandpass', f=1500, dur=.06, vol=.12)),
    'fall_damage': lambda s: (s.T(type='sine', f=130, f2=40, dur=.25, vol=.7), s.N(f=1200, f2=150, dur=.2, vol=.45)),
    # loot and interface
    'level_up': lambda s: ([s.T(type='triangle', f=f, dur=.3, vol=.28, delay=i * .09) for i, f in enumerate((523, 659, 784, 1047, 1319))],
                           s.T(type='sine', f=2093, dur=.6, vol=.1, delay=.45)),
    'item_pickup': lambda s: (s.T(type='sine', f=500, f2=1000, dur=.1, vol=.3), s.T(type='sine', f=700, f2=1400, dur=.08, vol=.15, delay=.03)),
    'chest_open': lambda s: (s.T(type='sawtooth', f=90, f2=260, dur=.45, vol=.25, lp=700, vib=14, vibDepth=20),
                             s.N(type='bandpass', f=1500, dur=.3, vol=.12, delay=.1), s.T(type='sine', f=100, f2=50, dur=.1, vol=.5, delay=.45)),
    'chest_close': lambda s: (s.T(type='sawtooth', f=240, f2=90, dur=.3, vol=.2, lp=700), s.T(type='sine', f=130, f2=45, dur=.12, vol=.7, delay=.28),
                              s.N(f=900, dur=.08, vol=.3, delay=.28)),
    'armor_equip': lambda s: (s.N(type='highpass', f=3000, dur=.1, vol=.3), s.T(type='square', f=1200, f2=900, dur=.07, vol=.12),
                              s.T(type='triangle', f=600, f2=400, dur=.1, vol=.2, delay=.04)),
    'totem': lambda s: ([s.T(type='sine', f=s.rnd(900, 3000), dur=.4, vol=.08, delay=i * .05) for i in range(14)],
                        [s.T(type='triangle', f=f, dur=.9, vol=.2, delay=.2 + i * .04) for i, f in enumerate((392, 494, 587, 784))]),
    'ui_click': lambda s: (s.T(type='square', f=900, f2=600, dur=.04, vol=.18), s.N(type='highpass', f=4000, dur=.02, vol=.2)),
    'door_open': lambda s: (s.T(type='sawtooth', f=110, f2=200, dur=.25, vol=.2, lp=600, vib=20, vibDepth=20),
                            s.T(type='triangle', f=180, f2=90, dur=.08, vol=.3, delay=.22)),
    'door_close': lambda s: (s.T(type='sine', f=160, f2=60, dur=.1, vol=.7), s.N(f=900, f2=200, dur=.1, vol=.45),
                             s.T(type='triangle', f=300, f2=150, dur=.05, vol=.2, delay=.04)),
    'bucket_fill': lambda s: (s.N(type='bandpass', f=600, f2=1800, dur=.45, vol=.4, q=1), s.T(type='sine', f=300, f2=700, dur=.4, vol=.15, vib=12, vibDepth=30)),
    'bell': lambda s: [s.T(type='sine', f=f, dur=2.2 - i * .5, vol=.28 / (i + 1)) for i, f in enumerate((880, 1760, 2637))],
    'ender_pearl': lambda s: (s.T(type='sine', f=300, f2=1400, dur=.3, vol=.3), s.N(type='bandpass', f=2000, f2=800, dur=.25, vol=.25, delay=.15, q=2),
                              s.T(type='triangle', f=1400, f2=400, dur=.2, vol=.2, delay=.25)),
    # world and weapons
    'explosion': lambda s: (s.N(f=2000, f2=60, dur=1.6, vol=.9), s.T(type='sine', f=90, f2=25, dur=1.2, vol=.9), s.N(f=300, f2=40, dur=2, vol=.5, delay=.1)),
    'thunder': lambda s: (s.N(f=500, f2=80, dur=3, vol=1, a=.05), s.N(f=200, f2=40, dur=3, vol=.6, a=.3, delay=.2)),
    'lava_pop': lambda s: (s.T(type='sine', f=150, f2=550, dur=.1, vol=.45), s.N(type='bandpass', f=900, dur=.1, vol=.3, q=2),
                           s.T(type='sine', f=200, f2=650, dur=.08, vol=.3, delay=.15)),
    'bow_shot': lambda s: (s.T(type='triangle', f=260, f2=90, dur=.12, vol=.5), s.N(type='bandpass', f=2500, f2=600, dur=.22, vol=.3, q=1.5),
                           s.T(type='sine', f=1200, f2=2200, dur=.2, vol=.05, delay=.05)),
    'arrow_hit': lambda s: (s.T(type='square', f=420, f2=160, dur=.06, vol=.3), s.T(type='triangle', f=240, f2=120, dur=.1, vol=.4),
                            s.N(type='highpass', f=3000, dur=.04, vol=.2)),
    # the universal block hit
    'block_hit': lambda s: (s.T(type='sine', f=150, f2=55, dur=.14, vol=.6), s.N(type='bandpass', f=1400, f2=500, dur=.09, vol=.45, q=.9),
                            s.N(f=2200, f2=300, dur=.07, vol=.25), s.T(type='triangle', f=620, f2=320, dur=.05, vol=.12)),
}
# Sounds made by mixing others, or the high part of one.
SFX_RECIPES['axe_swing'] = lambda s: (s.N(type='bandpass', f=2500, f2=600, dur=.22, vol=.42, q=1.5),
                                      s.T(type='sine', f=1200, f2=2200, dur=.2, vol=.07, delay=.05))
SFX_RECIPES['gun_shot'] = lambda s: (s.N(type='highpass', f=1800, f2=500, dur=.09, vol=.5),
                                     s.T(type='square', f=260, f2=70, dur=.08, vol=.3),
                                     s.N(type='bandpass', f=700, f2=200, dur=.16, vol=.25, q=.8, delay=.01))
SFX_RECIPES['collapse'] = lambda s: (s.layer('explosion', 1.0), s.layer('anvil', 0.6, delay=0.12), s.layer('thunder', 0.9, delay=0.08))
SFX_RECIPES['body_land'] = lambda s: (s.layer('fall_damage', 1.0), s.layer('break_stone', 0.8, delay=0.02))
for _i, _d in enumerate(PLING_SCALE):
    SFX_RECIPES['pling_%d' % _i] = _pling(_d)

SFX_LOOPS = ('laser',)      # sounds rendered as seamless loops (see SoundFX._render_loop)


class SoundFX:
    """Plays the game's sounds by name. All methods are safe to call with no audio: they do nothing."""
    GAP = {'step_grass': 0.1, 'step_sand': 0.1, 'step_stone': 0.07, 'break_gravel': 0.05, 'pickaxe_tink': 0.05,
           'break_stone': 0.06, 'break_wood': 0.06, 'glass_break': 0.08, 'body_land': 0.12, 'collapse': 1.2,
           'lava_pop': 0.0, 'ui_click': 0.02}
    VARIANTS = 4            # how many renders of a sound that uses random numbers

    def __init__(self):
        self.ok = False
        self.error = None
        self.bank = {}
        self.last = {}
        self.pending = []
        self.loops = {}
        self.eye = None
        self.yaw = 0.0
        self.sr = 44100
        self.ch = 2

    # ---- set-up ---------------------------------------------------------------------------------------
    def init(self, progress=None):
        """Open the mixer and render every sound. progress(text) is called now and then (the loading screen)."""
        self.ok = False
        self.error = None
        self.bank.clear()
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(44100, -16, 2, 512)
            self.sr, fmt, self.ch = pygame.mixer.get_init()
            if fmt != -16 or self.ch not in (1, 2):
                pygame.mixer.quit()
                pygame.mixer.init(44100, -16, 2, 512, allowedchanges=0)
                self.sr, fmt, self.ch = pygame.mixer.get_init()
            if fmt != -16:
                raise RuntimeError("mixer format %s is not 16-bit" % fmt)
            pygame.mixer.set_num_channels(32)
        except Exception as ex:
            self.error = "audio mixer unavailable: %s: %s" % (type(ex).__name__, ex)
            print("Sound: off (%s)" % self.error)
            return
        rng = _random.Random(1234)
        names = list(SFX_RECIPES)
        try:
            for k, name in enumerate(names):
                if progress and k % 10 == 0:
                    progress("%d of %d" % (k, len(names)))
                self.bank[name] = self._render(name, rng)
            for name in SFX_LOOPS:
                self.bank[name] = [self._render_loop(name, rng)]
            self.ok = True
            print("Sound: on (%d effects, %d Hz, %d channel%s)" %
                  (len(self.bank), self.sr, self.ch, "s" if self.ch != 1 else ""))
        except Exception as ex:
            self.bank.clear()
            self.error = "sound rendering failed: %s: %s" % (type(ex).__name__, ex)
            print("Sound: off (%s)" % self.error)

    def _synth(self, rng):
        s = _Synth(self.sr, rng)
        s.np_rng = np.random.default_rng(rng.randrange(1 << 30))
        return s

    def _sound(self, buf):
        y = buf * SFX_VOLUME
        pk = float(np.abs(y).max())
        if 1e-6 < pk < 0.8:
            y = y * (0.8 / pk) ** 0.7                        # quiet recipes (a footstep, thunder) are lifted part of the way
        y = np.tanh(y)
        a = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
        if self.ch == 2:
            a = np.ascontiguousarray(np.column_stack([a, a]))
        return pygame.sndarray.make_sound(a)

    def _render(self, name, rng):
        out = []
        for v in range(self.VARIANTS):
            s = self._synth(rng)
            SFX_RECIPES[name](s)
            out.append(self._sound(s.mix()))
            if not s.random:
                break                                        # nothing random in it: one render is enough
        return out

    def _render_loop(self, name, rng):
        """The laser: a Nether-portal style hum (steady tones with a slow wobble) and a bed of fire crackle. Every
        part repeats exactly over the loop, and the ends are crossfaded, so it can run for as long as the beam does."""
        sr = self.sr
        L = int(2.0 * sr)
        X = int(0.25 * sr)
        t = np.arange(L + X) / sr
        s = self._synth(rng)
        hum = np.zeros(L + X)
        for f, vol, vib, depth in ((110.0, .30, 3.0, 25.0), (222.0, .15, 4.0, 30.0)):
            ph = 2 * math.pi * (f * t - depth / (2 * math.pi * vib) * np.cos(2 * math.pi * vib * t))
            hum += vol * np.sin(ph)
        bed = _filtered(s.noise(L + X), 'bandpass', 700.0, None, 1.0, 2.0, sr) * (0.5 + 0.5 * np.sin(2 * math.pi * 0.5 * t)) * 0.5
        s.layers.append((0, hum + bed))
        for _ in range(26):
            s.N(type='highpass', f=rng.uniform(2000, 5000), dur=.03, vol=.28, delay=rng.uniform(0, 2.0))
        buf = s.mix()[:L + X]
        buf = np.pad(buf, (0, max(0, L + X - len(buf))))
        w = np.linspace(0.0, 1.0, X)
        out = buf[:L].copy()
        out[:X] = buf[:X] * w + buf[L:L + X] * (1.0 - w)
        return self._sound(out * 0.55)

    # ---- playing --------------------------------------------------------------------------------------
    def set_listener(self, eye, yaw):
        self.eye = np.asarray(eye, np.float64)
        self.yaw = float(yaw)

    def play(self, name, vol=1.0, pos=None, gap=None, delay=0.0):
        """Play a sound. pos = where it happens in the world (None = on the player). gap = least seconds since the
        last play of this same sound. delay = seconds from now."""
        if not self.ok:
            return None
        now = time.perf_counter()
        if delay > 0.0:
            self.pending.append((now + delay, name, vol, pos, gap))
            return None
        g = self.GAP.get(name, 0.03) if gap is None else gap
        if now - self.last.get(name, -9.0) < g:
            return None
        variants = self.bank.get(name)
        if not variants:
            if name not in self.bank and self.error is None:
                logging.warning("SFX name %r is missing from the sound bank", name)
            return None
        left = right = 1.0
        if pos is not None and self.eye is not None:
            d = np.asarray(pos, np.float64) - self.eye
            dist = float(np.linalg.norm(d))
            if dist > SFX_HEARD:
                return None
            vol *= 1.0 / (1.0 + (dist / SFX_FALLOFF) ** 2)
            flat = math.hypot(d[0], d[2])
            pan = 0.0 if flat < 0.5 else (d[0] * math.cos(self.yaw) + d[2] * math.sin(self.yaw)) / flat
            ang = (pan + 1.0) * math.pi / 4.0
            left, right = math.cos(ang) * 1.414, math.sin(ang) * 1.414
        vol = min(vol, 1.5)
        if vol < 0.01:
            return None
        self.last[name] = now
        ch = _random.choice(variants).play()
        if ch is not None:
            ch.set_volume(min(1.0, vol * left), min(1.0, vol * right))
        return ch

    def tick(self):
        """Start the sounds that were asked for with a delay. Call once a frame."""
        if not self.pending:
            return
        now = time.perf_counter()
        due = [p for p in self.pending if p[0] <= now]
        if due:
            self.pending = [p for p in self.pending if p[0] > now]
            for _, name, vol, pos, gap in due:
                self.play(name, vol, pos, gap)

    # ---- things that happen in the game ---------------------------------------------------------------------
    def step(self, sprint=False):
        self.play(_random.choice(('step_grass', 'step_grass', 'step_sand')), 0.55 if not sprint else 0.65)

    def land(self, speed):
        """The player came down at this speed (blocks per second)."""
        if speed > 14.0:
            self.play('fall_damage', min(1.0, 0.5 + (speed - 14.0) * 0.04), gap=0.15)
        elif speed > 5.0:
            self.play('place_block', min(0.9, 0.3 + (speed - 5.0) * 0.07), gap=0.15)

    def axe_hit(self, mat, pos):
        """The axe struck something: mat is the structure material hit, or None for terrain (dirt)."""
        d = 0.09                                             # just after the swing
        if mat is None or mat == DIRT:
            self.play(_random.choice(('break_gravel', 'break_grass')), 0.9, pos, delay=d)
        elif mat == PLANKS:
            self.play('break_wood', 0.9, pos, delay=d)
        elif mat == BRICK:
            self.play('pickaxe_tink', 0.9, pos, delay=d)
        elif mat == STEEL:
            self.play('pickaxe_tink', 0.9, pos, delay=d)
            self.play('anvil', 0.5, pos, delay=d)
        elif mat in (STONE, CONCRETE_WHITE):
            self.play('break_stone', 0.9, pos, delay=d)
        else:
            self.play('block_hit', 0.9, pos, delay=d)

    def crack(self, pos):
        """A bond in a structure gave way."""
        self.play(_random.choice(('pickaxe_tink', 'arrow_hit')), 0.35, pos, gap=0.09)

    def collapse(self, pos, volume):
        """A piece of a structure broke away; volume = how many blocks of it."""
        if volume >= 1.5:
            self.play('collapse', min(1.0, 0.35 + 0.06 * volume), pos)

    def body_land(self, pos, speed, volume):
        self.play('body_land', min(1.0, (0.15 + 0.07 * speed) * (0.5 + min(volume, 4.0) * 0.15)), pos)

    def crumble(self, pos):
        self.play('break_gravel', 0.6, pos, gap=0.06)

    def settle(self, pos, count):
        """Dirt cubes came to rest. Quiet, and only some of them are heard."""
        if _random.random() < 0.45:
            self.play(_random.choice(('step_sand', 'break_gravel')), min(0.3, 0.1 + 0.01 * count), pos, gap=0.12)

    def bounce(self, pos, speed):
        self.play('step_stone', min(1.0, 0.2 + speed * 0.05), pos, gap=0.05)


SFX = SoundFX()


def _mat_at(o, p, d=None):
    """Material of the cell of structure / body o at world point p (nudged a little along d, to be inside it)."""
    try:
        q = np.asarray(p, np.float64)
        if d is not None:
            q = q + np.asarray(d, np.float64) * 0.05 * o.cs
        ij = np.floor(((q - o.frame_pos) @ o.R - o.corner) / o.cs).astype(np.int64)
        ij = np.clip(ij, 0, np.array(o.alive.shape) - 1)
        return int(o.mat[tuple(ij)])
    except Exception:
        return None


class PlayerController:
    EPS = 1e-4

    def __init__(self, feet_pos):
        self.pos = np.array(feet_pos, dtype=np.float64)   # feet position
        self.vel = np.zeros(3)
        self.on_ground = False
        self.flying = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.stride = 0.0         # distance walked since the last footstep sound

    @property
    def eye(self):
        return self.pos + np.array([0.0, PLAYER_HEIGHT * 0.9, 0.0])

    def forward(self):
        cp = math.cos(self.pitch)
        return np.array([cp * math.sin(self.yaw), math.sin(self.pitch), -cp * math.cos(self.yaw)])

    def toggle_fly(self):
        self.flying = not self.flying
        self.vel[:] = 0.0
        SFX.play('ender_pearl' if self.flying else 'item_pickup', gap=0.1)

    def update(self, world, dt, wish, jump, sprint):
        """wish: desired direction (3-vector, length <= 1). Y is only used when flying."""
        dt = min(dt, 0.05)   # avoid tunnelling after a long frame
        mult = SPRINT_MULT if sprint else 1.0
        if self.flying:
            self.pos += wish * FLY_SPEED * mult * dt
            return

        self.vel[0] = wish[0] * MOVE_SPEED * mult
        self.vel[2] = wish[2] * MOVE_SPEED * mult
        if True:
            if jump and self.on_ground:
                self.vel[1] = JUMP_VEL
                SFX.play('jump', gap=0.25)
            self.vel[1] = max(self.vel[1] + GRAVITY * dt, -60.0)
        fall_speed = -float(self.vel[1])             # how fast it is coming down, for the landing sound

        was_grounded = self.on_ground
        self.on_ground = False
        dx, dz = self.vel[0] * dt, self.vel[2] * dt
        x0, z0 = self.pos[0], self.pos[2]
        self._move_axis(world, 0, dx)
        self._move_axis(world, 2, dz)
        wanted = dx * dx + dz * dz
        moved = (self.pos[0] - x0) ** 2 + (self.pos[2] - z0) ** 2
        if was_grounded and wanted > 1e-10 and moved < 0.5 * wanted:
            self._step_up(world, dx, dz)          # blocked: try to climb onto a step or slope
        self._move_axis(world, 1, self.vel[1] * dt)
        if self.on_ground:
            if not was_grounded:
                SFX.land(fall_speed)
            self.stride += math.hypot(self.pos[0] - x0, self.pos[2] - z0)
            stride_len = 2.6 if sprint else 2.0     # sprinting covers more ground per step, but the steps still come faster
            if self.stride >= stride_len:
                self.stride -= stride_len
                SFX.step(sprint)

    def _move_axis(self, world, axis, delta):
        """Move along one axis. If that puts the body inside the surface, back off to the closest
        free position along this axis, so the player slides along slopes and walls."""
        if delta == 0.0:
            return
        old = self.pos[axis]
        self.pos[axis] = old + delta
        if not _box_hits_solid(world, *self.pos):
            return
        free, blocked = old, old + delta
        for _ in range(12):                       # bisect to within a few thousandths of a block
            mid = 0.5 * (free + blocked)
            self.pos[axis] = mid
            if _box_hits_solid(world, *self.pos):
                blocked = mid
            else:
                free = mid
        self.pos[axis] = free
        if axis == 1:
            if delta < 0:
                self.on_ground = True             # landed on a surface
            self.vel[1] = 0.0

    def _step_up(self, world, dx, dz):
        """A horizontal move was blocked while on the ground. If there's a step or gentle slope
        ahead that's no higher than STEP_HEIGHT, climb onto it, then drop back to the surface."""
        base = self.pos.copy()
        self.pos[1] += STEP_HEIGHT
        if _box_hits_solid(world, *self.pos):     # no room above the step
            self.pos[:] = base
            return
        # Probe half a block ahead: a real step is no higher than STEP_HEIGHT there. Steep ground
        # (a cliff, smoothed into a ramp by the density field) is still too high, so it's refused.
        L = math.hypot(dx, dz)
        if L > 0.0:
            ahead = 0.5 / L
            if _box_hits_solid(world, self.pos[0] + dx * ahead, self.pos[1], self.pos[2] + dz * ahead):
                self.pos[:] = base
                return
        self.pos[0] += dx
        self.pos[2] += dz
        if _box_hits_solid(world, *self.pos):     # still blocked on top of it
            self.pos[:] = base
            return
        self._move_axis(world, 1, -(STEP_HEIGHT + 0.05))

# ═══════════════════════════════════════════════
#  MAIN LOOP (WASD + mouse, Space jump, F pause, V fly, 2-5 hotbar, LMB break, RMB place, Tab menu)
# ═══════════════════════════════════════════════
def _fit_window_to_display():
    """Shrink the window to fit the primary monitor. On multi-monitor setups the
    desktop info can cover every screen, so use the first display's size explicitly."""
    global WIDTH, HEIGHT
    try:
        primary_w, primary_h = pygame.display.get_desktop_sizes()[0]
    except (AttributeError, IndexError):
        info = pygame.display.Info()
        primary_w, primary_h = info.current_w, info.current_h
    if primary_w and primary_h:
        # Leave margin for the taskbar and title bar.
        scale = min(1.0, primary_w * 0.90 / WIDTH, primary_h * 0.85 / HEIGHT)
        WIDTH, HEIGHT = int(WIDTH * scale), int(HEIGHT * scale)

_GRID_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uVP;
void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

_GRID_FRAG = """
#version 330 core
out vec4 fragColor;
uniform vec4 uCol;
void main() {
    fragColor = uCol;
}
"""

# ═══════════════════════════════════════════════
#  DEBRIS SETTINGS
# ═══════════════════════════════════════════════
CONFIG = {'debris_on': True, 'debris_amount': 1.0, 'dirt_per_block': 1}
_NEIGH_OFFS = np.array([-1099512676353, -1099512676352, -1099512676351, -1099511627777, -1099511627776, -1099511627775, -1099510579201, -1099510579200, -1099510579199, -1048577, -1048576, -1048575, -1, 0, 1, 1048575, 1048576, 1048577, 1099510579199, 1099510579200, 1099510579201, 1099511627775, 1099511627776, 1099511627777, 1099512676351, 1099512676352, 1099512676353], np.int64)   # the 27 cells around a cube's cell, as key offsets
_rng = np.random.default_rng()


# ═══════════════════════════════════════════════
#  DUG DIRT: each dug block splits into mini cubes that fly out and land
# ═══════════════════════════════════════════════
WASH_DELAY     = 2.0                 # seconds a cube lies under water before it starts to wash away
WASH_FADE      = 1.5                 # seconds it then takes to shrink away
BODY_MAX       = 12000               # cubes alive at once; the oldest go first
MAX_AWAKE_CUBES = 500               # cubes simulated per frame; the slowest beyond this are put to sleep
HARD_AWAKE_CUBES = 700              # most cubes simulated in one frame even in flight; beyond this they take turns (see update)
BODY_GRAVITY   = 14.0                # blocks per second squared
BODY_MAX_FALL  = 9.0                 # terminal speed, blocks per second
BODY_FRICTION  = 6.0                 # sideways slow-down
SEP_WAKE_DEPTH  = 0.06                # blocks: overlap with a neighbour deeper than this keeps a cube awake (0.02 left packed crater piles awake for good)
JAM_MOVE       = 0.10                # blocks per second: a cube that is flagged as perched off-centre, or pushed hard, but moves less than this is wedged by its neighbours, not toppling or sliding
PROGRESS_R     = 0.05                # blocks: a cube that stays within this distance of one spot...
PROGRESS_TIME  = 1.0                 # ...for this many seconds is at rest, however much it shakes (0 switches the rule off). A pile in a narrow hole can vibrate for ever: each cube travels about 0.9 blocks a second and gets nowhere (0.005 blocks)
JAM_TIME_MUL   = 3.0                 # a wedged cube must stay put this many times longer than the usual sleep time before it sleeps (0 for JAM_MOVE switches the rule off)

def _sample_many(world, X, Y, Z):
    """Ground cubes at integer cube indices, for arrays: +1 solid, -1 air. Below the world is solid, above it is air,
    and unloaded chunks are air."""
    vals = np.full(len(X), -1.0, np.float64)
    vals[Y < 0] = 1.0
    inside = (Y >= 0) & (Y < NH)
    cx = np.floor_divide(X, NC)
    cz = np.floor_divide(Z, NC)
    ckey = (cx + 100000) * 200000 + (cz + 100000)                # 1-D keys: much faster than unique(axis=0)
    keys = ckey[:1] if len(ckey) and (ckey == ckey[0]).all() else np.unique(ckey)   # most batches sit in one chunk
    for kk in keys.tolist():
        kx, kz = kk // 200000 - 100000, kk % 200000 - 100000
        d = world.dens.get((kx, kz))
        if d is None:
            continue
        m = inside & (cx == kx) & (cz == kz)
        if not m.any():
            continue
        vals[m] = d.sample(X[m] - kx * NC, Y[m], Z[m] - kz * NC)
    return vals

def _density_many(world, P):
    """Solid (+1) or air (-1) at many world points: the ground cube each point is inside."""
    if len(P) == 0:
        return np.zeros(0, np.float64)
    c = np.floor(np.asarray(P, np.float64) / CELL).astype(np.int64)
    return _sample_many(world, c[:, 0], c[:, 1], c[:, 2])

def _cell_contact(world, Q, reach=0.06):
    """Where points sit against the ground cubes. For a point inside a cube: how far it is from the nearest face that opens
    to air, and the direction out. For a point just outside: the (negative) gap to the nearest solid face within `reach`.
    Returns (depth (m,), normal (m, 3), ok (m,)); ok is False for points that are neither inside nor within reach."""
    Q = np.asarray(Q, np.float64)
    m = len(Q)
    c = np.floor(Q / CELL).astype(np.int64)
    dirs = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.int64)
    here = _sample_many(world, c[:, 0], c[:, 1], c[:, 2]) > 0
    nb = np.stack([_sample_many(world, c[:, 0] + d[0], c[:, 1] + d[1], c[:, 2] + d[2]) > 0 for d in dirs], axis=1)   # (m, 6)
    lo = c * CELL
    hi = lo + CELL
    # distance from the point to each of the six faces of its own cube, and to the face it would meet in each neighbour
    face = np.stack([hi[:, 0] - Q[:, 0], Q[:, 0] - lo[:, 0], hi[:, 1] - Q[:, 1], Q[:, 1] - lo[:, 1],
                     hi[:, 2] - Q[:, 2], Q[:, 2] - lo[:, 2]], axis=1)                       # (m, 6)
    inside_d = np.where(~nb, face, np.inf)                                                  # leaving through a face that is open
    k_in = np.argmin(inside_d, axis=1)
    d_in = inside_d[np.arange(m), k_in]
    out_d = np.where(nb, face, np.inf)                                                      # a solid neighbour: the gap to its face
    k_out = np.argmin(out_d, axis=1)
    d_out = out_d[np.arange(m), k_out]
    deep = here & ~np.isfinite(d_in)                                                        # boxed in on every side: push up
    k_in = np.where(deep, 2, k_in)
    d_in = np.where(deep, face[:, 2] + CELL, d_in)
    depth = np.where(here, d_in, -d_out)
    k = np.where(here, k_in, k_out)
    n = dirs[k].astype(np.float64)
    n = np.where(here[:, None], n, -n)                                                      # inside: out through the open face; outside: away from the solid
    ok = here | (d_out <= reach)
    return depth, n, ok

def _density_batch(world, point_sets):
    """Several _density_many calls in one batched sample. Returns one array per point set.
    Each density value is independent of the others, so the results are the same as separate calls."""
    sizes = [len(p) for p in point_sets]
    if sum(sizes) == 0:
        return [np.zeros(0, np.float64) for _ in point_sets]
    out = _density_many(world, np.concatenate(point_sets))
    return np.split(out, np.cumsum(sizes)[:-1])

def _corners_touching(world, P, size):
    """(m, 4) bool: which of each cube's four bottom corners sit inside terrain."""
    m = len(P)
    pts = np.repeat(P, 4, axis=0)
    off = np.array([[0.4, 0.4], [0.4, -0.4], [-0.4, 0.4], [-0.4, -0.4]])
    s_rep = np.repeat(size, 4)
    pts[:, 0] += np.tile(off[:, 0], m) * s_rep
    pts[:, 2] += np.tile(off[:, 1], m) * s_rep
    pts[:, 1] -= np.repeat(size, 4) * 0.5
    return (_density_many(world, pts) > 0.0).reshape(m, 4)

def _struct_corners(P, size):
    """(m, 4) bool: which bottom corners sit inside a fixed structure or a loose rigid body."""
    m = len(P)
    out = np.zeros((m, 4), bool)
    objs = STRUCTS.objects()
    if m == 0 or not objs:
        return out
    pts = np.repeat(P, 4, axis=0)
    off = np.array([[0.4, 0.4], [0.4, -0.4], [-0.4, 0.4], [-0.4, -0.4]])
    s_rep = np.repeat(size, 4)
    pts[:, 0] += np.tile(off[:, 0], m) * s_rep
    pts[:, 2] += np.tile(off[:, 1], m) * s_rep
    pts[:, 1] -= s_rep * 0.5
    flat = out.reshape(-1)                     # view: writes land in out
    for O in objs:
        lo, hi = O.aabb()
        near = np.nonzero(((pts >= lo - 0.05) & (pts <= hi + 0.05)).all(axis=1))[0]
        if not len(near):
            continue
        sel, _, _ = O.probe(pts[near])
        flat[near[sel]] = True
    return out

def _support(world, P, size):
    """(m, 4) bool: which bottom corners are on the ground, counting a corner that is within 0.05 above it."""
    Q = P.copy()
    Q[:, 1] -= 0.05
    return _corners_touching(world, Q, size) | _struct_corners(Q, size)

def _downhill(world, P, size):
    """Unit horizontal direction pointing down the slope under each cube (zero on flat ground)."""
    e = 0.3 * size
    yb = P[:, 1] - size * 0.5 - 0.05
    def at(dx, dz):
        return np.stack([P[:, 0] + dx, yb, P[:, 2] + dz], axis=1)
    d = _density_batch(world, [at(e, 0.0), at(-e, 0.0), at(0.0, e), at(0.0, -e)])
    gx = d[0] - d[1]
    gz = d[2] - d[3]
    g = np.stack([gx, gz], axis=1)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return np.where(n > 1e-9, -g / np.maximum(n, 1e-9), 0.0)


def _struct_push(P, size):
    """For each cube whose bottom corners are inside a structure or loose body: the shortest move that takes it
    out (along the nearest exposed face's normal), and which cubes needed one. Returns (push (m,3), hit (m,))."""
    m = len(P)
    push = np.zeros((m, 3))
    hit = np.zeros(m, bool)
    objs = STRUCTS.objects()
    if m == 0 or not objs:
        return push, hit
    pts = np.repeat(P, 4, axis=0)
    off = np.array([[0.4, 0.4], [0.4, -0.4], [-0.4, 0.4], [-0.4, -0.4]])
    s_rep = np.repeat(size, 4)
    pts[:, 0] += np.tile(off[:, 0], m) * s_rep
    pts[:, 2] += np.tile(off[:, 1], m) * s_rep
    pts[:, 1] -= s_rep * 0.5
    best = np.zeros(m)
    for O in objs:
        lo, hi = O.aabb()
        near = np.nonzero(((pts >= lo - 0.05) & (pts <= hi + 0.05)).all(axis=1))[0]
        if not len(near):
            continue
        sel, depth, normal = O.probe(pts[near])
        if not len(sel):
            continue
        cube = near[sel] // 4
        for c, dp, nm in zip(cube.tolist(), depth.tolist(), normal):
            if dp > best[c]:
                best[c] = dp
                push[c] = nm * (dp + 0.01)
                hit[c] = True
    return push, hit

def _terrain_push(world, P, size):
    """For cubes with the underside middle or any bottom corner inside the ground: a move out of the cube that holds the deepest
    of those points. Catches cubes wedged against a wall by one corner. Returns (push (m,3), hit (m,))."""
    m = len(P)
    off = np.array([[0.0, 0.0], [0.4, 0.4], [0.4, -0.4], [-0.4, 0.4], [-0.4, -0.4]])
    pts = np.repeat(P, 5, axis=0)
    s5 = np.repeat(size, 5)
    pts[:, 0] += np.tile(off[:, 0], m) * s5
    pts[:, 2] += np.tile(off[:, 1], m) * s5
    pts[:, 1] -= s5 * 0.5
    inside = (_density_many(world, pts) > 0.0).reshape(m, 5)
    hit = inside.any(axis=1)
    push = np.zeros_like(P)
    if not hit.any():
        return push, hit
    depth, nrm, _ = _cell_contact(world, pts)
    depth = np.where(inside.reshape(-1), depth, -1.0).reshape(m, 5)
    k = np.argmax(depth, axis=1)
    sel = np.nonzero(hit)[0]
    kk = k[sel]
    d = depth[sel, kk]
    n = nrm.reshape(m, 5, 3)[sel, kk]
    push[sel] = n * (d + 0.005)[:, None]
    return push, hit

class DirtCubes:
    """Mini dirt cubes. Simulated as numpy arrays: gravity, collision with the ground cubes, and sleep
    for cubes at rest. Cubes do not collide with each other, so they can overlap."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.pos = np.zeros((0, 3), np.float64)
        self.vel = np.zeros((0, 3), np.float64)
        self.size = np.zeros(0, np.float64)
        self.shade = np.zeros(0, np.float64)
        self.awake = np.zeros(0, bool)
        self.still = np.zeros(0, np.float64)            # seconds each cube has been at rest
        self.wet = np.zeros(0, np.float64)              # seconds each cube has been under water: it washes away
        self.anc = np.zeros((0, 4), np.float64)         # progress anchor: x, y, z of the spot it was last somewhere new, then seconds it has stayed near that spot
        self._wet_acc = 0.0
        self.col = np.zeros((0, 3), np.float64)         # colour of each cube
        self.tile = np.empty(0, np.float32)             # terrain texture layer, or -1 for plain colour

    @property
    def n(self):
        return len(self.pos)

    def remove(self, mask):
        """Delete the cubes where `mask` is True."""
        keep = ~np.asarray(mask, bool)
        self.pos, self.vel = self.pos[keep], self.vel[keep]
        self.size, self.shade, self.awake = self.size[keep], self.shade[keep], self.awake[keep]
        self.still, self.wet, self.col = self.still[keep], self.wet[keep], self.col[keep]
        self.anc = self.anc[keep]
        self.tile = self.tile[keep]

    def wake_all(self):
        """Wake every cube so it re-checks what it rests on (called when the ground or a structure changes)."""
        self.awake[:] = True
        self.still[:] = 0.0

    def wake_sphere(self, centre, radius, column_up=0.0):
        """Wake only the cubes near `centre`: those whose centre lies within radius (plus their own size) of it.
        column_up > 0 also wakes cubes stacked above the centre in the same column, since a pile resting on a
        dug-out spot loses its support from below. Everything else stays asleep."""
        if self.n == 0:
            return
        c = np.asarray(centre, np.float64)
        d = self.pos - c
        reach2 = (radius + self.size) ** 2
        hit = (d * d).sum(axis=1) <= reach2
        if column_up > 0.0:
            column = ((d[:, 0] ** 2 + d[:, 2] ** 2) <= reach2) & (self.pos[:, 1] >= c[1] - radius - 1.0) \
                     & (self.pos[:, 1] <= c[1] + column_up)
            hit |= column
        self.awake[hit] = True
        self.still[hit] = 0.0

    def spawn(self, world, cell_groups, per_block, size_scale=1.0, rgb=None, cell=None, keep_all=False, material=None):
        """Split each dug ground cube (`cell` wide, given by its centre) into level x level x level smaller cubes (level 1 = one
        cube per ground cube, level 10 = 1,000). The debris amount setting keeps that share of the cubes.
        Nothing is thrown: the cubes start at rest in the dug space and fall. Cubes whose slot overlaps the
        ground are skipped, so none starts inside it. size_scale shrinks each cube inside its slot."""
        cell = CELL if cell is None else cell
        groups = [np.asarray(g, np.float64).reshape(-1, 3) for g in cell_groups if len(g)]
        level = int(max(1, min(10, round(per_block))))
        if not CONFIG['debris_on'] and not keep_all:
            return
        if not groups:
            return
        m = level
        slots = np.stack(np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing='ij'), -1).reshape(-1, 3)
        want = m ** 3 * (1.0 if keep_all else CONFIG['debris_amount'])                 # cubes kept per ground cube: a whole number plus a chance of one more
        blocks = np.concatenate(groups)
        keep = int(want) + (_rng.random(len(blocks)) < want - int(want)).astype(int)
        slot_c = (slots + 0.5) / m - 0.5                        # slot centres inside one ground cube
        P = []
        for block, k in zip(blocks, keep):
            if k > 0:
                pick = _rng.permutation(len(slots))[:k]        # which slots get a cube
                P.append(block + slot_c[pick] * cell)
        if not P:
            return
        P = np.concatenate(P)
        side = float(np.clip(size_scale, 0.1, 1.0)) * cell / m
        sizes = np.full(len(P), side)
        Pt = P + np.array([0.0, 0.02 * cell, 0.0])                # tested a hair higher: a cube sitting right on the floor is free
        free = ((_density_many(world, Pt) <= 0.0) & ~_corners_touching(world, Pt, sizes).any(axis=1)
                & ~_struct_corners(P, sizes).any(axis=1))
        if keep_all and not free.all():                           # none may be lost: a cube that starts inside a wall or the ground is lifted clear
            for _ in range(60):
                bad = ~free
                if not bad.any():
                    break
                P[bad] += np.array([0.0, cell, 0.0])
                Pt = P + np.array([0.0, 0.02 * cell, 0.0])
                free = ((_density_many(world, Pt) <= 0.0) & ~_corners_touching(world, Pt, sizes).any(axis=1)
                        & ~_struct_corners(P, sizes).any(axis=1))
        P, sizes = P[free], sizes[free]
        if len(P) == 0:
            return
        self.pos = np.concatenate([self.pos, P])
        self.vel = np.concatenate([self.vel, np.zeros((len(P), 3))])
        self.size = np.concatenate([self.size, sizes])
        self.shade = np.concatenate([self.shade, _rng.uniform(0.9, 1.1, len(P))])
        self.awake = np.concatenate([self.awake, np.ones(len(P), bool)])
        self.still = np.concatenate([self.still, np.zeros(len(P))])
        self.wet = np.concatenate([self.wet, np.zeros(len(P))])
        self.anc = np.concatenate([self.anc, np.zeros((len(P), 4))])
        self.col = np.concatenate([self.col, np.tile(np.array(_S_RGB[DIRT] if rgb is None else rgb, np.float64), (len(P), 1))])
        layer = _S_TEXLAYER[material] if material is not None else (_S_TEXLAYER[DIRT] if rgb is None else -1.0)
        self.tile = np.concatenate([self.tile, np.full(len(P), layer, np.float32)])
        if self.n > BODY_MAX:
            cut = self.n - BODY_MAX
            self.pos, self.vel = self.pos[cut:], self.vel[cut:]
            self.size, self.shade, self.awake = self.size[cut:], self.shade[cut:], self.awake[cut:]
            self.still = self.still[cut:]
            self.wet = self.wet[cut:]
            self.anc = self.anc[cut:]
            self.col = self.col[cut:]
            self.tile = self.tile[cut:]

    def spawn_chunks(self, P, V, sizes, rgb, world=None, escape_normal=None, material=None):
        """Chunks thrown from a broken structure: positions P, velocities V, sizes (disc diameters) and colours."""
        if not CONFIG['debris_on']:
            return
        P = np.asarray(P, np.float64).reshape(-1, 3)
        V = np.asarray(V, np.float64).reshape(-1, 3)
        sizes = np.broadcast_to(np.asarray(sizes, np.float64), (len(P),))
        rgb = np.broadcast_to(np.asarray(rgb, np.float64), (len(P), 3))
        tile = np.full(len(P), -1.0, np.float32) if material is None else np.broadcast_to(_S_TEXLAYER[material], (len(P),))
        if CONFIG['debris_amount'] < 1.0:
            sel = _rng.random(len(P)) < CONFIG['debris_amount']
            P, V, sizes, rgb, tile = P[sel], V[sel], sizes[sel], rgb[sel], tile[sel]
        if world is not None and len(P):
            def clear(Q, S):
                return (_density_many(world, Q) <= 0.0) & ~_corners_touching(world, Q, S).any(axis=1) \
                       & ~_struct_corners(Q, S).any(axis=1)
            ok = clear(P, sizes)
            if not ok.all():
                P = P.copy()
                if escape_normal is None:
                    P[~ok, 1] += 0.15
                    ok = clear(P, sizes)
                else:
                    # Take short steps toward the wall face until the whole cube fits.
                    step = max(0.025, float(sizes.min()) * 0.5)
                    steps = math.ceil(1.0 / step)
                    for i in range(1, steps + 1):
                        distance = i / steps
                        blocked = np.flatnonzero(~ok)
                        if not len(blocked):
                            break
                        Q = P[blocked] + distance * escape_normal
                        free = clear(Q, sizes[blocked])
                        P[blocked[free]] = Q[free]
                        ok[blocked[free]] = True
            P, V, sizes, rgb, tile = P[ok], V[ok], sizes[ok], rgb[ok], tile[ok]
        k = len(P)
        if k == 0:
            return
        self.pos = np.concatenate([self.pos, P])
        self.vel = np.concatenate([self.vel, np.asarray(V, np.float64).reshape(-1, 3)])
        self.size = np.concatenate([self.size, np.broadcast_to(np.asarray(sizes, np.float64), (k,))])
        self.shade = np.concatenate([self.shade, np.ones(k)])
        self.awake = np.concatenate([self.awake, np.ones(k, bool)])
        self.still = np.concatenate([self.still, np.zeros(k)])
        self.wet = np.concatenate([self.wet, np.zeros(k)])
        self.anc = np.concatenate([self.anc, np.zeros((k, 4))])
        self.col = np.concatenate([self.col, np.broadcast_to(np.asarray(rgb, np.float64), (k, 3))])
        self.tile = np.concatenate([self.tile, tile])
        if self.n > BODY_MAX:
            cut = self.n - BODY_MAX
            self.pos, self.vel = self.pos[cut:], self.vel[cut:]
            self.size, self.shade, self.awake = self.size[cut:], self.shade[cut:], self.awake[cut:]
            self.still = self.still[cut:]
            self.wet = self.wet[cut:]
            self.anc = self.anc[cut:]
            self.col = self.col[cut:]
            self.tile = self.tile[cut:]

    def _sleeper_index(self, cell, OFF, M):
        """Sorted neighbour keys and ids of the sleeping cubes. Cached: a sleeping cube does not move, so the index
        is rebuilt only when the awake set, the cube count, the cell size or a sleeping cube's position changes."""
        sl = np.nonzero(~self.awake)[0]
        pos_sl = self.pos[sl]
        c = getattr(self, '_sleep_cache', None)
        if (c is not None and c['cell'] == cell and np.array_equal(c['awake'], self.awake)
                and np.array_equal(c['pos'], pos_sl)):
            return c['keys'], c['ids']
        ci = np.floor(pos_sl / cell).astype(np.int64) + OFF
        keys = (ci[:, 0] * M + ci[:, 1]) * M + ci[:, 2]
        o = np.argsort(keys, kind='stable')
        keys, ids = keys[o], sl[o]
        self._sleep_cache = dict(cell=cell, awake=self.awake.copy(), pos=pos_sl.copy(), keys=keys, ids=ids)
        return keys, ids

    def _separate(self, passes=3, dt=1.0 / 60.0, grounded=None):
        """Round cube against round cube. Each cube is treated as a sphere with radius half its size.
        Overlapping pairs are pushed apart along the line between their centres. Sleeping cubes never
        move; when one of a pair is asleep, the awake one takes the whole correction. Motion that drives
        the pair into each other is removed, so piles settle instead of jittering."""
        self._contact = np.zeros(self.n, bool)                    # cubes that touched another cube this frame
        self._on_sleeper = np.zeros(self.n, bool)                 # cubes resting on a cube that is already asleep
        self._unstable = np.zeros(self.n, bool)                   # cubes perched off-centre on another cube: they slide off, never freeze or sleep
        if self.n < 2:
            return
        if grounded is None:
            grounded = np.zeros(self.n, bool)
        M = 1 << 20
        OFF = 1 << 19
        SEP_SLOP = 0.003
        k = max(1.0, dt * 60.0)          # slow frames fall further, so they are allowed to correct further
        SEP_SOFT = min(1.0, 0.35 * k)    # share of an overlap fixed per pass
        SEP_MAX_STEP = 0.02 * k          # most a pair is moved per pass, in blocks
        SEP_CUBE_CAP = 0.03 * k          # most one cube is moved per pass, however many neighbours push it
        # Only a pair with at least one awake cube can change anything, so the search starts from the awake
        # cubes alone and looks up their neighbours in the full index. Sleeping cubes are still indexed, so an
        # awake cube finds the sleeping ones it touches. The work now scales with the awake count, not self.n.
        aw_all = np.nonzero(self.awake)[0]
        if len(aw_all) == 0:
            return
        cell = float(self.size.max())                             # the same for every pass, so it is worked out once
        sleep_keys, sleep_ids = self._sleeper_index(cell, OFF, M)  # sleepers: cached, only re-sorted when they change
        for _ in range(passes):
            ci = np.floor(self.pos[aw_all] / cell).astype(np.int64) + OFF   # only the awake cubes are sorted each pass
            qk = (ci[:, 0] * M + ci[:, 1]) * M + ci[:, 2]
            ord_a = np.argsort(qk, kind='stable')
            ak = qk[ord_a]
            aid = aw_all[ord_a]
            # all 27 neighbouring cells in one batch per index, instead of 27 separate passes of small numpy calls
            nk = (qk[:, None] + _NEIGH_OFFS[None, :]).ravel()
            i_all = np.repeat(aw_all, len(_NEIGH_OFFS))
            I, J = [], []
            for idx_keys, idx_ids, awake_only in ((ak, aid, True), (sleep_keys, sleep_ids, False)):
                lo = np.searchsorted(idx_keys, nk, 'left')
                hi = np.searchsorted(idx_keys, nk, 'right')
                cnt = hi - lo
                total = int(cnt.sum())
                if total == 0:
                    continue
                i = np.repeat(i_all, cnt)
                starts = np.repeat(lo, cnt)
                within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                j = idx_ids[starts + within]
                # awake-awake pairs are found from both ends: keep them once; awake-sleeping pairs are kept
                keep = (i < j) if awake_only else np.ones(total, bool)
                I.append(i[keep])
                J.append(j[keep])
            if not I:
                return
            I = np.concatenate(I)
            J = np.concatenate(J)

            d = self.pos[J] - self.pos[I]
            # Debris cubes are axis-aligned boxes, so they are tested as boxes: a pair overlaps only when it overlaps
            # on all three axes, and it is pushed apart along the one axis with the least overlap. (A sphere test
            # missed corner-to-corner overlaps, which is what showed as cubes sunk into each other.)
            half = ((self.size[I] + self.size[J]) * 0.5)[:, None]              # sum of half-widths, one per pair
            ov = half - np.abs(d) - SEP_SLOP                                    # overlap on each axis, per pair
            # The axis with the least overlap, but vertical is favoured: a sideways shove pushes a cube into a wall,
            # and piles separate upward far more often than they should slide sideways.
            ax = np.argmin(ov * np.array([1.0, 0.7, 1.0]), axis=1)
            rows = np.arange(len(ov))
            depth = ov[rows, ax]
            aI = self.awake[I]
            aJ = self.awake[J]
            hit = (ov > 0.0).all(axis=1) & (aI | aJ)
            if not hit.any():
                return
            I, J, d, ax, depth = I[hit], J[hit], d[hit], ax[hit], depth[hit]
            aI, aJ = aI[hit], aJ[hit]

            n = np.zeros_like(d)                                    # unit normal from I towards J, on the chosen axis
            sd = d[np.arange(len(d)), ax]
            n[np.arange(len(d)), ax] = np.where(sd >= 0.0, 1.0, -1.0)
            wI = np.where(aI & aJ, 0.5, np.where(aI, 1.0, 0.0))     # share of the push each cube takes
            wJ = 1.0 - wI

            # Correct only part of the overlap, capped per pass. Fixing it all at once shoved a landing cube
            # up and sideways in one frame, which is what looked like a bounce.
            push = np.minimum(depth * SEP_SOFT, SEP_MAX_STEP)
            corr = np.zeros_like(self.pos)
            np.add.at(corr, J, n * (push * wJ)[:, None])
            np.add.at(corr, I, -n * (push * wI)[:, None])
            mag = np.linalg.norm(corr, axis=1)
            corr *= np.minimum(1.0, SEP_CUBE_CAP / np.maximum(mag, 1e-9))[:, None]    # a cube in a crowd moves at most this much
            self.pos += corr

            strong = depth > SEP_WAKE_DEPTH                      # a cube squeezed in deeper than this is still being shoved about; a shallow squeeze in a packed crater is just a pile at rest
            self.still[np.concatenate([I[strong], J[strong]])] = 0.0
            touched = np.unique(np.concatenate([I[aI], J[aJ]]))
            self._contact[touched] = True
            # Resting on a sleeping cube below it counts as held. Touching an awake cube, or a cube above, does not,
            # so a loose clump floating in mid-air keeps falling instead of freezing where it bumped.
            self._on_sleeper[I[aI & ~aJ & (n[:, 1] < -0.5)]] = True
            self._on_sleeper[J[aJ & ~aI & (n[:, 1] > 0.5)]] = True
            # Toppling. A cube resting on another cube with its centre past TOPPLE_MIN of a cube width off the lower
            # cube's centre is not balanced: it is pushed sideways, away from the lower cube's centre, harder the
            # further off it is. A cube sitting between two others is pushed both ways and stays put. Cubes that are
            # being toppled are flagged so static friction and sleep leave them alone until they have slid off.
            vert = (ax == 1)
            if vert.any():
                up_is_J = d[:, 1] > 0.0
                off = np.where(up_is_J[:, None], d[:, [0, 2]], -d[:, [0, 2]])        # upper cube's offset from the lower one, x and z
                avg = (self.size[I] + self.size[J]) * 0.5
                mag_off = np.max(np.abs(off), axis=1) / avg                          # in cube widths; 0 = dead centre
                tip = vert & (mag_off > TOPPLE_MIN)
                upper = np.where(up_is_J, J, I)
                tip &= self.awake[upper]
                if tip.any():
                    o = off[tip]
                    dirn = o / np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-9)
                    strength = np.clip(mag_off[tip] / TOPPLE_FULL, 0.0, 1.0)
                    u = upper[tip]
                    np.add.at(self.vel, (u, 0), dirn[:, 0] * strength * TOPPLE_ACCEL * dt / passes)
                    np.add.at(self.vel, (u, 2), dirn[:, 1] * strength * TOPPLE_ACCEL * dt / passes)
                    self._unstable[u] = True
            # contact friction: bleeds off sideways energy only. Slowing the fall too made debris drift down in slow motion.
            friction = touched[grounded[touched] | self._on_sleeper[touched]]
            self.vel[friction, 0] *= 0.85
            self.vel[friction, 2] *= 0.85

            # Remove only the velocity that drives the pair into each other.
            vI = np.einsum('ij,ij->i', self.vel[I], n)
            vJ = np.einsum('ij,ij->i', self.vel[J], n)
            # Every contact used the same pre-correction velocity, so a cube resting on two cubes had its fall
            # cancelled twice and came out moving UP. Each contact now takes only its share (1 / contacts), which
            # can never remove more than the approach speed.
            deg = (np.bincount(I, minlength=self.n) + np.bincount(J, minlength=self.n)).astype(np.float64)
            # Two unsupported cubes can share the same falling velocity. Cancel
            # only their closing speed, so contact cannot hold up the whole cloud.
            airborne = aI & aJ & ~grounded[I] & ~grounded[J]
            closing = airborne & (vI > vJ)
            impulse = (0.5 * (vI[closing] - vJ[closing])
                       / np.maximum(deg[I[closing]], deg[J[closing]]))
            delta = impulse[:, None] * n[closing]
            np.add.at(self.vel, I[closing], -delta)
            np.add.at(self.vel, J[closing], delta)
            kill_I = aI & ~airborne & (vI > 0)
            kill_J = aJ & ~airborne & (vJ < 0)
            np.add.at(self.vel, I[kill_I], -(vI[kill_I] / deg[I[kill_I]])[:, None] * n[kill_I])
            np.add.at(self.vel, J[kill_J], -(vJ[kill_J] / deg[J[kill_J]])[:, None] * n[kill_J])

    def update(self, world, dt, sleep_time=1.0):
        cap = int(BODY_MAX * CONFIG['debris_amount']) if CONFIG['debris_on'] else 0
        if self.n > cap:
            cut = self.n - cap
            self.pos, self.vel = self.pos[cut:], self.vel[cut:]
            self.size, self.shade, self.awake = self.size[cut:], self.shade[cut:], self.awake[cut:]
            self.still = self.still[cut:]
            self.wet = self.wet[cut:]
            self.anc = self.anc[cut:]
            self.col = self.col[cut:]
            self.tile = self.tile[cut:]
        if self.n == 0:
            return
        dt = min(dt, 0.05)
        aw = np.nonzero(self.awake)[0]
        if len(aw) > MAX_AWAKE_CUBES:
            # Too many moving cubes for one frame: the slowest RESTING cubes settle now. A cube in mid-air is never
            # frozen (slow at the top of its arc is not the same as resting), so it keeps falling.
            excess = len(aw) - MAX_AWAKE_CUBES
            speed = np.linalg.norm(self.vel[aw], axis=1)
            cand = aw[np.argsort(speed)[:excess * 4]]
            rest = _support(world, self.pos[cand], self.size[cand]).any(axis=1)
            drop = cand[rest][:excess]
            self.awake[drop] = False
            self.still[drop] = 0.0
            self.vel[drop] = 0.0
            aw = np.nonzero(self.awake)[0]
        if len(aw) > HARD_AWAKE_CUBES:
            # A big burst is all in mid-air, so nothing can be put to sleep. Rather than simulate every cube every
            # frame, split the airborne cubes into k groups and step one group per frame, with a k times longer step
            # (capped at 0.05 s), so the cloud keeps moving at nearly full speed while the frame cost stays flat.
            k = -(-len(aw) // HARD_AWAKE_CUBES)
            self._turn = getattr(self, '_turn', 0) + 1
            aw = aw[(np.arange(len(aw)) % k) == (self._turn % k)]
            dt = min(dt * k, 0.05)
        if len(aw):
            V = self.vel[aw]
            P = self.pos[aw]
            P_start = P.copy()
            V[:, 1] = np.maximum(V[:, 1] - BODY_GRAVITY * dt, -BODY_MAX_FALL)
            kf = math.exp(-BODY_FRICTION * dt)
            V[:, 0] *= kf
            V[:, 2] *= kf
            # Substeps are worked out per cube, not from the fastest cube in the whole batch: one fast fragment used to make every
            # slow cube repeat its terrain and structure checks up to six times. Cubes that need the same count move together.
            steps_i = np.clip(np.ceil(np.abs(V).max(axis=1) * dt * 16), 1, 6).astype(np.int64)
            size = self.size[aw]
            _tm = {}
            def _mark(key, t0):
                _tm[key] = _tm.get(key, 0.0) + (time.perf_counter() - t0)
            _t0 = time.perf_counter()
            # A cube that ended up inside a structure (a blast throws pieces out of one) is lifted out,
            # otherwise every move is blocked and it hangs there.
            for _ in range(4):
                push, stuck = _struct_push(P, size)
                if not stuck.any():
                    break
                L = np.linalg.norm(push[stuck], axis=1)
                step = push[stuck] * (np.minimum(L, 0.1) / np.maximum(L, 1e-9))[:, None]   # out the nearest face, not always up
                P[stuck] += step
                nrm = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
                into = np.minimum(np.einsum('ij,ij->i', V[stuck], nrm), 0.0)
                V[stuck] -= into[:, None] * nrm                                           # stop only the motion into the surface
            stuck = _struct_corners(P, size).any(axis=1)
            if stuck.any():
                P[stuck, 1] += 0.1 * size[stuck]                                          # last resort if the normals fail
            _mark('struct_push', _t0)
            # A cube whose centre is inside the ground (a push from a neighbour, or a blast that moved the surface)
            # is moved back out along the density gradient. It used to stay stuck there for good.
            _t_tp1 = time.perf_counter()
            for _ in range(2):
                tpush, tin = _terrain_push(world, P, size)
                if not tin.any():
                    break
                L = np.linalg.norm(tpush[tin], axis=1)
                P[tin] += tpush[tin] * (np.minimum(L, 0.08) / np.maximum(L, 1e-9))[:, None]
                V[tin, 1] = np.maximum(V[tin, 1], 0.0)                                   # stop it sinking back in
            _mark('terrain_push', _t_tp1)
            _t_move = time.perf_counter()
            for n_steps in np.unique(steps_i).tolist():
                gi = np.nonzero(steps_i == n_steps)[0]
                whole = len(gi) == len(steps_i)
                Pg, Vg, sizeg = (P, V, size) if whole else (P[gi], V[gi], size[gi])
                sdt = dt / n_steps
                for _ in range(n_steps):
                    for axis in (1, 0, 2):
                        new = Pg.copy()
                        new[:, axis] += Vg[:, axis] * sdt
                        if axis == 1:
                            under = new.copy()
                            under[:, 1] -= sizeg * 0.5
                            _t0 = time.perf_counter()
                            corners = _corners_touching(world, new, sizeg)
                            _mark('debris.move.terrain', _t0)
                            _t0 = time.perf_counter()
                            sc = _struct_corners(new, sizeg)
                            _mark('debris.move.struct', _t0)
                            _t0 = time.perf_counter()
                            blocked = (corners | sc).any(axis=1)                      # any bottom corner in terrain or a structure
                            blocked |= _density_many(world, under) > 0.0            # ...or the middle of its underside (a bump between the corners)
                            _mark('debris.move.terrain', _t0)
                        else:
                            _t0 = time.perf_counter()
                            dens = _density_many(world, new) > 0.0
                            _mark('debris.move.terrain', _t0)
                            _t0 = time.perf_counter()
                            sc = _struct_corners(new, sizeg).any(axis=1)
                            _mark('debris.move.struct', _t0)
                            blocked = dens | sc
                        go = ~blocked
                        Pg[go] = new[go]
                        Vg[blocked, axis] = 0.0
                if not whole:
                    P[gi] = Pg
                    V[gi] = Vg
            _mark('debris.move', _t_move)
            _t_sup = time.perf_counter()
            # A cube stops on its highest bottom corner, so its low side sinks slightly into the ground.
            # Where only some corners are supported, the cube slides downhill until its base is level.
            hits = _support(world, P, size)
            partial = ~hits.all(axis=1) & (V[:, 1] > -0.5)     # not fully on the ground, and not falling
            if partial.any():
                down = _downhill(world, P[partial], size[partial])
                V[partial, 0] += down[:, 0] * BODY_GRAVITY * 0.6 * dt
                V[partial, 2] += down[:, 1] * BODY_GRAVITY * 0.6 * dt
            _mark('debris.support', _t_sup)
            # Hard sleep timer: a cube that stays still for sleep_time seconds goes to sleep for good.
            self.pos[aw] = P
            self.vel[aw] = V
            _t_sep = time.perf_counter()
            grounded = np.zeros(self.n, bool)
            grounded[aw] = hits.any(axis=1)
            self.grounded_last = grounded                      # which cubes touched ground this step, so the game can reuse it
            self._separate(dt=dt, grounded=grounded)
            _mark('debris.separate', _t_sep)
            _t_fix = time.perf_counter()
            # Separation knows nothing about the ground, so a pile pressing down can push its bottom cubes into it.
            # Bring any cube whose underside went into the surface back out.
            for _ in range(2):
                tp, tin = _terrain_push(world, self.pos[aw], size)
                if not tin.any():
                    break
                L = np.linalg.norm(tp[tin], axis=1)
                self.pos[aw[tin]] += tp[tin] * (np.minimum(L, 0.1) / np.maximum(L, 1e-9))[:, None]
                self.vel[aw[tin], 1] = np.maximum(self.vel[aw[tin], 1], 0.0)
            _mark('debris.terrain_fix', _t_fix)
            # Static friction: a cube that is touching others and already moving slowly stops sliding sideways, so a
            # pile of round-ish cubes stops creeping and can go to sleep.
            ct = self._contact[aw]
            crawl = np.linalg.norm(self.vel[aw], axis=1) < 0.35
            held_contact = grounded[aw] | self._on_sleeper[aw]
            stick = aw[ct & crawl & held_contact & ~self._unstable[aw]]
            self.vel[stick, 0] = 0.0
            self.vel[stick, 2] = 0.0
            moved_speed = np.linalg.norm(self.pos[aw] - P_start, axis=1) / max(dt, 1e-6)   # real movement this frame, pushes included                                       # pushes cubes apart, stops motion into each other
            Vn = self.vel[aw]
            calm = (np.abs(Vn).max(axis=1) < 0.12) & (moved_speed < 0.25)         # crawling AND not being shoved around
            unstable_aw = self._unstable[aw]
            # A cube flagged as perched off-centre, or pushed hard (a slope pulling it down) against a neighbour that holds it, can be
            # busy without going anywhere. If it really has not moved, it is wedged in place: at rest, once it has stayed that way a while.
            busy = unstable_aw | ~calm
            wedged = busy & (moved_speed < JAM_MOVE)
            still_now = (calm & ~unstable_aw) | wedged                             # a cube that is still toppling or sliding is not at rest
            # A slow cube only counts as at rest if something is under it. Without this, a cube at the top of its arc passes the
            # "slow" test, has its velocity zeroed, and at high frame rates (gravity per frame below 0.12) gravity never builds
            # up again, so it hangs in mid-air. Airborne cubes keep their velocity and keep falling.
            # Progress test: judged over a second, not a frame. Cubes jammed in a pit shake in place (a long path, no progress), so the
            # one-frame speed tests above never call them at rest. Staying within PROGRESS_R of one spot for PROGRESS_TIME does.
            if PROGRESS_TIME > 0:
                self.anc[~self.awake, 3] = 0.0                                       # a cube that slept starts its count again when it wakes
                A = self.anc[aw]
                far = np.linalg.norm(self.pos[aw] - A[:, :3], axis=1) > PROGRESS_R
                A[far, :3] = self.pos[aw][far]
                A[:, 3] = np.where(far, 0.0, A[:, 3] + dt)
                self.anc[aw] = A
                prog = A[:, 3] >= PROGRESS_TIME
                still_now |= prog
            else:
                prog = np.zeros(len(aw), bool)
            cand = np.nonzero(still_now)[0]
            if len(cand):
                held_now = _support(world, self.pos[aw[cand]], size[cand]).any(axis=1) | self._on_sleeper[aw[cand]]
                still_now[cand[~held_now]] = False
            self.vel[aw[still_now]] = 0.0
            self.still[aw] = np.where(still_now, self.still[aw] + dt, 0.0)
            need = np.where(busy, sleep_time * JAM_TIME_MUL, sleep_time) if JAM_MOVE > 0 else sleep_time   # wedged cubes wait longer
            if PROGRESS_TIME > 0:
                need = np.where(prog, dt * 0.5, need)                              # it has already stayed put for a second: no further wait
            ready = np.nonzero(still_now & (self.still[aw] >= need))[0]                # about to fall asleep
            if len(ready):
                held = _support(world, self.pos[aw[ready]], size[ready]).any(axis=1) | self._on_sleeper[aw[ready]]
                lost = ready[~held]                                                    # nothing under it: keep it awake
                self.still[aw[lost]] = 0.0
            asleep = self.still[aw] >= need
            if asleep.any():
                SFX.settle(self.pos[aw[asleep]][0], int(asleep.sum()))
            self.awake[aw[asleep]] = False
            for k_, v_ in _tm.items():
                PROF.add(k_ if k_.startswith('debris.') else 'debris.' + k_, v_ * 1000.0)
        fading = self.wet > WASH_DELAY
        if fading.any():
            self.size[fading] *= math.exp(-3.0 * dt)
        keep = (self.pos[:, 1] > -4.0) & (self.wet < WASH_DELAY + WASH_FADE)     # also drops cubes that fell out of the world
        if not keep.all():
            self.pos, self.vel = self.pos[keep], self.vel[keep]
            self.size, self.shade, self.awake = self.size[keep], self.shade[keep], self.awake[keep]
            self.still = self.still[keep]
            self.wet = self.wet[keep]
            self.anc = self.anc[keep]
            self.col = self.col[keep]
            self.tile = self.tile[keep]

    def instances(self, textured=False):
        """Per cube: position, colour, side length, and optional texture layer."""
        out = np.empty((self.n, 8 if textured else 7), np.float32)
        out[:, :3] = self.pos
        out[:, 3:6] = self.col.astype(np.float32) * self.shade[:, None].astype(np.float32)
        out[:, 6] = self.size
        if textured:
            out[:, 7] = self.tile
        return out

BODIES = DirtCubes()

TOPPLE_MIN     = 0.18                # a cube resting on another slides off once its centre is this many cube widths off-centre
TOPPLE_FULL    = 0.5                 # ...and is pushed at full strength by this offset (the edge of the cube below)
TOPPLE_ACCEL   = BODY_GRAVITY * 1.2  # sideways acceleration at full strength, blocks per second squared

CUBE_ROW_STEP = 14                   # blocks between stamped cubes along a row

_DISC_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;        // unit cube vertex, -0.5..0.5
layout(location = 1) in vec3 aInstPos;    // cube centre
layout(location = 2) in vec3 aInstCol;    // cube colour
layout(location = 3) in float aInstSize;  // cube side length
layout(location = 4) in vec3 aNrm;        // face normal
layout(location = 5) in float aTile;
uniform mat4 uVP;
uniform float uLight;
out vec3 vCol;
out vec3 vNrm;
out vec3 vLocal;     // unit-cube position, for the edge wear
flat out vec3 vSeed;
flat out float vTile;
flat out float vSize;
void main() {
    vec3 p = aInstPos + aPos * aInstSize;
    vCol = aInstCol * uLight * 1.3;     // lifted to sit with the lit tiles
    vNrm = aNrm;
    vLocal = aPos;
    vSeed = aInstCol * 127.0 + vec3(13.0, 17.0, 31.0);
    vTile = aTile;
    vSize = aInstSize;
    gl_Position = uVP * vec4(p, 1.0);
}
"""

_DISC_FRAG = """
#version 330 core
in vec3 vCol;
in vec3 vNrm;
in vec3 vLocal;
flat in vec3 vSeed;
flat in float vTile;
flat in float vSize;
uniform sampler2DArray uTiles;
uniform bool uTextured;
out vec4 fragColor;
LIGHT_GLSL_PLACEHOLDER
float hash3d(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}
void main() {
    vec3 n = normalize(vNrm);
    vec3 light = dayLight(n, 1.0);
    // Edge wear: distance from this point to the nearest cube edge. Edges chip to a lighter, dustier tone.
    vec3 d = 0.5 - abs(vLocal);
    float edge = min(d.x + abs(n.x) * 9.0, min(d.y + abs(n.y) * 9.0, d.z + abs(n.z) * 9.0));
    float wear = 1.0 - smoothstep(0.0, 0.07, edge);
    // Fine speckle per face cell, so flat faces are not perfectly uniform.
    float speck = 0.96 + 0.08 * hash3d(floor((vLocal + 0.5) * 4.0) + n * 3.0);
    vec3 base = vCol * speck;
    if (uTextured && vTile >= 0.0) {
        vec2 faceUV = abs(n.y) > 0.5 ? vLocal.xz : (abs(n.x) > 0.5 ? vLocal.zy : vLocal.xy);
        vec2 patch = vec2(hash3d(vSeed), hash3d(vSeed.zxy));
        float scale = vTile == 1.0 ? 0.5 : 1.0;
        vec3 texel = texture(uTiles, vec3(patch + faceUV * vSize * scale, vTile)).rgb;
        if (vTile == 0.0) texel *= vec3(1.02, 0.86, 0.70);
        base = mix(base, texel, 0.72);
        light *= 0.82 + 0.18 * smoothstep(-0.5, 0.3, vLocal.y);
    }
    vec3 col = base * light;
    col = mix(col, col * 1.18 + vec3(0.02), wear * (uTextured ? 0.23 : 0.6));
    fragColor = vec4(col, 1.0);
}
"""
_DISC_FRAG = _DISC_FRAG.replace("LIGHT_GLSL_PLACEHOLDER", _LIGHT_GLSL)


def _cube_mesh():
    """36 vertices of a unit cube centred on the origin: position then face normal."""
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    v = []
    for n, u, w in faces:
        n, u, w = (np.array(t, np.float32) for t in (n, u, w))
        c = [n * 0.5 + (-u - w) * 0.5, n * 0.5 + (u - w) * 0.5, n * 0.5 + (u + w) * 0.5, n * 0.5 + (-u + w) * 0.5]
        for i in (0, 1, 2, 0, 2, 3):
            v.append(np.concatenate([c[i], n]))
    return np.array(v, np.float32)


_DISC_CORNERS = _cube_mesh()       # (kept under the old name: the mesh the renderer instances)

class DirtBodyRenderer:
    """Draws loose cubes with one instanced call."""
    def __init__(self):
        vs = _compile_shader(_DISC_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_DISC_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_tiles = gl.glGetUniformLocation(self.program, "uTiles")
        self.u_textured = gl.glGetUniformLocation(self.program, "uTextured")
        self.u_light = gl.glGetUniformLocation(self.program, "uLight")
        self.vao = gl.glGenVertexArrays(1)
        self.vbo_c = gl.glGenBuffers(1)
        self.vbo_i = gl.glGenBuffers(1)
        self._textured_layout = None
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_c)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, _DISC_CORNERS.nbytes, _DISC_CORNERS, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(4)
        gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_i)
        gl.glEnableVertexAttribArray(1)
        gl.glBindVertexArray(0)

    def render(self, vp, ambient, yaw, pitch, bodies=None):
        src = BODIES if bodies is None else bodies          # another set of cubes with .n and .instances() can be drawn the same way
        n = src.n
        if n == 0:
            return
        textured = src is BODIES
        inst = BODIES.instances(textured=True) if textured else src.instances()
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glUniform1i(self.u_textured, int(textured))
        if textured:
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tile_texture())
            gl.glUniform1i(self.u_tiles, 0)
        gl.glUniform1f(self.u_light, max(float(ambient), GFX['ambient_floor']))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_i)
        if self._textured_layout != textured:
            stride = inst.shape[1] * inst.itemsize
            for loc, count, offset in ((1, 3, 0), (2, 3, 12), (3, 1, 24)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, count, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(offset))
                gl.glVertexAttribDivisor(loc, 1)
            if textured:
                gl.glEnableVertexAttribArray(5)
                gl.glVertexAttribPointer(5, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(28))
                gl.glVertexAttribDivisor(5, 1)
            else:
                gl.glDisableVertexAttribArray(5)
            self._textured_layout = textured
        gl.glBufferData(gl.GL_ARRAY_BUFFER, inst.nbytes, inst, gl.GL_STREAM_DRAW)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 36, n)
        gl.glBindVertexArray(0)


# ═══════════════════════════════════════════════
#  STRUCTURES: breakable supports, rigid bodies, solid rotating geometry
# ═══════════════════════════════════════════════
# A structure is a box-shaped grid of solid cells (a pillar, a wall). Each cell has a material, and
# neighbouring cells are joined by bonds. Blasts destroy and crack cells. After each change the
# structure is checked: every bond across a cut plane carries its share of the weight above it, and
# a bond that is stretched or crushed past its strength breaks. A group of cells with no bond path to
# the ground becomes a RigidBody: a solid with a position, a rotation, mass and an inertia tensor.
# Bodies collide with the terrain, with structures and with each other, so a pillar that loses its
# footing tips over its remaining support instead of just dropping.

STRUCT_GRAVITY      = 24.0     # blocks per second squared
STRUCT_SUBSTEPS     = 3        # physics steps per frame for each awake body
STRUCT_SOLVER_ITERS = 8        # contact solver passes per step
STRUCT_FRICTION     = 0.7
STRUCT_RESTITUTION  = 0.0
STRUCT_SLOP         = 0.03    # penetration the solver leaves alone, in blocks
STRUCT_MAX_CONTACTS = 10       # contacts kept per body per step
# The Wobble setting (0-100%) drives these live. 50% is the original behaviour. Lower: loose pieces lose spin and
# rocking faster, grip harder, push more firmly out of overlaps and go to sleep sooner. Higher: they stay lively.
WOBBLE = dict(ang_damp=0.25, lin_damp=0.0, friction=STRUCT_FRICTION, iters=STRUCT_SOLVER_ITERS, sleep_k=1.0)


def apply_wobble(pct):
    t = max(0.0, min(1.0, float(pct) / 100.0))
    calm = max(0.0, 1.0 - 2.0 * t)                       # 1 at 0%, 0 at 50% and above
    WOBBLE['ang_damp'] = 0.25 * ((1.0 - t) / 0.5) ** 3   # 2.0 at 0%, 0.25 at 50%, 0 at 100%
    WOBBLE['lin_damp'] = 1.2 * calm                      # a little drag on sliding and drifting at the calm end
    WOBBLE['friction'] = STRUCT_FRICTION * (1.0 + 0.6 * calm)
    WOBBLE['iters'] = STRUCT_SOLVER_ITERS + int(round(6 * calm))
    WOBBLE['sleep_k'] = 2.0 ** ((0.5 - t) * 4.0)         # 4x looser sleep limits at 0%, 1x at 50%, 0.25x at 100%

STRUCT_SLEEP_LIN    = 0.15
STRUCT_SLEEP_ANG    = 0.20
STRUCT_SLEEP_TIME   = 0.6
STRUCT_SHATTER      = 0.25     # blast strength (after falloff) that destroys a cell of hardness 1
STRUCT_BLAST_IMPULSE = 140.0   # converts blast pressure on a body's surface to momentum
STRUCT_CRACK_STEP   = 0.05     # seconds between rounds of bond failures, so cracks spread visibly
STRUCT_BUDGET_S     = 0.0025   # most time per frame spent on the support checks
STRUCT_LOCAL_R      = 2.0      # blocks: a cut only re-checks the cells this close to it (cells past the edge count as held)
MESH_BRICK          = 16       # structure meshes are kept in bricks of this many cells a side; a cut rebuilds only the bricks it touches
STRUCT_MIN_BODY_CELLS = 6    # a loose group smaller than this crumbles into chunks instead of becoming a body
AXE_CUT_HALF_W = 0.45     # the axe's cut: a slot this many blocks either side of the swing line...
AXE_CUT_DEPTH = 0.75      # ...and this deep into the surface
AXE_BLUNT_FRAC = 0.5      # the middle of the axe is blunt: cells within this fraction of the cut's half-width are only cracked...
AXE_BLUNT_CRACK = 0.5     # ...by this much crack damage per swing (0.85 is the most a cell takes, so about three swings); only the outer edges cut cubes out
AXE_CUT_POWER = 0.2       # what the blade can cut: a cell is cut away if hardness * STRUCT_SHATTER is at most this (wood, planks, dirt, brick); harder cells only crack
AXE_MIN_PIECE_CELLS = 8     # the smallest piece an axe cut leaves as a loose log: anything smaller crumbles into cubes
AXE_IMPULSE = 0.05        # momentum of a swing (mass units x blocks/s: a 2 kg head at 8 m/s is about 0.016), pushed into a loose piece at the hit point
STRUCT_UNBREAKABLE_CELLS = 40000  # a fixed structure with more live cells than this never cracks or collapses under load (it can still be cut in two). Was 2000, which left anything over 16 blocks of cubes unable to collapse

# id: mass per block of volume, compressive, tensile and shear strength (force per face area), blast hardness.
# The numbers are tuned so a 2 x 2 pillar stands whole but cracks and topples once about a third of its
# section is gone. Blocks that are not listed behave like stone.
STRUCT_MATERIALS = {
    DIRT:   dict(rho=1.6, comp=1200.0,  tens=25.0,    shear=60.0,   hard=0.5),    # crumbly: weak in tension and shear
    STONE:  dict(rho=2.4, comp=3000.0,  tens=300.0,   shear=450.0,  hard=1.0),
    BRICK:  dict(rho=1.9, comp=1500.0,  tens=120.0,   shear=200.0,  hard=0.7),
    PLANKS: dict(rho=0.6, comp=500.0,   tens=400.0,   shear=100.0,  hard=0.4),
    STEEL:  dict(rho=7.8, comp=20000.0, tens=20000.0, shear=12000.0, hard=3.0, mu=0.5),
    CONCRETE_WHITE: dict(rho=2.3, comp=1600.0, tens=60.0, shear=350.0, hard=1.0, mu=0.75),   # heavy, strong squeezed, weak pulled
    WOOD:   dict(rho=0.7, comp=90000.0, tens=80000.0, shear=40000.0, hard=0.5),    # tree bark and branches: light, tough along its length
}
# Friction between a block and whatever it rests on (used for the sideways thrust of arches).
_S_MU = np.full(256, 0.6)
for _b, _m in STRUCT_MATERIALS.items():
    _S_MU[_b] = _m.get('mu', 0.6)
STRENGTH = {'k': 1.0}                 # global strength scale from the settings slider (1.0 = as tuned)
ARCH_JOINT_SHEAR = 0.25               # share of a material's shear strength a loose joint can use
ARCH_CONTACT = 0.5                    # share of an arch joint that actually carries the thrust
_S_RHO   = np.full(256, STRUCT_MATERIALS[STONE]['rho'])
_S_COMP  = np.full(256, STRUCT_MATERIALS[STONE]['comp'])
_S_TENS  = np.full(256, STRUCT_MATERIALS[STONE]['tens'])
_S_SHEAR = np.full(256, STRUCT_MATERIALS[STONE]['shear'])
_S_HARD  = np.full(256, STRUCT_MATERIALS[STONE]['hard'])
for _b, _m in STRUCT_MATERIALS.items():
    _S_RHO[_b], _S_COMP[_b], _S_TENS[_b] = _m['rho'], _m['comp'], _m['tens']
    _S_SHEAR[_b], _S_HARD[_b] = _m['shear'], _m['hard']
_S_RGB = np.full((256, 3), 0.6, np.float32)
for _b, _c in COLORS.items():
    _S_RGB[_b] = _c
_S_RGB[CONCRETE_WHITE] = (0.80, 0.80, 0.77)

# ═══════════════════════════════════════════════
#  MATERIAL TEXTURES
# ═══════════════════════════════════════════════
# One 256 x 256 seamless tile per structure material (JPEG, embedded here so the game stays a
# single file. Layer k of the GL texture array is TILE_MATS[k]. A face is mapped one tile per block, in the
# object's own frame, so a toppled wall carries its pattern with it. Materials not listed stay plain colour.
TILE_SIZE = 256
TILE_MATS = [DIRT, STONE, BRICK, PLANKS, STEEL, CONCRETE_WHITE, WOOD]
_TILE_PNG = {
    'dirt': (
        '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'
        'dv2p2+V2QDv9z+OpwzhfKVMiFvOf/9k='
    ),
    'stone': (
        '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'
    ),
    'brick': (
        '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'
        'I86rbWPtf9sUFCoNnNRfj8O6Dnv6L//Z'
    ),
    'planks': (
        '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/'
        '2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAEAAQADAREAAhEBAxEB/8QA'
        'HAAAAwEBAQEBAQAAAAAAAAAAAwQFAgEGAAcI/8QATBAAAgEDAgMGAwYCBgcECwEAAQIDBAURABIGITEHE0FRYXEUIoEIFTKRobEjQhYkM2KSwRcmQ3LR8PEY'
        'NFLhJTVTY3WCg5OjsrOi/8QAGgEBAQEBAQEBAAAAAAAAAAAAAQIAAwQFBv/EADoRAAIBAQUFBAkEAgMAAwAAAAABAhEDEiExQQQFUZGhExVTYRQiMkJDUrHR'
        '4WJxgaIz8FTB8SNEkv/aAAwDAQACEQMRAD8A/Iqfs67KWORwIkCY5NLep0J+nhr8A9p2rxeiP362TZfD6sLF2Q9mdUBOOEoZUycwxX6bGPLr+ej0zal8Toh9'
        'D2Z+51YnJ2SdmLVQc8BYR2KingutQQo9WLav0zaqf5OiIexbNX/H1Z2o7IeyrauOzhck43SXqfn6fj0Lbdr8Xojeg7L4XVii9jvZYkx+K7N4C780SPiKoRfr'
        '8/L11Xp210wtn/8AlfYO79l1surGh2I9kzoAOzVBI3UJxPUEL7fN++stv2zxf6r7G7u2Twv7P7m6bsf7HC2ZOyiSop1wryw8VTq4P+6W5/TQ9u21fG/qvsbu'
        '/Y/C/s/uMnsb7FZm7yLsrqKeBDtIqOJKlnY+GQGH5DU+n7cs7b+q+xu7tj8Lq/uff6G+xOWbYOy2r/Ccd3xFU8z9WOt6dt9K9t/VfYru/YvC/sw83YJ2Q7UE'
        'XZTU7xzPecUTqCNZbw2zW2/qiXu7ZNLL+zCv2D9jzRI47Ipoy3RP6YTZHvnW7w2zxv6o3d2yeF/Zkmu7Aezanmjkh7NhDBjmknFUjkn11feG1NY2v9UZbv2Z'
        'fD6s+TsQ7MoplFV2fxRRMM7qfiCSQj6Z1Ppu1PK16IfQdl1s+oU9jfZZFG/+oFQoJ+Rk4lYYHqM9db03an8X+oPYdm8PqDHZL2UdxtbgmqWo3Y/iXiV1Iz1P'
        'Pl7afTNs8TojehbJrZ9WfP2Xdk8JJXguadAcMy1dRtX0z4/TR6Vtj+J0Q+h7Kvh9WEg7OuylmJHAiQJjk0t6nQn6eGh7TtXi9EUtk2Xw+rCxdkPZnVATjhKG'
        'VMnMMV+mxjy6/no9M2pfE6IfQ9mfudWJydknZi1UHPAWEdiop4LrUEKPVi2r9M2qn+ToiHsWzV/x9WdqOyHsq2rjs4XJON0l6n5+n49C23a/F6I3oOy+F1Yo'
        'vY72WJMfiuzeAu/NEj4iqEX6/Py9dV6dtdMLZ/8A5X2Du/ZdbLqxodiPZM6ADs1QSN1CcT1BC+3zfvrLb9s8X+q+xu7tk8L+z+5um7H+xwtmTsokqKdcK8sP'
        'FU6uD/uluf00PbttXxv6r7G7v2Pwv7P7jJ7G+xWZu8i7K6ingQ7SKjiSpZ2PhkBh+Q1Pp+3LO2/qvsbu7Y/C6v7n3+hvsTlm2Dstq/wnHd8RVPM/VjrenbfS'
        'vbf1X2K7v2Lwv7MPN2CdkO1BF2U1O8cz3nFE6gjWW8Ns1tv6ol7u2TSy/swr9g/Y80SOOyKaMt0T+mE2R751u8Ns8b+qN3dsnhf2ZJruwHs2p5o5IezYQwY5'
        'pJxVI5J9dX3htTWNr/VGW79mXw+rPk7EOzKKZRVdn8UUTDO6n4gkkI+mdT6btTyteiH0HZdbPqFPY32WRRv/AKgVCgn5GTiVhgeoz11vTdqfxf6g9h2bw+oM'
        'dkvZR3G1uCapajdj+JeJXUjPU8+Xtp9M2zxOiN6Fsmtn1Zx+y7snhJK8FzToDhitXUbV9M+J9tHpW2P4nRD6Hsq+H1Z7l6WSFii11wnkc5iBcBl8wdy9M+Ov'
        'DXWiPfTzMSUFbIr97W1UoxiNS43KfHOE03lwCj4n0kdUsSqa2aNugO8ISfPmOetVVyGjNQxTyrGzzVEkiH5+8eN0B9DjUvB4CvMcMdyFQoL1O1cMzIYV3H/w'
        '8166Eo0NU7X0hijLR00xaUHfHLJExGDyzgc86y/czqT6S01VVG0stHDBCDhTsUMT4nIHIe2tKSWCZkm9Bn7tWmXFTU0Ybm0abZFPljrk++pvVyTFqh2JKkGQ'
        'mrSR2HIJ3iLjw2+g01XA1DUlvkjhMqzRrN0O+okbJ8tZNZGpqiZIK0VD5r1h24yxjnYZz4a6K7wIx4nCkTlC1ZFN4OrCQE+f4l1nXgP8i1wtkcs4mMiQQNyk'
        'Z49mBjr68vDVRbyCSPpeHbe9LH/3iZJRuDxSbQ2OY6D/AJ6aVOSZNxNBU4fgLI0JubeJ7qTJx9Ryxo7R60NcXmUaeyiHDwV99jVuiM4Af6DUO0rmkWoUybOP'
        'SyQsUWuuE8jnMQLgMvmDuXpnx1r1caI1KamJKCtkV+9raqUYxGpcblPjnCaby4BR8T6SOqWJVNbNG3QHeEJPnzHPWqq5DRmoYp5VjZ5qiSRD8/ePG6A+hxqX'
        'g8BXmOGO5CoUF6nauGZkMK7j/wCHmvXQlGhqna+kMUZaOmmLSg745ZImIweWcDnnWX7mdSfSWmqqo2llo4YIQcKdihifE5A5D21pSSwTMk3oM/dq0y4qamjD'
        'c2jTbIp8sdcn31N6uSYtUOxJUgyE1aSOw5BO8RceG30Gmq4Goakt8kcJlWaNZuh31EjZPlrJrI1NUTJBWiofNesO3GWMc7DOfDXRXeBGPE4UicoWrIpvB1YS'
        'Anz/ABLrOvAf5FrhbI5ZxMZEggblIzx7MDHX15eGqi3kEkfS8O296WP/ALxMko3B4pNobHMdB/z00qckybiaCpw/AWRoTc28T3UmTj6jljR2j1oa4vMo09lW'
        'La0NffY1bkEZwA/0GpdpXNItQpqxmS1tWGJn4goULgB1nrXOMdOo6ftrneuvIulVmUEsV2QqBdqKWlU5NRDdljZ/Icx4frovR4dDNP8A1hXNZHTSCaM1bA5W'
        'Vq6OfYvpy56MG/wGgRbhMKYFbNKT0BhSIB/XOpuriNfI4blXJtRrfVKTzLDuAceXP99FFx+o/wAGkvIqYwncvOWJ5zT08e0DzP7aHF1r9xvf7gbienrZjG1D'
        'UEMucLXU7pj0HXOijWKfRinpTqMwpbY5YY6i11gkC53irgwV8tT63H6lUXAxNV2li06pVd/uwE+IXco6Y+Xl01VJUBtVJMlZaKg1ET26VFU4PewNIefjnI11'
        'SmsakVjULRW22mPKLVKgbCBY5FVT+Z1LlIUohJ+GqUSyNJBcZATkdzA7qD6+P00qcqZoHFak9+CqGAxRb3Vpn3qk0roVOOrKwwPQartZZhcSBngRZnYr8XEi'
        'k7djl8eoC+et2wdmLrwjWQSndX3GmikbbtNPMQzeWQD+eq7RPRGutPNjbcKTQRAm6wd6Bkd48yEepyuhTroLQKS1tWGJn4goULgB1nrXOMdOo6ftrXrryNSq'
        'zKCWK7IVAu1FLSqcmohuyxs/kOY8P10Xo8Ohmn/rCuayOmkE0Zq2BysrV0c+xfTlz0YN/gNAi3CYUwK2aUnoDCkQD+udTdXEa+Rw3KuTajW+qUnmWHcA48uf'
        '76KLj9R/g0l5FTGE7l5yxPOaenj2geZ/bQ4utfuN7/cDcT09bMY2oaghlzha6ndMeg650UaxT6MU9KdRmFLbHLDHUWusEgXO8VcGCvlqfW4/Uqi4GJqu0sWn'
        'VKrv92AnxC7lHTHy8umqpKgNqpJkrLRUGoie3SoqnB72BpDz8c5GuqU1jUisahaK220x5RapUDYQLHIqqfzOpcpClEJPw1SiWRpILjICcjuYHdQfXx+mlTlT'
        'NA4rUnvwVQwGKLe6tM+9UmldCpx1ZWGB6DVdrLMLiQM8CLM7Ffi4kUnbscvj1AXz1u2DsxdeEayCU7q+400Ujbdpp5iGbyyAfz1XaJ6I11p5sbbhSaniBa6w'
        'd6Bkd68yEepyuhTroU0OT3e6LJIJKqjqI8fLtaOTPmTkeOoos0mFWSqlrnNEJFtNqqoee0yU/M+fTXRXVhVktSeNCatbXUqOJaKj7ksSYYaf5c+WPHXWkXky'
        'KtZoaa4VcVJFKvD9KiZAUzxsGY+GADqKKtLw1wyGP6Rd1EDJZdkjdBHBuAPoT11NzHCRd6mgxDxBTlmnkpKmnyACkduWUDHjj99Dg8q9RUln/wBDy8WUtSoU'
        'mqVQcMEssYZPU5OufZtf+jfT/wDBObiSx0ss6hqiVycE1VsUYYjw2nrqlZzkvyZziv8AwZp7xa61YysFBHM67WcrLAwHkeWProcZLiZSTxGTTWqdZJJJ1p2Q'
        'YWNbkHEg9ARnU1l/qHAsUFminiQ0dRWQcgWSUJJEx8idc3Jp4pFU4VFqi37pY5IZYQ6N1aKaMt77Wxj110UqZ/8ARLVTSU1XFHIFpBUBjuZkuEyBj7Mp1F6P'
        'HoWoyFJYKiESNNYLr3QUlZqKuVufkQcHVKUXlJfyiXGSzRI+8oISZDLxFSOBzeUyEKc8/wAPhro03wYVSzqPU15rRIY6TidpImGSZqpuf+IdNTdwxiF7gwc9'
        '3uiySCSqo6iPHy7Wjkz5k5HjqqLNJhVkqpa5zRCRbTaqqHntMlPzPn01aurCrJak8aE1a2upUcS0VH3JYkww0/y58seOutIvJkVazQ01wq4qSKVeH6VEyApn'
        'jYMx8MAHUUVaXhrhkMf0i7qIGSy7JG6CODcAfQnrqbmOEi71NBiHiCnLNPJSVNPkAFI7csoGPHH76HB5V6ipLP8A6Hl4spalQpNUqg4YJZYwyepydc+za/8A'
        'Rvp/+Cc3EljpZZ1DVErk4JqrYowxHhtPXVKznJfkznFf+DNPeLXWrGVgoI5nXazlZYGA8jyx9dDjJcTKSeIyaa1TrJJJOtOyDCxrcg4kHoCM6msv9Q4Figs0'
        'U8SGjqKyDkCyShJImPkTrm5NPFIqnCotUW/dLHJDLCHRurRTRlvfa2MeuuilTP8A6JaqaSmq4o5AtIKgMdzMlwmQMfZlOovR49C1GQpLBUQiRprBde6CkrNR'
        'Vytz8iDg6pSi8pL+US4yWaJH3lBCTIZeIqRwObymQhTnn+Hw10ab4MKpZ1Hqa81okaOk4naSJhkmeqYZ/wAQ6aLuGMQvcGHktNpZY6lxSOxP8MfD7h+Q8PXT'
        'WSqgVGxUZ+IZKKqggRfmKxK0Sk+flo85IcNCfVPVwbYxT0dRIeYcyEFh6+uqSTxqDbQrL9/zyOIqKAIGyJDMTu8yB4aqlmtSKz4D9JHc6lVkmrwq5x3YABz9'
        'fHUO6skdFV5sstQwU6ZqbnUJIuGLyyhQPfbqE65IqlNSReaqmIE9BFJVzP1mVmww8yxHPVxTykS8MUefW43CkkE2wb4mwI9+88/TGu12LwObk0PVXFbJAsVX'
        'VrDUyclecnaD/e5ctcuzxrFYFuapiUdyTwLJ3ltrHx8yYz9QcctTk9SswLTRxvteNKTa2Q1MzIpU+BweZ045ph/BYoYmmaUQXSshlfBQ00hXbgeKkEHXNya0'
        'KohS4Wq6tL3qXyWVtpUCqqCnLx5AY10U45OIXXxEIUeldxWUzXEY5yJcnOD5401TydP4Nis0Co7nYmqXbuKyKLHMLWOBn38tZxnRApItd/w3NTwkS4ZWz3cs'
        '+focjGpUbRA3Fh5LTaWWOpkFI7E/wx8PuH5Dw9dNZKqMrreAqM/EMlFVQQIvzFYlaJSfPy0eckOGhPqnq4NsYp6OokPMOZCCw9fXVJJ41BtoVl+/55HEVFAE'
        'DZEhmJ3eZA8NVSzWpFZ8B+kjudSqyTV4Vc47sAA5+vjqHdWSOiq82WWoYKdM1NzqEkXDF5ZQoHvt1CdckVSmpIvNVTECegikq5n6zKzYYeZYjnq4p5SJeGKP'
        'PrcbhSSCbYN8TYEe/eefpjXa7F4HNyaHqritkgWKrq1hqZOSvOTtB/vcuWuXZ41isC3NUxKO5J4Fk7y21j4+ZMZ+oOOWpyepWYFpo432vGlJtbIamZkUqfA4'
        'PM6cc0w/gsUMTTNKILpWQyvgoaaQrtwPFSCDrm5NaFUQpcLVdWl71L5LK20qBVVBTl48gMa6KccnELr4iEKPSu4rKZriMc5EuTnB88aap5On8GxWaBUdzsTV'
        'Lt3FZFFjmFrHAz7+Ws4zogUkWu/4bnp4cS4ZWz3c0+eXkcjGNSlaIG4smm6U8FX31LVwURcfPTyNlV+g1dG1SSqGWKDniqk2GOsuttnCsNyxHYR5Bj/lo7N5'
        'qLNfWrQxLxTQPuSNKQIw+R46lXYHzI8/TUqEhvRHKR4sq5qEIZMs5kUD0JydS0+A1WdRlJ2iDSw3qnaBuYRoVmAPoRz1LXGJSfBgeIKaoraanRbxEs7fNH/U'
        'iv5EDx/XlrpZ0WcepM6vJkxae4fBTSPVzSLEQZGRduxs4xz8fTWbjXBAq0zFKyK4FGqZaKFoYgWDtMUdcDmdw8f20pxyqaVaVaPKXxamnVMwbRUOBkS7w2fE'
        'A+mvTCj1yOE6rQLSUE0NGn4zEh5hSPnOfHx99EpJsuMXQLWirmcNHNVsV5mNtpUDzB89QmlnQ1G8i1aKKdkjWSvUBxzJmC41xlJaI6xXFlmltCxSLGLsjNID'
        'iJJQ5IHudS5Nr2TJUeZyspqCjy9TVp0KqksqjPsB/noV55IW0s2QqtLXUIrvNQtGefzS7znywNdo3loyHdZLtd6p6OpMccdNLhs7ZAMMPDXolFtVxOCaToeg'
        'N0poKvvqWrgoi4+ankbKr9B/ydeejapJVO+WKDniqk2GOsuttnCsNyxHYR5Bj/lo7N5qLNfWrQxLxTQPuSNKQIw+R46lXYHzI8/TUqEhvRHKR4sq5qEIZMs5'
        'kUD0JydS0+A1WdRlJ2iDSw3qnaBuYRoVmAPoRz1LXGJSfBgeIKaoraanRbxEs7fNH/Uiv5EDx/XlrpZ0WcepM6vJkxae4fBTSPVzSLEQZGRduxs4xz8fTWbj'
        'XBAq0zFKyK4FGqZaKFoYgWDtMUdcDmdw8f20pxyqaVaVaPKXxamnVMwbRUOBkS7w2fEA+mvTCj1yOE6rQLSUE0NGn4zEh5hSPnOfHx99EpJsuMXQLWirmcNH'
        'NVsV5mNtpUDzB89QmlnQ1G8i1aKKdkjWSvUBxzJmC41xlJaI6xXFlmltCxSLGLsjNIDiJJQ5IHudS5Nr2TJUeZyspqCjy9TVp0KqksqjPsB/noV55IW0s2Qq'
        'tLXUIrvNQtGefzS7znywNdo3loyHdZMtV5p6OqMaR0suGziRQQw16JRbVcTgmk6F6xW6orFUvS9whPJzGTn9Py15rSSWTPRFN5orT2G3zYLRQ7ujbqcKzLrk'
        'rSSLcYvEC9lsclUVlWKmG0MpWjOCPpqr9olh9SXGNRWqtXDELvI1PFNCBnvJaRxn1x0xqlO1eRLjDUH8VwrSSJ3lStKrAhY6dWQn6Dx1VLaWnM1bNagjxNHB'
        'UxClr7pXUsRwlPUwsF58h8x5403G1ikmF5J4Op5663+sqXqSlpp5Ep5wZJBLM5Zl5lWHLoOf567Rglm8zm5PREKkoqm/y1M9xrpC0kgjamVykZVssWAz+EAA'
        'e+u7cbNJRRwSc3iz2VntMUoEtPBXLUMoWFKd+8AwOjBug145yeToeqMVoNz0Ms1XJFWyTwyBQGGxQFHgQR0GpTSVYl00YzT8FWuIRqtfUxIo3ZEYkXPoT11x'
        'dpJ5pFKEVqO0nDo7wBaqrlp1/GWpo1JHnk8s6HJcOo0Fq2/8PWU91PULUSAZd5liJH0X9tdY2c5YpHNyjHNnnqrjuyy7/h4KyafntENJHtI8MltdVY2iz+pH'
        'aRyRKrrvPcQUiplp45eYIRd0fnkgdddIxSzdTNt5IJR0sij/ALjT4U57wqWYnzJ0t+ZKi+B6mxW6orApel7hCeTmMnP6flrzWkksmdopvNFaew2+bBaKHd0b'
        'dThWZdclaSRbjF4gXstjkqisqxUw2hlK0ZwR9NVftEsPqS4xqK1Vq4Yhd5Gp4poQM95LSOM+uOmNUp2ryJcYag/iuFaSRO8qVpVYELHTqyE/QeOqpbS05mrZ'
        'rUEeJo4KmIUtfdK6liOEp6mFgvPkPmPPGm42sUkwvJPB1PPXW/1lS9SUtNPIlPODJIJZnLMvMqw5dBz/AD12jBLN5nNyeiIVJRVN/lqZ7jXSFpJBG1MrlIyr'
        'ZYsBn8IAA99d242aSijgk5vFnsrPaYpQJaeCuWoZQsKU794BgdGDdBrxzk8nQ9UYrQbnoZZquSKtknhkCgMNigKPAgjoNSmkqxLpoxmn4KtcQjVa+piRRuyI'
        'xIufQnrri7STzSKUIrUdpOHR3gC1VXLTr+MtTRqSPPJ5Z0OS4dRoLVt/4esp7qeoWokAy7zLESPov7a6xs5yxSOblGObPPVXHdll3/DwVk0/PaIaSPaR4ZLa'
        '6qxtFn9SO0jkiVXXee4gpFTLTxy8wQi7o/PJA666Rilm6mbbyQWippEH/caf5TnvCpZifMnS2uJKi+B+hrc62pplWrp5513cvi68LgeGVTXkpFPD6HereaMz'
        'U8VRJ3lTBbQu4A5Zzt9evPUqVMFUtxqG+4KaoaUR3KnjYD/ZbsKPLGdF98Au+YKms8jRyoptshA5yVJlKv7eGNZvXE1HoO/dkEHdyVlkpqzltHw8g+QHyLDp'
        '5kaK/qNTiiXcKCjt1WamOOWgZRlPiYi8Xty8P+urjJyVHiS0ovACbhI8ik2uQSSAMJaWBJYZPJic5PLw8tN3zNXDISS2xLVsaK1CKqH4nWmIQeJypOB+2ul5'
        'tetIiiTwQ7SvNGojoa+2W95GJeSpn3SSHxPdoOufXXN01TZarTApxWedozJHXRtuBDzPTM+8+u7XNzjqupd1vUCLDeYZFP3y8RAOGp1RPlz4A/tpv2b90Lsl'
        'nIltQVO6RbhcmrUYYVqyTCr7gciPXV1T9mNP2JSazdSclgWF0nS02OYISN6sGY58dVebwvM11cEMR3GpppHjey25lHQwU0jcvXljQ4VxvMpTphRBKn4yqdHg'
        '4fEozg7KBvl8jnI5a0UlnLqDbehYpLpcbbSfDNZLhCi/7WnSKPb6YbJGpcIvG8gq1hRmkudbU0yrV08043cjV14XA8Mqg00inh9DVeqMzU8VRJ3lTBbQu4A5'
        'Zzt9evPUqVMFUtxqG+4KaoaUR3KnjYD/AGW7CjyxnRffALvmCprPI0cqKbbIQOclSZSr+3hjWb1xNR6Dv3ZBB3clZZKas5bR8PIPkB8iw6eZGiv6jU4ol3Cg'
        'o7dVmpjjloGUZT4mIvF7cvD/AK6uMnJUeJLSi8AJuEjyKTa5BJIAwlpYElhk8mJzk8vDy03fM1cMhJLbEtWxorUIqofidaYhB4nKk4H7a6Xm160iKJPBDtK8'
        '0aiOhr7Zb3kYl5KmfdJIfE92g659dc3TVNlqtMCnFZ52jMkddG24EPM9Mz7z67tc3OOq6l3W9QIsN5hkU/fLxEA4anVE+XPgD+2m/Zv3QuyWciW1BU7pFuFy'
        'atRhhWrJMKvuByI9dXVP2Y0/YlJrN1JyWBYXSdLTY5ghI3qwZjnx1V5vC8zXVwQxHcammkeN7LbmUdDBTSNy9eWNDhXG8ylOmFEEqfjKp0eDh8SjODsoG+Xy'
        'OcjlrRSWcuoNt6FikulxttJ8MbJcIVX/AGtOsUe30w2SNS4RbreQVawoyU3FFqrUZWmqgrqVZWoZMEH21dyUcvqClFi8lTZ+8Vo5atgq4CpSS5H5jlnz8tTS'
        'flzGsRcVlM7M60k80qnAjjppS3tk45Y/z1aTybBtDlFQTTUpaCKalhJ/htJBMGU+e3Q2k8WZJvIMliemV2lv92jDdTHSSYB9M/trOSfuoKNZseNvt1LREx8Z'
        'XR6wr/DE8JZM+qbfpo19hGab94VoaSaAExXG2ORyLx0hjcn1AOM+uNaTWqfMYp8UZq6ZW/hzVy90TudIpJNvtjGDoTWiNR6morHTSRFKaetjpXJOIECs59Ce'
        'ePbQ5OuNBu4DC8MGCOM0lhvMzBjlqqqOH9OvTQ51zkuQqPkzUdtqqSqYDhWmwoyslVXxk5z0wW6a1U17XQVVP2eozTXmso2AQcLUyD+WtrlJX0IAOda6nxf8'
        'DVrgEl4zuk8wWC8cJQuhCqtOZWI5eBwM6VBcGRV8UCl45usjsx4rtm8ZUxRUZbp5E+Wjsov3WVffEhVHaBdIqhVbiUrT55vDSryHoPEa6qyjSqiQ7R1xYQ8e'
        '0pX+s3m4XHHQCkA2/UaFZPRJA5panW4otVarK01UFdSrK1DJgg+2tcmsvqKlFi8lTZ+8Vo5atgq4CpSS5H5jlnz8tTSflzGsRcVlM7M60k80qnAjjppS3tk4'
        '5Y/z1aTybBtDlFQTTUpaCKalhJ/htJBMGU+e3Q2k8WZJvIMliemV2lv92jDdTHSSYB9M/trOSfuoKNZseNvt1LREx8ZXR6wr/DE8JZM+qbfpo19hGab94Voa'
        'SaAExXG2ORyLx0hjcn1AOM+uNaTWqfMYp8UZq6ZW/hzVy90TudIpJNvtjGDoTWiNR6morHTSRFKaetjpXJOIECs59CeePbQ5OuNBu4DC8MGCOM0lhvMzBjlq'
        'qqOH9OvTQ51zkuQqPkzUdtqqSqYDhWmwoyslVXxk5z0wW6a1U17XQVVP2eozTXmso2AQcLUyD+WtrlJX0IAOda6nxf8AA1a4BJeM7pPMFgvHCULoQqrTmViO'
        'XgcDOlQXBkVfFApeObrI7MeK7ZvGVMUVGW6eRPlo7KL91lX3xIVR2gXSKoVW4lK0+ebw0q8h6DxGuqso0qokO0dcWbPHtKV/rN5uFxx0ApANv1HPQrJ6JIHN'
        'LUBRtdA8byXG+yAkbtwp4SAfRjy1TuZUXUycuLK1VXzQsRHcbkHHVKmqp0CjxJI1CSei6l1fE1FX/P3iXWYxbcFhd4wG9enXUuPl0NXz6gqe6XCaYPT3SdHB'
        'IIe8REY9QP31rqWa6GUnXMLHe7whEY4lKxg9PvNGYn8uei7H5ehrz4j6V94ETyffFZK/Tcahceo6Z+uj1eBWJKmrJJn2yXMxTMcd21emM+ILFeR8c9Dpp5dA'
        'r5mKi6NE3dvflhhLZ7v48/OPM7YyD7DSo1yj0M21nI1LeqOXApr1FMoPymWqlAz44wgxpuv3o/Q15aMyY6SmkE09bbKtiMgPJVy7vXAGPrqk21gn0Ios211G'
        'KO4WuYs3w9jSReuaOo3Z8vmGs4yXHmjJp8Bt7xBRxs6Uthd+oC0jhn+m3pqaPz5hX9gQ4yglLGc21CB8gp7dMxPL8PJf11nB6V5jeVCfLe++n2pUiFJMg91Y'
        '3wPTJ8v11V3iuoXvPoS3qg1QO6mnjwx/sLMO8YeYDHV0wxXUK/7Q0bjXqrvBXX+SEDli1xKwPh9NF2OqXMbzWr5BaNrorxvJcb7ICw3bhTwkA+GGPLWdzKi6'
        'mTlxZWqq+aFiI7jcg46pU1VOgUeJJGoST0XUur4moq/5+8S6zGLbgsLvGA3r066lx8uhq+fUFT3S4TTB6e6To4JBD3iIjHqB++tdSzXQyk65hY73eEIjHEpW'
        'MHp95ozE/lz0XY/L0NefEfSvvAieT74rJX6bjULj1HTP10erwKxJU1ZJM+2S5mKZjju2r0xnxBYryPjnodNPLoFfMxUXRom7t78sMJbPd/Hn5x5nbGQfYaVG'
        'uUehm2s5Gpb1Ry4FNeoplB+Uy1UoGfHGEGNN1+9H6GvLRmTHSU0gmnrbZVsRkB5KuXd64Ax9dUm2sE+hFFm2uoxR3C1zFm+HsaSL1zR1G7Pl8w1nGS480ZNP'
        'gNveIKONnSlsLv1AWkcM/wBNvTU0fnzCv7AhxlBKWM5tqED5BT26ZieX4eS/rrOD0rzG8qE+W999PtSpEKSZB7qxvgemT5frqrvFdQvefQlvVBqgd1NPHhj/'
        'AGFmHeMPMBjq6YYrqFf9oaNwrwrtBXX+SEDkRa4lYHw+mi7HVLmN5rV8j+fx267I3jm4NqZo2/lemnyPqRr7r2DGqtFzR8Nbfxs3yYUfaIj3K03BFVOE5Dck'
        '2T6E7emju56Wi6D3gtbN8mP0v2oko5jIvZnLUpggx1Ecjp747vro7tfjLobvJaWT5MUX7TM8fOLgepjUNkRR0OBjyzszrd2t52q5m7ySysnyMSfapmjm3ydm'
        '0k2P/aRkH89mq7rqv8yIe80vhMBV/axklk3jsri34+VnlnU/moGrW6afH+hL3rX4L6gY/ta3WKAonZ1LFCeQENfVoP8APS90wbxt+iN3rJZWL6gB9rK/mpE6'
        'cI3LI5bFuVRtx/hzp7ps6UdquS+4d6zzVk+bKMn2yeIsANwTcUX/AOJ1A5+f4NQtz2S+MuS+4972ngvm/sBg+2LxNBIH/obWyE5/FcakZ8ug8NV3RZeMuSDv'
        'W00sX1GYvtk8UMx/1JuDZGDtuVXz/TR3RZr4y5IVvSbzsX1NN9rPisjc3B16R/BjXVeNvl01Pddn4q6Fd5y8J9Tr/aq4nqSD/RC6xgjHOprZP0yNHdkPFXQe'
        '8peE+oB/tIXWUjvOEatiBjLxVRJ+udbu2K+IuaHvJ+G+TF37eK2oZpJOC5TL1EgpqncPrq+74r4vVE+nt/DfJhKbt7uago3CNa278Ugp6jJ9SMaHsEfEXNG9'
        'Pl4b5M0O3XZG8c3BtTNG38r00+R9SNS9gxqrRc0Utv42b5MKPtER7labgiqnCchuSbJ9CdvTR3c9LRdB7wWtm+TH6X7USUcxkXszlqUwQY6iOR098d310d2v'
        'xl0N3ktLJ8mKL9pmePnFwPUxqGyIo6HAx5Z2Z1u7W87Vczd5JZWT5GJPtUzRzb5OzaSbH/tIyD+ezVd11X+ZEPeaXwmAq/tYySybx2Vxb8fKzyzqfzUDVrdN'
        'Pj/Ql71r8F9QMf2tbrFAUTs6lihPICGvq0H+el7pg3jb9EbvWSysX1AD7WV/NSJ04RuWRy2Lcqjbj/DnT3TZ0o7Vcl9w71nmrJ82UZPtk8RYAbgm4ov/AMTq'
        'Bz8/wahbnsl8Zcl9x73tPBfN/YDB9sXiaCQP/Q2tkJz+K41Iz5dB4aruiy8ZckHetppYvqMxfbJ4oZj/AKk3BsjB23Kr5/po7os18ZckK3pN52L6mm+1nxWR'
        'ubg69I/gxrqvG3y6anuuz8VdCu85eE+p1/tVcT1JB/ohdYwRjnU1sn6ZGjuyHiroPeUvCfUA/wBpC6ykd5wjVsQMZeKqJP1zrd2xXxFzQ95Pw3yYu/bxW1DN'
        'JJwXKZeokFNU7h9dX3fFfF6on09v4b5MJTdvdzUFG4RrW3fikFPUZPqRjQ9gjpaLmjeny1s3yZ/VR4j4hWNUS/XLfnDFZ3dfQDEXXX5tQs6+yj9Del8xOl4g'
        '4lNSVPE1yKfy/NLkeefk1d2zp7CJrP5mLTXviSOd1j4ruUp5FlaB32k+R2DWUbOnsIzv19pjj3ri0wCJuI7x3OP7JXZM/QAcvTU3bL5UVWfzMDU1XFce6SG7'
        '3IJt6YZm9TzIzpSstYoa2nEFDPxmkhlk4hvPMY/swoA8NvPw/XWfYvC6jLtFjeYaj4k40oUDVHFN2J5hY3mp0A9Pm/fS4WTeEF1NetNZMKnHvG0IITjK5y7x'
        'hlFVSnaPMALyOlWdk/cXU5uVp8z6HwvfFW4s3Fl5dcHmayFsfpppZfIhrN++xetvfEkRVpeM7vuIzyr6YcvLGNZKzfuLkxrNe+xR+Ibl+KfjCswT8jTXCnVu'
        'vTlqrsdIdCbz1kGi4muxdP8AW65TAcysNyi2keXQ6hwh8i5CpTzvsLJxvVCcYvlWRtx3clwizj6Y56OyjT2VyKdo6+0Cl4uuEy95JerlCrZAWGrQ9PYHVKzi'
        'soom/J6s1BxHc4iGHFVU0eCcSVMTk/4l/TRcj8hr7+YPbLxeZJe9/pDWzhhzAnRfqMIfy0SUKUumTda3ikeJOIVjVEv1yD5wxWd2X0AxF11ChZ6xRV6XzE6X'
        'iDiU1JU8TXIp/L80uR55+TV3bOnsIms/mYtNe+JI53WPiu5SnkWVoHfaT5HYNZRs6ewjO/X2mOPeuLTAIm4jvHc4/sldkz9ABy9NTdsvlRVZ/MwNTVcVx7pI'
        'bvcgm3phmb1PMjOlKy1ihracQUM/GaSGWTiG88xj+zCgDw28/D9dZ9i8LqMu0WN5hqPiTjShQNUcU3YnmFjeanQD0+b99LhZN4QXU1601kwqce8bQghOMrnL'
        'vGGUVVKdo8wAvI6VZ2T9xdTm5WnzPofC98VbizcWXl1weZrIWx+mmll8iGs377F6298SRFWl4zu+4jPKvphy8sY1krN+4uTGs177FH4huX4p+MKzBPyNNcKd'
        'W69OWqux0h0JvPWQaLia7F0/1uuUwHMrDcotpHl0OocIfIuQqU877Cycb1QnGL5Vkbcd3JcIs4+mOejso09lcinaOvtApeLrhMveSXq5Qq2QFhq0PT2B1Ss4'
        'rKKJvyerNQcR3OIhhxVVNHgnElTE5P8AiX9NFyPyGvv5g9tvF5kl73+kNbOGHMCdFHuMIfy0SjClLpk3X2jtRaqOmm+SguxMi8g8dUFPrjOpUpPVdDNR/wBq'
        'KC0GX5Y6Cv2KDzWnkwf8Tapt51NRIDHZ7wlWrRWmqcP+BZIwAcebF+utWNMWGuQRbNcpah/9XKZnJwZJJEUE+h389a9FL2jUfAK1prnP9ZstogjUHDzSCTf7'
        'fMfy1LlFZSZSi+COtQ1caBCvDsdORyBj8PHBGhSj51FqXkBgSrnUrS1VhECj5tsQVj6Et4aqsNUwpLijNXFGshNS1krJVIxEtXEMnwJ5YJ99VGWGFUS1pgae'
        'ptyxkyUNseUc3EUcRGPcL9Nb1q4NjRcETVraWlYpDZoacvzxTwocfUrqvWeLkSkloZivEVD39QtjqWLNtOTDzPouCdLTeF41dboa13Frok2y0yKQcgsVQAfl'
        'qZK68WKdcka7x2rEhqAaKAdC+zbn1bbn20aVWI64lFPgKWQkXOCJ5Wx3aFySfAnC4Goq3oUrq1GBQUFUTFLeI1UncVJkG76AaLzWKRmk82NLZ6BYs/eNdLGh'
        '6QpUsffkNa/LgjUXEVqLVR003yUF2zIOQeOqCn1xnSpSeq6EtRFBaDL8sdBX7FB5rTyYP+JtU286mokBjs94SrVorTVOH/AskYAOPNi/XWrGmLDXIItmuUtQ'
        '/wDq5TM5ODJJIign0O/nrXope0aj4BWtNc5/rNltEEag4eaQSb/b5j+WpcorKTKUXwR1qGrjQIV4djpyOQMfh44I0KUfOotS8gMCVc6laWqsIgUfNtiCsfQl'
        'vDVVhqmFJcUZq4o1kJqWslZKpGIlq4hk+BPLBPvqoywwqiWtMDT1NuWMmShtjyjm4ijiIx7hfpretXBsaLgiatbS0rFIbNDTl+eKeFDj6ldV6zxciUktDMV4'
        'ioe/qFsdSxZtpyYeZ9FwTpabwvGrrdDWu4tdEm2WmRSDkFiqAD8tTJXXixTrkjXeO1YkNQDRQDoX2bc+rbc+2jSqxHXEop8BSyEi5wRPK2O7QuST4E4XA1FW'
        '9CldWowKCgqiYpbxGqk7ipMg3fQDReaxSM0nmxoWegWLP3jXTRoekKVJPvyGtflwRqLiOVXHdVTU0ay1NHI2PmLNLiPyxz5nQoVeCJboQ5uLpqio2rfLbTr6'
        'xyOD/wA+OruUXssL1XmN1Fes38N73a2VhjvI6SZgD58tSl+l8y6+YKC3VTviHiag7voVFA/5dNVVaw6m/kpQ8PV/dK9PfqNnK81W1Fvyzjnri5RrRx6l0byf'
        'QZtvBzLCkclXVUsmCwiFBEqyf3lBJOplaRrl1FRZRp7HNSSYMNbXzbeeO5hQDwPTWvJ+RrrNVDGjgieClgkbAaQSywnHLny26yo9QxJs13rKmOV/ulEi/E7/'
        'ABCLjA5HAAHTTRZJjV8BJeJq+GUtFCBCw6jdIy+TcuXPVXE0TeaZxuIKmRe9aRl2nGViKD3zouLIb1RKLiaWarMcq1k7r/NsLqf2xq+zSVcCL+NGMxRVddLi'
        'QIwb51jKBOnv5eeh0RWLKDWi97kMREW5ss3eDp4D21CcXmOOhUhhvjVALTUTSpzUNXBcA+GP8tTSIVdAFxut4oJWDV1sXlnu0qmfafpq0ovRgTqrjuqpaZFl'
        'qaOR8fMWaXEfljn11ahV4IhuhDm4umqKjat8ttOvrHI4P/Pjq7lF7LC9V5jdRXrN/De92tlYY7yOkmYA+fLUpfpfMuvmCgt1U74h4moO76FRQP8Al01VVrDq'
        'b+SlDw9X90r09+o2crzVbUW/LOOeuLlGtHHqXRvJ9Bm28HMsKRyVdVSyYLCIUESrJ/eUEk6mVpGuXUVFlGnsc1JJgw1tfNt547mFAPA9Na8n5Gus1UMaOCJ4'
        'KWCRsBpBLLCccufLbrKj1DEmzXesqY5X+6USL8Tv8Qi4wORwAB000WSY1fASXiavhlLRQgQsOo3SMvk3Llz1VxNE3mmcbiCpkXvWkZdpxlYig986LiyG9USi'
        '4mlmqzHKtZO6/wA2wup/bGr7NJVwIv40YzFFV10uJAjBvnWMoE6e/l56HRFYsoNaL3uQxERbmyzd4OngPbUJxeY46FSGG+NUAtNRNKnNQ1cFwD4Y/wAtTSIV'
        'dAFxut4oJWDV9sXlkRpVM+0/TVpRejAj1VIlDTMjXu0wKn4z3Iyw8eZPPVrF+ywf7ifwlpKl346t1KgH+wCbhy9uuqrLLs2GHzHyV9rTbDTcW1j4HOSJQCw9'
        'OX1Ot6yzgNV8xynu9ibZE964jrirbmxIQHP5dNLU+CQJx4sfqKixSqGSj4tlROZ7uqUBvUgnrrk1LiuRdVwYs8tkeSKVrFfJIyPlapq03/TB5aVGdKXlyF3e'
        'DD0hold5Kax1kO4YVJ5RKR9c6HXWQYLJGklqV7uQRpCIznaaKMg+5J0OmQ4jkl7nePaKwLEgJGzuUTPkDjnoUVwNUFU8RMgBS+SxoQd0ZqByP/yrpUf0g35k'
        'qS7UkSxk1iMx5hRM8gP6dNXdfAKriAS8UEkikRkbOTJAso3++R4aqjoTgVIqkrOu6jpZhGuV+InyGU9QCOhGophmIc3yVpNkdlsiKBlDJWMc/kNa6uLNXyGG'
        'o7jeYZAlp4dWQEFZO+ZgDnmDy8vHR6qebGreiE5+GKxxtmr7TSs5yy00ZOB5DJ66b0dKhRvMHVUqUVMyNe7TAqfjPcjLDx5k89KxfsszyzE/hLSVLvx1bqVA'
        'P9gE3Dl7ddVWWXZsMPmPkr7Wm2Gm4trHwOckSgFh6cvqdb1lnAar5jlPd7E2yJ71xHXFW3NiQgOfy6aWp8EgTjxY/UVFilUMlHxbKicz3dUoDepBPXXJqXFc'
        'i6rgxZ5bI8kUrWK+SRkfK1TVpv8Apg8tKjOlLy5C7vBh6Q0Su8lNY6yHcMKk8olI+udDrrIMFkjSS1K93II0hEZztNFGQfck6HTIcRyS9zvHtFYFiQEjZ3KJ'
        'nyBxz0KK4GqCqeImQApfJY0IO6M1A5H/AOVdKj+kG/MlSXakiWMmsRmPMKJnkB/Tpq7r4BVcQCXigkkUiMjZyZIFlG/3yPDVUdCcCpFUlZ13UdLMI1yvxE+Q'
        'ynqAR0I1FMMxDm+StJsjstkRQMoZKxjn8hrXVxZq+Qw1HcbzDIEtPDqyAgrJ3zMAc8weXl46PVTzY1b0QnPwxWMNs9faKVmOWWmjPIeQyeum9HSoUbzPLJxF'
        'b4kkU01GCo+Vo6YSZ8+WvRdlxITRqK6GtkDxW0VCBcAiniiVfb/jqWqZsU08kcqHeZ1Q0yRc+slQoVfyHM6lLzKw4GXlgSXbI1MykYASokOPfl00pPMKpBVt'
        '1JUsHa6TUzAbCscfyHyJ3Hr661WtDUT1KtLwtSIdj3askbqQrxgYPt09NS5POhkvMZm4Ns7SIXNwq48fMxrQBn1xz1PaSXAbqEaqwWWjeRpYKNYUOFgmqJCf'
        'ZjnmdN+b1Ndis0BilsqxnurZTNKAcIlVtjx7sOQOl39Wb1dETpeJLYCncladwCMAbx+YHM66XJE1joaprvmKL7vWSp+bOAhT9TocPmwG9wKm/iCTYdtPTxtj'
        'KyTM7N+Q1FLNG9dizWW8AkPWJFIx3fw6VmHp8zcs+GqUocApLiYktVXNCHF2rqhwSGhjSKAjzwcapSin7KJab1Fms1NUPHDNFWSgnaWqLixP+FMDGntGsVTk'
        'ZRTzA1TW6zTOojolZiEAmJOznzI59capOUlqS7sRhOIrfEkimmowVHytHTCTPny1N2XEtNGoroa2QPFbRUIFwCKeKJV9v+OpapmxTTyRyod5nVDTJFz6yVCh'
        'V/IczqUvMrDgZeWBJdsjUzKRgBKiQ49+XTSk8wqkFW3UlSwdrpNTMBsKxx/IfIncevrrVa0NRPUq0vC1Ih2PdqyRupCvGBg+3T01Lk86GS8xmbg2ztIhc3Cr'
        'jx8zGtAGfXHPU9pJcBuoRqrBZaN5Glgo1hQ4WCaokJ9mOeZ035vU12KzQGKWyrGe6tlM0oBwiVW2PHuw5A6Xf1ZvV0ROl4ktgKdyVp3AIwBvH5gczrpckTWO'
        'hqmu+Yovu9ZKn5s4CFP1Ohw+bAb3Aqb+IJNh209PG2MrJMzs35DUUs0b12LNZbwCQ9YkUjHd/DpWYenzNyz4apShwCkuJiS1Vc0IcXauqHBIaGNIoCPPBxql'
        'KKfsolpvUWazU1Q8cM0VZKCdpaouLE/4UwMae0axVORlFPMDVNbrNMyiOiVmKxgTEnZz5kc+uNUnKS1Jd2IBbhKiHLSQRYCssUYySfDl+w1LRSDNR22aPEtb'
        'GmRyMkbBg3gdTWS0LonqUTYqD4X5q6TcB0gp2OT6f8dCk3oLiggstL8PGY56l3zlnenY/p/lrVdQoUKKkt0coDrX1LYxIFoevuT10NOmBqlRYqal7ySGCsSk'
        'YgMi0ah/3/TU4vCoZHnOKrn8NEqUEMyZOWldEQgeSjOukIY1kDlgeJErVcjpVXI08e7PeSkcgfDI6H116HRLBHKlcGz0Vr4ft0347uauLoTlefn1GuEpyXu0'
        'O8YLiehhtlipj3ZlgeXHy8wz/T2153K0eJ0UY5F2lrrPHCpmrHTZgLulABPt0z6a5u89BdCXcr29fVOhradokxtCyKDj1I8fbXVQosjm5ECvrpYCX3SmI9Ga'
        'Rjka7RjU5uTJUlyr7p3fw0eQCQHchUA8z4n212UYxzObbeQWLhi6SOHmrJZIcZMFMpQDP8wI/bWvx0QXZPNn0lnhtFOzx26NmYci4LSOfL5vHU3nLBsu6loY'
        'W4Sohy8kEWArCKMZJPh/5alxKTDNR22aPEtbGmRyMkbBg3gdTWS0LonqUTYqD4X5q6TcB0gp2OT6f8dCk3oLiggstL8PGY56l3zlnenY/p/lrVdQoUKKkt0c'
        'oDrX1LYxIFoevuT10NOmBqlRYqal7ySGCsSkYgMi0ah/3/TU4vCoZHnOKrn8NEqUEMyZOWldEQgeSjOukIY1kDlgeJErVcjpVXI08e7PeSkcgfDI6H116HRL'
        'BHKlcGz0Vr4ft0347uauLoTlefn1GuEpyXu0O8YLiehhtlipj3ZlgeXHy8wz/T2153K0eJ0UY5F2lrrPHCpmrHTZgLulABPt0z6a5u89BdCXcr29fVOhrado'
        'kxtCyKDj1I8fbXVQosjm5ECvrpYCX3SmI9GaRjka7RjU5uTJUlyr7p3fw0eQCQHchUA8z4n212UYxzObbeQWLhi6SOHmrJZIcZMFMpQDP8wI/bWvx0QXZPNn'
        'ZLPDZ6dnjt0ZZhyLgtI58vm8dS5OWDZd1LQdqoyzCS4WyuikZvlljbcnoMAcuXL11C4RaOn7o+evoFURxxU1I5zunkjk3HHgD0HrqWpa4lJoXN+uEZAir0Cr'
        'y5yZAHnzGqUVwC8+IKS9V1Vl6e+iasUkhhUbQPQrjpqklrHAh1eTOx3W+1MnO709CmMbkbvXJ8dueQHrqmoLJVJ9Z5soJa56t1R7xVTuFyXkdRn8j11zv090'
        'pxwzGxwXS1UO+aL4ojkfkMhY+w1LtmsnQtWaehOrOBqiod0gtbwRogBaoptpJ9D4jSrVLNmuaJCb8OXqyTKxphFCD8x7wMPpj9Brp2kJo53JRCJXVklXE0kN'
        'hkfae7WppZEYL/v9AT+upajTUVJ10PRUdLea+lAhpbAQ5wEySo9z6a87uJ4tnasmskaXhG60sr9/BYdxOQYojuz5ddX2kHk2RdlqkLQW+qw0FUttqKNDlgC4'
        'wP8Aj5atuOlakpPJ0JUvDPel5YUpqXDHYIp2b5c9T5HzGr7TjiRc4FCS33ymp1SnC1CRLnHed2T/AMR6a5pweZdGsjRud5qJEWttSzyfyyd6CSPEE6Go6SFS'
        'eqEaqMswkuFsropGb5ZYm3J6DAHLly9dWuEWjfuj56+gVRHHFTUjnO6eSOTcceAPQeupalriUmhc364RkCKvQKvLnJkAefMapRXALz4gpL1XVWXp76JqxSSG'
        'FRtA9CuOmqSWscCHV5M7Hdb7Uyc7vT0KYxuRu9cnx255AeuqagslUn1nmyglrnq3VHvFVO4XJeR1GfyPXXO/T3SnHDMbHBdLVQ75oviiOR+QyFj7DUu2aydC'
        '1Zp6E6s4GqKh3SC1vBGiAFqim2kn0PiNKtUs2a5okJvw5erJMrGmEUIPzHvAw+mP0GunaQmjnclEIldWSVcTSQ2GR9p7tamlkRgv+/0BP66lqNNRUnXQ9FR0'
        't5r6UCGlsBDnATJKj3Pprzu4ni2dqyayRpeEbrSyv38Fh3E5BiiO7Pl11faQeTZF2WqQtBb6rDQVS22oo0OWALjA/wCPlq246VqSk8nQlS8M96XlhSmpcMdg'
        'inZvlz1PkfMavtOOJFzgUJLffKanVKcLUJEucd53ZP8AxHprmnB5l0ayNNc71UOi1tqWeTHyyd6CSPEE6KR0kKk9UTlt1wZDEhv0AT5VRK5HUDXW+vLkRcfm'
        'FjtFdTKskcVfCI1JctUK24Y8R+vrqG0+B0WAt8bWNOp+762cthiwiA3L4AjPMaFFUzQXnXIPW3GQRq5t1aykc0FCCx9A3jqox0b6hJ+QG1XCeoXatPVUSkc1'
        'eAAL+Q1U1TWpKddAi0duppS8ix1EQPNZImG1vMEddTWTRVEW6W601K6mOaOgOMJ3VQ6FD45z11yabzxLTSKbVtRXQmGHjGUOw5q1QvyjHqOWuaVM4F5+8Dqq'
        'S5RRrJHxBTzSAbS8rROuPLI8dZOLfsmo+Jn7ouVWhaS4WKcN1apnH/6g6qsVoyaS8hBuB2h/j/H0dM/Qii3AH6eOq7VZUf8AIOzpiKwcFz4dYbpWxhz82KXf'
        'n6t4+uqdquCJ7Ouph7dV0UaU63BKyeNjlai2SRqfLmOp03k3WnUKNYVr/AVVro0KrS29pJeTMGePn6g/pobVc2ViDSjuVA/cyI0u4fhhrdyjn0II66qqeRFG'
        'gtVbqitp0ElLWKR+P+tqjHHny/XQpJPMc9CYtuuLRmNGv0AT5VRK5HUa6Oa8uRNx+YWO0V1MqyRxV8IjUly1QrbhjxH6+uobT4HRYC3xtY06n7vrZy2GLCID'
        'cvgCM8xoUVTNBedcg9bcZBGrm3VrKRzQUILH0DeOqjHRvqEn5AbVcJ6hdq09VRKRzV4AAv5DVTVNakp10CLR26mlLyLHURA81kiYbW8wR11NZNFURbpbrTUr'
        'qY5o6A4wndVDoUPjnPXXJpvPEtNIptW1FdCYYeMZQ7DmrVC/KMeo5a5pUzgXn7wOqpLlFGskfEFPNIBtLytE648sjx1k4t+yaj4mfui5VaFpLhYpw3Vqmcf/'
        'AKg6qsVoyaS8hBuB2h/j/H0dM/Qii3AH6eOq7VZUf8g7OmIrBwXPh1hulbGHPzYpd+fq3j66p2q4Ins66mHt1XRRpTrcErJ42OVqLZJGp8uY6nTeTdadQo1h'
        'Wv8AAVVro0KrS29pJeTMGePn6g/pobVc2ViDSjuVA/cyI0u4fhhrdyjn0II66qqeRFGgtXbqitplElLWKR+P+tqjHHn/AMdCkk8xpUgzT3L5GgvFKQBtyKBg'
        'W9QQddUorOPUhuWjPqP4kRymS50c8ZPQz4fPj1/bQ6aIU3xNQVd0jSTvJAmeQ+GlTLg+p6e2hqOgpy1Y4kdzYLtNwafoqrUqy/kPDRVeRsdB1Z7gpRJTXxqQ'
        'NxjqVQBvAEEZ1LppQatHI6q87yFSSnjQ53tOGPpnA0O4KbLKXaqYR75Z5XwCwNXEAfQApri0jrVjj1qPTyNL38czjASNoZDIfLIHXz1KFfsAarCondW65NE3'
        'J90lMB7bT104av6hXyEqmKyVG7vrZMkyt8zvQwyfkY2B/TVpzWT+pHq6onuLR34KQiN8Bk7ymmTcM9RtY/lqvX49UT6p0zWwR4mkcyH+ZJaleXmOX6ayv6f9'
        'DWKzNz1sVOqpBeLvRoBzalld0A8BucZHvqkm80gr5sUe7RFZImvNzdcbi88aSbh9P89Vd/SgrTURS91dMJXF0mZGXKxmjZw5HReQ6fXV3U8KdSLwBLlU1MXe'
        'z8RQRSOQTE1CVKt4DBOquqvs9SVL9Rqae5ZRoLxSsANuRQMC3qCDqUo6x6lNy4nKP4kRymS50c8ZPQz4fPj1/bWdNEKb4moKu6RpJ3kgTPIfDSplwfU9PbQ1'
        'HQU5ascSO5sF2m4NP0VVqVZfyHhoqvI2Og6s9wUokpr41IG4x1KoA3gCCM6l00oNWjkdVed5CpJTxoc72nDH0zgaHcFNllLtVMI98s8r4BYGriAPoAU1xaR1'
        'qxx61Hp5Gl7+OZxgJG0MhkPlkDr56lCv2ANVhUTurdcmibk+6SmA9tp66cNX9Qr5CVTFZKjd31smSZW+Z3oYZPyMbA/pq05rJ/Uj1dUT3Fo78FIRG+Ayd5TT'
        'JuGeo2sfy1Xr8eqJ9U6ZrYI8TSOZD/MktSvLzHL9NZX9P+hrFZm562KnVUgvF3o0A5tSyu6AeA3OMj31STeaQV82KPdoiskTXm5uuNxeeNJNw+n+equ/pQVp'
        'qIpe6umEri6TMjLlYzRs4cjovIdPrq7qeFOpF4AlyqamPvZ+IoIpHILRNQlSp8BgnVXVX2epKl+o/Fl7VO0mGUSrwTfA6/hb4KX5fyGvsPZNlfxVzR8f0vak'
        '/wDC+QcdsvaO0exuzWuqx1Pe2eRiT5k7dT6Hsvjrmh9N2nwHyCU/blx3S0wgk7LKxuZO77hlz+edS9g2duq2hc0XHb7dLGwfIA/b7xwZvm7KpJdn4R9yT/kc'
        'Nqlu/Z/+QuaIe8bfwHyZz/tK8eU47t+yyNnI+XfZqlG9uT89V3bs7+P1RPeVv4L5MDJ9pPjp0YHsliEh/nFvrAR9N+NPduz/API6oO8rfwOjA1H2iuO6hBs7'
        'M5qdh/LFQ1WCfP8AFoW7dnWduuaKe8ren+B8mLt9oDtCLDvOy2SWPbtCtR1i8/PkdX3bsvj9UQ95bT4HRgZO3TjppAx7KZAhBzGaas2E+DY89Pd2zf8AI+hP'
        'eO0vHsH1AN218XnIPZGyv5rT1YwfPVd32C/+x1Qd4bR4H1N0vblxxDu3dmVUxPMlYKtT78tD3fszyt1zQreG0rOwfJh4u3jjT5dnZfXEDqdtaxP66nu7Z/8A'
        'kL+o94W/gPqdHbJxhUBu97Or3Gh/kigqmA/PT6DYLK2XNG9PtXnYvkwc/atxb3m49nV3VfBJKCbn6k4ydPodh4y5ofTbZ/BfJh4u13jcKyU3Z9dKUvggQ0dU'
        'MHzxjGpex7PrbLmhW22+SsXyY2vHvaIrrOOAr4rDmsht8pI/MaFYbNl2q5j6RtGfZPkFXtU7SYZRMOCb4HX8LfBS/L+Q1L2TZWv8q5ofS9qTr2L5Bx2y9o7R'
        '7G7Na6rHU97Z5GJPmTt1Poey+OuaH03afAfIJT9uXHdLTCCTssrG5k7vuGXP551L2DZ26raFzRcdvt0sbB8gD9vvHBm+bsqkl2fhH3JP+Rw2qW79n/5C5oh7'
        'xt/AfJnP+0rx5Tju37LI2cj5d9mqUb25Pz1Xduzv4/VE95W/gvkwMn2k+OnRgeyWISH+cW+sBH0340927P8A8jqg7yt/A6MDUfaK47qEGzszmp2H8sVDVYJ8'
        '/wAWhbt2dZ265op7yt6f4HyYu32gO0IsO87LZJY9u0K1HWLz8+R1fduy+P1RD3ltPgdGBk7dOOmkDHspkCEHMZpqzYT4Njz093bN/wAj6E947S8ewfUA3bXx'
        'ecg9kbK/mtPVjB89V3fYL/7HVB3htHgfU3S9uXHEO7d2ZVTE8yVgq1Pvy0Pd+zPK3XNCt4bSs7B8mHi7eONPl2dl9cQOp21rE/rqe7tn/wCQv6j3hb+A+p0d'
        'snGFQG73s6vcaH+SKCqYD89PoNgsrZc0b0+1edi+TBz9q3Fvebj2dXdV8EkoJufqTjJ0+h2HjLmh9Ntn8F8mHi7XeNwrJTdn10pS+CBDR1QwfPGMal7Hs+ts'
        'uaFbbb5KxfJja8e9oius44CvisMFZDb5SR+Y1K2fZsu1XMfSNoz7J8j9afiz7T1HKFnHCcTEbtv3rTjHviU6+f2O63rLke3tN48I8wicZ/adnCEUnB7Z/mNw'
        'jOR6/wAbR2G61rLl+DdtvLhHmdqOKftPRRNI83BMEYOP/WMXy+WP4ulWO6v1cjO13l+nmIzX77UMYMi3HhSn6jfFdIkP5mXVqy3UtJcvwTf3m/l5k9+O/tT0'
        'rYF7s2VGc/ecZ5e5k1Sst0vSXL8A570/TzNw9oX2rKlwiVlln5ZybhDg/wD5eus7HdCzvcvwZWu9dFHn+ROt4n+08JJKiqmsNO2efeXWJc/Tv9UrPdOUb3L8'
        'Eue9M2o8/wAgqbtC+04oxBc7EpBxuF2hBPpzm1XYbq1UuT+xu13pkrvP8hh2qfaopZgpv1oc8+RukDD/APro7HdL0lyf2Jv7zWkea+4FO1L7Uk0veC8Wsluq'
        'tdIQD64Munsd1JZS5P7GVpvTS7zX3O1HGv2oCyvJVWVWk/nFyh5+/wDG0Kz3V+rk/sV2m9eEef5Ax8a/aclCpBW2bcx5CK5w5Y//AHtPZbqXzcn9g7Xen6ef'
        '5Ktu40+1XA7RhbLUkDJeproW2/XvhqJWW6OMuX4FT3rwj/v8jJ7UvtRS57un4WLA4/h1ETHl/wDWOp7DdK1ly/BV/enyx/3+Q8PaD9qhiJX/AKNbTy2S1sCr'
        '+Xe6l2W6uMuX4KU956xjz/If+nf2ozN3aU/B8hI/EtdAcD3M3LW7LdVMXLl+Av7yrlHmdk4s+09RyhZxwnExG7b9604x74lOp7Ldb1lyK7TePCPM2nGf2nZw'
        'hFJwe2f5jcIzkev8bR2G61rLl+DdtvLhHmdqOKftPRRNI83BMEYOP/WMXy+WP4ulWO6v1cjO13l+nmIzX77UMYMi3HhSn6jfFdIkP5mXVqy3UtJcvwTf3m/l'
        '5k9+O/tT0rYF7s2VGc/ecZ5e5k1Sst0vSXL8A570/TzNw9oX2rKlwiVlln5ZybhDg/8A5eus7HdCzvcvwZWu9dFHn+ROt4n+08JJKiqmsNO2efeXWJc/Tv8A'
        'VKz3TlG9y/BLnvTNqPP8gqbtC+04oxBc7EpBxuF2hBPpzm1XYbq1UuT+xu13pkrvP8hh2qfaopZgpv1oc8+RukDD/wDro7HdL0lyf2Jv7zWkea+4FO1L7Uk0'
        'veC8WsluqtdIQD64Munsd1JZS5P7GVpvTS7zX3O1HGv2oCyvJVWVWk/nFyh5+/8AG0Kz3V+rk/sV2m9eEef5Ax8a/aclCpBW2bcx5CK5w5Y//e09lupfNyf2'
        'Dtd6fp5/kq27jT7VcDtGFstSQMl6muhbb9e+GolZbo4y5fgVPevCP+/yMntS+1FLnu6fhYsDj+HURMeX/wBY6nsN0rWXL8FX96fLH/f5Dw9oP2qGIlf+jW08'
        'tktbAq/l3updlurjLl+ClPeesY8/yH/p19qNpu7Sn4PckfiWugIA9zNy1uy3XTOXL8Bf3lwjzC0lrtTybu+pnTbmRIqBmY/3SMa8cpy/1n0VGOaN22wUcZl7'
        '2rgigUEx4iMb5J6cxzA1V9si6kFqLDRMkaiamwpLENCxGPcDw1u0aG6mbqLLSTNEJmstTuG1TMZV2j15aFN6VKuryFpLbQQ04Ie17EYfwoZ2bf8AQr0035V1'
        'NdXA2a239yFnp7SueSkVCZHt0JI1k5VwqbDyEZ6DhyWlZjmSdWyGQxjb7Hnq1KaZzuxA1NhtVVTkJS10lVtJUp3Uqkegx8pOm/JahcT0Cw2emFIvxFnukT4+'
        'YR0UTIcfXl7ai+64SXMq6qYoah4WoJwkx4frpM9GS2Rsc+p38tDtZVpf6ioReN0Zg4YNPK5jsdzM5yoiazIV2nl+INy0O0qsZLma7+noDThCkldkm4bqIp1I'
        'YVENBtCjz/Fy9jrdq1lPqa4nnE5UWGkp5goo5jExxIhoEOT6Dfovt69S7qWhiqho4GCLTmkckYcW0IRjyw356ItvWv8AJnRaBIa2gafEjRiQnCJLbc5Hhjrq'
        'mpUw+pKarRn1VDSPIXMdIVHM95QlM/ppjJrKvMGkCpLXamk3CamdNuZEioGZj/dIxqZWkv8AWUoxzRu22CjjMve1cEUCgmPERjfJPTmOYGqvtkXUgtRYaJkj'
        'UTU2FJYhoWIx7geGt2jQ3UzdRZaSZohM1lqdw2qZjKu0evLQpvSpV1eQtJbaCGnBD2vYjD+FDOzb/oV6ab8q6murgbNbb+5Cz09pXPJSKhMj26Ekaycq4VNh'
        '5CM9Bw5LSsxzJOrZDIYxt9jz1alNM53Ygamw2qqpyEpa6Sq2kqU7qVSPQY+UnTfktQuJ6BYbPTCkX4iz3SJ8fMI6KJkOPry9tRfdcJLmVdVMUNQ8LUE4SY8P'
        '10mejJbI2OfU7+Wh2sq0v9RUIvG6MwcMGnlcx2O5mc5URNZkK7Ty/EG5aHaVWMlzNd/T0BpwhSSuyTcN1EU6kMKiGg2hR5/i5ex1u1ayn1NcTzicqLDSU8wU'
        'UcxiY4kQ0CHJ9Bv0X29epd1LQxVQ0cDBFpzSOSMOLaEIx5Yb89EW3rX+TOi0CQ1tA0+JGjEhOESW25yPDHXVNSph9SU1WjPqqGkeQuyUhUcz3lCUz+mmMmsq'
        '8waRep+/nfIo5T0JeOvaEEeuOevNRavod8dCzQlKn5o7XV1xTKrEL6nJumSHGcafVWf0Jak8glNRVj/EItuvffRDc6CvhcKPTz9tS3HDFCq61E47fKwaV6Di'
        'mGnA5sKZGH5ddNUtUOPmDiqKHvSe+4gplA/HU2p1z6AgctDTppzNeXnyIdWaCoidGvss4Vs/NbtpQZ6ZZeX+euivJ+z1IbT16Gaijt9VTKkF7pIEk5FpaCN9'
        '2Oe0HAxrJtPGPUzSawfQp0NLE0kKpd7My7cZa2qBy9mzobwdYvmWl5jVPRVs9YYozw5Ii88ywOUI9g2prFKtGbGtKorw2m6imPd2rhuJFJZe775Nx8z836a5'
        'Xot6lUfkDFJeqkO6cN2RivyoO/qm8efIMNXegtWRSTyFazhe+yzrJ9z0Stj+zpWkTJ8zuc5+ut2kOJShISm4MvgrA1Tw9QGn28wRKWXl4EHnp7SFM3U12XDA'
        'Vm4fdKlKaqsVDAWHed4IJXyvpk6VNUqpA4vJoD8DHHIBJb6QyMSSyULDZz8Mtqr1cn1Iu+QxM8Qpss0MQ3Aq4pnVnPvk51GtDo8j6n7+d8ijmOcEvHXtECPX'
        'HPTRLN9Ax0LNCUqfmjtdXXFMqsQvqcm6ZIcZxp9VZ/QlqTyCU1FWP8Qi26999ENzoK+Fwo9PP21LccMUKrrUTjt8rBpXoOKYacDmwpkYfl101S1Q4+YOKooe'
        '9J77iCmUD8dTanXPoCBy0NOmnM15efIh1ZoKiJ0a+yzhWz81u2lBnpll5f566K8n7PUhtPXoZqKO31VMqQXukgSTkWloI33Y57QcDGsm08Y9TNJrB9CnQ0sT'
        'SQql3szLtxlraoHL2bOhvB1i+ZaXmNU9FWz1hijPDkiLzzLA5Qj2DamsUq0Zsa0qivDabqKY93auG4kUll7vvk3HzPzfprlei3qVR+QMUl6qQ7pw3ZGK/Kg7'
        '+qbx58gw1d6C1ZFJPIVrOF77LOsn3PRK2P7OlaRMnzO5zn663aQ4lKEhKbgy+CsDVPD1AafbzBEpZeXgQeentIUzdTXZcMBWbh90qUpqqxUMBYd53gglfK+m'
        'TpU1SqkDi8mgPwMccgElvpDIxJLJQsNnPwy2qvVyfUi75DEskS02WaGIbgVcUzqzn3yc6jWh0eQ8tyqWICmh75gNsi08bAr5cm8fLRcRr43BV32SFo1ksjqn'
        'P57Yqvj3z4eetSCxx5hWTBz1FbCkcVXRWR42YFcxtGCT48j+ulKLybM66h6gNsJamoyIz8yw1soyfTnz0YCkxillq0kT4ZrrEj9AtzDxr6kE51GGtORWPmAp'
        'Ki9QwiJbhcGiyctWwI4Yk+eqai8aIE5cRl7o4iEFXWWqeRHwKaagwVx5ldFFWqTBtg2mglX4hLTwvMh+YrKsiZ8x15Aa3lVlfwgLvaHcmt4G4fl3/KHgq5QQ'
        'TprJZTZFIt4xQqtRw1EBIeCKSPB2lo6yQY58j8x8dNZ/Oa7Fe6EFbw8HY/0XB65IqsgY6/kOZ0ev8w0XABPNw+J073hmsp425ZhVpT78j+nrppJ5SCsVmhSo'
        'NggLfDW26RE+MiSgD1wW1XrvNo3q6IE9TaizLNdKijZgMF4pQW+vh/10UlnQaxyqM7rVBSK8d9r1kB+dSSVPqoIyNakn7qHDizJ4k7uDfS8QySwxdUdUkUDP'
        'kRkaez4xJv4YMYFyqWYBTQ983JZFp42G3y5N1Plqbi8xvjcFXfZIWjWSyOqc/ntiq+PfPh561ILHHmFZMHPUVsKRxVdFZHjZgVzG0YJPjyP66UovJszrqHqA'
        '2wlqajIjPzLDWyjJ9OfPRgKTGKWWrSRPhmusSP0C3MPGvqQTnUYa05FY+YCkqL1DCIluFwaLJy1bAjhiT56pqLxogTlxGXujiIQVdZap5EfAppqDBXHmV0UV'
        'apMG2DaaCVfiEtPC8yH5isqyJnzHXkBreVWV/CAu9odya3gbh+Xf8oeCrlBBOmsllNkUi3jFCq1HDUQEh4IpI8HaWjrJBjnyPzHx01n85rsV7oQVvDwdj/Rc'
        'HrkiqyBjr+Q5nR6/zDRcAE83D4nTveGaynjblmFWlPvyP6eumknlIKxWaFKg2CAt8NbbpET4yJKAPXBbVeu82jerogT1NqLMs10qKNmAwXilBb6+H/XRSWdB'
        'rHKozutUFIrx32vWQH51JJU+qgjI1qSfuocOLOf0j7uEPS8QySwxYyjqkigZ8iMjT2fGJN/DBnl5J4IKh1qrfJBBnlIIAFPuV/PXpUW1gzhVJ4obZ7dcDG0c'
        'cSKTj+2kDehwTqaSjmV6reAZrdFSd5HFUS06Dnymyo8wM6i83mNEh/bU1BVlrY2VVyiSU7Dn4n5PPXKqWh0o+J2mvqUxZZqQtLtyz06rIMfXBHtrONcUwUuK'
        'KtPdrVWIhkdceCudm33B8da7KI1iytVJaKpVLdxGxGDKWCsyjxyP31y9ZF0ixOXh7h+FX+JggneYbg8VWH5Y64HTVKc9H0NdiKJwpwU+wwS1sJP/AL44/Px1'
        'TtLVcCVCBiThu1rIUjq6tFOTteRZUPqef5aL8nwG7FagqizimhLJVSbRyUtKsYcdOgPTw1SddAoLVP3lEoKVlZGchMxBZQTjkoJ66Uo8CavRkqaO7u5Mlwqi'
        '78h3kKDGB4e366v1VoFHxBw2e9TU3evNG4bn/Hj2k+R5cs+mm9CtKGpLiGlsl1BWYtTNO42hZZiowPAeo/XQpRyGjCtZ66AK89EiMCA8wIdfPB5Y9eeteTyY'
        'UpoSXnp4ahxVW+SngzykEACn3K/nrqotrBnJtJ4obZ7dcDG0ccSKTj+2kDehwTopKOZXqt4Bmt0VJ3kcVRLToOfKbKjzAzqLzeY0SH9tTUFWWtjZVXKJJTsO'
        'fifk89cqpaHSj4naa+pTFlmpC0u3LPTqsgx9cEe2s41xTBS4oq092tVYiGR1x4K52bfcHx1rsojWLK1UloqlUt3EbEYMpYKzKPHI/fXL1kXSLE5eHuH4Vf4m'
        'CCd5huDxVYfljrgdNUpz0fQ12IonCnBT7DBLWwk/++OPz8dU7S1XAlQgYk4btayFI6urRTk7XkWVD6nn+Wi/J8BuxWoKos4poSyVUm0clLSrGHHToD08NUnX'
        'QKC1T95RKClZWRnITMQWUE45KCeulKPAmr0ZKmju7uTJcKou/Id5CgxgeHt+ur9VaBR8QcNnvU1N3rzRuG5/x49pPkeXLPppvQrShqS4hpbJdQVmLUzTuNoW'
        'WYqMDwHqP10KUchowjWeugVXnokRgRvmBDr54PLHrz1qp5MKU0NI9Z3hporjHCZCWkiWkJB88ZP6aqipVrqRV5JjX3OKyMmealkQdZPgiB+YOov0eH1Lu1zI'
        'dRYYKWUyw3eGnkD8kk3IoHtz5a7X28HE5uNMmDS9EsUnq1RXbYJIZCFb3IHLPnrdnwQKejLVripqMxSotNL82JVklfkP7uRzPnnXGVXhU6rDGhUuF8gqpFFR'
        'RwSQH5FWVfl/bnojFrJi2tUTK+3WakVJKWeiimY8446l8bfVcctWpTeZDUVkT4O6qWaNa2NJgflEZkAx9F1bqtAquJpLQSU7mskmjLbQVcuqH2K51qrVGo9G'
        'YrOGqqP+GKuZz+Lmgz9enLSpx4BdlxMw2y6iJu7rEVuiZgUgnxB5/XQ5QrijJSpgzSrc4XRJbnCNo5BqAk+ucN101g8l1BKer6FSlt1xudMpNyiV1+bElvLY'
        '8ue7prjJxi6U6nVXnr0G1gvCLGBcqAAnAEltkIB88A6lXM6PmV63HoHqaa+kNTvX26TPJl+6mG0f4tZXM6PmDvavoarIb+tF8O1bbJIUwe6NueNjjp8245+u'
        'snZ1rR8wd/iuRDR6zvDTRXGOEyEvJGtISD54yf011os2upCbyTGvucVkZM81LIg6yfBED8wdRfo8PqXdrmQ6iwwUsplhu8NPIH5JJuRQPbny12vt4OJzcaZM'
        'Gl6JYpPVqiu2wSQyEK3uQOWfPW7PggU9GWrXFTUZilRaaX5sSrJK/If3cjmfPOuMqvCp1WGNCpcL5BVSKKijgkgPyKsq/L+3PRGLWTFtaomV9us1IqSUs9FF'
        'Mx5xx1L42+q45atSm8yGorInwd1Us0a1saTA/KIzIBj6Lq3VaBVcTSWgkp3NZJNGW2gq5dUPsVzrVWqNR6MxWcNVUf8ADFXM5/FzQZ+vTlpU48Auy4mYbZdR'
        'E3d1iK3RMwKQT4g8/rocoVxRkpUwZpVucLoktzhG0cg1ASfXOG66aweS6glPV9CpS2643OmUm5RK6/NiS3lseXPd01xk4xdKdTqrz16DawXhFjAuVAATgCS2'
        'yEA+eAdSrmdHzK9bj0D1NNfSGp3r7dJnky/dTDaP8WsrmdHzB3tX0NVkN/Wi+HattksKYPdG3PGxx0+bcc6ydnWtHzB36ZrkSIzN3gqZIfhwrYRJpQNwPXmO'
        'mRrphSlSaVdaHoKJ6epppJJIqmNIOqmoiweXLn4+muEqpnVYo83f+IrfVStBRGreYLtKKY2Gf7zAddd4Wc1jKhznOLwRCtvD1e0haRZRuOQqHl1/5667ytI0'
        'wOUYS1Pb0tuudKsQFFOkajJ31OA36chrwNxep6kmhyG3mabfVU9PJMfBqhm2+nPlorTJjTiLyxG21UkqA0seMF4lTaB6+J1S9ZYktXTAmWdGd65YmPRnfJHq'
        'wxrpgskc8XifUnfJOjTVKUrKRJG3eAguPbx8h00Su6FJSpiU5njWMmrudNDLIuSdse8Dx6n9dc15IrHViBrKdHEUNwE8P8uKaIlD6beZ0tVxoCwwqSa6iqK+'
        'WSIQyTRseXybD+/XXSMlHEl1bKdmsUscBcxJgDDJ8Q0Z/wCmonOrpUuMaIepresELNPu70thIvvPYMev+Worw+hWQ5FGUD7qVEQePxu/P1xpr5g0SpuI3pd8'
        'Hw0EhZdu5a0O+PYjVXa416E3qYEKMzbxUSQ/DhW2ok0oG4HrzHTI12qqUqc6VdaHoKJ6epppJJIqmNIOqmoiweXLn4+muEqpnVYo83f+IrfVStBRGreYLtKK'
        'Y2Gf7zAddd4Wc1jKhznOLwRCtvD1e0haRZRuOQqHl1/5667ytI0wOUYS1Pb0tuudKsQFFOkajJ31OA36chrwNxep6kmhyG3mabfVU9PJMfBqhm2+nPlorTJj'
        'TiLyxG21UkqA0seMF4lTaB6+J1S9ZYktXTAmWdGd65YmPRnfJHqwxrpgskc8XifUnfJOjTVKUrKRJG3eAguPbx8h00Su6FJSpiU5njWMmrudNDLIuSdse8Dx'
        '6n9dc15IrHViBrKdHEUNwE8P8uKaIlD6beZ0tVxoCwwqSa6iqK+WSIQyTRseXybD+/XXSMlHEl1bKdmsUscBcxJgDDJ8Q0Z/6aic6ulS4xoh6mt6wQs0+7vS'
        '2Ei+89gx6/5aivD6FZDkUZQPupURB4/G78/XGmvmDRKn4ielD0/w0Ehddu5a0O+PYjV3a416E3qYHnPhuIapjHVTvtbluoYwxA8fxa61sljHqc6TeYWDgOh2'
        'I801VUHd+KpcKgz5gdPL9dPbSrRG7OOo/NSWnh7erNQLLjEYjn3lh5FR++prOfE1IxCQcUTx5pqEVlSDhFEMOXHLmA3TGc6HZrOWAqfA3LT8R10EcUcVZTog'
        '5iuqwfrtXw1NbNMr1tBOo4VuSuklTW1BiMeP6ojS4/8ALVq0jTBBceoGk4KpEmSoqDLLvJ3PLHzTyyPA65ytW3RHRWazH4aS2UTOk1E8zEYXdUlV9iBrXpSy'
        'M0o5hYeH6aaYuvDcNSNuATVMxx/4ees5y+YEktA8HBc9UzSJ2f24SfytU1CKxHhzJzrdo1h2jNdT9wbjt14tSKH4ctlDGu7JWujcKfYHp+2pbjJ1vN/wajXu'
        '0FHnuNupga77mQuw+aqqCCB5AKf31V2L9mpKclnQWqeNqKkVkq6iySLKMxoklQT054CgnWVk3ik+hnNatCS9oVIx2W+wG6zAbXkpaCYqD/vPjXTsX7zp/JHa'
        'LRV/g1NxRxBWohh4eqYNmVEc8sUOfPxJzoULJYOf1Zm5v3SX94cSPMnc0FrpgWx33xCytnyOBro1Y09pk/8AycEE+F4hqmMdVO+G5FqGMMQPH8WprZLGPUqk'
        '9QsHAdDsR5pqqoO78VS4VBnzA6eX66e2lWiN2cdR+aktPD29WagWXGIxHPvLDyKj99TWc+JqRiEg4onjzTUIrKkHCKIYcuOXMBumM50OzWcsBU+BuWn4jroI'
        '4o4qynRBzFdVg/Xavhqa2aZXraCdRwrcldJKmtqDEY8f1RGlx/5atWkaYILj1A0nBVIkyVFQZZd5O55Y+aeWR4HXOVq26I6KzWY/DSWyiZ0monmYjC7qkqvs'
        'QNa9KWRmlHMLDw/TTTF14bhqRtwCapmOP/Dz1nOXzAkloHg4LnqmaROz+3CT+VqmoRWI8OZOdbtGsO0ZrqfuDcduvFqRQ/DlsoY13ZK10bhT7A9P21LcZOt5'
        'v+DUa92go89xt1MDXfcyF2HzVVQQQPIBT++quxfs1JTks6C1TxtRUislXUWSRZRmNEkqCenPAUE6ysm8Un0M5rVoSXtCpGOy32A3WYDa8lLQTFQf958a6di/'
        'edP5I7RaKv8ABqbijiCtRDDw9UwbMqI55Yoc+fiTnQoWSwc/qzNzfuksXDiSSZO5oLXTAtjv/iFlbPkcDXRqyp7TZP8A8nBFOWqFwpwtZPea7Jy1PGdoPodu'
        'Mj66hK6/VSRda4OpRpbTJUgLT8H1vcsObyBUB/Nv+uud6mc0VSuUWMLYbwsqtQcN26m2KV7yqnjQn05Z5a1+FMZMbrWUTaW6/O6H4qw07qmDHHM8mB45wdFY'
        'cGPreRNa610chL8T0cblShSjoFJI93Y8/pqqR+XqGL94JRXfv5Ghhvl0/hrzWCMBfYbV5e2pceMUKelRtbC1UBItPdp8MWZ3jbDsfHwz9dTepwGleJXt9nqq'
        'KknVLVc96NuLgQxZ+rDXJtN5otg3W4SusM9TJTN+KMTXuJFTzwF10os19CKsXuTW6n2tX3CgPLAVbl3hJP8AdHj66qMZPJdAcksWxGS2cMSowZrSSMFu8Yye'
        'PkOei9aJ6mpBqo3RQ2HYZKSnjnK8m7i2M59vmHTQ5T1fUyUeA9TyhEeaipa2Z5Du72SnSALjoOfl4Y1P7sr9gVVDWTW9pIqJJE37d09SAFPieWqSinmFWeYu'
        'NYtPN3Dm3UcwYfLEQzMCeQLHl666qNcVVkNhKi6C3kw095maQfMwphGw98AdfTWUK5xBypkwEtV8fAErJ7zXZOWpoztB9DtxkfXVpXX6qSCtc6lGltMlSAtP'
        'wfW9yw5vIFQH82/6653qZzRVK5RYwthvCyq1Bw3bqbYpXvKqeNCfTlnlrX4UxkxutZRNpbr87ofirDTuqYMcczyYHjnB0VhwY+t5E1rrXRyEvxPRxuVKFKOg'
        'Ukj3djz+mqpH5eoYv3glFd+/kaGG+XT+GvNYIwF9htXl7alx4xQp6VG1sLVQEi092nwxZneNsOx8fDP11N6nAaV4le32eqoqSdUtVz3o24uBDFn6sNcm03mi'
        '2DdbhK6wz1MlM34oxNe4kVPPAXXSizX0Iqxe5Nbqfa1fcKA8sBVuXeEk/wB0ePrqoxk8l0BySxbEZLZwxKjBmtJIwW7xjJ4+Q56L1onqakGqjdFDYdhkpKeO'
        'crybuLYzn2+YdNDlPV9TJR4D1PKER5qKlrZnkO7vZKdIAuOg5+XhjU/uyv2BVUNZNb2kiokkTft3T1IAU+J5apKKeYVZ5i41i083cObdRzBh8sRDMwJ5AseX'
        'rrqo1xVWQ2EqLoLeTDT3mZpBzYUwjYe+AOvprKFcXEHKmTD1HFN5rZGWGvcSuBueloAq58AGOBn9tCs4LNdRc29QFJLBUQE1NXfJJzncpZSr+xzqmmngkCdc'
        '2xCChjjqQ3eVVQWztjqJIwPyzzGrcnQm75nbPRQNUTPFQtUuuULFkwMnxw3XTKTpiwSVS1DJWw0rR0lHSUuz5ch4Q36nXF0eZ1Seh9R3CthjaNaymiI6x/Fq'
        'rZ9dp1LpwMqsxHb7lNVmaWqhpQ39mGklm3e3Pn78hrXlQKOo5Bw9DVB/irvE+zmcUrOM+PIseWi9TJFXeLGqeyWeByBcqSNn/nltjA58hmPGdZylw6ikuPQb'
        'jt1rotypekV1HPbbOf6qNRek/d6jgtegJ6f4hlCXS4VCMcgxW0Rj057empfnFcyl+4FqGlqGk33W/wAzr8uxC0YI8+S5xpUmtEDhXNsmf0fooXaWKlvGFBId'
        'llYN59SNd70ms0cWkuIWKey0MatU22sd5Ccd+C236Z/fWpN5MyaWhOqLqaaraaiopKaJT8rfDxkex3Z56UuLM3wRqfjhqKeN1aiiqBzbbSKSB55Xx0qyclQ1'
        '6jC1PFN5rZGWGvcSuBueloAq58AGOBn9tSrOCxa6i5t6gKSWCogJqau+STnO5SylX9jnVNNPBIE65tiEFDHHUhu8qqgtnbHUSRgflnmNW5OhN3zO2eigaome'
        'KhapdcoWLJgZPjhuumUnTFgkqlqGSthpWjpKOkpdny5Dwhv1OuLo8zqk9D6juFbDG0a1lNER1j+LVWz67TqXTgZVZiO33KarM0tVDShv7MNJLNu9ufP35DWv'
        'KgUdRyDh6GqD/FXeJ9nM4pWcZ8eRY8tF6mSKu8WNU9ks8DkC5UkbP/PLbGBz5DMeM6zlLh1FJceg3HbrXRblS9IrqOe22c/1Uai9J+71HBa9AT0/xDKEulwq'
        'EY5BitojHpz29NS/OK5lL9wLUNLUNJvut/mdfl2IWjBHnyXONKk1ogcK5tkz+j9FC7SxUt4woJDssrBvPqRrvek1mji0lxCxT2WhjVqm21jvITjvwW2/TP76'
        '1JvJmTS0J1RdTTVbTUVFJTRKflb4eMj2O7PPSlxZm+CNTccNRVEbo1FDUDm2KRSQPPK+OlWTktQvUZ+cQ/aJ4EdwrrUKoGFdjE2fXaX173u/aNP+zxLbtn4g'
        'T25cB02GS6yJvb5o1t0DBB6fxOen0LaX7vVh6bs+sugwv2jez+CnkpzX3IQsclqa004Zv7uSentqVu/aW60XNlen7Mteh1/tN9mMMBRLlVwM/wCICz0obHrh'
        '9Pdm1vOPVk947L83Qzb/ALUXZxTq3fXisUdFKWCnGR6nd1Gs92bU8o9WC3lsy97odH2tezW3TOUvvELIVwvwlrponTzwxJ6+mnurape6ubB7z2Ve8wf/AGv+'
        'yoSErUcVyEDlJULTsxJ8OnTVd07XwQLemy8WBP2u+zF5kUm+rF/PI6Q5HkAFPPW7p2vgh702R5sPU/a67N6gKqXasC4/2tJP8p9lfQt0bXm4rmhe9NkphI5H'
        '9qjs0qIGD3eYYH4Xt7HP1L51u69rT9nqPeeyP3hKf7TvAUgxBe+7BGAi01UoPvhtL3XtOseqIW89mWUvqci+0zwYqSGO80xkJwDJHNjHsX0d17QvdfQe8tn+'
        'Ywn2j+EIw3/pa2sGPRqdzjl4fOdU93bR8jDvDZvnQan+0twNJCsct0jhjU8xR28ZYepZv+uh7u2mtbvUpbw2antG/wDT52bTzmT42sXnkKKGHmR0bm+pW79q'
        'SxXUfTtneTCy/aF4FlYK9zrDLnLOKOmTP1B5ay3ftFMvqb06wr7QWH7RPAkjgOtQqgYV2MTf/wCS+h7v2jT/ALFbds/ECe3LgOmwyXWRN7fNGtugYIPT+Jz0'
        '+hbS/d6sPTdn1l0GF+0b2fwU8lOa+5CFjktTWmnDN/dyT09tSt37S3Wi5sr0/Zlr0Ov9pvsxhgKJcquBn/EBZ6UNj1w+nuza3nHqye8dl+boZt/2ouzinVu+'
        'vFYo6KUsFOMj1O7qNZ7s2p5R6sFvLZl73Q6Pta9mtumcpfeIWQrhfhLXTROnnhiT19NPdW1S91c2D3nsq95g/wDtf9lQkJWo4rkIHKSoWnZiT4dOmq7p2vgg'
        'W9Nl4sCftd9mLzIpN9WL+eR0hyPIAKeet3TtfBD3psjzYep+112b1AVUu1YFx/taSf5T7K+hbo2vNxXNC96bJTCRyP7VHZpUQMHu8wwPwvb2OfqXzrd17Wn7'
        'PUe89kfvCU/2neApBiC992CMBFpqpQffDaXuvadY9UQt57MspfU5F9pngxUkMd5pjITgGSObGPYvo7r2he6+g95bP8xhPtH8IRhv/S1tYMejU7nHLw+c6p7u'
        '2j5GHeGzfOg1P9pbgaSFY5bpHDGp5ijt4yw9Szf9dD3dtNa3epS3hs1PaN/6fOzaecyfG1i88hRQw8yOjc31K3ftSWK6j6ds7yYWX7QvAsrAPc6wyg5ZxR0y'
        'Z+oPLWW79oSy+pvTrCvtH//Z'
    ),
    'steel': (
        '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'
    ),
    'concrete': (
        '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'
        '2Q=='
    ),
    'wood': (
        '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'
    ),
}

def _decode_tiles():
    import base64, io
    out = []
    for n in ('dirt', 'stone', 'brick', 'planks', 'steel', 'concrete', 'wood'):
        raw = base64.b64decode(''.join(_TILE_PNG[n]))
        surf = pygame.image.load(io.BytesIO(raw), n + '.jpg')
        assert surf.get_size() == (TILE_SIZE, TILE_SIZE)
        out.append(np.frombuffer(pygame.image.tostring(surf, 'RGB'), np.uint8).reshape(TILE_SIZE, TILE_SIZE, 3).copy())
    return np.stack(out)

TILE_RGB = _decode_tiles()                         # (layers, 256, 256, 3) uint8
_S_TEXLAYER = np.full(256, -1.0, np.float32)       # block id -> texture layer, -1 = plain colour
for _k, _b in enumerate(TILE_MATS):
    _S_TEXLAYER[_b] = _k
    _S_RGB[_b] = TILE_RGB[_k].reshape(-1, 3).mean(axis=0) / 255.0    # a tile's average colour: debris and chips match it

_TILE_TEX = [0]

def tile_texture():
    """The GL texture array holding every material tile (made on first use, needs a GL context)."""
    if not _TILE_TEX[0]:
        t = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, t)
        data = np.ascontiguousarray(TILE_RGB)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, gl.GL_RGB8, TILE_SIZE, TILE_SIZE, len(data), 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, data)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D_ARRAY)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        try:
            aniso = gl.glGetFloatv(0x84FF)                                 # GL_MAX_TEXTURE_MAX_ANISOTROPY
            gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, 0x84FE, min(8.0, float(aniso)))
        except Exception:
            pass
        _TILE_TEX[0] = t
    return _TILE_TEX[0]

# Face order everywhere: -x, +x, -y, +y, -z, +z.
_FACE_DIRS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], np.int64)


def _q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _q_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _cross(a, b):
    """Row-wise cross product, faster than np.cross for the small arrays the solver uses."""
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], axis=-1)


def _label_cells_xp(xp, alive, bonds):
    """Connected groups of live cells. Cells share a label when a chain of bonds joins them; empty cells
    get a label larger than any real one."""
    big = alive.size + 1
    lab = xp.where(alive, xp.arange(alive.size).reshape(alive.shape), big)
    for _ in range(alive.size + 1):
        old = lab.copy()
        for a in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[a], hi[a] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            m = xp.minimum(lab[lo], lab[hi])
            b = bonds[a]
            lab[lo] = xp.where(b, m, lab[lo])
            lab[hi] = xp.where(b, m, lab[hi])
        for a in range(2, -1, -1):                      # sweep back, so a label crosses the grid in few passes
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[a], hi[a] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            m = xp.minimum(lab[lo], lab[hi])
            b = bonds[a]
            lab[hi] = xp.where(b, m, lab[hi])
            lab[lo] = xp.where(b, m, lab[lo])
        if bool(xp.array_equal(old, lab)):
            break
    return lab


def _label_scipy(alive, bonds, cut=None):
    """Connected groups with scipy. While every neighbouring pair of live cells is still bonded, the groups are
    plain flood-filled blobs (ndimage.label); with `cut` = (axis, plane) the two sides of that plane are labelled
    separately. Once some bonds have broken, the bonds themselves are handed to a graph labeller. Empty cells get
    a label larger than any real one, the same as the numpy version."""
    big = alive.size + 1
    full = True
    for a in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        if not np.array_equal(bonds[a], alive[tuple(lo)] & alive[tuple(hi)]):
            full = False
            break
    if full:
        if cut is None:
            lab, _ = _ndi.label(alive)
        else:
            a, p = cut
            first = tuple(slice(0, p + 1) if k == a else slice(None) for k in range(3))
            second = tuple(slice(p + 1, None) if k == a else slice(None) for k in range(3))
            l1, n1 = _ndi.label(alive[first])
            l2, _ = _ndi.label(alive[second])
            lab = np.concatenate([l1, np.where(l2 > 0, l2 + n1, 0)], axis=a)
        return np.where(lab > 0, lab, big).astype(np.int64)
    idx = np.arange(alive.size, dtype=np.int64).reshape(alive.shape)
    us, vs = [], []
    for a in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        m = bonds[a]
        if cut is not None and cut[0] == a:
            m = m.copy()
            m[tuple(cut[1] if k == a else slice(None) for k in range(3))] = False
        us.append(idx[tuple(lo)][m])
        vs.append(idx[tuple(hi)][m])
    u, v = np.concatenate(us), np.concatenate(vs)
    g = _coo_matrix((np.ones(len(u), np.int8), (u, v)), shape=(alive.size, alive.size)).tocsr()
    _, comp = _connected_components(g, directed=False)
    lab = comp.reshape(alive.shape).astype(np.int64)
    lab[~alive] = big
    return lab


def _label_cells(alive, bonds, cut=None):
    """Connected groups of live cells (see _label_cells_xp). `cut` = (axis, plane) leaves out the bonds between
    that plane's two layers. Uses scipy when it is installed. Otherwise large grids are labelled on the GPU when
    it is available and switched on (any GPU error turns it off), and the CPU does the rest."""
    global GPU_OK
    if HAS_SCIPY:
        try:
            return _label_scipy(alive, bonds, cut)
        except Exception as err:
            print(f"scipy labelling failed ({type(err).__name__}: {err}); using the slow path")
    if cut is not None:
        bonds = [b.copy() for b in bonds]
        bonds[cut[0]][tuple(cut[1] if k == cut[0] else slice(None) for k in range(3))] = False
    if GFX['physics_device'] == 'gpu' and GPU_OK and alive.size >= GPU_MIN_CELLS:
        try:
            lab = _label_cells_xp(cp, cp.asarray(alive), [cp.asarray(b) for b in bonds])
            return cp.asnumpy(lab)
        except Exception as err:
            GPU_OK = False
            print(f"GPU physics failed ({type(err).__name__}: {err}); continuing on the CPU")
    return _label_cells_xp(np, alive, bonds)


def _group_mass_com(cell_mass, grp, cs):
    """Mass and centre of mass (in the grid's own frame) of the cells selected by the boolean mask `grp`,
    from per-axis profiles of the grid instead of listing every cell."""
    w = np.where(grp, cell_mass, 0.0)
    mass = float(w.sum())
    if mass <= 0.0:
        return 0.0, None
    com = np.empty(3)
    for k in range(3):
        other = tuple(i for i in range(3) if i != k)
        profile = w.sum(axis=other)
        com[k] = float((profile * (np.arange(w.shape[k]) + 0.5)).sum()) * cs / mass
    return mass, com


class GridObject:
    """Cells on a regular grid, in a frame that can be moved and rotated. A structure's frame stays put;
    a rigid body's frame is its centre of mass, turned by its rotation matrix. Local coordinates of a
    cell corner are  cs * grid_index + corner."""

    def __init__(self, mat, cdmg, tint, cs):
        self.mat = np.ascontiguousarray(mat, np.int16)
        self.cdmg = np.ascontiguousarray(cdmg, np.float32)
        self.tint = np.ascontiguousarray(tint, np.float32)
        self.cs = float(cs)
        self.stress = np.zeros(self.mat.shape, np.float32)   # stress / strength per cell from the last support pass (1 = about to break)
        self.stress_ver = 0
        self.frame_pos = np.zeros(3)
        self.R = np.eye(3)
        self.corner = np.zeros(3)
        self.mesh_ver = 0
        self._mesh_cache = None
        self._bricks = None
        self._brick_dirty = set()
        self._aabb = None
        self._geometry()

    # ---- geometry, rebuilt after every change to the cells -------------------------------------------
    def _geometry(self, box=None):
        """Rebuild the exposed-face flags, cell masses and bounds. With `box` = (lo, hi), the cells in [lo, hi) are the only ones
        that changed, so only that box (plus a one-cell margin, where a neighbour's faces change) is redone."""
        inc = (box is not None and getattr(self, 'expo', None) is not None and getattr(self, 'alive', None) is not None
               and self.alive.shape == self.mat.shape and self.expo.shape[:3] == self.mat.shape)
        if not inc:
            self.alive = self.mat > 0
            self._live = int(self.alive.sum())
        n = self.alive.shape
        if inc:
            Ub = tuple(slice(max(int(box[0][k]), 0), min(int(box[1][k]), n[k])) for k in range(3))
            before = int(self.alive[Ub].sum())
            self.alive[Ub] = self.mat[Ub] > 0                      # only the changed cells; the rest of `alive` stands
            self._live += int(self.alive[Ub].sum()) - before
            lo = [max(int(box[0][k]) - 1, 0) for k in range(3)]
            hi = [min(int(box[1][k]) + 1, n[k]) for k in range(3)]
            U = tuple(slice(lo[k], hi[k]) for k in range(3))
            ext = tuple(hi[k] - lo[k] for k in range(3))
            lp = np.zeros(tuple(e + 2 for e in ext), bool)       # the box with one cell of neighbours round it
            src, dst = [], []
            for k in range(3):
                a0, a1 = lo[k] - 1, hi[k] + 1
                s0, s1 = max(a0, 0), min(a1, n[k])
                src.append(slice(s0, s1))
                dst.append(slice(s0 - a0, s1 - a0))
            lp[tuple(dst)] = self.alive[tuple(src)]
            au = self.alive[U]
            for f, (dx, dy, dz) in enumerate(_FACE_DIRS):
                nb = lp[1 + dx:1 + dx + ext[0], 1 + dy:1 + dy + ext[1], 1 + dz:1 + dz + ext[2]]
                self.expo[U + (f,)] = au & ~nb
            self.cell_mass[U] = np.where(au, _S_RHO[self.mat[U]] * self.cs ** 3, 0.0)
            if self._bricks is not None:                            # only the bricks this box touches need a new mesh
                B = MESH_BRICK
                for bx in range(lo[0] // B, (hi[0] - 1) // B + 1):
                    for by in range(lo[1] // B, (hi[1] - 1) // B + 1):
                        for bz in range(lo[2] // B, (hi[2] - 1) // B + 1):
                            self._brick_dirty.add((bx, by, bz))
        else:
            nx, ny, nz = n
            pad = np.pad(self.alive, 1)
            expo = np.zeros((nx, ny, nz, 6), bool)
            for f, (dx, dy, dz) in enumerate(_FACE_DIRS):
                nb = pad[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz]
                expo[..., f] = self.alive & ~nb
            self.expo = expo
            self.cell_mass = np.where(self.alive, _S_RHO[self.mat] * self.cs ** 3, 0.0)
            self._bricks = None                                     # everything changed: the mesh is rebuilt from scratch
            self._brick_dirty = set()
        al = self.alive
        if inc and getattr(self, 'cell_lo', None) is not None:
            # Cells were only removed, so the bounds can only shrink: step each face inwards while its layer is empty.
            lo_, hi_ = [int(v) for v in self.cell_lo], [int(v) for v in self.cell_hi]
            for ax in range(3):
                while lo_[ax] < hi_[ax] and not al[tuple(lo_[ax] if k == ax else slice(None) for k in range(3))].any():
                    lo_[ax] += 1
                while hi_[ax] > lo_[ax] and not al[tuple(hi_[ax] - 1 if k == ax else slice(None) for k in range(3))].any():
                    hi_[ax] -= 1
            if all(lo_[k] < hi_[k] for k in range(3)):
                self.cell_lo, self.cell_hi = np.array(lo_, np.int64), np.array(hi_, np.int64)
            else:
                self.cell_lo = self.cell_hi = np.zeros(3, np.int64)
        elif al.any():                                      # the live cells' bounds, from per-axis projections (no full-grid argwhere)
            lo_, hi_ = [], []
            for ax in range(3):
                nzv = np.nonzero(al.any(axis=tuple(a for a in range(3) if a != ax)))[0]
                lo_.append(int(nzv[0]))
                hi_.append(int(nzv[-1]) + 1)
            self.cell_lo, self.cell_hi = np.array(lo_, np.int64), np.array(hi_, np.int64)
        else:
            self.cell_lo = self.cell_hi = np.zeros(3, np.int64)
        self._surf = None                                   # the surface points are rebuilt on first use (see _build_surf)
        self.mesh_ver += 1
        self._mesh_cache = None
        self._aabb = None

    def _build_surf(self):
        """Surface sample points (cell corners and face centres), their normals and which are corners. Only loose
        bodies colliding with this object need them, so they are built when first asked for, not on every cut."""
        pts, nrm = [], []
        for f in range(6):
            idx = np.argwhere(self.expo[..., f])
            if not len(idx):
                continue
            ax, sg = f // 2, (1 if f % 2 else -1)
            u, v = (ax + 1) % 3, (ax + 2) % 3
            base = idx * 2 + 1
            base[:, ax] = idx[:, ax] * 2 + (2 if sg > 0 else 0)
            for du, dv in ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                p = base.copy()
                p[:, u] += du
                p[:, v] += dv
                pts.append(p)
                n = np.zeros((len(p), 3))
                n[:, ax] = sg
                nrm.append(n)
        if pts:
            P = np.concatenate(pts)
            N = np.concatenate(nrm)
            M = 1 << 14                                       # pack each point into one integer: a 1-D unique is far
            key = (P[:, 0] * M + P[:, 1]) * M + P[:, 2]        # faster than unique(axis=0) on a big grid
            ukey, inv = np.unique(key, return_inverse=True)
            inv = inv.reshape(-1)
            uniq = np.stack([ukey // (M * M), (ukey // M) % M, ukey % M], axis=1)
            sumn = np.stack([np.bincount(inv, weights=N[:, k], minlength=len(uniq)) for k in range(3)], axis=1)
            ln = np.linalg.norm(sumn, axis=1, keepdims=True)
            g = uniq / 2.0                          # grid units
            n_ = np.where(ln > 1e-9, sumn / np.maximum(ln, 1e-9), 0.0)
            c_ = (uniq % 2 == 0).all(axis=1)    # cell corners; the rest are face centres
            self._surf = (g, n_, c_)
        else:
            self._surf = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, bool))

    @property
    def live_count(self):
        """Number of live cells, kept up to date by _geometry (a full count of a big grid costs tens of milliseconds)."""
        return self._live

    def worst_cell_load(self):
        """Highest load (stress or cracks) of any cell, 0..1. Small grids are scanned (and the answer kept until they change);
        a big grid answers from numbers the structure keeps as it is checked and hit, since scanning it costs far too much."""
        if self.alive.size > 300_000 and hasattr(self, 'worst_load'):
            return min(1.0, max(self.worst_load, self.cdmg_peak / 0.85))
        key = (self.mesh_ver, self.stress_ver)
        c = getattr(self, '_worst_c', None)
        if c is not None and c[0] == key:
            return c[1]
        w = float(np.where(self.alive, np.maximum(self.stress, self.cdmg / 0.85), 0.0).max()) if self._live else 0.0
        self._worst_c = (key, min(1.0, w))
        return self._worst_c[1]

    @property
    def surf_g(self):
        if self._surf is None:
            self._build_surf()
        return self._surf[0]

    @property
    def surf_n(self):
        if self._surf is None:
            self._build_surf()
        return self._surf[1]

    @property
    def surf_corner(self):
        if self._surf is None:
            self._build_surf()
        return self._surf[2]

    @property
    def surf_loc(self):
        return self.surf_g * self.cs + self.corner

    def surface_world(self):
        return self.frame_pos + self.surf_loc @ self.R.T

    def surface_normals_world(self):
        return self.surf_n @ self.R.T

    def aabb(self):
        """World bounds (lo, hi) of the live cells. Cached until the cells or the pose change."""
        if self._aabb is not None:
            return self._aabb
        if not self.alive.any():
            self._aabb = (self.frame_pos.copy(), self.frame_pos.copy())
            return self._aabb
        c = np.array([[x, y, z] for x in (self.cell_lo[0], self.cell_hi[0])
                      for y in (self.cell_lo[1], self.cell_hi[1])
                      for z in (self.cell_lo[2], self.cell_hi[2])], np.float64)
        w = self.frame_pos + (c * self.cs + self.corner) @ self.R.T
        self._aabb = (w.min(0), w.max(0))
        return self._aabb

    def velocity_at(self, P):
        return np.zeros((len(P), 3))

    # ---- where is solid -----------------------------------------------------------------------------
    def probe(self, Pw):
        """World points -> (indices of points inside a live cell, depth to the nearest exposed face, outward
        normal of that face in world space)."""
        q = ((Pw - self.frame_pos) @ self.R - self.corner) / self.cs
        ij = np.floor(q).astype(np.int64)
        nx, ny, nz = self.alive.shape
        ok = (ij >= 0).all(axis=1) & (ij[:, 0] < nx) & (ij[:, 1] < ny) & (ij[:, 2] < nz)
        sel = np.nonzero(ok)[0]
        if not len(sel):
            return sel, None, None
        c = ij[sel]
        live = self.alive[c[:, 0], c[:, 1], c[:, 2]]
        sel, c = sel[live], c[live]
        if not len(sel):
            return sel, None, None
        f = q[sel] - c
        dist = np.stack([f[:, 0], 1 - f[:, 0], f[:, 1], 1 - f[:, 1], f[:, 2], 1 - f[:, 2]], axis=1)
        ex = self.expo[c[:, 0], c[:, 1], c[:, 2]]
        masked = np.where(ex, dist, np.inf)
        k = np.argmin(masked, axis=1)
        buried = ~np.isfinite(masked[np.arange(len(k)), k])           # no exposed face: use the nearest face
        if buried.any():
            k[buried] = np.argmin(dist[buried], axis=1)
        depth = dist[np.arange(len(k)), k] * self.cs
        normal = _FACE_DIRS[k].astype(np.float64) @ self.R.T
        return sel, depth, normal

    # ---- mesh ---------------------------------------------------------------------------------------
    def mesh(self, mode=None):
        """(n, 10) float32 vertices, 6 per exposed face: position in the frame, normal, colour, texture layer (-1 = none).
        A textured face carries its colour as a multiplier on the tile: 1.0 for sound cells, darker and greyer for cracked ones.
        The normal mesh is kept in bricks, so a cut rebuilds only the bricks it touched."""
        if mode is not None:
            return self._mesh_region((0, 0, 0), self.alive.shape, mode)
        if self._mesh_cache is not None:
            return self._mesh_cache
        B = MESH_BRICK
        shape = self.alive.shape
        nb = [-(-shape[k] // B) for k in range(3)]
        if self._bricks is None:
            self._bricks = {}
            keys = [(x, y, z) for x in range(nb[0]) for y in range(nb[1]) for z in range(nb[2])]
        else:
            keys = [k for k in self._brick_dirty if k[0] < nb[0] and k[1] < nb[1] and k[2] < nb[2]]
        self._brick_dirty = set()
        for key in keys:
            lo = [key[k] * B for k in range(3)]
            hi = [min(lo[k] + B, shape[k]) for k in range(3)]
            part = self._mesh_region(lo, hi)
            if len(part):
                self._bricks[key] = part
            else:
                self._bricks.pop(key, None)
        res = np.concatenate([self._bricks[k] for k in sorted(self._bricks)]) if self._bricks else np.zeros((0, 10), np.float32)
        self._mesh_cache = res
        return res

    def _mesh_region(self, lo, hi, mode=None):
        """Vertices for the exposed faces of the cells in [lo, hi)."""
        out = []
        cs = self.cs
        sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        off = np.array([int(v) for v in lo], np.int64)
        for f in range(6):
            idx = np.argwhere(self.expo[sl + (f,)])
            if not len(idx):
                continue
            idx = idx + off
            ax, sg = f // 2, (1.0 if f % 2 else -1.0)
            u, v = (ax + 1) % 3, (ax + 2) % 3
            base = idx.astype(np.float64)
            base[:, ax] += 1.0 if sg > 0 else 0.0
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = base.copy()
                p[:, u] += du
                p[:, v] += dv
                corners.append(p)
            order = (0, 1, 2, 0, 2, 3) if sg > 0 else (0, 2, 1, 0, 3, 2)
            tri = np.stack([corners[o] for o in order], axis=1)          # (n, 6, 3)
            pos = tri * cs + self.corner
            ii = (idx[:, 0], idx[:, 1], idx[:, 2])
            mi = self.mat[ii]
            lay = _S_TEXLAYER[mi]
            dm = np.zeros((len(idx), 1), np.float32)      # cracks no longer darken the surface
            plain = _S_RGB[mi] * self.tint[ii][:, None] * (1.0 - 0.6 * dm) + np.array([0.20, 0.18, 0.17], np.float32) * (0.6 * dm)
            # Textured faces get a neutral darkening only: the tile already carries the colour, so damage no longer tints it.
            textured = self.tint[ii][:, None] * (1.0 - 0.6 * dm)
            col = np.where((lay >= 0)[:, None], textured, plain)
            if mode == 'integrity':
                col = self._integrity_rgb(ii)
                lay = np.full(len(idx), -1.0, np.float32)
            nrm = np.zeros(3)
            nrm[ax] = sg
            v9 = np.empty((len(idx), 6, 10), np.float32)
            v9[:, :, 0:3] = pos
            v9[:, :, 3:6] = nrm
            v9[:, :, 6:9] = col[:, None, :]
            v9[:, :, 9] = lay[:, None]
            out.append(v9.reshape(-1, 10))
        return np.concatenate(out) if out else np.zeros((0, 10), np.float32)

    def load_of(self, ii):
        """How close each cell is to failing, 0..1: the worst of its stress and its cracks."""
        return np.clip(np.maximum(self.stress[ii], self.cdmg[ii] / 0.85), 0.0, 1.0)

    def _integrity_rgb(self, ii):
        """Green (safe) through yellow to red (about to fail)."""
        x = self.load_of(ii)
        return np.stack([np.interp(x, [0.0, 0.5, 1.0], [0.12, 0.95, 0.95]),
                         np.interp(x, [0.0, 0.5, 1.0], [0.75, 0.85, 0.12]),
                         np.interp(x, [0.0, 0.5, 1.0], [0.30, 0.12, 0.08])], axis=1).astype(np.float32)


class Structure(GridObject):
    """A fixed structure: cells, bonds between neighbours, and the cells that stand on the ground."""

    def __init__(self, mat, origin, cs, cdmg=None, tint=None):
        mat = np.asarray(mat)
        if cdmg is None:
            cdmg = np.zeros(mat.shape, np.float32)
        if tint is None:
            tint = np.ones(mat.shape, np.float32)     # uniform colour: no per-block variation, so no visible grid
        super().__init__(mat, cdmg, tint, cs)
        self.frame_pos[:] = origin
        self.bond = [self.alive[:-1] & self.alive[1:],
                     self.alive[:, :-1] & self.alive[:, 1:],
                     self.alive[:, :, :-1] & self.alive[:, :, 1:]]
        self.anch = np.zeros(self.alive.shape, bool)
        self.unanch = np.zeros(self.alive.shape, bool)   # feet that gave way: never ground again
        self.hot = np.asarray(origin, np.float64).copy()
        self.todo = []                    # (axis, plane) cuts still to be checked in this pass
        self.dirty_all = True             # the next support pass looks at the whole structure (new, or the ground changed)
        self.dirty_box = None             # otherwise just this box of cells, (lo, hi): where cuts have been made
        self.win = None                   # the box the pass in progress is checking (None = the whole structure)
        self.pass_broke = False
        self.stress_dirty = True          # a new structure is checked once before it is trusted
        self.detach_dirty = False
        self.wait = 0.0
        self.max_ratio = 0.0              # highest stress / strength seen in the last pass (1.0 = about to break)
        self.worst_load = 0.0             # highest stress in the area the last pass looked at (for the overlay)
        self.cdmg_peak = 0.0              # highest crack damage dealt so far
        self.crumbs = []                  # (cell centres, materials) of scraps too small to be a body
        self.stress_new = np.zeros(self.mat.shape, np.float32)   # stress being gathered by the pass that is running
        self._pass_open = False

    # ---- ground ---------------------------------------------------------------------------------------
    def refresh_anchors(self, world):
        """A cell with nothing under it is anchored when the terrain is solid just below its bottom face.
        Samples in chunks that are not loaded keep their old answer, so unloading never drops a structure."""
        below = np.zeros_like(self.alive)
        below[:, 1:, :] = self.alive[:, :-1, :]
        cand = self.alive & ~below
        idx = np.argwhere(cand)
        new = np.zeros_like(self.anch)
        if len(idx):
            P = self.frame_pos + (idx + np.array([0.5, 0.0, 0.5])) * self.cs - np.array([0.0, 0.03, 0.0])
            known = np.array([(int(math.floor(x / CHUNK_SIZE)), int(math.floor(z / CHUNK_SIZE))) in world.dens
                              for x, z in zip(P[:, 0].tolist(), P[:, 2].tolist())])
            solid = _density_many(world, P) > 0.0
            old = self.anch[idx[:, 0], idx[:, 1], idx[:, 2]]
            val = np.where(known, solid, old)
            new[idx[:, 0], idx[:, 1], idx[:, 2]] = val
        new &= ~self.unanch
        if not np.array_equal(new, self.anch):
            self.anch = new
            self.detach_dirty = True
            self.stress_dirty = True
            self.dirty_all = True

    # ---- damage ---------------------------------------------------------------------------------------
    def _note_dirty(self, lo, hi):
        """Cells in [lo, hi) changed: the next support pass checks that box widened by STRUCT_LOCAL_R."""
        r = int(math.ceil(STRUCT_LOCAL_R / self.cs))
        n = self.alive.shape
        self._union_box([max(int(lo[k]) - r, 0) for k in range(3)], [min(int(hi[k]) + r, n[k]) for k in range(3)])

    def _union_box(self, lo, hi):
        if self.dirty_box is None:
            self.dirty_box = (list(lo), list(hi))
        else:
            a, b = self.dirty_box
            self.dirty_box = ([min(a[k], lo[k]) for k in range(3)], [max(b[k], hi[k]) for k in range(3)])

    def _remove(self, kill, box=None, add=None, add_mat=0):
        """Take the cells in `kill` out of the structure. Without `box`, `kill` is a mask over the whole grid. With `box` = (lo, hi),
        `kill` is a mask over just that box of cells, and only the box (and its neighbours) is updated."""
        if box is None:
            if not kill.any():
                return
            lo, hi = [], []
            for ax in range(3):
                nzv = np.nonzero(kill.any(axis=tuple(k for k in range(3) if k != ax)))[0]
                lo.append(int(nzv[0]))
                hi.append(int(nzv[-1]) + 1)
        else:
            lo, hi = [int(x) for x in box[0]], [int(x) for x in box[1]]
        n = self.alive.shape
        Bx = tuple(slice(lo[k], hi[k]) for k in range(3))
        kb = kill[Bx] if box is None else kill
        self.mat[Bx][kb] = 0
        ch = kb
        if add is not None:                                          # cells filled in the same step (a dent's raised rim)
            self.mat[Bx][add] = add_mat
            ch = kb | add
        for arr in (self.cdmg, self.stress, self.stress_new):    # mostly zero: writing a zero over a zero would only touch fresh memory pages
            part = arr[Bx]
            if part[ch].any():
                part[ch] = 0
        R = tuple(slice(max(lo[k] - 1, 0), min(hi[k] + 1, n[k])) for k in range(3))
        a = self.mat[R] > 0
        off = [R[k].start for k in range(3)]
        if add is not None:
            addR = np.zeros(a.shape, bool)
            addR[tuple(slice(lo[k] - off[k], hi[k] - off[k]) for k in range(3))] = add
        for ax in range(3):
            lo_s = [slice(None)] * 3
            hi_s = [slice(None)] * 3
            lo_s[ax], hi_s[ax] = slice(0, -1), slice(1, None)
            pair = a[tuple(lo_s)] & a[tuple(hi_s)]                   # bonds inside the box and its margin
            bsl = tuple(slice(off[k], off[k] + pair.shape[k]) for k in range(3))
            self.bond[ax][bsl] &= pair
            if add is not None:
                self.bond[ax][bsl] |= pair & (addR[tuple(lo_s)] | addR[tuple(hi_s)])
        self.anch[R] &= a
        self.unanch[R] &= a
        self._geometry((lo, hi))
        self.detach_dirty = True
        self.stress_dirty = True
        self._note_dirty(lo, hi)

    DENT_RIM_FILL = 0.4       # share of the pressed-out cells that reappear as rim (the rest is squashed flat), one layer only
    DENT_RIM_PEAK = 1.2      # the rim's crest sits this many crater radii from the centre
    DENT_RIM_REACH = 1.6      # no rim cell further out than this many crater radii

    def dent(self, centre, direction, radius, mat=STEEL):
        """Push metal aside instead of breaking it: the `mat` cells inside a sphere of `radius` at the hit are squeezed
        out into a raised rim round the crater (up to as many cells as were pressed out, so the volume is conserved
        where there is room). Nothing is thrown, so nothing detaches. Returns the number of cells pressed out."""
        centre = np.asarray(centre, np.float64)
        d = np.asarray(direction, np.float64)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        c = centre - d * 0.3 * radius                      # a little outside the surface, so the crater is a shallow bowl
        n = self.alive.shape
        g0 = (c - self.frame_pos) / self.cs
        rr = radius * self.DENT_RIM_REACH / self.cs + 2.0
        lo = [max(int(math.floor(g0[k] - rr)), 0) for k in range(3)]
        hi = [min(int(math.ceil(g0[k] + rr)), n[k]) for k in range(3)]
        if any(hi[k] <= lo[k] for k in range(3)):
            return 0
        sl = tuple(slice(lo[k], hi[k]) for k in range(3))
        ax = [self.frame_pos[k] + (np.arange(lo[k], hi[k]) + 0.5) * self.cs - c[k] for k in range(3)]
        dist = np.sqrt(ax[0][:, None, None] ** 2 + ax[1][None, :, None] ** 2 + ax[2][None, None, :] ** 2)
        crater = (self.mat[sl] == mat) & (dist < radius)
        total = int(crater.sum())
        if total == 0:
            return 0
        live = self.alive[sl] & ~crater
        added = np.zeros(crater.shape, bool)
        zone = (dist >= radius) & (dist <= self.DENT_RIM_REACH * radius)
        score = np.abs(dist - self.DENT_RIM_PEAK * radius)
        todo = total
        quota = int(round(total * self.DENT_RIM_FILL))
        cur = live
        nb = np.zeros_like(cur)                             # empty cells touching a filled one (face to face)
        nb[1:] |= cur[:-1]; nb[:-1] |= cur[1:]
        nb[:, 1:] |= cur[:, :-1]; nb[:, :-1] |= cur[:, 1:]
        nb[:, :, 1:] |= cur[:, :, :-1]; nb[:, :, :-1] |= cur[:, :, 1:]
        cand = np.argwhere(nb & ~cur & zone)
        if quota > 0 and len(cand):
            sc = score[cand[:, 0], cand[:, 1], cand[:, 2]]
            take = cand[np.argsort(sc, kind='stable')[:quota]]
            added[take[:, 0], take[:, 1], take[:, 2]] = True
        self.hot = centre.copy()
        self._remove(crater, (lo, hi), add=added, add_mat=mat)
        return total

    def blast(self, centre, radius, power, shatter=False):
        """Blast at a world point. Cells close enough are destroyed; cells further out are cracked, which
        weakens their bonds. Returns (centres, materials) of the destroyed cells. Only the cells in the blast's own
        box are looked at, not the whole structure."""
        none = (np.zeros((0, 3)), np.zeros(0, np.int64))
        centre = np.asarray(centre, np.float64)
        n = self.alive.shape
        g0 = (centre - self.frame_pos) / self.cs
        rr = radius / self.cs + 1.0
        lo = [max(int(math.floor(g0[k] - rr)), 0) for k in range(3)]
        hi = [min(int(math.ceil(g0[k] + rr)), n[k]) for k in range(3)]
        if any(hi[k] <= lo[k] for k in range(3)):
            return none
        idx = np.argwhere(self.alive[tuple(slice(lo[k], hi[k]) for k in range(3))])
        if not len(idx):
            return none
        idx = idx + np.array(lo, np.int64)
        cc = self.frame_pos + (idx + 0.5) * self.cs
        dist = np.linalg.norm(cc - centre, axis=1)
        near = dist < radius
        if not near.any():
            return none
        idx, cc, dist = idx[near], cc[near], dist[near]
        f = power * (1.0 - dist / radius) ** 2
        ii = (idx[:, 0], idx[:, 1], idx[:, 2])
        m = self.mat[ii]
        thr = _S_HARD[m] * STRUCT_SHATTER
        kill = f >= thr
        if shatter:
            kill = np.ones_like(kill)                   # every cell in range is destroyed, whatever its hardness
        hurt = ~kill
        if hurt.any():
            jj = (idx[hurt, 0], idx[hurt, 1], idx[hurt, 2])
            self.cdmg[jj] = np.minimum(0.85, self.cdmg[jj] + 0.7 * f[hurt] / thr[hurt])
            self.cdmg_peak = max(self.cdmg_peak, float(self.cdmg[jj].max()))
        killed_c, killed_m = cc[kill], m[kill]
        self.hot = centre.copy()
        if kill.any():
            ik = idx[kill]
            lo_k, hi_k = ik.min(0), ik.max(0) + 1
            loc = np.zeros(tuple(int(v) for v in hi_k - lo_k), bool)       # a mask over just the cells hit, not the whole grid
            loc[ik[:, 0] - lo_k[0], ik[:, 1] - lo_k[1], ik[:, 2] - lo_k[2]] = True
            self._remove(loc, (lo_k, hi_k))
        else:
            self.mesh_ver += 1                          # only cracks changed: the integrity overlay redraws, the mesh is as it was
            self._note_dirty(idx.min(0), idx.max(0) + 1)
        self.stress_dirty = True
        return killed_c, killed_m

    # ---- does it still stand? -------------------------------------------------------------------------
    def _queue_planes(self):
        """Every cut plane between two layers of cells in the area to check, nearest the last blast first. The area is the box
        round the cuts made since the last pass, or the whole structure when it is new or the ground changed."""
        n = self.alive.shape
        if self.dirty_all or self.dirty_box is None:
            lo, hi = [0, 0, 0], list(n)
            self.win = None
        else:
            lo, hi = list(self.dirty_box[0]), list(self.dirty_box[1])
            self.win = (lo, hi)
        self.dirty_all = False
        self.dirty_box = None
        planes = []
        for a in range(3):
            for p in range(lo[a], hi[a] - 1):
                sl = tuple(p if k == a else slice(lo[k], hi[k]) for k in range(3))
                if self.bond[a][sl].any():
                    mid = self.frame_pos[a] + (p + 1) * self.cs + (self.corner[a])
                    planes.append((abs(mid - self.hot[a]), a, p))
        planes.sort()
        self.todo = [(a, p) for _, a, p in planes]
        self.pass_broke = False
        if self.win is None:
            self.stress_new = np.zeros(self.mat.shape, np.float32)
        else:
            self.stress_new = self.stress                            # a windowed pass rewrites its own area in place; outside it the last answer stands
            self.stress_new[tuple(slice(lo[k], hi[k]) for k in range(3))] = 0.0
        self._pass_open = True

    def _check_plane_win(self, a, p):
        """check_plane on the pass's area only. The area is cut out as a small structure that shares the big one's arrays
        (so every change lands in the real structure), with the cells on its outer faces treated as held, since they join the
        rest of the structure, which this pass is not looking at."""
        if self.win is None:
            return self.check_plane(a, p)
        lo, hi = self.win
        n = self.alive.shape
        W = tuple(slice(lo[k], hi[k]) for k in range(3))
        sub = object.__new__(Structure)
        sub.cs = self.cs
        sub.mat, sub.cdmg, sub.alive = self.mat[W], self.cdmg[W], self.alive[W]
        sub.cell_mass, sub.unanch, sub.stress_new = self.cell_mass[W], self.unanch[W], self.stress_new[W]
        anch = self.anch[W].copy()
        for k in range(3):
            if lo[k] > 0:
                anch[tuple(0 if j == k else slice(None) for j in range(3))] = True
            if hi[k] < n[k]:
                anch[tuple(-1 if j == k else slice(None) for j in range(3))] = True
        sub.anch = anch
        sub.bond = [self.bond[ax][tuple(slice(lo[k], hi[k] - 1 if k == ax else hi[k]) for k in range(3))] for ax in range(3)]
        sub.max_ratio = self.max_ratio
        sub.detach_dirty = False
        broke = sub.check_plane(a, p - lo[a])
        if broke:
            self.cdmg_peak = max(self.cdmg_peak, 0.9)
        self.max_ratio = max(self.max_ratio, sub.max_ratio)
        if sub.detach_dirty:
            self.detach_dirty = True
        self.anch[W] &= ~self.unanch[W]                              # feet that gave way are no longer ground
        return broke

    def check_plane(self, a, p):
        """Cut the structure along one plane and look at what the cut separates from the ground. The part
        with no ground of its own hangs on the bonds that cross the plane. Its weight acts at its centre of
        mass; the bonds share it like a bolted joint: a uniform push or pull plus a linear change across the
        plane that balances the turning moment. A bond pulled apart beyond its tensile strength or crushed
        beyond its compressive strength breaks (the worst one first), and a moment the bonds cannot balance
        at all, because they all lie on one line, breaks the section. Returns True if a bond broke."""
        sl = tuple(p if k == a else slice(None) for k in range(3))
        cross = self.bond[a][sl]
        if not cross.any():
            return False
        lab = _label_cells(self.alive, self.bond, cut=(a, p))
        anchored = np.unique(lab[self.anch & self.alive])
        ij = np.argwhere(cross)
        lo = np.insert(ij, a, p, axis=1)
        hi = np.insert(ij, a, p + 1, axis=1)
        l_lo = lab[lo[:, 0], lo[:, 1], lo[:, 2]]
        l_hi = lab[hi[:, 0], hi[:, 1], hi[:, 2]]
        a_lo, a_hi = np.isin(l_lo, anchored), np.isin(l_hi, anchored)
        broke = False
        if a != 1:
            broke = self._check_thrust(a, p, lab, lo, hi, l_lo, l_hi, a_lo, a_hi)
        for L in np.unique(np.concatenate([l_lo[~a_lo], l_hi[~a_hi]])):
            # the loaded group L must hang only on anchored ground
            if np.any(((l_lo == L) & ~a_hi & (l_hi != L)) | ((l_hi == L) & ~a_lo & (l_lo != L))):
                continue
            use = ((l_lo == L) & a_hi) | ((l_hi == L) & a_lo)
            if not use.any():
                continue
            sgn = -1.0 if np.any((l_lo == L) & use) else 1.0        # support-to-load direction along axis a
            n = np.zeros(3)
            n[a] = sgn
            lo_u, hi_u = lo[use], hi[use]
            cm = self.cell_mass
            mass, com = _group_mass_com(cm, lab == L, self.cs)
            if mass <= 0:
                continue
            fg = np.array([0.0, -mass * STRUCT_GRAVITY, 0.0])
            pos = (lo_u + hi_u + 1) * 0.5 * self.cs                 # centres of the bonds' shared faces
            A = np.full(len(pos), self.cs * self.cs)
            Asum = A.sum()
            c = (A[:, None] * pos).sum(0) / Asum
            rho = pos - c
            sig0 = -float(fg @ n) / Asum
            mg = np.cross(com - c, fg)
            mg = mg - (mg @ n) * n                                   # turning about the plane's normal is carried by shear
            u, v = [k for k in range(3) if k != a]
            cols = []
            for k in (u, v):
                cols.append(((A * rho[:, k])[:, None] * np.cross(rho, n)).sum(0))
            Mm = np.stack(cols, axis=1)                              # (3, 2)
            g, *_ = np.linalg.lstsq(Mm, -mg, rcond=None)
            resid = float(np.linalg.norm(Mm @ g + mg))
            s = sig0 + g[0] * rho[:, u] + g[1] * rho[:, v]           # stress on each bond, push positive
            ml = self.mat[lo_u[:, 0], lo_u[:, 1], lo_u[:, 2]]
            mh = self.mat[hi_u[:, 0], hi_u[:, 1], hi_u[:, 2]]
            fac = 1.0 - np.maximum(self.cdmg[lo_u[:, 0], lo_u[:, 1], lo_u[:, 2]],
                                   self.cdmg[hi_u[:, 0], hi_u[:, 1], hi_u[:, 2]])
            fac = np.maximum(fac, 0.02)
            sk = STRENGTH['k']
            comp = np.minimum(_S_COMP[ml], _S_COMP[mh]) * fac * sk
            tens = np.minimum(_S_TENS[ml], _S_TENS[mh]) * fac * sk
            shear = np.minimum(_S_SHEAR[ml], _S_SHEAR[mh]) * fac * sk
            vt = fg - (fg @ n) * n
            tau = float(np.linalg.norm(vt)) / Asum
            ratio = np.maximum(np.maximum(s / comp, -s / tens), tau / shear)
            if resid > 0.02 * (float(np.linalg.norm(mg)) + 1e-9) and resid > 1.0:
                ratio = np.maximum(ratio, 2.0)                       # one line of bonds cannot hold a turning load
            r_c = np.minimum(ratio, 1.5).astype(np.float32)
            np.maximum.at(self.stress_new, (lo_u[:, 0], lo_u[:, 1], lo_u[:, 2]), r_c)
            np.maximum.at(self.stress_new, (hi_u[:, 0], hi_u[:, 1], hi_u[:, 2]), r_c)
            self.max_ratio = max(self.max_ratio, float(ratio.max()))
            for w in np.nonzero(ratio > 1.0)[0]:
                a_i, b_i = tuple(lo_u[w]), tuple(hi_u[w])
                self.bond[a][a_i] = False                 # the bond is indexed by the lower of its two cells
                self.cdmg[a_i] = min(0.9, float(self.cdmg[a_i]) + 0.12)
                self.cdmg[b_i] = min(0.9, float(self.cdmg[b_i]) + 0.12)
                broke = True
        return broke

    def _check_thrust(self, a, p, lab, lo, hi, l_lo, l_hi, a_lo, a_hi):
        """Arches. A piece that stands on the ground at one end and meets another grounded piece at this cut (the
        two halves of an arch at the crown) leans on its neighbour. Turning about the inner edge of its foot, its
        weight W at horizontal distance e from that edge is balanced by a sideways thrust H = W e / h pushed in at
        the top of the joint, h above the foot. The joint must take H in compression, and the foot must not slide:
        H <= friction x W. Where the joint is a loose slab (a lintel) it can instead carry W as shear; the piece
        stands if either way works. A piece that fails loses its ground for good and falls."""
        pair = a_lo & a_hi & (l_lo != l_hi)
        if not pair.any():
            return False
        broke = False
        opened = False                                            # a half has already failed at this joint
        g = STRUCT_GRAVITY
        sk = STRENGTH['k']
        for side, labs in ((0, l_lo), (1, l_hi)):
            for L in np.unique(labs[pair]):
                sel = pair & (labs == L)
                layer = p if side == 0 else p + 1
                lsl = tuple(layer if k == a else slice(None) for k in range(3))
                if np.any(self.anch[lsl] & (lab[lsl] == L)):
                    continue                                          # it stands on the ground right at the cut
                grp = lab == L
                foot = grp & self.anch & ~self.unanch
                if not foot.any():
                    continue
                mass, com = _group_mass_com(self.cell_mass, grp, self.cs)
                if mass <= 0:
                    continue
                fi = np.argwhere(foot)
                s_ = 1.0 if side == 0 else -1.0
                x_in = ((fi[:, a].max() + 1) if side == 0 else fi[:, a].min()) * self.cs
                e = s_ * (com[a] - x_in)
                if e <= 0.0:
                    continue
                W = mass * g
                y_foot = fi[:, 1].min() * self.cs
                cells = (lo if side == 0 else hi)[sel]
                y_top = (cells[:, 1].max() + 1) * self.cs
                h = max(y_top - y_foot, self.cs)
                H = W * e / h
                m_j = self.mat[cells[:, 0], cells[:, 1], cells[:, 2]]
                fac = max(0.02, 1.0 - float(self.cdmg[cells[:, 0], cells[:, 1], cells[:, 2]].max()))
                area = len(cells) * self.cs * self.cs
                comp = float(_S_COMP[m_j].min()) * fac * sk
                r_comp = (H / (area * ARCH_CONTACT)) / comp
                mu = float(_S_MU[self.mat[fi[:, 0], fi[:, 1], fi[:, 2]]].min())
                r_fric = H / (mu * W)
                r_thrust = max(r_comp, r_fric)
                shear = float(_S_SHEAR[m_j].min()) * fac * sk * ARCH_JOINT_SHEAR
                r_beam = (W / area) / shear
                ratio = min(r_thrust, r_beam)
                if opened:
                    ratio = max(ratio, 2.0)                           # the other half lost its prop, so it tips too
                r_c = np.float32(min(ratio, 1.5))
                np.maximum.at(self.stress_new, (cells[:, 0], cells[:, 1], cells[:, 2]), r_c)
                self.max_ratio = max(self.max_ratio, float(ratio))
                if ratio > 1.0:
                    self.unanch |= foot                               # the feet give way
                    self.anch &= ~self.unanch
                    ai = tuple(slice(None) if k != a else p for k in range(3))
                    self.bond[a][ai][...] = False                     # and the joint opens
                    self.cdmg[cells[:, 0], cells[:, 1], cells[:, 2]] = np.minimum(0.9, self.cdmg[cells[:, 0], cells[:, 1], cells[:, 2]] + 0.12)
                    self.detach_dirty = True
                    broke = True
                    opened = True
        return broke

    def update(self, world, dt, budget=STRUCT_BUDGET_S):
        """Run the support checks within a time budget. Returns the rigid bodies that came loose."""
        new = []
        if (self.stress_dirty or self.todo) and int(self.alive.sum()) > STRUCT_UNBREAKABLE_CELLS:
            self.stress_dirty = False                       # too big to crack under load: no stress checks. A cut all the way
            self.todo = []                                  # through still sends the cut-off part away (the detach step below)
            self._pass_open = False
        if self.detach_dirty:
            with Timed('structs.detach', gpu=False):
                new += self._detach()
        self.wait = max(0.0, self.wait - dt)
        if not self.todo and self.stress_dirty and self.wait <= 0.0:
            self._queue_planes()
            self.stress_dirty = False
        if self.todo:
            # One plane check labels the whole grid and can cost far more than the budget on a fine grid, and the
            # loop below cannot stop one halfway. So each check's cost goes on a debt that is paid down at the
            # budget's share of real time; no new check starts until the debt is clear. Average cost ~ budget.
            self._chk_debt = max(0.0, getattr(self, '_chk_debt', 0.0) - dt * budget * 60.0)
            t0 = time.perf_counter()
            with Timed('structs.check', gpu=False):
              while self.todo and self._chk_debt <= 0.0 and time.perf_counter() - t0 < budget:
                a, p = self.todo.pop(0)
                t_c = time.perf_counter()
                broke_c = self._check_plane_win(a, p)
                self._chk_debt += time.perf_counter() - t_c
                if broke_c:
                    SFX.crack(self.frame_pos + 0.5 * np.array(self.alive.shape) * self.cs)
                    self.pass_broke = True
                    self.detach_dirty = True
                    self._mesh_cache = None
                    self.mesh_ver += 1
            if not self.todo and self._pass_open:
                self._pass_open = False
                self.stress = self.stress_new
                self.worst_load = float(self.stress[tuple(slice(self.win[0][k], self.win[1][k]) for k in range(3))].max()
                                        if self.win is not None else self.stress.max())
                self.stress_ver += 1                  # the integrity overlay redraws from this
            if not self.todo and self.pass_broke:
                self.stress_dirty = True              # something broke, so look at the same area again
                if self.win is None:
                    self.dirty_all = True
                else:
                    self._union_box(*self.win)
                self.wait = STRUCT_CRACK_STEP
                self.pass_broke = False
            if self.detach_dirty:
                with Timed('structs.detach', gpu=False):
                    new += self._detach()
        return new

    def _detach(self):
        """Cells with no chain of bonds to an anchored cell leave as rigid bodies."""
        self.detach_dirty = False
        lab = _label_cells(self.alive, self.bond)
        ground = np.unique(lab[self.anch & self.alive])
        free = self.alive & ~np.isin(lab, ground)
        if not free.any():
            return []
        bodies = []
        for L in np.unique(lab[free]):
            mask = lab == L
            min_cells = max(6, int(round(STRUCT_MIN_BODY_CELLS * (0.5 / self.cs) ** 3)))   # same physical size at any detail
            if int(mask.sum()) < min_cells:
                idx = np.argwhere(mask)
                self.crumbs.append((self.frame_pos + (idx + 0.5) * self.cs, self.mat[mask].copy()))
            else:
                b = RigidBody.from_cells(self, mask)
                b.topple(self.hot)                 # the piece leans away from the blast before it falls
                bodies.append(b)
        self._remove(free)
        return bodies


class RigidBody(GridObject):
    """A solid chunk with a centre of mass, an orientation, velocity and spin."""

    def __init__(self, mat, cdmg, tint, cs, com_local):
        super().__init__(mat, cdmg, tint, cs)
        self.corner = -np.asarray(com_local, np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.asleep = False
        self.still = 0.0
        self._top = float(TERRAIN_MAX_H)
        self.last_contacts = 0
        self.supported = False
        self._mass_props()
        self._sync()

    @classmethod
    def from_cells(cls, st, mask):
        idx = np.argwhere(mask)
        lo, hi = idx.min(0), idx.max(0) + 1
        sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        m = np.where(mask[sl], st.mat[sl], 0)
        cm = np.where(mask[sl], st.cell_mass[sl], 0.0)
        cells = np.argwhere(m > 0)
        com = ((cells + 0.5) * cm[m > 0][:, None]).sum(0) / cm.sum() * st.cs
        b = cls(m, st.cdmg[sl].copy(), st.tint[sl].copy(), st.cs, com)
        b.frame_pos = st.frame_pos + lo * st.cs + com
        return b

    def _mass_props(self):
        cells = np.argwhere(self.alive)
        m = self.cell_mass[self.alive]
        self.mass = float(m.sum())
        self.invm = 1.0 / max(self.mass, 1e-9)
        r = (cells + 0.5) * self.cs + self.corner
        r2 = np.einsum('ij,ij->i', r, r)
        I = np.eye(3) * float((m * r2).sum()) - np.einsum('i,ij,ik->jk', m, r, r)
        I += np.eye(3) * float((m * self.cs ** 2 / 6.0).sum())
        self.Ib_inv = np.linalg.inv(I)

    def _sync(self):
        self.R = _q_mat(self.q)
        self.Iinv = self.R @ self.Ib_inv @ self.R.T

    def velocity_at(self, P):
        return self.v + _cross(self.w, P - self.frame_pos)

    def wake(self):
        self.asleep = False
        self.still = 0.0

    def topple(self, centre):
        """Start a loose piece leaning away from the point it came off (the last blast): a push away from that point
        and a spin about the horizontal axis that turns its top away, so it tips over instead of dropping straight down."""
        up = np.array([0.0, 1.0, 0.0])
        d = np.asarray(self.frame_pos, np.float64) - np.asarray(centre, np.float64)
        d[1] = 0.0
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        self.wake()                                  # split-off pieces keep the parent's velocity and spin; nothing is added

    # ---- blast --------------------------------------------------------------------------------------
    def blast(self, centre, radius, power):
        """Push on the surface that faces the blast. Each surface point takes a pressure that falls off with
        distance, so the impulse acts away from the centre of mass and the body spins as well as moves."""
        P = self.surface_world()
        N = self.surface_normals_world()
        d = P - centre
        dist = np.linalg.norm(d, axis=1)
        ok = (dist < radius) & (dist > 1e-6) & (np.linalg.norm(N, axis=1) > 0.5)
        if not ok.any():
            return
        P, N, d, dist = P[ok], N[ok], d[ok], dist[ok]
        facing = -np.einsum('ij,ij->i', N, d / dist[:, None])
        ok = facing > 0.05
        if not ok.any():
            return
        P, N, dist, facing = P[ok], N[ok], dist[ok], facing[ok]
        pressure = power * (1.0 - dist / radius) ** 2 * facing * (0.5 * self.cs * self.cs)
        J = -N * (pressure * STRUCT_BLAST_IMPULSE)[:, None]
        self.v = self.v + J.sum(0) * self.invm
        self.w = self.w + self.Iinv @ np.cross(P - self.frame_pos, J).sum(0)
        self.wake()

    # ---- contacts -----------------------------------------------------------------------------------
    def _contacts(self, world, obstacles):
        """Points where this body touches or is about to touch something. Returns arrays of the contact
        point, the normal that pushes the body out, the depth (negative: a small gap), and the velocity
        of whatever it touches."""
        Pw = self.surface_world()
        pts, nrm, dep, vob = [], [], [], []
        # terrain: the density field, with its gradient giving the surface normal
        Pc = Pw[self.surf_corner]
        if len(Pc):
            Pc = Pc[Pc[:, 1] < self._top + 0.4]            # only points low enough to reach the ground
        if len(Pc):
            cdep, cn, okc = _cell_contact(world, Pc)
            if okc.any():
                pts.append(Pc[okc])
                nrm.append(cn[okc])
                dep.append(cdep[okc])
                vob.append(np.zeros((int(okc.sum()), 3)))
        lo, hi = Pw.min(0) - 0.05, Pw.max(0) + 0.05
        for O in obstacles:
            olo, ohi = O.aabb()
            if (olo > hi).any() or (ohi < lo).any():
                continue
            # our points inside the other object
            m = ((Pw >= olo - 0.05) & (Pw <= ohi + 0.05)).all(axis=1)
            if m.any():
                ids = np.nonzero(m)[0]
                sel, depth, normal = O.probe(Pw[ids])
                if len(sel):
                    P = Pw[ids[sel]]
                    pts.append(P)
                    nrm.append(normal)
                    dep.append(depth)
                    vob.append(O.velocity_at(P))
            # its points inside us
            Ow = O.surface_world()
            m = ((Ow >= lo) & (Ow <= hi)).all(axis=1)
            if m.any():
                ids = np.nonzero(m)[0]
                sel, depth, normal = self.probe(Ow[ids])
                if len(sel):
                    P = Ow[ids[sel]]
                    pts.append(P)
                    nrm.append(-normal)
                    dep.append(depth)
                    vob.append(O.velocity_at(P))
        if not pts:
            return None
        return (np.concatenate(pts), np.concatenate(nrm), np.concatenate(dep), np.concatenate(vob))

    def _ground_top(self, world):
        """Highest the ground can be under this body. The ground is never above the flat top (dug cubes only ever lower it), so
        points above that skip the terrain test."""
        return float(TERRAIN_MAX_H)

    def _reduce(self, pts, nrm, dep, vob):
        """Keep the deepest contacts, spread out, so the solver has a few well-placed points instead of dozens."""
        n = len(pts)
        if n <= STRUCT_MAX_CONTACTS:
            return pts, nrm, dep, vob
        band = dep >= dep.max() - 0.12
        ids = np.nonzero(band)[0]
        if len(ids) <= STRUCT_MAX_CONTACTS:
            rest = np.setdiff1d(np.arange(n), ids)
            extra = rest[np.argsort(-dep[rest])][:STRUCT_MAX_CONTACTS - len(ids)]
            ids = np.concatenate([ids, extra])
        else:
            chosen = [int(ids[np.argmax(dep[ids])])]
            mind = np.linalg.norm(pts[ids] - pts[chosen[0]], axis=1)
            while len(chosen) < STRUCT_MAX_CONTACTS:
                k = int(np.argmax(mind))
                chosen.append(int(ids[k]))
                mind = np.minimum(mind, np.linalg.norm(pts[ids] - pts[ids[k]], axis=1))
            ids = np.array(chosen)
        return pts[ids], nrm[ids], dep[ids], vob[ids]

    def _solve(self, pts, nrm, dep, vob, h):
        """Sequential impulses on the contacts: a push along each normal that never pulls, and friction along
        two tangents limited by the push. Gives the body its new velocity and spin."""
        m = len(pts)
        r = pts - self.frame_pos
        ref = np.where(np.abs(nrm[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        t1 = _cross(nrm, ref)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = _cross(nrm, t1)
        D = np.concatenate([nrm, t1, t2])                          # (3m, 3) directions
        Ang = _cross(np.concatenate([r, r, r]), D)
        K = self.invm * (D @ D.T) + Ang @ self.Iinv @ Ang.T
        diag = np.diag(K).copy()
        diag = np.maximum(diag, 1e-9)
        jv = D @ self.v + Ang @ self.w - np.einsum('ij,ij->i', D, np.concatenate([vob, vob, vob]))
        vn0 = jv[:m].copy()
        c = (dep - STRUCT_SLOP) / h
        c = np.where(c > 0, 0.0, c)                  # no velocity bias for overlaps (was adding outward speed); gaps still allow approach
        bounce = vn0 < -1.5
        c = np.where(bounce, np.maximum(c, -STRUCT_RESTITUTION * vn0), c)
        lam = [0.0] * (3 * m)
        cl = c.tolist()
        dg = diag.tolist()
        for _ in range(WOBBLE['iters']):
            moved = 0.0
            for i in range(m):
                new = max(0.0, lam[i] - (jv[i] - cl[i]) / dg[i])
                dl = new - lam[i]
                if dl:
                    lam[i] = new
                    jv += K[:, i] * dl
                    moved += abs(dl)
                lim = WOBBLE['friction'] * lam[i]
                for row in (m + i, 2 * m + i):
                    new = min(max(lam[row] - jv[row] / dg[row], -lim), lim)
                    dl = new - lam[row]
                    if dl:
                        lam[row] = new
                        jv += K[:, row] * dl
                        moved += abs(dl)
            if moved < 1e-5:
                break
        lam = np.array(lam)
        self.v = self.v + self.invm * (D.T @ lam)
        self.w = self.w + self.Iinv @ (Ang.T @ lam)

    def step(self, world, dt, obstacles):
        if self.asleep:
            return
        h = dt / STRUCT_SUBSTEPS
        fall_v = -float(self.v[1])                           # downward speed at the start of the step, for the landing sound
        self._top = self._ground_top(world) + 0.5 * float(np.linalg.norm(self.v)) * dt
        contact_steps = 0
        support = False
        for _ in range(STRUCT_SUBSTEPS):
            self.v[1] -= STRUCT_GRAVITY * h
            self.w *= math.exp(-WOBBLE['ang_damp'] * h)
            if WOBBLE['lin_damp']:
                kd = math.exp(-WOBBLE['lin_damp'] * h)
                self.v[0] *= kd                                  # sideways only: a fall is not slowed
                self.v[2] *= kd
            con = self._contacts(world, obstacles)
            if con is not None:
                pts, nrm, dep, vob = self._reduce(*con)
                self._solve(pts, nrm, dep, vob, h)
                # Position fix: the impulses only stop the approach, so an overlap that already exists is never
                # pushed out and lasts for good. Move the body out along its deepest contact, a little each substep.
                over = dep - STRUCT_SLOP
                if len(over) and float(over.max()) > 0.0:
                    i = int(np.argmax(over))
                    self.frame_pos = self.frame_pos + nrm[i] * min(float(over[i]), 0.05) * 0.6
                contact_steps += 1
                self.last_contacts = len(pts)
                cp = pts - pts.mean(0)
                support = len(pts) >= 3 and np.linalg.svd(cp, compute_uv=False)[1] > 0.12
            self.frame_pos = self.frame_pos + self.v * h
            wq = np.array([0.0, *(self.w * h * 0.5)])
            self.q = self.q + _q_mul(wq, self.q)
            self.q /= np.linalg.norm(self.q)
            self._sync()
        self.supported = support
        self._aabb = None
        if contact_steps and not getattr(self, '_touching', False) and fall_v > 2.5:
            SFX.body_land(self.frame_pos, fall_v, float(self.alive.sum()) * self.cs ** 3)
        self._touching = bool(contact_steps)
        k = WOBBLE['sleep_k']
        slow = np.linalg.norm(self.v) < STRUCT_SLEEP_LIN * k and np.linalg.norm(self.w) < STRUCT_SLEEP_ANG * k
        if slow and support:
            self.still += dt
            if self.still >= STRUCT_SLEEP_TIME / k:
                self.asleep = True
                self.v[:] = 0.0
                self.w[:] = 0.0
        else:
            self.still = 0.0


MAX_LOOSE_BODIES = 80      # more loose pieces than this and the smallest become debris, to keep the physics cost bounded


def _chop_object(o, centre, d, half_w, depth, power, hot):
    """An axe cut into one structure or loose body: the cells in a slot along the swing direction `d` (half_w either side,
    `depth` deep from the hit point) are cut away if the blade can (hardness at most `power`) and only cracked if not.
    Returns (centres, materials) of the cells cut away and whether the slot touched the object at all."""
    none = (np.zeros((0, 3)), np.zeros(0, np.int64), False)
    cs = o.cs
    reach = max(half_w, depth) + 2.0 * cs
    if isinstance(o, Structure):
        n = o.alive.shape
        g0 = (centre - o.frame_pos) / cs
        rr = reach / cs + 1.0
        lo = [max(int(math.floor(g0[k] - rr)), 0) for k in range(3)]
        hi = [min(int(math.ceil(g0[k] + rr)), n[k]) for k in range(3)]
        if any(hi[k] <= lo[k] for k in range(3)):
            return none
        idx = np.argwhere(o.alive[tuple(slice(lo[k], hi[k]) for k in range(3))])
        if not len(idx):
            return none
        idx = idx + np.array(lo, np.int64)
        cc = o.frame_pos + (idx + 0.5) * cs
    else:
        idx = np.argwhere(o.alive)
        if not len(idx):
            return none
        cc = o.frame_pos + ((idx + 0.5) * cs + o.corner) @ o.R.T
    rel = cc - centre
    along = rel @ d
    lat = np.linalg.norm(rel - along[:, None] * d, axis=1)
    cut = (along >= -0.75 * cs) & (along <= depth) & (lat <= half_w)
    if not cut.any():
        return none
    idx, cc, lat = idx[cut], cc[cut], lat[cut]
    ii = (idx[:, 0], idx[:, 1], idx[:, 2])
    m = o.mat[ii]
    thr = _S_HARD[m] * STRUCT_SHATTER
    blunt = lat <= AXE_BLUNT_FRAC * half_w                  # the blunt middle of the axe: it never cuts a cube out, it only cracks
    kill = (thr <= power) & ~blunt                          # the sharp edges cut out whatever the blade can cut
    hurt = ~kill
    if hurt.any():
        jj = (idx[hurt, 0], idx[hurt, 1], idx[hurt, 2])
        add = np.full(int(hurt.sum()), AXE_BLUNT_CRACK)
        edge_hard = ~blunt[hurt]                            # an edge on something too hard to cut cracks it the old way
        add[edge_hard] = 0.7 * power / thr[hurt][edge_hard]
        o.cdmg[jj] = np.minimum(0.85, o.cdmg[jj] + add)
        if isinstance(o, Structure):
            o.cdmg_peak = max(o.cdmg_peak, float(o.cdmg[jj].max()))
    if isinstance(o, Structure):
        o.hot = np.asarray(hot, np.float64).copy()
        if kill.any():
            ik = idx[kill]
            lo_k, hi_k = ik.min(0), ik.max(0) + 1
            loc = np.zeros(tuple(int(v) for v in hi_k - lo_k), bool)
            loc[ik[:, 0] - lo_k[0], ik[:, 1] - lo_k[1], ik[:, 2] - lo_k[2]] = True
            o._remove(loc, (lo_k, hi_k))
        else:
            o.mesh_ver += 1
            o._note_dirty(idx.min(0), idx.max(0) + 1)
        o.stress_dirty = True
    elif kill.any():
        mask = np.zeros(o.alive.shape, bool)
        mask[idx[kill, 0], idx[kill, 1], idx[kill, 2]] = True
        o._remove(mask)
    return cc[kill], m[kill], True


class StructureManager:
    """Every structure and loose body in the world."""

    def __init__(self):
        self.structs = []
        self.bodies = []

    def clear(self):
        self.structs = []
        self.bodies = []
        self.extra_dirt = 0

    # ---- building -----------------------------------------------------------------------------------
    def ground_y(self, world, x, z):
        """Height of the top of the highest ground cube at column (x, z), or None if there is none (or it is not loaded)."""
        ix, iz = math.floor(x / CELL), math.floor(z / CELL)
        d = world.dens.get((ix // NC, iz // NC))
        if d is None:
            return None
        n = int(d.tops()[ix - (ix // NC) * NC, iz - (iz // NC) * NC])
        return float(n * CELL) if n > 0 else None

    def add(self, world, mat, x0, z0, cs=0.5):
        """Put a grid of materials (0 = empty) on the ground with its lowest corner at (x0, z0)."""
        nx, ny, nz = mat.shape
        y = self.ground_y(world, x0 + nx * cs * 0.5, z0 + nz * cs * 0.5)
        if y is None:
            return None
        st = Structure(np.asarray(mat, np.int16), np.array([x0, y, z0]), cs)
        st.refresh_anchors(world)
        self.structs.append(st)
        return st

    def add_ground_body(self, world, cells, mats, centre, max_cells=5000):
        """A cut-off slab of ground becomes one loose rigid body that falls, tumbles and lands as a piece. `cells` are the (n, 3)
        ground cube indices (already removed from the world), `mats` the material of each, `centre` the blast it came off.
        A slab with too many cubes is rebuilt from bigger cells (2x2x2, 3x3x3 ...) so the physics stays affordable.
        Returns the body, or None if the slab is too small to be a body (the caller then makes debris of it)."""
        idx = np.asarray(cells, np.int64)
        mats = np.asarray(mats)
        if len(idx) == 0:
            return None
        f = 1
        while len(idx) / float(f ** 3) > max_cells and f < 6:
            f += 1
        lo = idx.min(0)
        c = (idx - lo) // f
        shape = tuple(int(v) + 1 for v in c.max(0))
        flat = np.ravel_multi_index((c[:, 0], c[:, 1], c[:, 2]), shape)
        total = int(np.prod(shape))
        count = np.bincount(flat, minlength=total)
        stone = np.bincount(flat, weights=(mats == STONE).astype(np.float64), minlength=total)
        alive = (count * 2 >= f ** 3) if f > 1 else (count > 0)       # a bigger cell is solid if at least half its cubes were
        if int(alive.sum()) < STRUCT_MIN_BODY_CELLS:
            return None
        mat = np.where(alive, np.where(stone * 2 > count, STONE, DIRT), 0).astype(np.int16).reshape(shape)
        cs = f * CELL
        cells_a = np.argwhere(mat > 0)
        rho = _S_RHO[mat[mat > 0]]
        com = ((cells_a + 0.5) * rho[:, None]).sum(0) / rho.sum() * cs
        body = RigidBody(mat, np.zeros(shape, np.float32), np.ones(shape, np.float32), cs, com)
        body.frame_pos = lo * CELL + com
        # A small push away from the blast, and a spin that tips the top away from it: the slab leans out as it lets go.
        up = np.array([0.0, 1.0, 0.0])
        d = body.frame_pos - np.asarray(centre, np.float64)
        d[1] = 0.0
        nd = float(np.linalg.norm(d))
        d = d / nd if nd > 1e-6 else np.array([1.0, 0.0, 0.0])
        size = max(1.0, float(body.mass) ** (1.0 / 3.0))
        body.v = d * 1.2
        body.w = np.cross(up, d) * (0.9 / size ** 0.5)
        body.wake()
        self.bodies.append(body)
        SFX.collapse(body.frame_pos, float(body.alive.sum()) * cs ** 3)
        self._cap_bodies(world)
        return body

    # ---- events -------------------------------------------------------------------------------------
    def wake_all(self):
        for b in self.bodies:
            b.wake()

    def _wake_bodies_near(self, centre, radius):
        """Wake only the loose bodies whose box comes within radius of centre."""
        c = np.asarray(centre, np.float64)
        for b in self.bodies:
            if b.asleep:
                lo, hi = b.aabb()
                if np.linalg.norm(np.clip(c, lo, hi) - c) <= radius:
                    b.wake()

    def _wake_region(self, centre, radius, column_up=0.0):
        """Wake the loose bodies and the debris cubes near one spot. Nothing outside the region is touched."""
        self._wake_bodies_near(centre, radius)
        BODIES.wake_sphere(centre, radius, column_up)

    def on_terrain_changed(self, world, centre=None, radius=None):
        """The ground changed: structures re-check what they stand on and loose bodies wake up. With a centre and
        radius only the bodies and cubes near that spot wake; without them everything wakes, as before."""
        for s in self.structs:
            s.refresh_anchors(world)
        if centre is None:
            self.wake_all()
            BODIES.wake_all()
        else:
            self._wake_region(centre, radius, column_up=radius * 3.0)

    def blast(self, world, centre, radius, power=1.0, shatter=False, throw=True, debris_normal=None):
        """Damage structures, push bodies, throw chunks. The caller draws the fireball. throw=False: what is destroyed is
        vaporised (the melting laser), so no chunks are thrown."""
        centre = np.asarray(centre, np.float64)
        self._wake_region(centre, radius * 2.0, column_up=radius * 3.0)
        broken = []
        for s in self.structs:
            lo, hi = s.aabb()
            if (centre < lo - radius).any() or (centre > hi + radius).any():
                continue
            kc, km = s.blast(centre, radius, power, shatter)
            for c, m in zip(kc, km):
                d = c - centre
                broken.append((1.0, c[0] - 0.5, c[1] - 0.5, c[2] - 0.5, int(m), d[0], d[1], d[2]))
            if len(kc) and throw:
                self._throw_chunks(kc, km, centre, radius, world=world, cs=s.cs, launch_normal=debris_normal)
        for b in self.bodies:
            b.blast(centre, radius, power)
            if np.linalg.norm(b.frame_pos - centre) < radius * 3:
                b.wake()
        centre = np.asarray(centre, np.float64)
        for b in list(self.bodies):                       # loose pieces take damage too, then split apart
            lo, hi = b.aabb()
            if (centre < lo - radius).any() or (centre > hi + radius).any():
                continue
            if b in self.bodies:                          # the cap may have turned it into debris already
                self._break_loose(world, b, centre, radius, power, shatter, debris_normal=debris_normal)
        return broken

    def chop(self, world, centre, direction, half_w=AXE_CUT_HALF_W, depth=AXE_CUT_DEPTH, power=AXE_CUT_POWER, impulse=AXE_IMPULSE):
        """A blunt axe cut: a slot along the swing is cut out of whatever is at the hit point and the cut cubes drop out, shoved
        aside and a little along the swing, not blown away. There is no blast: a fixed structure only loses those cubes (and
        falls over, away from the swing, if that was enough to cut it through) and a loose piece takes the swing's momentum at
        the hit point, so it tips and slides the way a real hit on it would."""
        centre = np.asarray(centre, np.float64)
        d = np.asarray(direction, np.float64)
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        reach = max(half_w, depth) + 0.5
        dh = np.array([d[0], 0.0, d[2]])
        dh = dh / max(float(np.linalg.norm(dh)), 1e-9)
        hot = centre - dh * 1.0                      # a piece cut loose leans away from this point, so it falls the way the axe pushed
        self._wake_region(centre, 0.8, column_up=0.5)
        for s in self.structs:
            lo, hi = s.aabb()
            if (centre < lo - reach).any() or (centre > hi + reach).any():
                continue
            kc, km, _t = _chop_object(s, centre, d, half_w, depth, power, hot)
            if len(kc):
                self._chips(world, kc, km, centre, d, s.cs)
        for b in list(self.bodies):
            lo, hi = b.aabb()
            if (centre < lo - reach).any() or (centre > hi + reach).any():
                continue
            kc, km, touched = _chop_object(b, centre, d, half_w, depth, power, hot)
            if not touched:
                continue
            J = d * impulse                          # the swing's momentum, applied at the hit point
            b.v = b.v + J * b.invm
            b.w = b.w + b.Iinv @ np.cross(centre - b.frame_pos, J)
            b.wake()
            if len(kc):
                self._chips(world, kc, km, centre, d, b.cs)
                new, debris = _split_body(b, AXE_MIN_PIECE_CELLS)      # a cut log stays a log, however small, so it can be cut again
                for pos, mats in debris:
                    self._throw_chunks(pos, mats, pos.mean(0), 1.5, slow=True, world=world, cs=b.cs)
                if not b.alive.any() and b in self.bodies:
                    self.bodies.remove(b)
                self.bodies.extend(new)
        self._cap_bodies(world)

    def _chips(self, world, pos, mats, centre, d, cs):
        """Cubes cut out by the axe: each is pushed sideways out of the cut and a little along the swing, gently, then falls."""
        if not STRUCT_SHARDS:
            return
        pos, mats = np.asarray(pos), np.asarray(mats)
        if len(pos) > 80:
            sel = _rng.choice(len(pos), 80, replace=False)
            pos, mats = pos[sel], mats[sel]
        n = len(pos)
        rel = pos - centre
        along = rel @ d
        lat = rel - along[:, None] * d
        ln = np.maximum(np.linalg.norm(lat, axis=1, keepdims=True), 1e-6)
        t = np.clip(along / AXE_CUT_DEPTH, 0.0, 1.0)[:, None]      # 0 at the surface, 1 at the bottom of the cut
        # Momentum goes along the swing, so cubes near the surface leave that way. Deeper ones have uncut wood behind them, so
        # (as with a wedge-shaped blade) they are shoved out sideways instead of being driven into it. About 0.5 to 1.5 blocks/s.
        V = d * _rng.uniform(0.5, 1.5, (n, 1)) * (1.0 - t) + lat / ln * _rng.uniform(0.3, 0.8, (n, 1)) * (0.2 + 0.8 * t)
        V[:, 1] = np.minimum(V[:, 1], 0.0)                      # never upward: they fall
        col = _S_RGB[mats] * _rng.uniform(0.8, 1.1, (n, 1)).astype(np.float32)
        BODIES.spawn_chunks(pos, V, np.full(n, cs), col, world, material=mats)                   # every cube is the size of one cell: the cube detail setting is global

    def _break_loose(self, world, body, centre, radius, power, shatter=False, debris_normal=None):
        """A blast or drill hit on a loose body: cells near the centre crack or are destroyed, then whatever is no
        longer connected breaks away."""
        kc, km = body.damage_at(centre, radius, power, shatter)
        self._finish_body(world, body, kc, km, centre, radius, debris_normal=debris_normal)

    def _finish_body(self, world, body, kc, km, centre, radius, debris_normal=None):
        """Throw out destroyed cells, split off pieces that are no longer connected, and keep the body count under
        the cap. Pieces split off a body topple away from the hit, like detached structure."""
        if len(kc):
            self._throw_chunks(kc, km, centre, radius, world=world, cs=body.cs, launch_normal=debris_normal)
        new, debris = _split_body(body)
        for child in new:
            child.topple(centre)
        for pos, mats in debris:
            self._throw_chunks(pos, mats, pos.mean(0), 1.5, slow=True, world=world, cs=body.cs)
        if not body.alive.any() and body in self.bodies:      # every cell was destroyed: the body is gone
            self.bodies.remove(body)
        self.bodies.extend(new)
        self._cap_bodies(world)

    def _cap_bodies(self, world):
        """Too many loose pieces: the smallest become debris, so the physics cost stays bounded."""
        while len(self.bodies) > MAX_LOOSE_BODIES:
            small = min(self.bodies, key=lambda b: int(b.alive.sum()))
            self.bodies.remove(small)
            pos, mats = _cells_to_debris(small, small.alive)
            if len(pos):
                self._throw_chunks(pos, mats, pos.mean(0), 1.5, slow=True, world=world, cs=small.cs)

    def _throw_chunks(self, pos, mats, centre, radius, slow=False, world=None, cs=0.5, launch_normal=None):
        if not STRUCT_SHARDS:
            return                                                # no cloud of loose cubes (see STRUCT_SHARDS)
        # A big hit at a high cube detail can break thousands of small cells at once; spawning a piece for each
        # of them bogs the frame down. Keep a random sample of them as debris and let the rest vanish.
        MAX_CHUNK_CELLS = 300
        pos, mats = np.asarray(pos), np.asarray(mats)
        count = MAX_CHUNK_CELLS if launch_normal is None else (2000 if STRUCT_DEBRIS_SPLIT <= 1 else MAX_CHUNK_CELLS // 8)
        if len(pos) > count:
            sel = _rng.choice(len(pos), count, replace=False)
            pos, mats = pos[sel], mats[sel]
        if launch_normal is None:
            P = pos + _rng.uniform(-0.05, 0.05, pos.shape) * cs
        elif STRUCT_DEBRIS_SPLIT > 1:                         # old behaviour: every destroyed cell is thrown as eight half-size cubes
            offsets = (_CUBE_UNIT - 0.5) * (cs * 0.5)
            P = (pos[:, None, :] + offsets).reshape(-1, 3)
            mats = np.repeat(mats, len(offsets))
            cs *= 0.5
        else:                                                 # one destroyed cell = one thrown cube, the same size as the cell
            P = pos.copy()
        d = P - centre
        dist = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 0.2)
        if launch_normal is None:
            speed = _rng.uniform(0.5, 2.0, (len(P), 1)) if slow else _rng.uniform(4.0, 12.0, (len(P), 1))
            V = d / dist * speed * (1.2 - 0.5 * np.minimum(dist / radius, 1.0))
            V[:, 1] = np.minimum(V[:, 1], 1.0)
        else:
            spread = _rng.uniform(-1.5, 1.5, P.shape)
            spread[:, 1] += 2.0
            spread -= (spread @ launch_normal)[:, None] * launch_normal
            speed = _rng.uniform(5.0, 9.0, (len(P), 1))
            V = (speed * launch_normal + spread) * (1.0 - 0.35 * np.minimum(dist / radius, 1.0))
        col = _S_RGB[mats]
        if launch_normal is None:
            col = col * _rng.uniform(0.8, 1.1, (len(P), 1)).astype(np.float32)
        if launch_normal is not None and abs(launch_normal[1]) < 0.7:
            # Start wall fragments past the struck face. A launch from inside the
            # cavity can hit the intact rim immediately and lose its outward kick.
            # Preserve the spacing of the original depth layers so they can separate.
            normal_depth = (P - centre) @ launch_normal
            clearance = cs * 0.5 * np.abs(launch_normal).sum() + 0.02
            out_depth = clearance + 1.10 * np.abs(normal_depth)
            P = P + (out_depth - normal_depth)[:, None] * launch_normal
        BODIES.spawn_chunks(P, V, np.full(len(P), cs), col, world, escape_normal=launch_normal, material=mats)

    def update(self, world, dt):
        dt = min(dt, 0.05)
        for s in list(self.structs):
            new = s.update(world, dt)
            if new:
                self.bodies.extend(new)
                SFX.collapse(new[0].frame_pos, sum(float(b.alive.sum()) * b.cs ** 3 for b in new))
                for b in new:                         # a piece broke away: cubes and bodies resting near it must re-check
                    lo, hi = b.aabb()
                    c = (lo + hi) * 0.5
                    self._wake_region(c, float(np.linalg.norm(hi - lo)) * 0.5 + 0.5, column_up=1.0)
            for cc, mm in s.crumbs:
                SFX.crumble(cc.mean(0))
                self._throw_chunks(cc, mm, cc.mean(0) - np.array([0.0, 0.4, 0.0]), 1.5, slow=True, world=world, cs=s.cs)
            s.crumbs = []
            if not s.alive.any():
                self.structs.remove(s)
        awake = [b for b in self.bodies if not b.asleep]
        with Timed('structs.bodies', gpu=False):
            for b in awake:
                others = self.structs + [o for o in self.bodies if o is not b]
                b.step(world, dt, others)
        self.bodies = [b for b in self.bodies if b.frame_pos[1] > -30.0 and b.alive.any()]

    # ---- queries --------------------------------------------------------------------------------------
    def objects(self):
        return self.structs + self.bodies

    def eject_player(self, world, player):
        """A body that has come down on the player cannot be walked through: step the player out of it, to the
        nearest free spot, or on top of it."""
        r, h = PLAYER_RADIUS, PLAYER_HEIGHT
        if not (self.structs or self.bodies) or not self.hits_box(player.pos[0], player.pos[1], player.pos[2], r, h):
            return False
        for dist in (0.4, 0.8, 1.2, 1.8, 2.6, 3.6):
            for k in range(12):
                a = k * math.pi / 6.0
                x, z = player.pos[0] + dist * math.cos(a), player.pos[2] + dist * math.sin(a)
                if not _box_hits_solid(world, x, player.pos[1], z):
                    player.pos[0], player.pos[2] = x, z
                    player.vel[:] = 0.0
                    return True
        for up in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
            if not _box_hits_solid(world, player.pos[0], player.pos[1] + up, player.pos[2]):
                player.pos[1] += up
                player.vel[:] = 0.0
                return True
        return False

    def hits_box(self, x, y, z, r, h):
        """True if the player's box at feet position (x, y, z) is inside a structure or a body."""
        if not (self.structs or self.bodies):
            return False
        lo_b = np.array([x - r, y, z - r])
        hi_b = np.array([x + r, y + h, z + r])
        P = None
        for o in self.objects():
            lo, hi = o.aabb()
            if (lo > hi_b).any() or (hi < lo_b).any():
                continue
            if P is None:
                P = np.array([[x + dx, ys, z + dz] for ys in (y + 0.05, y + 0.5 * h, y + h - 0.05)
                              for dx in (-r, 0.0, r) for dz in (-r, 0.0, r)])
            if len(o.probe(P)[0]):
                return True
        return False

    def raycast(self, origin, direction, reach, step=0.05):
        """First point along a ray that is inside a structure or body, or None. Returns (t, last point outside, object hit)."""
        best = None
        t = np.arange(0.0, reach, step)
        P = np.asarray(origin, np.float64) + np.outer(t, np.asarray(direction, np.float64))
        for o in self.objects():
            sel = o.probe(P)[0]
            if len(sel):
                k = int(sel.min())
                if best is None or k < best[0]:
                    best = (k, o)
        if best is None:
            return None
        k = best[0]
        return float(t[k]), P[max(k - 1, 0)], best[1]


STRUCTS = StructureManager()



# ---- the garden bed -------------------------------------------------------------------------------------------
BED_X, BED_Z = 3.0, -4.0          # lowest corner of the bed's outer wall, beside the start point
BED_INNER = (2.0, 0.8, 1.0)       # inside size in blocks: along x, soil depth, along z. At the default cube size that is 10 x 4 x 5 = 200 cubes
BED_WALL_H = 1.0                  # wall height, a little above the soil


class GardenBed:
    """A planked bed with no floor (the ground is its floor). Loose dirt cubes poured into it are counted; once it holds its full
    amount and they have all settled, they are replaced by one block of soil: a single structure made of dirt cells."""

    def __init__(self, wall, lo, size, res):
        self.wall = wall                       # the plank structure
        self.soil = None                       # the soil structure, once the bed is full
        self.lo = np.asarray(lo, np.float64)   # lowest corner of the inside
        self.size = size                       # inside size in cells (x, y, z)
        self.res = res
        self.cs = 1.0 / res
        self.cap = int(size[0] * size[1] * size[2])
        self.count = 0
        self.full_t = 0.0                      # seconds the bed has held its full amount
        self.slide = None                      # (cube indices, start positions, end positions) while the cubes glide into place
        self.slide_t = 0.0

    @property
    def ready(self):
        return self.soil is not None

    def parts(self):
        return [p for p in (self.wall, self.soil) if p is not None]

    def near(self, p, margin=1.2):
        """True if the point is over the bed or within `margin` blocks of it (sideways only)."""
        hi = self.lo + np.array([self.size[0], 0.0, self.size[2]]) * self.cs
        return (self.lo[0] - self.cs - margin <= p[0] <= hi[0] + self.cs + margin and
                self.lo[2] - self.cs - margin <= p[2] <= hi[2] + self.cs + margin)

    def pour_points(self, n):
        """Where n cubes start so that they all fall straight down inside the walls: one column of cubes over every place in the
        bed, layer by layer from just above the rim."""
        nx, ny, nz = self.size
        j = np.arange(n)
        layer, c = j // (nx * nz), j % (nx * nz)
        i, k = c // nz, c % nz
        inset = 0.02 * self.cs                                       # a hair inside the walls, so no cube starts touching one
        x = self.lo[0] + inset + (i + 0.5) * (nx * self.cs - 2 * inset) / nx
        z = self.lo[2] + inset + (k + 0.5) * (nz * self.cs - 2 * inset) / nz
        y = self.lo[1] + BED_WALL_H + 0.1 + layer * self.cs
        return np.stack([x, y, z], axis=1)

    def state(self):
        return (self.cap if (self.ready or self.slide is not None) else self.count, self.cap, self.ready)

    def _inside(self):
        hi = self.lo + np.array([self.size[0], self.size[1], self.size[2]]) * self.cs
        P = BODIES.pos
        return ((P[:, 0] >= self.lo[0]) & (P[:, 0] <= hi[0]) & (P[:, 2] >= self.lo[2]) & (P[:, 2] <= hi[2])
                & (P[:, 1] >= self.lo[1] - 0.3) & (P[:, 1] <= self.lo[1] + BED_WALL_H + 0.6))

    SETTLE_S = 1.5                              # a heap of loose cubes never goes fully still, so full for this long counts as settled
    SLIDE_S = 1.2                               # seconds the cubes take to glide into their places

    def _slots(self):
        """Centre of every cube-sized place inside the bed, bottom layer first."""
        nx, ny, nz = self.size
        g = np.stack(np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz), indexing='ij'), -1).reshape(-1, 3)   # (y, x, z) order
        return self.lo + (g[:, [1, 0, 2]] + 0.5) * self.cs

    def _start_slide(self, m):
        idx = np.nonzero(m)[0]
        P = BODIES.pos[idx]
        if len(idx) > self.cap:                                         # extras beyond the bed's amount stay loose
            idx = idx[np.argsort(P[:, 1], kind='stable')[:self.cap]]
            P = BODIES.pos[idx]
        order = np.lexsort((P[:, 2], P[:, 0], np.round((P[:, 1] - self.lo[1]) / self.cs)))     # layer by layer, then along x, then z
        idx = idx[order]
        self.slide = (idx, BODIES.pos[idx].copy(), self._slots())      # which cubes, where they began, where each ends up
        self.slide_t = 0.0

    def _finish(self, world, idx):
        m = np.zeros(BODIES.n, bool)
        m[idx[idx < BODIES.n]] = True
        BODIES.remove(m)
        self.fill_now(world)

    def fill_now(self, world):
        """Fill the bed with soil straight away (used by the crafting table)."""
        mat = np.full(self.size, DIRT, np.int16)
        soil = Structure(mat, self.lo.copy(), self.cs, tint=np.full(mat.shape, 0.42, np.float32))     # darkened to the loose dirt's brown
        soil.refresh_anchors(world)
        STRUCTS.structs.append(soil)
        self.soil = soil
        self.slide = None
        SFX.play('level_up', 0.6, gap=0.0)

    def update(self, world, dt=0.0):
        if self.ready:
            return
        if self.slide is not None:                                      # every cube glides to its own place, then they fuse
            idx, start, end = self.slide
            self.slide_t += dt
            f = min(1.0, self.slide_t / self.SLIDE_S)
            e = f * f * (3.0 - 2.0 * f)
            ok = idx < BODIES.n
            BODIES.pos[idx[ok]] = start[ok] + (end[ok] - start[ok]) * e
            BODIES.vel[idx[ok]] = 0.0
            BODIES.awake[idx[ok]] = False
            if f >= 1.0:
                self._finish(world, idx)
            return
        if BODIES.n == 0:
            self.count, self.full_t = 0, 0.0
            return
        m = self._inside()
        self.count = int(m.sum())
        self.full_t = self.full_t + dt if self.count >= self.cap else 0.0
        if self.full_t >= self.SETTLE_S:                              # full and settled: the cubes slide into place and fuse into soil
            self._start_slide(m)


def bed_dims(res):
    """The garden bed's inside size in cubes (x, y, z) and its wall height in cells, for `res` cubes per block."""
    res = max(1, int(res))
    return (int(round(BED_INNER[0] * res)), max(1, int(round(BED_INNER[1] * res))), int(round(BED_INNER[2] * res)),
            max(2, int(math.ceil(BED_WALL_H * res))))


def bed_capacity(res):
    """How many cubes the bed holds at this cube size. The shovel's scoop is a quarter of it, so it always takes four digs."""
    nx, ny, nz, _ = bed_dims(res)
    return nx * ny * nz


def build_garden_bed(world, res, x=None, z=None):
    """Stand the bed on the ground at (x, z) (its lowest corner; default beside the start point). Returns None if the ground there is not loaded."""
    x = BED_X if x is None else x
    z = BED_Z if z is None else z
    res = max(1, int(res))
    cs = 1.0 / res
    nx, ny, nz, nh = bed_dims(res)
    m = np.full((nx + 2, nh, nz + 2), PLANKS, np.int16)
    m[1:-1, :, 1:-1] = 0                                          # hollow: four walls, open top, no floor
    wall = STRUCTS.add(world, m, x, z, cs)
    if wall is None:
        return None
    lo = wall.frame_pos + np.array([cs, 0.0, cs])
    return GardenBed(wall, lo, (nx, ny, nz), res)


# ---- trees ----------------------------------------------------------------------------------------------------
NUM_TREES = 18
TREE_AVOID = ((0.5, 0.5, 4.0), (4.2, -3.3, 5.0))      # (x, z, radius) no tree this close to the start point or the garden bed


def tree_spots(count=NUM_TREES, seed=7):
    """Where the trees stand, as (x, z, seed): scattered round the start point, never too close to each other or to the bed."""
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < count and tries < 4000:
        tries += 1
        ang, dist = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(9.0, 60.0)
        x, z = dist * math.cos(ang), dist * math.sin(ang)
        if any(math.hypot(x - ax, z - az) < ar for ax, az, ar in TREE_AVOID):
            continue
        hh = float(terrain_height(x, z))
        if hh <= WATER_Y + 2.0 or hh >= DIRT_ALT - 2.0 or any(float(terrain_height(x + dx, z + dz)) != hh for dx in (-1.6, 1.6) for dz in (-1.6, 1.6)):
            continue
        if any(math.hypot(x - ox, z - oz) < 6.5 for ox, oz, _ in out):
            continue
        out.append((float(x), float(z), int(rng.integers(1, 1_000_000))))
    return out


TREE_HEIGHT_MULT = 3.5                                         # trunk height multiplier (1.0 = the original size)
TREE_TRUNK_MULT = 2.2                                          # trunk and branch thickness multiplier


def make_tree(res, seed, scale=1.0):
    """One tree as a grid of cubes, `res` cubes per block. A bark trunk that flares at the root, narrows with height and leans a little,
    two or three bare branches reaching up and out.
    Returns (materials, tint, (cells from the grid's low x edge to the trunk centre, same for z))."""
    rng = np.random.default_rng(seed)
    cs = 1.0 / res
    H = rng.uniform(3.4, 4.8) * scale * TREE_HEIGHT_MULT                           # trunk height
    r0, r1 = rng.uniform(0.34, 0.46) * scale * TREE_TRUNK_MULT, rng.uniform(0.16, 0.22) * scale * TREE_TRUNK_MULT   # trunk radius at the base and near the top
    R = rng.uniform(1.7, 2.3) * scale                           # canopy radius
    Rv = R * rng.uniform(0.95, 1.15)                            # canopy height radius
    lean = rng.uniform(0.0, 2.0 * math.pi)
    lean_amt = rng.uniform(0.05, 0.22)
    cyc = H + Rv * 0.55                                         # canopy centre height
    lobes = [(0.0, cyc, 0.0, R)]
    for _ in range(int(rng.integers(4, 7))):
        a, dist = rng.uniform(0.0, 2.0 * math.pi), R * rng.uniform(0.45, 0.85)
        lobes.append((dist * math.cos(a), cyc + Rv * rng.uniform(-0.35, 0.45), dist * math.sin(a), R * rng.uniform(0.55, 0.8)))
    hw = 1.7 * R + 0.4
    n = int(math.ceil(2 * hw / cs))
    ny = int(math.ceil((cyc + 1.5 * R) / cs))
    X = ((np.arange(n) + 0.5) * cs - hw)[:, None, None] * np.ones((1, ny, n))
    Y = ((np.arange(ny) + 0.5) * cs)[None, :, None] * np.ones((n, 1, n))
    Z = ((np.arange(n) + 0.5) * cs - hw)[None, None, :] * np.ones((n, ny, 1))
    # trunk: a leaning, tapering, flared cylinder. Never thinner than about 3 cubes across, or the canopy's turning load has
    # no spread-out section to hold it and the support check breaks it off.
    floor_r = cs * (0.7 + 1.05 * min(1.0, scale * 1.56))
    t = np.clip(Y / H, 0.0, 1.0)
    rad = np.maximum(r1 + (r0 - r1) * (1.0 - t) ** 1.4 + 0.22 * scale * TREE_TRUNK_MULT * np.exp(-Y / 0.35), floor_r)
    ox, oz = lean_amt * t * t * H * math.cos(lean), lean_amt * t * t * H * math.sin(lean)
    trunk = ((X - ox) ** 2 + (Z - oz) ** 2 <= rad ** 2) & (Y <= cyc)
    # branches: thin bark cylinders from the upper trunk out to the canopy lobes
    wood = trunk.copy()
    for lx, ly, lz, lr in lobes[1:4]:
        p0 = np.array([lean_amt * 0.8 * H * math.cos(lean), H * rng.uniform(0.68, 0.85), lean_amt * 0.8 * H * math.sin(lean)])
        p1 = np.array([lx * 0.8, ly - 0.15, lz * 0.8])
        d = p1 - p0
        w = np.stack([X - p0[0], Y - p0[1], Z - p0[2]], axis=-1)
        u = np.clip((w @ d) / float(d @ d), 0.0, 1.0)
        dist2 = ((w - u[..., None] * d) ** 2).sum(axis=-1)
        wood |= dist2 <= (max(rng.uniform(0.10, 0.15) * scale * TREE_TRUNK_MULT, 0.5 * floor_r) * (1.0 - 0.4 * u)) ** 2
    mat = np.zeros(X.shape, np.int16)
    mat[wood] = WOOD
    try:                                                        # drop any bit not joined to the trunk: it would just fall off
        from scipy import ndimage
        lab, _n = ndimage.label(mat > 0)
        base = lab[:, 0, :][mat[:, 0, :] == WOOD]
        if len(base):
            mat[lab != np.bincount(base).argmax()] = 0
    except ImportError:
        pass
    alive = mat > 0
    ix, iy, iz = np.nonzero(alive)
    lo = (ix.min(), 0, iz.min())
    hi = (ix.max() + 1, iy.max() + 1, iz.max() + 1)
    sl = (slice(lo[0], hi[0]), slice(0, hi[1]), slice(lo[2], hi[2]))
    mat = np.ascontiguousarray(mat[sl])
    tint = np.ones(mat.shape, np.float32)
    tint[mat == WOOD] = (0.85 + 0.25 * rng.random(int((mat == WOOD).sum()))).astype(np.float32)
    return mat, tint, (int(n // 2 - lo[0]), int(n // 2 - lo[2]))


TREES = []        # every standing tree: dict(st=structure, wood0=wood cells at the start, done=seeds already given)


def build_tree(world, res, x, z, seed, scale=1.0, y=None):
    """Stand one tree with its trunk centred on (x, z), on the ground (or at height y). Returns the structure, or None if the ground
    there is not loaded yet."""
    gy = STRUCTS.ground_y(world, x, z) if y is None else y
    if gy is None:
        return None
    cs = 1.0 / max(1, int(res))
    mat, tint, (cx, cz) = make_tree(res, seed, scale)
    origin = np.array([x - (cx + 0.5) * cs, gy, z - (cz + 0.5) * cs])
    st = Structure(mat, origin, cs, tint=tint)
    st.refresh_anchors(world)
    STRUCTS.structs.append(st)
    TREES.append(dict(st=st, wood0=int((mat == WOOD).sum()), done=False))
    return st


def build_trees(world, res):
    """Stand every tree in tree_spots(). Returns the number placed (0 if the ground is not loaded yet)."""
    n = 0
    for x, z, sd in tree_spots(max(3, NUM_TREES - 2 * max(0, int(res) - 5))):      # finer cubes mean far more cells per tree, so fewer trees
        if build_tree(world, res, x, z, sd) is not None:
            n += 1
    return n


# ---- seeds, apples and planting ---------------------------------------------------------------------------------
INV = {'seeds': 0, 'apples': 0, 'wood': 0, 'dirt': 0, 'msg': '', 'msg_t': 0.0}
BED_WOOD_COST = 8                            # wood to craft a garden bed; the dirt cost is the bed's capacity in cubes
GROW_STAGES = (0.18, 0.32, 0.48, 0.64)       # tree size at each stage of growing from a seed
GROW_STAGE_S = 18.0                          # seconds per stage


def inv_note(msg, secs=4.0):
    INV['msg'], INV['msg_t'] = msg, secs


def check_chopped(dt):
    """A tree that has lost over 40% of its wood is chopped down: it drops seeds and wood once."""
    for t in list(TREES):
        st = t['st']
        gone = st not in STRUCTS.structs
        wood = 0 if gone else int(((st.mat == WOOD) & st.alive).sum())
        if gone or wood < 0.6 * t['wood0']:
            TREES.remove(t)
            sd = _random.randint(2, 3)
            wd = _random.randint(6, 9)
            INV['seeds'] += sd
            INV['wood'] += wd
            inv_note(f"Tree chopped: +{wd} wood, +{sd} seeds")
            SFX.play('level_up', 0.5, gap=0.0)


class Planting:
    """One apple seed in the soil bed. It grows through GROW_STAGES; the full-grown tree can be chopped for more seeds."""
    def __init__(self, bed, seed):
        self.bed, self.seed = bed, seed
        self.stage, self.t, self.st = -1, 0.0, None

    def _show(self, world, stage):
        if self.st is not None:
            for t in list(TREES):
                if t['st'] is self.st:
                    TREES.remove(t)
            if self.st in STRUCTS.structs:
                STRUCTS.structs.remove(self.st)
        b = self.bed
        top = float(b.lo[1] + b.size[1] * b.cs)
        cx = float(b.lo[0] + b.size[0] * b.cs * 0.5)
        cz = float(b.lo[2] + b.size[2] * b.cs * 0.5)
        self.st = build_tree(world, b.res, cx, cz, self.seed, GROW_STAGES[stage], y=top)
        self.stage = stage

    def update(self, world, dt):
        self.t += dt
        want = min(len(GROW_STAGES) - 1, int(self.t / GROW_STAGE_S))
        if self.stage < 0 or want != self.stage:
            if self.stage >= 0 and (self.st is None or self.st not in STRUCTS.structs):
                return False                                # it was chopped down: the bed is empty again
            self._show(world, want)
        elif self.st is None or self.st not in STRUCTS.structs:
            return False
        return True


def try_plant(world, bed, player, plant):
    """T: plant an apple seed in the finished soil bed if the player is next to it and it is empty."""
    if bed is None or not bed.ready:
        inv_note("Fill the garden bed with soil first")
    elif plant is not None:
        inv_note("Something is already growing there")
    elif INV['seeds'] <= 0:
        inv_note("No seeds - chop a tree to get some")
    elif not bed.near(player.pos, 3.0):
        inv_note("Stand next to the garden bed to plant")
    else:
        INV['seeds'] -= 1
        inv_note("Planted an apple seed")
        return Planting(bed, _random.randint(1, 1_000_000))
    return plant



_SOLID_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aCol;
layout(location = 3) in float aMat;   // texture layer, -1 = plain colour
uniform mat4 uVP;
uniform mat4 uModel;       // the object's pose: rotation and position
out vec3 vCol;
flat out float vMat;
out vec3 vNrm;
out vec3 vLocal;     // object-space position, for the block joints
out vec3 vWorldP;    // world position, for the shadow lookup
out float vDepth;
void main() {
    vec4 w = uModel * vec4(aPos, 1.0);
    vCol = aCol;
    vMat = aMat;
    vNrm = mat3(uModel) * aNrm;
    vLocal = aPos;
    vWorldP = w.xyz;
    gl_Position = uVP * w;
    vDepth = gl_Position.w;
}
"""

_SOLID_FRAG = """
#version 330 core
in vec3 vCol;
in vec3 vNrm;
in vec3 vLocal;
in vec3 vWorldP;
in float vDepth;
flat in float vMat;
out vec4 fragColor;
uniform float uLight;
uniform float uAlpha;
uniform float uGrid;      // 1 = draw the cube grid lines (toggled with X)
uniform float uCell;      // side length of one cell, in block units
uniform vec3 uCorner;     // the object's corner offset, so the lines sit on its cells
uniform vec3 uEye;        // camera position, so the haze takes the sky's colour in that direction
uniform sampler2DArray uTiles;      // the material tiles, bound to unit 2
SHADOW_GLSL_PLACEHOLDER
float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
LIGHT_GLSL_PLACEHOLDER
void main() {
    vec3 n = normalize(vNrm);
    float sh = sunShadow(vWorldP, n);
    vec3 light = dayLight(n, sh);
    // Block faces: the two in-plane axes give the face's own coordinates. Joints are darker lines at whole-block edges.
    vec3 an = abs(n);
    vec2 uv = an.y > 0.5 ? vLocal.xz : (an.x > 0.5 ? vLocal.zy : vLocal.xy);
    vec2 f = fract(uv);
    vec2 cell = floor(uv);
    float speck = 1.0;                                       // flat colour: no speckle, block outlines or per-block tint
    vec3 base = vCol * speck;
    // One tile per block, except stone and concrete: one tile per two blocks, so their larger pattern is not busy.
    float tscale = (abs(vMat - 1.0) < 0.5 || vMat > 4.5) ? 0.5 : 1.0;
    if (vMat > -0.5) base = min(texture(uTiles, vec3(uv * tscale, vMat)).rgb * vCol * 1.3, vec3(1.0));   // the tile
    vec3 col = base * light * uLight;
    // Cube grid: a dark line on every cell edge of the face, when X has turned the grid on.
    vec3 g = (vLocal - uCorner) / uCell;
    vec2 gv = an.y > 0.5 ? g.xz : (an.x > 0.5 ? g.zy : g.xy);
    vec2 e = abs(fract(gv + 0.5) - 0.5);                  // distance to the nearest cell edge, 0..0.5
    float edge = 1.0 - smoothstep(0.0, 0.045, min(e.x, e.y));
    col = mix(col, col * 0.4, edge * uGrid);
    if (uAlpha >= 0.99) {                                    // no haze on the integrity overlay
        col = mix(col, fogColor(normalize(vWorldP - uEye)), 1.0 - exp(-vDepth * 0.0105));
    }
    fragColor = vec4(col, uAlpha);
}
"""
_SOLID_FRAG = (_SOLID_FRAG.replace("SHADOW_GLSL_PLACEHOLDER", _SHADOW_GLSL)
               .replace("LIGHT_GLSL_PLACEHOLDER", _LIGHT_GLSL))


class ShadowMap:
    """Depth map from the sun. Each frame the terrain and structures are drawn from the sun's side into this
    texture; the terrain and structure shaders then compare each pixel's depth against it to find shadow."""
    def __init__(self):
        vs = _compile_shader(_DEPTH_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_DEPTH_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_lvp = gl.glGetUniformLocation(self.program, "uLVP")
        self.u_model = gl.glGetUniformLocation(self.program, "uModel")
        self.res = SHADOW_RES        # texels along each side; the map is reallocated by set_quality
        self.on = True               # False: no shadows (the map is cleared to "nothing in the way")
        self.every = 1               # redraw the map every this many frames
        self.level = 2
        self.count = 0
        self.lvp = None              # the light matrix the map was last drawn with
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, SHADOW_RES, SHADOW_RES, 0,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        for pname in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, pname, gl.GL_NEAREST)
        for pname in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, pname, gl.GL_CLAMP_TO_EDGE)
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def set_quality(self, level, every):
        """level indexes SHADOW_LEVELS (0 = off); every = redraw interval in frames. Reallocates the map only
        when its size changes. The texture object stays the same, so the renderers keep pointing at it."""
        level = int(level)
        self.every = max(1, int(every))
        if level == self.level:
            return
        self.level = level
        self.on = level > 0
        self.res = SHADOW_LEVELS[level] if self.on else 4
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, self.res, self.res, 0,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.tex, 0)
        if not self.on:                                  # nothing is ever in the way: depth 1.0 everywhere
            gl.glViewport(0, 0, self.res, self.res)
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, _FRAME_FBO)
        self.lvp = None                                  # draw it again at the new size on the next update

    def update(self, renderer, struct_gl, eye):
        """Redraw the map when it is due and return the light matrix the shaders must use. Between redraws the
        old matrix is kept, so the shadows stay where they were drawn."""
        if not self.on:
            if self.lvp is None:
                self.lvp = light_matrix(eye, self.res)
            return self.lvp
        self.count += 1
        if self.lvp is None or self.count >= self.every:
            self.count = 0
            self.lvp = light_matrix(eye, self.res)
            self.render(renderer, struct_gl, self.lvp)
        return self.lvp

    def render(self, renderer, struct_gl, lvp):
        """Draw the terrain and the structures' last-uploaded meshes from the sun into the depth map. Loose
        dirt cubes do not cast shadows."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.res, self.res)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_lvp, 1, gl.GL_TRUE, lvp.astype(np.float32))
        gl.glUniformMatrix4fv(self.u_model, 1, gl.GL_TRUE, np.eye(4, dtype=np.float32))
        n = renderer._n                                 # terrain: the same visible chunk ranges the main pass uses
        if n:
            gl.glBindVertexArray(renderer.vao)
            gl.glMultiDrawArrays(gl.GL_TRIANGLES, renderer._first[:n], renderer._count[:n], n)
        model = np.eye(4, dtype=np.float32)
        for ent in struct_gl.cache.values():            # structures: each object's mesh, posed by its rigid transform
            if not ent[2]:
                continue
            o = ent[4]
            if not struct_gl.in_view(o):
                continue
            model[:3, :3] = o.R
            model[:3, 3] = o.frame_pos
            gl.glUniformMatrix4fv(self.u_model, 1, gl.GL_TRUE, model)
            gl.glBindVertexArray(ent[0])
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, ent[2])
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, _FRAME_FBO)
        gl.glViewport(0, 0, WIDTH, HEIGHT)


class GroundDmgView(GridObject):
    """The damaged ground near the look point as a cell grid, so the structure renderer can draw it coloured by damage."""

    def mesh(self, mode=None):
        out = super().mesh(mode)
        if mode is not None and len(out):
            out = np.array(out, np.float32)
            out[:, 0:3] += out[:, 3:6] * 0.012                 # lifted off the surface so it does not flicker against the terrain
        return out


def make_ground_dmg_views(world, centre, chunk_reach=3, hot=()):
    """The damaged ground as a list of GroundDmgView, one per damaged chunk within `chunk_reach` chunks of `centre` (empty if none).
    Built straight from the sparse damage layer, so the cost follows how much ground is damaged, not how far you can see.
    `hot` is a list of (n, 3) cube-index arrays (ledges about to give way); those cubes are drawn as about-to-fall."""
    if not world.gdmg:
        return []
    pcx, pcz = math.floor(float(centre[0]) / CHUNK_SIZE), math.floor(float(centre[2]) / CHUNK_SIZE)
    hot = [np.asarray(h, np.int64) for h in hot if len(h)]
    views = []
    for key, (a0, arr) in list(world.gdmg.items()):
        if abs(key[0] - pcx) > chunk_reach or abs(key[1] - pcz) > chunk_reach:
            continue
        d = world.dens.get(key)
        if d is None:
            continue
        rows = arr.shape[1]
        dm = arr.astype(np.float32)
        mask = (d.box(0, NC - 1, a0, a0 + rows - 1, 0, NC - 1) > 0) & (dm > 0.02)
        if not mask.any():
            continue
        b0, b1 = [], []
        for ax in range(3):
            nz = np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0]
            b0.append(int(nz[0]))
            b1.append(int(nz[-1]) + 1)
        sl = tuple(slice(b0[k], b1[k]) for k in range(3))
        mask, dm = mask[sl], dm[sl]
        lo2 = np.array([key[0] * NC, a0, key[1] * NC], np.int64) + np.array(b0)
        mat = np.where(mask, DIRT, 0).astype(np.int16)
        obj = GroundDmgView(mat, np.where(mask, dm, 0.0).astype(np.float32), np.ones(mat.shape, np.float32), CELL)
        for h in hot:
            r = h - lo2
            ok = ((r >= 0) & (r < np.array(mat.shape))).all(axis=1)
            r = r[ok]
            if len(r):
                obj.stress[r[:, 0], r[:, 1], r[:, 2]] = 1.0
        ctr = (lo2 + np.array(mat.shape) * 0.5) * CELL
        obj.frame_pos = ctr
        obj.corner = lo2 * CELL - ctr
        views.append(obj)
    return views


def make_ground_dmg_view(world, centre, half=(12.0, 10.0, 12.0), hot=()):
    """Kept for older callers: the first damaged chunk's view near `centre`, or None."""
    v = make_ground_dmg_views(world, centre, chunk_reach=max(1, int(max(half) // CHUNK_SIZE) + 1), hot=hot)
    return v[0] if v else None


class StructureRenderer:
    """Draws every structure and loose body as solid geometry: one mesh per object, placed by a model
    matrix, so a body that has turned over is drawn turned over. A mesh is uploaded again only when the
    object's cells change."""

    def __init__(self):
        vs = _compile_shader(_SOLID_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_SOLID_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_model = gl.glGetUniformLocation(self.program, "uModel")
        self.u_light = gl.glGetUniformLocation(self.program, "uLight")
        self.u_alpha = gl.glGetUniformLocation(self.program, "uAlpha")
        self.u_lvp = gl.glGetUniformLocation(self.program, "uLVP")
        self.u_shadow = gl.glGetUniformLocation(self.program, "uShadow")
        self.u_stexel = gl.glGetUniformLocation(self.program, "uShadowTexel")
        self.shadow_texel = 1.0 / SHADOW_RES
        self.u_tiles = gl.glGetUniformLocation(self.program, "uTiles")
        self.u_grid = gl.glGetUniformLocation(self.program, "uGrid")
        self.u_cell = gl.glGetUniformLocation(self.program, "uCell")
        self.u_corner = gl.glGetUniformLocation(self.program, "uCorner")
        self.u_eye = gl.glGetUniformLocation(self.program, "uEye")
        self.light_vp = np.eye(4, dtype=np.float32)   # set each frame by the main loop
        self.shadow_tex = 0
        self.view = None     # (player chunk x, player chunk z, chunk radius): objects further out than this are not drawn, like the terrain
        self.icache = {}     # integrity colouring: id(object) -> [vao, vbo, count, (mesh ver, stress ver), object]
        self.cache = {}      # id(object) -> [vao, vbo, vertex count, mesh version, object]

    def in_view(self, o):
        """False for an object whose chunk is outside the square of chunks the terrain is drawn for. Without this a
        tower still showed up out where the ground had stopped, floating against the sky."""
        if self.view is None:
            return True
        pcx, pcz, r = self.view
        S = CHUNK_SIZE
        return (abs(math.floor(o.frame_pos[0] / S) - pcx) <= r and abs(math.floor(o.frame_pos[2] / S) - pcz) <= r)

    def _make(self):
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 1, 36)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, 40, ctypes.c_void_p(off))
        gl.glBindVertexArray(0)
        return vao, vbo

    def render(self, vp, ambient, overlay=False, grid=False, extra=()):
        objs = STRUCTS.objects()
        extra = list(extra) if overlay else []
        live = {id(o) for o in objs} | {id(o) for o in extra}
        for k in [k for k in self.icache if k not in live]:
            vao, vbo = self.icache.pop(k)[:2]
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteBuffers(1, [vbo])
        for k in [k for k in self.cache if k not in live]:
            vao, vbo = self.cache.pop(k)[:2]
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteBuffers(1, [vbo])
        if not objs and not extra:
            return
        gl.glUseProgram(self.program)
        gl.glUniform1f(self.u_stexel, self.shadow_texel)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        eye_ = _eye_from_vp(vp)
        gl.glUniform3f(self.u_eye, float(eye_[0]), float(eye_[1]), float(eye_[2]))
        gl.glUniform1f(self.u_light, max(float(ambient), GFX['ambient_floor']))
        gl.glUniform1f(self.u_alpha, 1.0)
        gl.glUniform1f(self.u_grid, 1.0 if grid else 0.0)
        gl.glUniformMatrix4fv(self.u_lvp, 1, gl.GL_TRUE, self.light_vp.astype(np.float32))
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.shadow_tex)
        gl.glUniform1i(self.u_shadow, 1)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tile_texture())
        gl.glUniform1i(self.u_tiles, 2)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        model = np.eye(4, dtype=np.float32)
        for o in objs:
            if not self.in_view(o):
                continue
            ent = self.cache.get(id(o))
            if ent is None:
                vao, vbo = self._make()
                ent = self.cache[id(o)] = [vao, vbo, 0, -1, o]
            if ent[3] != o.mesh_ver:
                data = np.ascontiguousarray(o.mesh(), np.float32)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, ent[1])
                gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
                ent[2], ent[3] = len(data), o.mesh_ver
            if not ent[2]:
                continue
            model[:3, :3] = o.R
            model[:3, 3] = o.frame_pos
            gl.glUniformMatrix4fv(self.u_model, 1, gl.GL_TRUE, model)
            gl.glUniform1f(self.u_cell, float(o.cs))
            gl.glUniform3f(self.u_corner, *(float(c) for c in o.corner))
            gl.glBindVertexArray(ent[0])
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, ent[2])
        if overlay:
            # Second pass over the same faces, coloured by how close each cell is to failing.
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glUniform1f(self.u_alpha, 0.72)
            gl.glUniform1f(self.u_light, 1.0)
            for o in list(objs) + extra:
                if not self.in_view(o):
                    continue
                ent = self.icache.get(id(o))
                if ent is None:
                    vao, vbo = self._make()
                    ent = self.icache[id(o)] = [vao, vbo, 0, None, o]
                key = (o.mesh_ver, o.stress_ver)
                if ent[3] != key:
                    data = np.ascontiguousarray(o.mesh('integrity'), np.float32)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, ent[1])
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
                    ent[2], ent[3] = len(data), key
                if not ent[2]:
                    continue
                model[:3, :3] = o.R
                model[:3, 3] = o.frame_pos
                gl.glUniformMatrix4fv(self.u_model, 1, gl.GL_TRUE, model)
                gl.glUniform1f(self.u_cell, float(o.cs))
                gl.glUniform3f(self.u_corner, *(float(c) for c in o.corner))
                gl.glBindVertexArray(ent[0])
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, ent[2])
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDisable(gl.GL_BLEND)
            gl.glUniform1f(self.u_alpha, 1.0)
        gl.glBindVertexArray(0)





def _body_remove(self, kill):
    """A loose body loses cells. It is re-centred on its new centre of mass so its spin and inertia stay consistent,
    and it wakes up."""
    self.mat[kill] = 0
    self.cdmg[kill] = 0
    self._geometry()
    if self.alive.any():
        self.recentre()
        self.wake()


def _recentre(self):
    """Move the body's origin to its centre of mass without moving any cell in the world. The origin's velocity is
    adjusted to match, so the body's motion does not change."""
    idx = np.argwhere(self.alive)
    m = self.cell_mass[self.alive]
    local = (idx + 0.5) * self.cs + self.corner
    com = (m[:, None] * local).sum(0) / m.sum()
    shift = com @ self.R.T
    self.frame_pos = self.frame_pos + shift
    self.v = self.v + np.cross(self.w, shift)
    self.corner = self.corner - com
    self._aabb = None
    self._mass_props()
    self._sync()


def _body_damage(self, centre, radius, power, shatter=False):
    """A blast or drill hit on this loose body. Cells near the centre crack or are destroyed, using the same falloff
    and hardness rule as fixed structures. Returns (centres, materials) of the cells destroyed."""
    none = (np.zeros((0, 3)), np.zeros(0, np.int64))
    idx = np.argwhere(self.alive)
    if not len(idx):
        return none
    local = (idx + 0.5) * self.cs + self.corner
    cc = self.frame_pos + local @ self.R.T
    dist = np.linalg.norm(cc - centre, axis=1)
    near = dist < radius
    if not near.any():
        return none
    idx, cc, dist = idx[near], cc[near], dist[near]
    self.wake()
    f = power * (1.0 - dist / radius) ** 2
    ii = (idx[:, 0], idx[:, 1], idx[:, 2])
    m = self.mat[ii]
    thr = _S_HARD[m] * STRUCT_SHATTER
    kill = f >= thr
    if shatter:
        kill = np.ones_like(kill)
    hurt = ~kill
    if hurt.any():
        jj = (idx[hurt, 0], idx[hurt, 1], idx[hurt, 2])
        self.cdmg[jj] = np.minimum(0.85, self.cdmg[jj] + 0.7 * f[hurt] / thr[hurt])
    killed_c, killed_m = cc[kill], m[kill]
    if kill.any():
        mask = np.zeros(self.alive.shape, bool)
        mask[idx[kill, 0], idx[kill, 1], idx[kill, 2]] = True
        self._remove(mask)
    return killed_c, killed_m


def _cells_to_debris(body, mask):
    """World positions and materials of the cells in mask, for throwing out as loose chunks."""
    idx = np.argwhere(mask)
    if not len(idx):
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    local = (idx + 0.5) * body.cs + body.corner
    return body.frame_pos + local @ body.R.T, body.mat[mask].copy()


def _piece_of(body, piece):
    """A new rigid body made of the cells in piece, keeping the parent's pose and velocity."""
    child = RigidBody(np.where(piece, body.mat, 0), np.where(piece, body.cdmg, 0),
                      body.tint.copy(), body.cs, np.zeros(3))
    child.corner = body.corner.copy()
    child.frame_pos = body.frame_pos.copy()
    child.q = body.q.copy()
    child.v = body.v.copy()
    child.w = body.w.copy()
    child._sync()                      # R must follow q before recentre() uses it
    child.recentre()
    return child


def _split_body(body, min_cells=None):
    """Cells removed from a loose body can leave several pieces that no longer touch. The largest stays as `body`;
    each other piece becomes its own body, or debris if it is tiny. Returns (new bodies, debris), where debris is a
    list of (positions, materials) pairs."""
    alive = body.alive
    if not alive.any():
        return [], []
    bonds = [alive[:-1] & alive[1:], alive[:, :-1] & alive[:, 1:], alive[:, :, :-1] & alive[:, :, 1:]]
    lab = _label_cells(alive, bonds)
    labels = [int(L) for L in np.unique(lab[alive])]
    if len(labels) <= 1:
        return [], []
    sizes = {L: int((lab == L).sum()) for L in labels}
    main = max(labels, key=lambda L: sizes[L])
    if min_cells is None:
        min_cells = max(6, int(round(STRUCT_MIN_BODY_CELLS * (0.5 / body.cs) ** 3)))
    new_bodies, debris = [], []
    for L in labels:
        if L == main:
            continue
        piece = (lab == L) & alive
        if sizes[L] < min_cells:
            debris.append(_cells_to_debris(body, piece))
        else:
            new_bodies.append(_piece_of(body, piece))
    body._remove(alive & (lab != main))     # the pieces split off are now gone from the parent
    return new_bodies, debris


RigidBody._remove = _body_remove
RigidBody.recentre = _recentre
RigidBody.damage_at = _body_damage


_MAT_NAMES = {b: n for b, n in ((DIRT, 'Dirt'), (STONE, 'Stone'), (BRICK, 'Brick'), (PLANKS, 'Planks'), (STEEL, 'Steel'), (CONCRETE_WHITE, 'Concrete'), (WOOD, 'Wood'))}

class ChunkGrid:
    """Debug overlay (shown with G): outlines every chunk on the ground across the drawn area,
    so chunk boundaries and mesh spikes can be matched to a position in the world."""
    LIFT = 0.02    # sits just above the grass top so the lines don't flicker

    def __init__(self):
        vs = _compile_shader(_GRID_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_GRID_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_col = gl.glGetUniformLocation(self.program, "uCol")
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

    def draw(self, vp, pcx, pcz, radius):
        y = FLAT_SURFACE_Y + 1 + self.LIFT
        xs = np.arange(pcx - radius, pcx + radius + 2) * CHUNK_SIZE      # chunk boundaries along x
        zs = np.arange(pcz - radius, pcz + radius + 2) * CHUNK_SIZE      # chunk boundaries along z
        z0, z1 = zs[0], zs[-1]
        x0, x1 = xs[0], xs[-1]

        # Lines running along z (one per x boundary), then lines running along x (one per z boundary).
        lines_z = np.zeros((len(xs), 2, 3), dtype=np.float32)
        lines_z[:, :, 0] = xs[:, None]
        lines_z[:, 0, 2] = z0
        lines_z[:, 1, 2] = z1
        lines_x = np.zeros((len(zs), 2, 3), dtype=np.float32)
        lines_x[:, :, 2] = zs[:, None]
        lines_x[:, 0, 0] = x0
        lines_x[:, 1, 0] = x1
        verts = np.concatenate([lines_z.reshape(-1, 3), lines_x.reshape(-1, 3)])
        verts[:, 1] = y

        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glUniform4f(self.u_col, 1.0, 0.9, 0.2, 0.55)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_LINES, 0, len(verts))
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)

class SceneTarget:
    """An offscreen colour and depth target. When the resolution is not 100%, the 3D world is drawn into it at the chosen size
    and then scaled to the window, so the window itself never changes. At 100% it is not used at all."""
    def __init__(self):
        self.fbo = self.tex = self.rb = None
        self.size = (0, 0)
        self.broken = False                      # set if the card refuses the target: the resolution then stays at 100%

    def _free(self):
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo])
            gl.glDeleteTextures(1, [self.tex])
            gl.glDeleteRenderbuffers(1, [self.rb])
            self.fbo = self.tex = self.rb = None

    def _make(self, w, h):
        self._free()
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.rb = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, w, h)
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.rb)
        ok = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, _FRAME_FBO)
        self.size = (w, h)
        return ok

    def begin(self, scale, force=False):
        """Start drawing the world at `scale` (1.0 = window size). Returns False when the window itself should be drawn into.
        `force` uses the offscreen target even at 100%, for effects (bloom, contrast) that need the finished picture."""
        w, h = max(16, int(WIDTH * scale)), max(16, int(HEIGHT * scale))
        if self.broken or ((w, h) == (WIDTH, HEIGHT) and not force):
            return False
        try:
            if (w, h) != self.size or self.fbo is None:
                if not self._make(w, h):
                    self.broken = True
                    self._free()
                    print("Resolution scaling is not available on this graphics card: staying at 100%")
                    return False
        except Exception as err:
            self.broken = True
            print(f"Resolution scaling is not available: {err}")
            return False
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, w, h)
        return True

    def finish(self):
        """Scale what was drawn up (or down) to the window and go back to drawing into the window."""
        w, h = self.size
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _FRAME_FBO)
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, WIDTH, HEIGHT, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, _FRAME_FBO)
        gl.glViewport(0, 0, WIDTH, HEIGHT)


_FRAME_FBO = 0        # what "the screen" is while drawing: 0 is the window itself, or the offscreen frame while fullscreen


_POST_VERT = """
#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));   // one triangle that covers the screen
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_POST_DOWN = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec2 uTexel;      // one texel of the source
uniform float uThresh;    // brightness below which nothing glows (a very low value just copies and shrinks)
vec3 tap(vec2 uv) {
    vec3 c = texture(uTex, uv).rgb;
    float l = max(c.r, c.g * 0.9);                  // red/yellow weighted: fire, sparks and flashes glow, the blue sky and the grass do not
    return c * smoothstep(uThresh, uThresh + 0.18, l);
}
void main() {
    vec3 c = tap(vUV + uTexel * vec2(-0.5, -0.5)) + tap(vUV + uTexel * vec2(0.5, -0.5))
           + tap(vUV + uTexel * vec2(-0.5, 0.5)) + tap(vUV + uTexel * vec2(0.5, 0.5));
    fragColor = vec4(c * 0.25, 1.0);
}
"""

_POST_BLUR = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
uniform vec2 uDir;        // one texel along the blur direction
void main() {
    vec3 c = texture(uTex, vUV).rgb * 0.227027027;
    c += texture(uTex, vUV + uDir * 1.3846153846).rgb * 0.3162162162;
    c += texture(uTex, vUV - uDir * 1.3846153846).rgb * 0.3162162162;
    c += texture(uTex, vUV + uDir * 3.2307692308).rgb * 0.0702702703;
    c += texture(uTex, vUV - uDir * 3.2307692308).rgb * 0.0702702703;
    fragColor = vec4(c, 1.0);
}
"""

_POST_COMP = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;
uniform sampler2D uBloomA;   // half size glow: tight
uniform sampler2D uBloomB;   // quarter size glow: wide
uniform float uBloom;
uniform float uContrast;
uniform float uSharp;     // 0 = off; higher = crisper edges (unsharp mask)
uniform vec2 uTexel;
void main() {
    vec3 c = texture(uScene, vUV).rgb;
    if (uSharp > 0.0) {
        vec3 n = texture(uScene, vUV + vec2(uTexel.x, 0.0)).rgb + texture(uScene, vUV - vec2(uTexel.x, 0.0)).rgb
               + texture(uScene, vUV + vec2(0.0, uTexel.y)).rgb + texture(uScene, vUV - vec2(0.0, uTexel.y)).rgb;
        c += (c - n * 0.25) * uSharp;
    }
    c = (c - 0.5) * uContrast + 0.5;
    float lm = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(lm), c, 0.92);                                                                   // slightly less saturated
    c *= mix(vec3(0.97, 0.99, 1.04), vec3(1.04, 1.00, 0.95), smoothstep(0.25, 0.75, lm));         // cool shadows, warm highlights
    vec3 b = texture(uBloomA, vUV).rgb * 0.7 + texture(uBloomB, vUV).rgb * 1.0;
    c += b * uBloom;
    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""


class PostFX:
    """Bloom and contrast, applied to the finished 3D world (the HUD is drawn afterwards, so it is not touched). The bright parts
    of the picture are shrunk to half and quarter size, blurred, and added back on top, which makes fire and sparks glow."""
    def __init__(self):
        self.broken = False
        self.size = (0, 0)
        self.t = []
        self.vao = gl.glGenVertexArrays(1)           # an empty one: the corners are made in the vertex shader
        def prog(src):
            return _link_program(_compile_shader(_POST_VERT, gl.GL_VERTEX_SHADER), _compile_shader(src, gl.GL_FRAGMENT_SHADER))
        self.p_down, self.p_blur, self.p_comp = prog(_POST_DOWN), prog(_POST_BLUR), prog(_POST_COMP)
        loc = gl.glGetUniformLocation
        self.u = dict(
            down=(loc(self.p_down, "uTex"), loc(self.p_down, "uTexel"), loc(self.p_down, "uThresh")),
            blur=(loc(self.p_blur, "uTex"), loc(self.p_blur, "uDir")),
            comp=tuple(loc(self.p_comp, n) for n in ("uScene", "uBloomA", "uBloomB", "uBloom", "uContrast", "uSharp", "uTexel")))

    def _free(self):
        for fbo, tex, _w, _h in self.t:
            gl.glDeleteFramebuffers(1, [fbo])
            gl.glDeleteTextures(1, [tex])
        self.t = []

    def _make(self, w, h):
        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                     (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex, 0)
        ok = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, _FRAME_FBO)
        if not ok:
            raise RuntimeError("bloom target not complete")
        return (fbo, tex, w, h)

    def _ensure(self, w, h):
        if (w, h) == self.size and self.t:
            return
        self._free()
        hw, hh, qw, qh = max(8, w // 2), max(8, h // 2), max(4, w // 4), max(4, h // 4)
        self.t = [self._make(hw, hh), self._make(hw, hh), self._make(qw, qh), self._make(qw, qh)]
        self.size = (w, h)

    def _pass(self, prog, target, src_tex):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, target[0])
        gl.glViewport(0, 0, target[2], target[3])
        gl.glUseProgram(prog)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, src_tex)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def _blur(self, a, b):
        ub = self.u['blur']
        gl.glUseProgram(self.p_blur)
        gl.glUniform1i(ub[0], 0)
        gl.glUniform2f(ub[1], 1.0 / a[2], 0.0)
        self._pass(self.p_blur, b, a[1])
        gl.glUniform2f(ub[1], 0.0, 1.0 / a[3])
        self._pass(self.p_blur, a, b[1])

    def apply(self, scene, bloom, contrast, sharp=0.0):
        """Draw `scene` (a SceneTarget that has just been drawn into) onto the screen with bloom (0 = none), contrast (1 = unchanged) and sharpness (0 = none)."""
        if self.broken:
            scene.finish()
            return
        w, h = scene.size
        try:
            self._ensure(w, h)
        except Exception as err:
            self.broken = True
            print(f"Bloom and contrast are not available on this graphics card: {err}")
            scene.finish()
            return
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(self.vao)
        A, B, C, D = self.t
        if bloom > 0.0:
            ud = self.u['down']
            gl.glUseProgram(self.p_down)
            gl.glUniform1i(ud[0], 0)
            gl.glUniform2f(ud[1], 1.0 / w, 1.0 / h)
            gl.glUniform1f(ud[2], 0.82)
            self._pass(self.p_down, A, scene.tex)                   # the bright parts, at half size
            self._blur(A, B)
            gl.glUseProgram(self.p_down)
            gl.glUniform2f(ud[1], 1.0 / A[2], 1.0 / A[3])
            gl.glUniform1f(ud[2], -2.0)                              # no threshold: just shrink the glow to quarter size
            self._pass(self.p_down, C, A[1])
            self._blur(C, D)
            self._blur(C, D)                                         # twice: a wide, soft halo
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, _FRAME_FBO)
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        uc = self.u['comp']
        gl.glUseProgram(self.p_comp)
        for unit, tex in ((0, scene.tex), (1, A[1]), (2, C[1])):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(uc[unit], unit)
        gl.glUniform1f(uc[3], float(bloom))
        gl.glUniform1f(uc[4], float(contrast))
        gl.glUniform1f(uc[5], float(sharp))
        gl.glUniform2f(uc[6], 1.0 / w, 1.0 / h)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)


class FrameTarget:
    """While fullscreen, the whole frame is drawn offscreen at the game's own size and then scaled up to the screen, with
    black bars if the shapes differ, so nothing in the game has to know the screen is a different size."""
    def __init__(self):
        self.fbo = self.tex = self.rb = None
        self.active = False

    def enable(self):
        global _FRAME_FBO
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, WIDTH, HEIGHT, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.rb = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, WIDTH, HEIGHT)
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.rb)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            self.disable()
            return False
        _FRAME_FBO = self.fbo
        self.active = True
        return True

    def disable(self):
        global _FRAME_FBO
        _FRAME_FBO = 0
        self.active = False
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if self.fbo is not None:
            gl.glDeleteFramebuffers(1, [self.fbo])
        if self.tex is not None:
            gl.glDeleteTextures(1, [self.tex])
        if self.rb is not None:
            gl.glDeleteRenderbuffers(1, [self.rb])
        self.fbo = self.tex = self.rb = None
        gl.glViewport(0, 0, WIDTH, HEIGHT)

    def bind(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, WIDTH, HEIGHT)

    def present(self):
        """Put the finished frame on the screen, as large as it will go without changing its shape."""
        ww, wh = pygame.display.get_window_size()
        k = min(ww / WIDTH, wh / HEIGHT)
        dw, dh = int(WIDTH * k), int(HEIGHT * k)
        dx, dy = (ww - dw) // 2, (wh - dh) // 2
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glViewport(0, 0, ww, wh)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glClearColor(SKY_RGB[0], SKY_RGB[1], SKY_RGB[2], 1.0)
        gl.glBlitFramebuffer(0, 0, WIDTH, HEIGHT, dx, dy, dx + dw, dy + dh, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, WIDTH, HEIGHT)


def _clear_cache(world, renderer, player_pos, keep):
    """Free the chunks and the stored meshes away from the player. Edited chunks keep their edits."""
    pcx, pcz = math.floor(player_pos[0] / CHUNK_SIZE), math.floor(player_pos[2] / CHUNK_SIZE)
    unload_far_chunks(world, renderer, pcx, pcz, keep=keep)
    with CACHE.lock:
        CACHE.mesh.clear()
        CACHE.mesh_bytes = 0
        CACHE.hit_m = CACHE.miss_m = 0
    import gc
    gc.collect()


def _set_vsync(on):
    """Turn waiting for the screen refresh on or off while the game runs. Returns True if it took."""
    iv = 1 if on else 0
    try:
        from OpenGL.WGL.EXT.swap_control import wglSwapIntervalEXT       # Windows
        if wglSwapIntervalEXT(iv):
            return True
    except Exception:
        pass
    try:
        import glob, ctypes
        root = os.path.dirname(pygame.__file__)
        for pat in ('SDL2*.dll', 'libSDL2*.so*', 'libSDL2*.dylib', os.path.join('..', 'pygame.libs', 'libSDL2*'),
                    os.path.join('..', 'pygame.libs', 'SDL2*')):
            for path in glob.glob(os.path.join(root, pat)):
                lib = ctypes.CDLL(path)
                lib.SDL_GL_SetSwapInterval.argtypes = [ctypes.c_int]
                if lib.SDL_GL_SetSwapInterval(iv) == 0:
                    return True
    except Exception:
        pass
    return False


class Backdrop:
    """The last 3D frame, copied into a texture, so the Tab menu can be drawn over a frozen world
    without re-rendering the world every frame."""
    def __init__(self):
        vs = _compile_shader(_OVL_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_OVL_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_tex = gl.glGetUniformLocation(self.program, "uTex")

        quad = np.array([-1, -1, 0, 0,   1, -1, 1, 0,   1, 1, 1, 1,   -1, 1, 0, 1], dtype=np.float32)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))
        gl.glBindVertexArray(0)

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def capture(self):
        """Copy the frame currently in the back buffer into the texture."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glCopyTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 0, 0, WIDTH, HEIGHT, 0)

    def draw(self):
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glUniform1i(self.u_tex, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)

# ═══════════════════════════════════════════════
#  MONITOR (G): every stage of the frame, the GPU, the world and the process, in one overlay
# ═══════════════════════════════════════════════
PROF_WINDOW = 240              # frames kept for every graph and average (4 s at 60 fps)
PROF_TICK_S = 0.25             # the monitor's numbers and graphs refresh this often
MON_W, MON_H = 1200, 633
FRAME_BUDGET_MS = 1000.0 / 60.0

# CPU stages, in the order a frame runs them. PROF_TOP stages do not overlap and add up to the work time.
# PROF_SUB stages are parts of 'terrain' and are listed indented under it.
PROF_TOP = ['input', 'chunks', 'player', 'debris', 'structs', 'weapon', 'eject', 'terrain',
            'shadow', 'struct_draw', 'debris_draw', 'overlay', 'monitor', 'unload']
PROF_SUB = ['terrain.mesh', 'terrain.upload', 'terrain.list', 'terrain.collect', 'terrain.call']
# Parts of the debris stage ('debris'), timed inside BODIES.update(). They overlap each other, so they are not summed.
DEBRIS_SUB = ['debris.struct_push', 'debris.terrain_push', 'debris.move', 'debris.move.terrain',
             'debris.move.struct', 'debris.support', 'debris.separate', 'debris.terrain_fix']
STRUCTS_SUB = ['structs.check', 'structs.detach', 'structs.bodies']   # parts of the structs stage, timed inside STRUCTS.update()
PROF_FLIP = 'flip'
PROF_LABEL = {
    'input': 'input', 'chunks': 'chunk load + layout', 'player': 'player',
    'debris': 'debris physics', 'structs': 'structs + bodies', 'weapon': 'weapon (hit + dig)',
    'eject': 'eject player', 'terrain': 'terrain total', 'shadow': 'shadow pass',
    'struct_draw': 'structs draw', 'debris_draw': 'debris draw',
    'overlay': 'grid, crosshair, bar', 'monitor': 'monitor overlay', 'unload': 'unload far chunks',
    'terrain.mesh': 'mesh submit', 'terrain.upload': 'gpu upload', 'terrain.list': 'draw list',
    'terrain.collect': 'mesh collect', 'terrain.call': 'draw call', PROF_FLIP: 'flip (vsync wait)',
    'debris.struct_push': 'structure probes (push out)', 'debris.terrain_push': 'terrain push (in ground)',
    'debris.move': 'substep motion (all)', 'debris.move.terrain': 'motion: terrain samples',
    'debris.move.struct': 'motion: structure probes', 'debris.support': 'support + downhill',
    'debris.separate': 'cube separation', 'debris.terrain_fix': 'terrain push (after sep)',
    'structs.check': 'support checks', 'structs.detach': 'detach (label + split)', 'structs.bodies': 'rigid body steps',
}
PROF_COLOR = {
    'input': (120, 120, 130), 'chunks': (90, 140, 200), 'player': (200, 200, 90),
    'debris': (200, 110, 160), 'structs': (120, 200, 140), 'weapon': (230, 190, 90),
    'eject': (80, 160, 120), 'terrain': (70, 190, 230), 'shadow': (150, 120, 230),
    'struct_draw': (110, 220, 190), 'debris_draw': (230, 120, 120),
    'overlay': (170, 170, 200), 'monitor': (255, 255, 255), 'unload': (100, 100, 100),
    'terrain.mesh': (110, 200, 230), 'terrain.upload': (110, 200, 230), 'terrain.list': (110, 200, 230),
    'terrain.collect': (110, 200, 230), 'terrain.call': (110, 200, 230), PROF_FLIP: (150, 150, 160),
    'debris.struct_push': (220, 150, 190), 'debris.terrain_push': (220, 150, 190), 'debris.move': (220, 150, 190),
    'debris.move.terrain': (220, 150, 190), 'debris.move.struct': (220, 150, 190), 'debris.support': (220, 150, 190),
    'debris.separate': (220, 150, 190), 'debris.terrain_fix': (220, 150, 190),
    'structs.check': (150, 215, 170), 'structs.detach': (150, 215, 170), 'structs.bodies': (150, 215, 170),
}
GPU_STAGES = ['shadow', 'terrain', 'struct_draw', 'debris_draw', 'overlay', 'monitor']


class Timed:
    """Times one stage of the frame. Always on the CPU; on the GPU too when gpu is True, with a timer query.
    GL runs one timer query at a time, so GPU stages must never nest."""
    __slots__ = ('name', 'gpu', 't0')

    def __init__(self, name, gpu=True):
        self.name = name
        self.gpu = gpu
        self.t0 = 0.0

    def __enter__(self):
        if self.gpu:
            GT.begin(self.name)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        PROF.add(self.name, (time.perf_counter() - self.t0) * 1000.0)
        if self.gpu:
            GT.end()
        return False


class Profiler:
    """CPU milliseconds per stage for every frame, over the last PROF_WINDOW frames.
    A stage that did not run in a frame counts as zero, so averages stay honest."""

    def __init__(self):
        self.cur = {}
        self.hist = {name: collections.deque(maxlen=PROF_WINDOW) for name in PROF_TOP + PROF_SUB + DEBRIS_SUB + STRUCTS_SUB + [PROF_FLIP]}
        self.frame_ms = collections.deque(maxlen=PROF_WINDOW)
        self.work_ms = collections.deque(maxlen=PROF_WINDOW)

    def add(self, name, ms):
        self.cur[name] = self.cur.get(name, 0.0) + ms

    def push(self, frame_ms, work_ms):
        for name in self.cur:
            if name not in self.hist:
                self.hist[name] = collections.deque(maxlen=PROF_WINDOW)
        for name, dq in self.hist.items():
            dq.append(self.cur.get(name, 0.0))
        self.frame_ms.append(frame_ms)
        self.work_ms.append(work_ms)
        self.cur = {}


def _new_query():
    q = gl.glGenQueries(1)
    return int(q[0]) if hasattr(q, '__len__') else int(q)


class GpuTimers:
    """GPU milliseconds per stage, from GL_TIME_ELAPSED queries. Results arrive a frame or two late and are
    read only once available, so reading them never stalls the pipeline."""

    def __init__(self):
        self.enabled = True
        self.open_name = None
        self.open_q = None
        self.pending = []        # (stage name, query) issued but not read yet
        self.free = []           # queries ready for reuse
        self.hist = {}           # stage name -> deque of ms
        self.latest = {}         # stage name -> most recent ms

    def begin(self, name):
        if not self.enabled or self.open_name is not None:
            return
        q = self.free.pop() if self.free else _new_query()
        gl.glBeginQuery(gl.GL_TIME_ELAPSED, q)
        self.open_name, self.open_q = name, q

    def end(self):
        if self.open_name is None:
            return
        gl.glEndQuery(gl.GL_TIME_ELAPSED)
        self.pending.append((self.open_name, self.open_q))
        self.open_name = self.open_q = None

    def poll(self):
        if not self.pending:
            return
        keep = []
        for name, q in self.pending:
            if gl.glGetQueryObjectuiv(q, gl.GL_QUERY_RESULT_AVAILABLE):
                ms = gl.glGetQueryObjectuiv(q, gl.GL_QUERY_RESULT) / 1e6
                self.free.append(q)
                self.latest[name] = ms
                self.hist.setdefault(name, collections.deque(maxlen=PROF_WINDOW)).append(ms)
            else:
                keep.append((name, q))
        self.pending = keep

    def total(self):
        return sum(self.latest.values())


def _window(dq, n):
    """The last n values of a rolling window, padded on the left with zeros."""
    v = list(dq)[-n:] if n else []
    return [0.0] * (n - len(v)) + v


def _stats(vals):
    """(latest, average, worst 1%) of a list of values."""
    if not vals:
        return 0.0, 0.0, 0.0
    a = np.asarray(vals, dtype=np.float64)
    return float(a[-1]), float(a.mean()), float(np.percentile(a, 99))


def _fmt(v, spec='.0f', unit=''):
    return 'n/a' if v is None else f"{v:{spec}}{unit}"


PROF = Profiler()
GT = GpuTimers()


def _proj(vp, p):
    """World point -> (screen x, screen y, depth), or None when the point is behind the camera."""
    c = vp @ np.array([p[0], p[1], p[2], 1.0])
    w = float(c[3])
    if w <= 0.1:
        return None
    return ((c[0] / w + 1.0) * 0.5 * WIDTH, (1.0 - c[1] / w) * 0.5 * HEIGHT, w)


def _march_terrain(world, origin, d, reach, step=0.1):
    """First solid terrain sample along a ray: (distance, point), or None."""
    t = 0.0
    while t <= reach:
        p = origin + d * t
        v = world.cell(math.floor(p[0] / CELL), math.floor(p[1] / CELL), math.floor(p[2] / CELL))
        if v is None:
            return None              # ran into chunks that aren't loaded
        if v > 0:
            return t, p
        t += step
    return None


def _first_hit(world, origin, d, reach):
    """Nearest solid thing along a ray: (distance, point, structure or body or None for terrain), or None."""
    origin = np.asarray(origin, np.float64)
    d = np.asarray(d, np.float64)
    best = None
    m = _march_terrain(world, origin, d, reach)
    if m is not None:
        best = (m[0], m[1], None)
    s = STRUCTS.raycast(origin, d, reach)
    if s is not None and (best is None or s[0] < best[0]):
        best = (s[0], origin + d * s[0], s[2])
    return best


_CUBE_UNIT = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)], np.float64)
_CUBE_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
_CUBE_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def _ray_cubes(origin, d, tmax):
    """Nearest loose dirt cube (BODIES) along a ray: (distance, index) or None. Cubes are axis aligned."""
    n = BODIES.n
    if n == 0:
        return None
    h = BODIES.size[:, None] * 0.5
    lo, hi = BODIES.pos - h, BODIES.pos + h
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1, t2 = (lo - origin) * inv, (hi - origin) * inv
    tn = np.minimum(t1, t2).max(axis=1)
    tf = np.maximum(t1, t2).min(axis=1)
    ok = (tf >= np.maximum(tn, 0.0)) & (tn <= tmax)
    if not ok.any():
        return None
    idx = np.nonzero(ok)[0]
    k = idx[int(np.argmin(np.maximum(tn[idx], 0.0)))]
    return float(max(tn[k], 0.0)), int(k)


def structure_hit_face(origin, direction, point, body):
    # Recover the entry face of the live cell found by the structure raycast.
    local_origin = (origin - body.frame_pos) @ body.R
    local_direction = direction @ body.R
    local_point = (point - body.frame_pos) @ body.R
    cell = np.floor((local_point - body.corner) / body.cs)
    lo = body.corner + cell * body.cs
    hi = lo + body.cs
    entering = np.where(local_direction > 0.0, lo, hi)
    moving = np.abs(local_direction) > 1e-9
    t = np.full(3, -np.inf)
    np.divide(entering - local_origin, local_direction, out=t, where=moving)
    axis = int(np.argmax(t))
    normal = -np.sign(local_direction[axis]) * body.R[:, axis]
    return origin + max(0.0, t[axis]) * direction, normal


def aim_target(world, origin, d, hit, reach):
    """The cube under the crosshair for the highlight: dict(corners (8,3) in world space, label, kind) or None.
    `hit` is what _first_hit found (distance, point, structure or None for terrain) and may be None."""
    origin = np.asarray(origin, np.float64)
    d = np.asarray(d, np.float64)
    best_t = hit[0] if hit is not None else reach
    c = _ray_cubes(origin, d, best_t)
    if c is not None and c[0] <= best_t:
        k = c[1]
        lo = BODIES.pos[k] - BODIES.size[k] * 0.5
        return dict(corners=lo + _CUBE_UNIT * BODIES.size[k], label='Dirt cube', kind='debris')
    if hit is None:
        return None
    p, o = np.asarray(hit[1], np.float64), hit[2]
    if o is None:                                            # terrain: the ground cube the ray entered
        cell = np.floor(p / CELL) * CELL
        return dict(corners=cell + _CUBE_UNIT * CELL, label='Ground', kind='terrain')
    for nudge in (0.05, 0.15, 0.3, -0.05):                   # structure or loose piece
        q = p + d * nudge * o.cs
        ij = np.floor(((q - o.frame_pos) @ o.R - o.corner) / o.cs).astype(np.int64)
        if (ij >= 0).all() and (ij < np.array(o.alive.shape)).all() and o.alive[tuple(ij)]:
            lo = o.corner + ij * o.cs
            w = o.frame_pos + (lo + _CUBE_UNIT * o.cs) @ o.R.T
            m = _MAT_NAMES.get(int(o.mat[tuple(ij)]), 'Stone')
            return dict(corners=w, label=f"{m} ({'fixed' if isinstance(o, Structure) else 'loose'})", kind='structure')
    return None


_FX_VERT = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aCol;
out vec4 vCol;
void main() {
    vCol = aCol;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""
_FX_FRAG = """
#version 330 core
in vec4 vCol;
out vec4 fragColor;
void main() {
    fragColor = vCol;
}
"""


class WeaponFX:
    """Shots, tracers and the aim highlight. Every primitive is in world space and is projected with the frame's view-projection
    matrix, so the effects stay put while the camera moves. Drawn as GL triangles (thick lines are screen-space quads): nothing
    is rendered to a window-sized image and uploaded, which used to cost several milliseconds a frame."""
    def __init__(self):
        self.program = _link_program(_compile_shader(_FX_VERT, gl.GL_VERTEX_SHADER), _compile_shader(_FX_FRAG, gl.GL_FRAGMENT_SHADER))
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(8))
        gl.glBindVertexArray(0)

    def draw(self, vp, prims):
        verts = []                                       # x, y (pixels), r, g, b, a per vertex, three per triangle
        for pr in prims:
            kind = pr[0]
            if kind == 'line':
                pa, pb = _proj(vp, pr[1]), _proj(vp, pr[2])
                if not (pa and pb):
                    continue
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                ln = math.hypot(dx, dy)
                if ln < 1e-6:
                    continue
                hw = pr[4] * 0.5
                nx, ny = -dy / ln * hw, dx / ln * hw
                col = tuple(c / 255.0 for c in pr[3])
                q = ((pa[0] + nx, pa[1] + ny), (pa[0] - nx, pa[1] - ny), (pb[0] - nx, pb[1] - ny), (pb[0] + nx, pb[1] + ny))
                for k in (0, 1, 2, 0, 2, 3):
                    verts.append((q[k][0], q[k][1]) + col)
            elif kind == 'poly':                         # a filled, translucent polygon (the face of a highlighted cube)
                qs = [_proj(vp, v) for v in pr[1]]
                if not all(qs):
                    continue
                col = tuple(c / 255.0 for c in pr[2])
                for k in range(1, len(qs) - 1):
                    for q in (qs[0], qs[k], qs[k + 1]):
                        verts.append((q[0], q[1]) + col)
        if not verts:
            return
        a = np.asarray(verts, np.float32)
        a[:, 0] = a[:, 0] / WIDTH * 2.0 - 1.0
        a[:, 1] = 1.0 - a[:, 1] / HEIGHT * 2.0
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, a.nbytes, a, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(a))
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)


class Arsenal:
    """Hand-tool slot. The axe and shovel (models, swing, digging and chopping) have been removed to be rebuilt; what is left is
    the crosshair cube highlight (`target`), which the target label also reads."""
    REACH = 4.5
    MARK_TIME = 0.5

    def __init__(self):
        self.fx = WeaponFX()
        self.reset()

    def reset(self):
        self.extra = []            # more things to draw this frame, set by the main loop
        self.drawn = False
        self.target = None         # the cube under the crosshair right now: dict(corners, label, kind, dir)
        self.pulse = 0.0

    def release(self):
        pass

    def update(self, world, player, dt, fire_now=False, weapon=0):
        eye = np.asarray(player.eye, np.float64)
        fwd = np.asarray(player.forward(), np.float64)
        fwd = fwd / np.linalg.norm(fwd)
        self.pulse = max(0.0, self.pulse - dt)
        self.target = aim_target(world, eye, fwd, _first_hit(world, eye, fwd, self.REACH), self.REACH)
        if self.target is not None:
            self.target['dir'] = fwd.copy()

    def draw(self, vp, fov):
        prims = list(self.extra)
        tg = self.target
        if tg is not None:
            k = 0.55 + 0.45 * min(1.0, self.pulse / 0.25)
            C = tg['corners']
            col = (255, 196, 64)
            ctr = C.mean(axis=0)
            best = max(_CUBE_FACES, key=lambda f: float(np.dot(C[list(f)].mean(axis=0) - ctr, -tg['dir'])))
            prims.append(('poly', [C[i] for i in best], (col[0], col[1], col[2], int(70 * k)), 0))
            for a, b in _CUBE_EDGES:
                prims.append(('line', C[a], C[b], (col[0], col[1], col[2], int(70 * k)), 7))
            for a, b in _CUBE_EDGES:
                prims.append(('line', C[a], C[b], (255, 240, 190, int(255 * k)), 2))
        if not prims and not self.drawn:
            return
        self.drawn = bool(prims)
        self.fx.draw(vp, prims)


'''
# ===== ENGINE END =====


if __name__ == "__mp_main__":
    install_engine()          # a mesh worker process: the engine functions it is sent live in this module
elif __name__ == "__main__":
    main()
