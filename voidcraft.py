# game.py
# The Voidcraft game: settings panel, HUD overlays, loading screen and the main loop.
# The engine (world, physics, meshing, sound, OpenGL renderer) is built into engineui.py, which runs this
# game on it and shows the loading progress itself (see _emit / _window_ready below).
SPAWN_TREES = False   # set True to bring the trees back
SPAWN_HUT = False     # set True to bring the hut back

import math, os, sys, time, ctypes, threading, multiprocessing
import numpy as np
import pygame
from pygame.locals import *
from OpenGL import GL as gl

# When engineui.py starts this game it shows the loading screen itself: this process reports each loading step
# on stdout, keeps its own window hidden until loading is finished, then shows it.
_UI_LOADING = bool(os.environ.get("VOIDENGINE_UI"))


def _emit(*parts):
    if _UI_LOADING:
        print("@@LOAD|" + "|".join(str(p).replace("|", "/").replace("\n", " ") for p in parts), flush=True)


_WIN = None


def _window_ready():
    """Loading is finished: show the hidden game window, take the mouse, and tell engineui."""
    if not _UI_LOADING:
        return
    try:
        from pygame._sdl2.video import Window
        global _WIN
        _WIN = Window.from_display_module()      # kept for the whole run: freeing the wrapper corrupts memory
        _WIN.show()
    except Exception as err:
        print("could not show the game window:", err)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    _emit("done")


try:
    import psutil
except ImportError:
    psutil = None

try:
    import voidcraft_engine as _E    # provided by engineui.py while the game runs. WIDTH, HEIGHT, GPU_OK, GPU_NAME change at run time, so they are read as _E.NAME
except ImportError:
    raise SystemExit("Open this game from engineui.py: it provides the engine.")
from voidcraft_engine import (
    apply_wobble, Arsenal, Backdrop, BODIES, BODY_MAX, build_trees, CACHE, check_chopped, CHUNK_SIZE,
    ChunkGrid, CONFIG, DEBRIS_AMOUNT, DirtBodyRenderer, FOV,
    FRAME_BUDGET_MS, FrameTarget, GFX, GT, INV, inv_note,
    load_around, LOD_NAMES, MESH_MAX_PENDING, MESH_WORKERS, PlayerController, PROF, PROF_FLIP, QUALITY_KEYS, QUALITY_NAMES, quality_snapshot, Renderer,
    PostFX, SceneTarget, set_ground_res, SFX, SHADOW_NAMES, ShadowMap, SKY_RGB, STRENGTH, STRENGTH_PCT,
    STRUCT_RES, STRUCTS, Structure, StructureRenderer, sync_quality, TARGET_FPS,
    terrain_height, Timed, TREES, unload_far_chunks, VERT_STRIDE,
    WOBBLE_PCT, World, _compile_shader, _fit_window_to_display,
    _gpu_init, _link_program, _MAT_NAMES, _OVL_FRAG, _OVL_VERT, _S_HARD, _set_vsync,
    _window,
)

GAME = {
    "name": "Voidcraft",
    "version": "1.0",
    "engine_min": 1,
    "requires": [
    "apply_wobble",
            "Arsenal",
        "Backdrop",
        "BODIES",
        "BODY_MAX",
        "build_trees",
        "CACHE",
        "check_chopped",
        "CHUNK_SIZE",
        "ChunkGrid",
        "CONFIG",
        "DEBRIS_AMOUNT",
        "DirtBodyRenderer",
        "FOV",
        "FRAME_BUDGET_MS",
        "FrameTarget",
        "GFX",
        "GT",
        "INV",
        "inv_note",
        "load_around",
        "LOD_NAMES",
        "MESH_MAX_PENDING",
        "MESH_WORKERS",
        "PlayerController",
        "PROF",
        "PROF_FLIP",
        "QUALITY_KEYS",
        "QUALITY_NAMES",
        "quality_snapshot",
        "Renderer",
        "PostFX",
        "SceneTarget",
        "set_ground_res",
        "SFX",
        "SHADOW_NAMES",
        "ShadowMap",
        "SKY_RGB",
        "STRENGTH",
        "STRENGTH_PCT",
        "STRUCT_RES",
        "STRUCTS",
        "Structure",
        "StructureRenderer",
        "sync_quality",
        "TARGET_FPS",
        "terrain_height",
        "Timed",
        "TREES",
        "unload_far_chunks",
        "VERT_STRIDE",
        "WOBBLE_PCT",
        "World",
        "_compile_shader",
        "_fit_window_to_display",
        "_gpu_init",
        "_link_program",
        "_MAT_NAMES",
        "_OVL_FRAG",
        "_OVL_VERT",
        "_S_HARD",
        "_set_vsync",
        "_window",
    ],
}

TNT_MAX = 2   # most charges that can be planted at once


# ═══════════════════════════════════════════════
#  GRAPHICS SETTINGS MENU (Tab)
# ═══════════════════════════════════════════════
# Tabs of the settings panel. Each row is a header, a slider, a toggle or an info line.
# Sliders change with Left/Right within [min, max] in steps of `step`.
# Toggles flip with Left/Right or Enter. Headers and info lines are skipped by the cursor.
SETTINGS_TABS = [
    ("DISPLAY", [
        {'type': 'header', 'text': 'RESOLUTION'},
        {'type': 'slider', 'key': 'render_scale', 'label': 'Resolution', 'min': 25, 'max': 150, 'step': 5,
         'fmt': lambda v: f"{v}%  {int(_E.WIDTH * v / 100)} x {int(_E.HEIGHT * v / 100)}",
         'help': 'How big the 3D world is drawn, then scaled to fit. Lower is faster, above 100% is sharper.'},
        {'type': 'slider', 'key': 'sharpness', 'label': 'Sharpness', 'min': 0, 'max': 100, 'step': 5,
         'fmt': lambda v: 'off' if v == 0 else f"{v}%",
         'help': 'Crisper edges. Helps when the resolution is turned down; too much looks harsh.'},
        {'type': 'header', 'text': 'FRAME RATE'},
        {'type': 'slider', 'key': 'fps_limit', 'label': 'FPS limit', 'min': 0, 'max': 240, 'step': 30,
         'fmt': lambda v: 'uncapped' if v == 0 else f"{v} fps",
         'help': 'Frame rate cap. Uncapped shows the true frame time but uses a full CPU core.'},
        {'type': 'toggle', 'key': 'vsync', 'label': 'VSync',
         'help': 'Waits for the screen refresh before each frame: no tearing, a little more delay.'},
    ]),
    ("GRAPHICS", [
        {'type': 'header', 'text': 'TERRAIN'},
        {'type': 'toggle', 'key': 'cube_unlock', 'label': 'Change cube detail',
         'help': 'Off: cube detail is locked. On: the slider below works. Changing it rebuilds the whole map, so everything on it is reset.'},
        {'type': 'slider', 'key': 'dirt_cubes', 'label': 'Cube detail', 'min': 1, 'max': 50, 'step': 1, 'locked': True,
         'fmt': lambda v: "1 cube = 1 block" if int(v) == 1 else f"{int(v)} cubes per block",
         'help': 'Finer cubes look smoother but cost more. A launch command (cubedetail-N) also sets it for one run.'},
        {'type': 'header', 'text': 'CACHE'},
        {'type': 'toggle', 'key': 'cache_terrain', 'label': 'Cache terrain generation', 'tick': lambda: CACHE.tick('terrain'),
         'help': lambda: CACHE.text('terrain')},
        {'type': 'toggle', 'key': 'cache_meshes', 'label': 'Cache meshes', 'tick': lambda: CACHE.tick('mesh'),
         'help': lambda: CACHE.text('mesh')},
        {'type': 'button', 'label': 'Delete cache', 'text': 'DELETE CACHE', 'action': 'clear_cache',
         'help': 'Removes every cached file from the voidcraft_cache folder. They are rebuilt as you play.'},
    ]),
    ("SCANS", [
        {'type': 'scans'},                      # drawn and clicked by SettingsPanel: the list of scanned objects (.cells)
    ]),
    ("CONTROLS", [
        {'type': 'header', 'text': 'KEYS'},
        {'type': 'info', 'b': ('W A S D', 'Move')},
        {'type': 'info', 'b': ('Space', 'Jump')},
        {'type': 'info', 'b': ('Mouse', 'Look around')},
        {'type': 'info', 'b': ('R', 'Reset menu (damage / spawn)')},
        {'type': 'info', 'b': ('V', 'Fly on / off (Space up, Ctrl down)')},
        {'type': 'info', 'b': ('X', 'Grid lines')},
        {'type': 'info', 'b': ('Left click (hold)', f'Aim an arc, let go to throw TNT (up to {TNT_MAX})')},
        {'type': 'info', 'b': ('Right click', 'Detonate all placed TNT')},
        {'type': 'info', 'b': ('Tab', 'Settings menu (SCANS tab: place a scanned object)')},
        {'type': 'info', 'b': ('Esc', 'Quit the game')},
    ]),
]
GFX_TAB = 1   # index of the GRAPHICS tab in SETTINGS_TABS
SCANS_TAB = [n for n, _r in SETTINGS_TABS].index('SCANS')
SCAN_TOP, SCAN_ROW_H, SCAN_VIS = 176, 58, 9   # the scan list: first row's y in the panel, row pitch, rows shown at once





# ---- disk cache (the engine's: terrain generation and chunk meshes) ------------------------------------------------------
# The caching lives in engineui.py's engine (CACHE: DiskCache). This game only switches it on and off, shows it in the Tab menu
# (GRAPHICS tab) and deletes it. An older engine without it gets a stand-in that says so.
class _NoCache:
    terrain_on = mesh_on = False

    def tick(self, kind):
        return False

    def text(self, kind):
        return 'This engineui.py has no cache. Use the updated engineui.py.'

    def refresh(self):
        pass

    def clear(self):
        pass


CACHE = getattr(_E, 'DISK_CACHE', None) or _NoCache()


CUBE_DETAIL_DEFAULT = 5   # cubes per block along each side. Whole numbers only: the ground grid and the chunk size (15 blocks) must divide into whole cubes, so 4.5 is not possible (the engine would cut it to 4)


def default_settings():
    return {'cache_chunks': True, 'render_distance': 8, 'fov': FOV, 'show_hud': False,
            'fps_limit': TARGET_FPS, 'debris_yield': 1.0, 'debris_size': 1.0, 'dirt_cubes': CUBE_DETAIL_DEFAULT, 'cube_unlock': False, 'cache_terrain': True, 'cache_meshes': True,
            'dirt_cube_size': 1.0, 'cube_sleep_time': 0.3, 'debris_on': True, 'debris_amount': DEBRIS_AMOUNT,
            'struct_res': STRUCT_RES, 'quality': 3, 'lod': 0, 'shadow_quality': 1, 'shadow_update': 3, 'render_scale': 100, 'vsync': False, 'fullscreen': False, 'gpu_physics': True, 'wobble': WOBBLE_PCT, 'strength': STRENGTH_PCT,
            'bloom': 70, 'contrast': 103, 'sharpness': 45, 'shake': 200}   # bloom, contrast and shake have no menu slider now: change them here

class TabMenu:
    """Cursor state for the settings panel: which tab, and which row in it."""
    def __init__(self, settings):
        self.settings = settings
        self.open = False
        self.tab = 0
        self.collapsed = {(t, r['text']) for t, (_n, rows) in enumerate(SETTINGS_TABS) for r in rows if r['type'] == 'header' and _n != 'CONTROLS'}   # every section starts folded shut
        self.row = self.selectable()[0]
        self.drag = None     # row index of a slider being dragged with the mouse
        self.scan_sel = 0    # SCANS tab: the highlighted scan and the first one shown
        self.scan_scroll = 0

    @property
    def on_scans(self):
        return self.tab == SCANS_TAB

    def scan_fit(self, n):
        """Keep the highlighted scan inside the list and inside the visible rows."""
        self.scan_sel = max(0, min(n - 1, self.scan_sel)) if n else 0
        self.scan_scroll = max(0, min(max(0, n - SCAN_VIS), self.scan_scroll))
        if self.scan_sel < self.scan_scroll:
            self.scan_scroll = self.scan_sel
        elif self.scan_sel >= self.scan_scroll + SCAN_VIS:
            self.scan_scroll = self.scan_sel - SCAN_VIS + 1

    def scan_move(self, step):
        self.scan_sel += step
        self.scan_fit(len(SCAN_LIB.items))

    def scan_wheel(self, step):
        """Scroll the list; the highlight stays on screen by moving with it."""
        n = len(SCAN_LIB.items)
        self.scan_scroll = max(0, min(max(0, n - SCAN_VIS), self.scan_scroll + step))
        self.scan_sel = max(self.scan_scroll, min(self.scan_sel, self.scan_scroll + SCAN_VIS - 1, max(0, n - 1)))

    def rows(self):
        """Rows of the current tab that are shown now. A row with 'when' only shows while that setting is on."""
        out, folded = [], False
        for r in SETTINGS_TABS[self.tab][1]:
            if r.get('when') is not None and not self.settings[r['when']]:
                continue
            if r['type'] == 'header':
                folded = (self.tab, r['text']) in self.collapsed
                out.append(r)
            elif not folded:                       # the rows under a folded heading are hidden
                out.append(r)
        return out

    def toggle_section(self, i, fold=None):
        """Fold or unfold the section whose heading is row i (fold=None flips it)."""
        r = self.rows()[i]
        if r['type'] != 'header':
            return
        key = (self.tab, r['text'])
        shut = key in self.collapsed if fold is None else not fold
        if shut:
            self.collapsed.discard(key)
        else:
            self.collapsed.add(key)
        self.row = i

    def selectable(self):
        return [i for i, r in enumerate(self.rows())
                if r['type'] in ('header', 'slider', 'toggle', 'button') and not r.get('locked')]

    def move(self, step):
        idx = self.selectable()
        if idx:
            pos = idx.index(self.row) if self.row in idx else 0
            self.row = idx[(pos + step) % len(idx)]

    def switch_tab(self, step):
        self.tab = (self.tab + step) % len(SETTINGS_TABS)
        idx = self.selectable()
        self.row = idx[0] if idx else 0

    def adjust(self, step):
        r = self.rows()[self.row]
        if r.get('locked'):
            return
        if r['type'] == 'header':
            self.toggle_section(self.row, fold=(step < 0))     # right opens the section, left closes it
        elif r['type'] == 'slider':
            v = round(self.settings[r['key']] + step * r['step'], 4)
            self.settings[r['key']] = max(r['min'], min(r['max'], v))
        elif r['type'] == 'toggle':
            self.settings[r['key']] = not self.settings[r['key']]
            self.fix_row()

    def fix_row(self):
        """Move the cursor off a row that has just been hidden."""
        idx = self.selectable()
        if idx and self.row not in idx:
            self.row = idx[0]

    def activate(self):
        r = self.rows()[self.row]
        if r['type'] == 'button':
            self.action = r['action']
        elif r['type'] == 'header':
            self.toggle_section(self.row)
        else:
            self.adjust(1)

    def take_action(self):
        a = getattr(self, 'action', None)
        self.action = None
        return a

    # Mouse actions. Each one moves the cursor to what was clicked, then applies the change.
    def set_tab(self, i):
        self.tab = i
        idx = self.selectable()
        self.row = idx[0] if idx else 0

    def select_row(self, i):
        self.row = i

    def set_toggle(self, i, on):
        r = self.rows()[i]
        if r.get('locked') or r['type'] != 'toggle':
            return
        self.row = i
        self.settings[r['key']] = on

    def set_slider_fraction(self, i, frac):
        """Set a slider from a 0..1 position along its track, snapped to its step."""
        self.row = i
        r = self.rows()[i]
        if r.get('locked'):
            return
        steps = round(frac * (r['max'] - r['min']) / r['step'])
        v = round(r['min'] + steps * r['step'], 4)
        self.settings[r['key']] = max(r['min'], min(r['max'], v))


INFO_HEAD, INFO_ROW_H, KEY_COL_W = 34, 44, 230   # the key table: header row height, height of each key row, width of the KEY column
PANEL_W, PANEL_H = 1000, 790   # settings panel (Tab); tall enough for every tab

# Settings panel colours.
UI_BG      = (16, 18, 26, 245)
UI_BORDER  = (58, 66, 86, 255)
UI_TEXT    = (226, 230, 238)
UI_DIM     = (132, 140, 158)
UI_ACCENT  = (70, 190, 230)
UI_TRACK   = (40, 46, 60)
UI_SEG_OFF = (32, 37, 50)
UI_SEG_ON  = (46, 128, 166)
UI_ROW_SEL = (36, 42, 58)
UI_LINE    = (48, 56, 74)

def _mono(size):
    return pygame.font.SysFont("consolas,couriernew,monospace", size)

class Overlay:
    """A translucent texture drawn over the 3D view at a fixed screen rectangle.
    Subclasses draw into self.surf in _render(); the texture is re-uploaded on every draw."""
    def __init__(self, w, h, x, y):
        self.w, self.h = w, h
        self.font_big = pygame.font.Font(None, 60)
        self.font = pygame.font.Font(None, 42)
        self.font_small = pygame.font.Font(None, 30)

        vs = _compile_shader(_OVL_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_OVL_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_tex = gl.glGetUniformLocation(self.program, "uTex")

        # Quad covering the rectangle (x, y is its top-left in pixels), in normalised device coordinates.
        x0, x1 = x / _E.WIDTH * 2 - 1, (x + w) / _E.WIDTH * 2 - 1
        y0, y1 = 1 - (y + h) / _E.HEIGHT * 2, 1 - y / _E.HEIGHT * 2
        quad = np.array([
            x0, y0, 0.0, 0.0,
            x1, y0, 1.0, 0.0,
            x1, y1, 1.0, 1.0,
            x0, y1, 0.0, 1.0,
        ], dtype=np.float32)
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
        self.surf = pygame.Surface((w, h), pygame.SRCALPHA)

    def _render(self, *args):
        raise NotImplementedError

    def draw(self, *args):
        self._render(*args)
        data = pygame.image.tostring(self.surf, "RGBA", True)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        gl.glUniform1i(self.u_tex, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)

class ResetPanel(Overlay):
    """The R menu: a small centred panel with two buttons, reset all damage and put the player back at spawn."""
    W, H = 460, 270
    def __init__(self):
        super().__init__(self.W, self.H, (_E.WIDTH - self.W) // 2, (_E.HEIGHT - self.H) // 2)
        self.f_title = _mono(30)
        self.f_btn = _mono(22)
        self.f_help = _mono(16)
        self.x0 = (_E.WIDTH - self.W) // 2
        self.y0 = (_E.HEIGHT - self.H) // 2
        self.buttons = [('damage', 'RESET ALL DAMAGE', pygame.Rect(30, 80, self.W - 60, 56)),
                        ('spawn',  'RESET POSITION TO SPAWN', pygame.Rect(30, 150, self.W - 60, 56))]

    def mouse_down(self, screen_pos):
        """'damage', 'spawn', or 'close' (a click outside the panel); None for a click on empty panel."""
        p = (screen_pos[0] - self.x0, screen_pos[1] - self.y0)
        for action, _, rect in self.buttons:
            if rect.collidepoint(p):
                return action
        return None if pygame.Rect(0, 0, self.W, self.H).collidepoint(p) else 'close'

    def _render(self, screen_pos):
        s = self.surf
        s.fill((0, 0, 0, 0))
        pygame.draw.rect(s, UI_BG, (0, 0, self.w, self.h), border_radius=12)
        pygame.draw.rect(s, UI_BORDER, (0, 0, self.w, self.h), 2, border_radius=12)
        title = self.f_title.render("RESET", True, UI_TEXT)
        s.blit(title, ((self.w - title.get_width()) // 2, 24))
        p = (screen_pos[0] - self.x0, screen_pos[1] - self.y0)
        for _, label, rect in self.buttons:
            hot = rect.collidepoint(p)
            pygame.draw.rect(s, UI_SEG_ON if hot else UI_ROW_SEL, rect, border_radius=8)
            pygame.draw.rect(s, UI_ACCENT if hot else UI_LINE, rect, 2, border_radius=8)
            t = self.f_btn.render(label, True, UI_TEXT)
            s.blit(t, t.get_rect(center=rect.center))
        hint = self.f_help.render("R or Esc to close", True, UI_DIM)
        s.blit(hint, ((self.w - hint.get_width()) // 2, self.h - 40))

class SettingsPanel(Overlay):
    """The Tab menu: tabbed settings with sliders and toggles, centred on screen."""
    def __init__(self):
        super().__init__(PANEL_W, PANEL_H, (_E.WIDTH - PANEL_W) // 2, (_E.HEIGHT - PANEL_H) // 2)
        self.f_title = _mono(34)
        self.f_tab = _mono(22)
        self.f_head = _mono(20)
        self.f_label = _mono(22)
        self.f_value = _mono(20)
        self.f_help = _mono(16)
        self.f_exit = _mono(20)
        self.exit_rect = pygame.Rect(self.w - 24 - 170, 16, 170, 40)   # EXIT GAME, above the tab bar on the right
        self.x0 = (_E.WIDTH - PANEL_W) // 2     # panel's top-left on screen, for mouse coordinates
        self.y0 = (_E.HEIGHT - PANEL_H) // 2

    def layout(self, menu):
        """Clickable regions as (kind, row or tab index, rect), in panel pixels.
        Positions match _render, so what you see is what you click."""
        regions = []
        tab_y = 74
        x = 24
        for i, (name, _) in enumerate(SETTINGS_TABS):
            tw = self.f_tab.size(name)[0] + 36
            regions.append(('tab', i, pygame.Rect(x, tab_y, tw, 44)))
            x += tw + 8
        if menu.on_scans:
            n = len(SCAN_LIB.items)
            for k in range(SCAN_VIS):
                i = menu.scan_scroll + k
                if i >= n:
                    break
                regions.append(('scan', i, pygame.Rect(16, SCAN_TOP + k * SCAN_ROW_H, self.w - 32 - 16, SCAN_ROW_H - 6)))
            return regions
        y = tab_y + 66
        n_info = 0
        n_keys = sum(1 for r in menu.rows() if r['type'] == 'info')
        for i, row in enumerate(menu.rows()):
            if row['type'] == 'header':
                regions.append(('head', i, pygame.Rect(16, y - 4, self.w - 32, 32)))
                y += 34
                continue
            if row['type'] == 'info':
                if n_info == 0:
                    y += INFO_HEAD + INFO_ROW_H * n_keys + 8     # the key table is drawn once: it takes this much height
                n_info += 1
                continue
            if row.get('locked'):          # locked rows have no click targets at all
                y += 62
                continue
            regions.append(('row', i, pygame.Rect(16, y - 8, self.w - 32, 58)))
            if row['type'] == 'slider':
                regions.append(('slider', i, pygame.Rect(360, y + 10, 320, 8).inflate(0, 24)))
            else:
                if row['type'] == 'button':
                    regions.append(('button', i, pygame.Rect(360, y, 220, 34)))
                else:
                    regions.append(('off', i, pygame.Rect(360, y, 108, 34)))
                    regions.append(('on', i, pygame.Rect(476, y, 108, 34)))
            y += 62
        return regions

    def _local(self, screen_pos):
        return screen_pos[0] - self.x0, screen_pos[1] - self.y0

    @staticmethod
    def _fraction(rect, x):
        return max(0.0, min(1.0, (x - rect.x) / rect.w))

    def mouse_down(self, menu, screen_pos):
        p = self._local(screen_pos)
        if self.exit_rect.collidepoint(p):
            return 'exit'
        regions = self.layout(menu)
        for kind, idx, rect in regions:
            if kind == 'head' and rect.collidepoint(p):
                menu.toggle_section(idx)
                return
        for kind, idx, rect in regions:
            if kind == 'button' and rect.collidepoint(p):
                menu.select_row(idx)
                return menu.rows()[idx].get('action', 'reset_damage')
        # Controls first, so a click on a toggle or slider isn't taken by the row behind it.
        for kind, idx, rect in regions:
            if kind in ('off', 'on') and rect.collidepoint(p):
                menu.set_toggle(idx, kind == 'on')
                return
            if kind == 'slider' and rect.collidepoint(p):
                menu.drag = idx
                menu.set_slider_fraction(idx, self._fraction(rect, p[0]))
                return
        for kind, idx, rect in regions:
            if kind == 'scan' and rect.collidepoint(p):
                menu.scan_sel = idx
                return ('place', idx)
        for kind, idx, rect in regions:
            if kind == 'tab' and rect.collidepoint(p):
                menu.set_tab(idx)
                return
        for kind, idx, rect in regions:
            if kind == 'row' and rect.collidepoint(p):
                menu.select_row(idx)
                return

    def mouse_move(self, menu, screen_pos):
        """SCANS tab: the entry under the pointer becomes the highlighted one."""
        p = self._local(screen_pos)
        for kind, idx, rect in self.layout(menu):
            if kind == 'scan' and rect.collidepoint(p):
                menu.scan_sel = idx
                return

    def mouse_drag(self, menu, screen_pos):
        p = self._local(screen_pos)
        for kind, idx, rect in self.layout(menu):
            if kind == 'slider' and idx == menu.drag:
                menu.set_slider_fraction(idx, self._fraction(rect, p[0]))
                return

    def _slider(self, x, y, row, value, locked=False):
        s = self.surf
        track = pygame.Rect(x, y + 8, 320, 8)
        frac = (value - row['min']) / (row['max'] - row['min'])
        fill_col = (84, 86, 98) if locked else UI_ACCENT
        knob_col = (110, 112, 124) if locked else UI_TEXT
        pygame.draw.rect(s, UI_TRACK, track, border_radius=4)
        filled = track.copy()
        filled.w = max(8, int(track.w * frac))
        pygame.draw.rect(s, fill_col, filled, border_radius=4)
        pygame.draw.circle(s, knob_col, (track.x + int(track.w * frac), track.centery), 10)
        value_txt = self.f_value.render(row['fmt'](value), True, UI_DIM if locked else UI_TEXT)
        s.blit(value_txt, (track.right + 22, y + 2))

    def _toggle(self, x, y, on):
        s = self.surf
        for i, (text, active) in enumerate((("OFF", not on), ("ON", on))):
            r = pygame.Rect(x + i * 116, y, 108, 34)
            pygame.draw.rect(s, UI_SEG_ON if active else UI_SEG_OFF, r, border_radius=5)
            t = self.f_value.render(text, True, UI_TEXT if active else UI_DIM)
            s.blit(t, t.get_rect(center=r.center))

    def _key_table(self, keys, x, y):
        """The controls as a grid: a KEY column and an ACTION column, one ruled row per key."""
        s = self.surf
        w = self.w - 2 * x
        top = y + INFO_HEAD
        bottom = top + INFO_ROW_H * len(keys)
        pygame.draw.rect(s, UI_ROW_SEL, (x, y, w, INFO_HEAD), border_top_left_radius=6, border_top_right_radius=6)
        s.blit(self.f_help.render("KEY", True, UI_DIM), (x + 16, y + (INFO_HEAD - 16) // 2))
        s.blit(self.f_help.render("ACTION", True, UI_DIM), (x + KEY_COL_W + 16, y + (INFO_HEAD - 16) // 2))
        for i, r in enumerate(keys):
            ry = top + i * INFO_ROW_H
            if i % 2 == 0:
                pygame.draw.rect(s, UI_SEG_OFF, (x, ry, w, INFO_ROW_H))
            k, d = r['b']
            kt = self.f_value.render(k, True, (255, 220, 120))
            dt_ = self.f_value.render(d, True, UI_TEXT)
            s.blit(kt, (x + 16, ry + (INFO_ROW_H - kt.get_height()) // 2))
            s.blit(dt_, (x + KEY_COL_W + 16, ry + (INFO_ROW_H - dt_.get_height()) // 2))
        for i in range(len(keys) + 1):                                    # grid lines: between the rows, and between the columns
            ly = top + i * INFO_ROW_H
            pygame.draw.line(s, UI_LINE, (x, ly), (x + w - 1, ly), 1)
        pygame.draw.line(s, UI_LINE, (x + KEY_COL_W, y), (x + KEY_COL_W, bottom), 1)
        pygame.draw.rect(s, UI_BORDER, (x, y, w, bottom - y + 1), 1, border_radius=6)

    def _render_scans(self, menu):
        """The SCANS tab: every .cells file the game can find, newest first. The list is live (see ScanLibrary.poll)."""
        s = self.surf
        items = SCAN_LIB.items
        n = len(items)
        menu.scan_fit(n)
        sub = self.f_help.render("%d scan%s found   live: new scans from the phone page appear here by themselves" % (n, '' if n == 1 else 's'), True, UI_DIM)
        s.blit(sub, (32, SCAN_TOP - 30))
        now = time.time()
        if not n:
            s.blit(self.f_label.render("No .cells files found yet.", True, UI_TEXT), (32, SCAN_TOP + 10))
            s.blit(self.f_value.render("Export a scan from the scanner page, or drop a .cells file in one of these folders:", True, UI_DIM), (32, SCAN_TOP + 44))
            y = SCAN_TOP + 78
            for d in SCAN_DIRS:
                shown = d if len(d) < 90 else '...' + d[-87:]
                s.blit(self.f_value.render(shown, True, (255, 220, 120)), (48, y))
                y += 26
        for k in range(SCAN_VIS):
            i = menu.scan_scroll + k
            if i >= n:
                break
            it = items[i]
            r = pygame.Rect(16, SCAN_TOP + k * SCAN_ROW_H, self.w - 32 - 16, SCAN_ROW_H - 6)
            if i == menu.scan_sel:
                pygame.draw.rect(s, UI_SEG_ON if it['ok'] else (120, 48, 52), r, border_radius=8)
                pygame.draw.rect(s, UI_ACCENT, r, 2, border_radius=8)
            else:
                pygame.draw.rect(s, UI_ROW_SEL, r, border_radius=8)
            name = it['name']
            while len(name) > 4 and self.f_label.size(name)[0] > r.w - 160:
                name = name[:-4].rstrip() + '...'
            s.blit(self.f_label.render(name, True, UI_TEXT if it['ok'] else (255, 150, 140)), (r.x + 14, r.y + 5))
            if it['ok']:
                info = "%d cells   %.1f x %.1f x %.1f blocks   %s   %s" % (it['n'], it['w'], it['h'], it['dp'], _age_text(now - it['mtime']), it['folder'])
            else:
                info = "cannot be read: %s" % it['err']
            s.blit(self.f_help.render(info[:96], True, UI_TEXT if i == menu.scan_sel else UI_DIM), (r.x + 14, r.y + 30))
            if it['path'] not in SCAN_LIB.opened:                       # arrived after the menu was opened
                tag = self.f_help.render(" NEW ", True, (20, 24, 32))
                tr = tag.get_rect(topright=(r.right - 12, r.y + 8))
                pygame.draw.rect(s, (255, 220, 120), tr.inflate(4, 2), border_radius=4)
                s.blit(tag, tr)
        if n > SCAN_VIS:                                                 # scroll bar
            track = pygame.Rect(self.w - 26, SCAN_TOP, 6, SCAN_VIS * SCAN_ROW_H - 6)
            pygame.draw.rect(s, UI_TRACK, track, border_radius=3)
            kh = max(20, int(track.h * SCAN_VIS / n))
            ky = track.y + int((track.h - kh) * menu.scan_scroll / max(1, n - SCAN_VIS))
            pygame.draw.rect(s, UI_ACCENT, (track.x, ky, track.w, kh), border_radius=3)

    def _render(self, menu):
        s = self.surf
        s.fill((0, 0, 0, 0))
        pygame.draw.rect(s, UI_BG, (0, 0, self.w, self.h), border_radius=12)
        pygame.draw.rect(s, UI_BORDER, (0, 0, self.w, self.h), 2, border_radius=12)

        title = self.f_title.render("SETTINGS", True, UI_TEXT)
        s.blit(title, ((self.w - title.get_width()) // 2, 16))

        # Tab bar: the selected tab gets a lighter background and an accent underline.
        tab_y = 74
        x = 24
        for i, (name, _) in enumerate(SETTINGS_TABS):
            selected = i == menu.tab
            label = self.f_tab.render(name, True, UI_TEXT if selected else UI_DIM)
            tw = label.get_width() + 36
            if selected:
                pygame.draw.rect(s, UI_ROW_SEL, (x, tab_y, tw, 40), border_radius=6)
                pygame.draw.rect(s, UI_ACCENT, (x, tab_y + 36, tw, 4))
            s.blit(label, (x + 18, tab_y + 8))
            x += tw + 8
        pygame.draw.line(s, UI_LINE, (24, tab_y + 44), (self.w - 24, tab_y + 44), 1)
        pygame.draw.rect(s, (120, 48, 52), self.exit_rect, border_radius=6)
        exit_txt = self.f_exit.render("EXIT GAME", True, UI_TEXT)
        s.blit(exit_txt, exit_txt.get_rect(center=self.exit_rect.center))

        if menu.on_scans:
            self._render_scans(menu)
            foot = self.f_help.render("[ ] switch tab   Up/Down select   Enter or click: place it in front of you   wheel scroll",
                                      True, UI_DIM)
            s.blit(foot, (24, self.h - 34))
            return

        # Rows for the current tab.
        y = tab_y + 66
        n_info = 0
        for i, row in enumerate(menu.rows()):
            if row['type'] == 'header':
                head = self.f_head.render(row['text'], True, UI_ACCENT)
                shut = (menu.tab, row['text']) in menu.collapsed
                if i == menu.row:
                    pygame.draw.rect(s, UI_ROW_SEL, (16, y - 4, self.w - 32, 32), border_radius=6)
                cy = y + head.get_height() // 2
                if shut:                                          # a triangle: points right when folded, down when open
                    pygame.draw.polygon(s, UI_ACCENT, [(28, cy - 6), (28, cy + 6), (37, cy)])
                else:
                    pygame.draw.polygon(s, UI_ACCENT, [(27, cy - 4), (39, cy - 4), (33, cy + 6)])
                s.blit(head, (48, y))
                ly = cy
                pygame.draw.line(s, UI_LINE, (48 + head.get_width() + 14, ly), (row.get('width', self.w - 24), ly), 1)
                y += 34
                continue

            if row['type'] == 'info':
                if n_info == 0:                                   # the whole key list is one table, drawn at its first row
                    keys = [r for r in menu.rows() if r['type'] == 'info']
                    self._key_table(keys, 24, y)
                    y += INFO_HEAD + INFO_ROW_H * len(keys) + 8
                n_info += 1
                continue
            if i == menu.row:
                pygame.draw.rect(s, UI_ROW_SEL, (16, y - 8, self.w - 32, 58), border_radius=6)
            locked = bool(row.get('locked'))
            s.blit(self.f_label.render(row['label'], True, UI_DIM if locked else UI_TEXT), (32, y))
            help_txt = row['help']() if callable(row['help']) else row['help']
            while len(help_txt) > 4 and self.f_help.size(help_txt)[0] > self.w - 64:      # never run off the panel's edge
                help_txt = help_txt[:-4].rstrip() + "..."
            s.blit(self.f_help.render(help_txt, True, UI_DIM), (32, y + 34))
            if row['type'] == 'slider':
                self._slider(360, y + 2, row, menu.settings[row['key']], locked)
            elif row['type'] == 'button':
                btn = pygame.Rect(360, y, 220, 34)
                pygame.draw.rect(s, UI_SEG_ON, btn, border_radius=5)
                t = self.f_value.render(row.get('text', 'RESET'), True, UI_TEXT)
                s.blit(t, t.get_rect(center=btn.center))
            else:
                self._toggle(360, y, menu.settings[row['key']])
            if row.get('tick') is not None:                            # a tick box beside the toggle: ticked when something is cached
                ok = bool(row['tick']())
                box = pygame.Rect(612, y + 2, 30, 30)
                pygame.draw.rect(s, (70, 200, 120) if ok else UI_LINE, box, 2, border_radius=4)
                if ok:
                    pygame.draw.lines(s, (70, 200, 120), False, [(box.x + 6, box.y + 16), (box.x + 12, box.y + 23), (box.x + 24, box.y + 7)], 4)
                lab = self.f_help.render('cached' if ok else 'not cached', True, (70, 200, 120) if ok else UI_DIM)
                s.blit(lab, (box.right + 10, y + 8))
            y += 62

        foot = self.f_help.render("[ ] switch tab   Up/Down select   Left/Right change   Enter toggle / fold",
                                  True, UI_DIM)
        s.blit(foot, (24, self.h - 34))




class _CachedOverlay(Overlay):
    """An overlay that redraws its texture only when its state changes."""
    def draw(self, state):
        if state != getattr(self, '_state', None):
            self._state = state
            self.surf.fill((0, 0, 0, 0))
            self._render(state)
            self._data = pygame.image.tostring(self.surf, "RGBA", True)
            self._dirty = True
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        if self._dirty:                                  # the texture keeps its pixels, so it is uploaded only after a redraw
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self._data)
            self._dirty = False
        gl.glUniform1i(self.u_tex, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)






def aimed_cell(player):
    """The structure cell under the crosshair as a dict of facts, or None."""
    eye = np.asarray(player.eye, np.float64)
    fwd = np.asarray(player.forward(), np.float64)
    so = STRUCTS.raycast(eye, fwd, 40.0)
    if so is None:
        return None
    o = so[2]
    p = eye + fwd * (so[0] + 0.02)
    ij = np.floor(((p - o.frame_pos) @ o.R - o.corner) / o.cs).astype(np.int64)
    if (ij < 0).any() or (ij >= np.array(o.alive.shape)).any() or not o.alive[tuple(ij)]:
        return None
    c = tuple(int(v) for v in ij)
    m = int(o.mat[c])
    return dict(mat=_MAT_NAMES.get(m, 'Stone'), fixed=isinstance(o, Structure),
                stress=float(o.stress[c]), cracks=float(o.cdmg[c]),
                hard=float(_S_HARD[m]), mass=float(o.cell_mass[c]),
                grounded=bool(o.anch[c]) if isinstance(o, Structure) else None)


def integrity_state(player):
    """Everything the panel shows as a tuple of strings, so it is redrawn only when something changes."""
    worst = 0.0
    for o in STRUCTS.objects():
        worst = max(worst, o.worst_cell_load())
    a = aimed_cell(player)
    lines = [f"{len(STRUCTS.structs)} fixed   {len(STRUCTS.bodies)} loose   worst cell {min(worst, 1.0) * 100:.0f}%"]
    if a is None:
        lines.append("crosshair: no structure")
    else:
        lines += [f"crosshair: {a['mat']}, {'fixed' if a['fixed'] else 'loose piece'}",
                  f"stress {min(a['stress'], 1.5) * 100:.0f}% of limit",
                  f"cracks {a['cracks'] * 100:.0f}%",
                  f"hardness {a['hard']:.1f}   mass {a['mass']:.2f}" + ("" if a['grounded'] is None else f"   grounded {'yes' if a['grounded'] else 'no'}")]
    return tuple(lines)


class Crosshair(_CachedOverlay):
    """Centre-screen aiming mark: four arms with a gap around a centre dot, dark outline under white."""
    SIZE = 64
    def __init__(self):
        super().__init__(self.SIZE, self.SIZE, (_E.WIDTH - self.SIZE) // 2, (_E.HEIGHT - self.SIZE) // 2)

    def _render(self, state):
        c = self.SIZE // 2
        arms = (((c - 22, c), (c - 8, c)), ((c + 8, c), (c + 22, c)),
                ((c, c - 22), (c, c - 8)), ((c, c + 8), (c, c + 22)))
        for col, wd in (((0, 0, 0, 220), 6), ((255, 255, 255, 255), 3)):   # dark outline, then white
            for a, b in arms:
                pygame.draw.line(self.surf, col, a, b, wd)
        pygame.draw.circle(self.surf, (0, 0, 0, 220), (c, c), 5)
        pygame.draw.circle(self.surf, (255, 255, 255, 255), (c, c), 2)


class TargetLabel(_CachedOverlay):
    """The name of the cube an action just landed on, under the crosshair."""
    W, H = 340, 30
    def __init__(self):
        super().__init__(self.W, self.H, (_E.WIDTH - self.W) // 2, _E.HEIGHT // 2 + 40)
        self.font = pygame.font.SysFont(None, 24)

    def _render(self, state):
        text, a = state
        if not text:
            return
        img = self.font.render(text, True, (255, 226, 150))
        sh = self.font.render(text, True, (0, 0, 0))
        img.set_alpha(a)
        sh.set_alpha(int(a * 0.8))
        x = (self.W - img.get_width()) // 2
        self.surf.blit(sh, (x + 1, 8))
        self.surf.blit(img, (x, 7))


class InvHud(_CachedOverlay):
    """Top left of the screen: the latest message (TNT placed, detonated...). Nothing is drawn while there is no message."""
    W, H = 700, 36

    def __init__(self):
        super().__init__(self.W, self.H, 14, 14)
        self.f_small = _mono(16)

    def _render(self, st):
        msg = st[0]
        if not msg:
            return
        s = self.surf
        pygame.draw.rect(s, (*UI_BG[:3], 205), (0, 0, self.W, self.H), border_radius=10)
        pygame.draw.rect(s, UI_BORDER, (0, 0, self.W, self.H), 2, border_radius=10)
        s.blit(self.f_small.render(msg, True, UI_TEXT), (14, 8))


class DebrisTint:
    """The loose cubes recoloured for the G overlay: awake cubes orange (they are what costs frame time), sleeping ones a dim blue.
    Drawn by the same renderer as the real cubes (it only needs .n and .instances())."""
    AWAKE = np.array([1.0, 0.55, 0.12], np.float32)
    ASLEEP = np.array([0.30, 0.42, 0.62], np.float32)

    @property
    def n(self):
        return BODIES.n

    def instances(self):
        out = BODIES.instances()
        if len(out):
            out[:, 3:6] = np.where(BODIES.awake[:, None], self.AWAKE, self.ASLEEP)
        return out


# ═══════════════════════════════════════════════
#  G monitor: a small translucent panel in the top-right corner, drawn over the picture (the game is not resized). The graphs are
#  drawn on the GPU: each frame only 16 floats go up (one column of a small ring texture) and a shader draws the stacked bars.
#  The text is redrawn four times a second, not every frame, so watching the monitor costs the CPU almost nothing. (Measuring the
#  CPU stages has to happen on the CPU; the GPU stages come from GL timer queries.) H switches between compact and detailed.
# ═══════════════════════════════════════════════
MON_N = 240            # frames across each graph
MON_ROWS = 16          # per frame: 10 CPU stages, then 5 GPU stages, one spare
MON_SMOOTH = 0.30      # how much of each new frame is mixed into the smoothed value the graph shows
MON_CPU = [            # (PROF key, label, colour). The last row is whatever the named stages do not add up to.
    ('debris',      'debris physics',    (214, 106, 160)),
    ('structs',     'slabs + collapse',  (110, 200, 130)),
    ('weapon',      'blast fx + tools',  (232, 190,  90)),
    ('terrain',     'terrain draw',      ( 70, 170, 230)),
    ('shadow',      'shadow map',        (150, 120, 230)),
    ('struct_draw', 'damage + structs',  (100, 220, 190)),
    ('debris_draw', 'debris draw',       (235, 125, 110)),
    ('overlay',     'HUD',               (170, 170, 200)),
    ('panel',       'this monitor',      (255, 255, 255)),
    ('other',       'everything else',   (150, 156, 172)),
]
MON_DIM = (178, 186, 204)      # label grey, lighter than the menus use because the panel sits over a changing picture
MON_GPU = ['shadow', 'terrain', 'struct_draw', 'debris_draw', 'overlay']       # the stages that have GPU timer queries

_MON_FRAG = """
#version 330 core
in vec2 vUV;
out vec4 o;
uniform sampler2D uRing;
uniform int uHead;        // the newest column of the ring
uniform int uN;
uniform int uRow0;        // first row of the stack in the ring texture
uniform int uRows;
uniform float uMaxMs;
uniform float uAlpha;
uniform vec3 uCol[10];
uniform vec2 uGrid;       // two guide lines, in ms (-1 = none)
uniform vec2 uPx;         // one pixel in uv units
void main() {
    int c = min(int(vUV.x * float(uN)), uN - 1);
    int col = (uHead + 1 + c) % uN;                  // oldest on the left, newest on the right
    float ms = vUV.y * uMaxMs;
    float pxMs = uPx.y * uMaxMs;
    vec3 rgb = vec3(0.055, 0.060, 0.082);
    float a = uAlpha * 0.8;
    float acc = 0.0;
    for (int i = 0; i < uRows; i++) {
        float v = texelFetch(uRing, ivec2(col, uRow0 + i), 0).r;
        if (ms >= acc && ms < acc + v) { rgb = uCol[i]; a = uAlpha; }
        acc += v;
    }
    if (abs(ms - uGrid.x) < pxMs * 0.5 || abs(ms - uGrid.y) < pxMs * 0.5) { rgb = mix(rgb, vec3(1.0), 0.35); a = uAlpha; }
    if (vUV.x < uPx.x || vUV.x > 1.0 - uPx.x || vUV.y < uPx.y || vUV.y > 1.0 - uPx.y) { rgb = vec3(0.30, 0.34, 0.44); a = 1.0; }
    o = vec4(rgb, a);
}
"""


class PerfOverlay:
    P = 360                # panel width in pixels
    M = 10                 # inner margin
    EDGE = 12              # gap to the corner of the screen
    NICE = (8.0, 16.7, 33.3, 50.0, 100.0, 200.0, 400.0, 800.0)

    def __init__(self):
        self.prog_tex = _link_program(_compile_shader(_OVL_VERT, gl.GL_VERTEX_SHADER), _compile_shader(_OVL_FRAG, gl.GL_FRAGMENT_SHADER))
        self.prog_gr = _link_program(_compile_shader(_OVL_VERT, gl.GL_VERTEX_SHADER), _compile_shader(_MON_FRAG, gl.GL_FRAGMENT_SHADER))
        self.u_tex = gl.glGetUniformLocation(self.prog_tex, "uTex")
        self.u = {n: gl.glGetUniformLocation(self.prog_gr, n) for n in ('uRing', 'uHead', 'uN', 'uRow0', 'uRows', 'uMaxMs', 'uAlpha', 'uCol[0]', 'uGrid', 'uPx')}
        quad = np.array([-1, -1, 0, 0,   1, -1, 1, 0,   1, 1, 1, 1,   -1, 1, 0, 1], np.float32)      # fills whatever viewport is set
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
        self.ring = gl.glGenTextures(1)                                           # the history: one column per frame, one row per stage
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ring)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, MON_N, MON_ROWS, 0, gl.GL_RED, gl.GL_FLOAT, np.zeros((MON_ROWS, MON_N), np.float32))
        for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST),
                     (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
        self.tex = gl.glGenTextures(1)                                            # the text and table, redrawn four times a second
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        self.hist = np.zeros((MON_N, MON_ROWS), np.float32)                       # a CPU copy of the ring, for the averages in the text
        self.head = MON_N - 1
        self.prev = None
        self.pending = []
        self.detail = False                                                       # H: all stages and the debris sub-stages
        self.scale = 33.3
        self.low_since = None
        self.grid = (16.7, -1.0)
        self.drawn = None                                                         # (detail, height) the text texture was last made for
        self.t_text = 0.0
        self.f_title, self.f, self.f_small = _mono(15), _mono(13), _mono(11)
        cols = np.zeros((10, 3), np.float32)
        for i, (_k, _l, c) in enumerate(MON_CPU):
            cols[i] = np.array(c, np.float32) / 255.0
        self.cols_cpu = cols
        gcols = np.zeros((10, 3), np.float32)
        keys = [k for k, _l, _c in MON_CPU]
        for j, k in enumerate(MON_GPU):
            gcols[j] = cols[keys.index(k)]
        self.cols_gpu = gcols

    def sample(self, cur, work_ms, gpu_latest):
        """One frame's numbers: CPU stages from the profiler's running totals, GPU stages from the timer queries."""
        n = len(MON_CPU) - 1
        raw = np.zeros(MON_ROWS, np.float32)
        for i in range(n):
            raw[i] = cur.get(MON_CPU[i][0], 0.0)
        raw[n] = max(0.0, work_ms - float(raw[:n].sum()))
        for j, k in enumerate(MON_GPU):
            raw[10 + j] = gpu_latest.get(k, 0.0)
        self.prev = raw.copy() if self.prev is None else np.ascontiguousarray(self.prev + (raw - self.prev) * MON_SMOOTH, np.float32)
        self.head = (self.head + 1) % MON_N
        self.hist[self.head] = self.prev
        self.pending.append((self.head, self.prev))                                 # uploaded at the start of the next panel draw

    def layout(self):
        """Where everything sits inside the panel (x, y, w, h from its top-left), and the panel's height."""
        M, P = self.M, self.P
        gh = 64 if self.detail else 48
        gw = P - 2 * M - 30
        yA = 130
        yB = yA + gh + 24
        y_tab = yB + gh + 10
        rows = (len(MON_CPU) if self.detail else 5) + 1
        y_subs = y_tab + 16 + 15 * rows + 8
        y_leg = y_subs + (16 + 14 * 8 + 8 if self.detail else 0)
        return dict(A=(M, yA, gw, gh), B=(M, yB, gw, gh), tab=y_tab, rows=rows, subs=y_subs if self.detail else None, leg=y_leg, H=y_leg + 44)

    def _refresh(self, lay, c, fps, frame_ms):
        P, H, M = self.P, lay['H'], self.M
        idx = (self.head - np.arange(30)) % MON_N
        avg = self.hist[idx].mean(axis=0)
        cpu, gpu = float(avg[:10].sum()), float(avg[10:15].sum())
        peak = max(float(self.hist[:, :10].sum(axis=1).max()), float(self.hist[:, 10:15].sum(axis=1).max()))
        want = next((s for s in self.NICE if s >= peak * 1.05), self.NICE[-1])         # a graph scale that only grows quickly and shrinks slowly
        now = time.perf_counter()
        if want > self.scale:
            self.scale, self.low_since = want, None
        elif want < self.scale:
            if self.low_since is None:
                self.low_since = now
            elif now - self.low_since > 2.0:
                self.scale, self.low_since = want, None
        else:
            self.low_since = None
        g = [v for v in (16.7, 33.3, 100.0, 200.0, 400.0) if v < self.scale * 0.98][-2:]
        self.grid = (g[0] if g else -1.0, g[1] if len(g) > 1 else -1.0)

        surf = pygame.Surface((P, H), pygame.SRCALPHA)
        pygame.draw.rect(surf, (12, 14, 22, 206), (0, 0, P, H), border_radius=8)             # see-through: the game shows behind it
        pygame.draw.rect(surf, (58, 66, 86, 230), (0, 0, P, H), 1, border_radius=8)
        f, fs = self.f, self.f_small
        def put(font, x, y, text, col=UI_TEXT, right=False):
            im = font.render(text, True, col)
            surf.blit(im, (x - im.get_width() if right else x, y))
        put(self.f_title, M, 5, 'MONITOR   G close   H ' + ('compact' if self.detail else 'detail'), UI_ACCENT)
        ch = c['dmg_chunks']
        put(f, M, 26, f"{ch} damaged chunk{'' if ch == 1 else 's'}   ledges: " + (f"{c['ledges']} (next {c['next_t']:.1f} s)" if c['ledges'] else 'none'),
            (240, 120, 90) if c['ledges'] else UI_TEXT)
        put(f, M, 42, f"slabs {c['slabs_awake']}/{c['slabs']} awake   structures {c['structs']}")
        put(f, M, 58, f"debris cubes {c['cubes_awake']} awake / {c['cubes']}")
        put(f, M, 78, f'{frame_ms:.0f} ms ({fps:.0f} fps)   CPU {cpu:.1f}   GPU {gpu:.1f} ms')
        if cpu >= 1.5 * gpu and cpu > 0.5:
            put(f, M, 94, 'held back by the CPU (GPU waiting)', (240, 170, 90))
        elif gpu >= 1.5 * cpu and gpu > 0.5:
            put(f, M, 94, 'held back by the GPU', (240, 120, 90))
        else:
            put(f, M, 94, 'CPU and GPU about even', MON_DIM)
        for key, title in (('A', f'CPU ms per frame (stacked), 0-{self.scale:g}'), ('B', 'GPU ms per frame (stacked), same scale')):
            gx, gy, gw, gh = lay[key]
            put(fs, gx, gy - 14, title, MON_DIM)
            for gv in self.grid:                                                  # guide-line labels sit just right of the graph
                if gv > 0:
                    put(fs, gx + gw + 3, int(gy + gh * (1.0 - gv / self.scale)) - 6, f'{gv:g}', MON_DIM)
        y = lay['tab']
        put(fs, M + 18, y, 'stage', MON_DIM)
        put(fs, P - 80, y, 'CPU ms', MON_DIM, right=True)
        put(fs, P - M, y, 'GPU ms', MON_DIM, right=True)
        order = list(range(len(MON_CPU))) if self.detail else sorted(range(len(MON_CPU)), key=lambda i: -avg[i])[:5]   # compact: just the five slowest
        for r, i in enumerate(order):
            k, label, col = MON_CPU[i]
            yy = y + 16 + 15 * r
            pygame.draw.rect(surf, col, (M, yy + 3, 10, 10))
            put(f, M + 18, yy, label)
            put(f, P - 80, yy, f'{avg[i]:5.1f}', right=True)
            if k in MON_GPU:
                put(f, P - M, yy, f'{avg[10 + MON_GPU.index(k)]:5.1f}', right=True)
        yy = y + 16 + 15 * len(order)
        put(f, M + 18, yy, 'total', UI_ACCENT)
        put(f, P - 80, yy, f'{cpu:5.1f}', UI_ACCENT, right=True)
        put(f, P - M, yy, f'{gpu:5.1f}', UI_ACCENT, right=True)
        if lay['subs'] is not None:
            ys = lay['subs']
            put(fs, M, ys, 'inside debris physics (ms, they overlap)', MON_DIM)
            labels = getattr(_E, 'PROF_LABEL', {})
            for i, k in enumerate(getattr(_E, 'DEBRIS_SUB', [])[:8]):
                dq = PROF.hist.get(k)
                v = float(np.mean(list(dq)[-30:])) if dq else 0.0
                put(fs, M + 12, ys + 16 + 14 * i, labels.get(k, k))
                put(fs, P - M, ys + 16 + 14 * i, f'{v:5.1f}', right=True)
        yl = lay['leg']
        pygame.draw.rect(surf, (255, 140, 30), (M, yl + 2, 10, 10))
        put(fs, M + 15, yl + 1, 'awake', MON_DIM)
        pygame.draw.rect(surf, (77, 107, 158), (M + 70, yl + 2, 10, 10))
        put(fs, M + 85, yl + 1, 'asleep', MON_DIM)
        x0, w = M, P - 2 * M                                                       # the damage scale, the same colours the overlay draws (built once)
        if getattr(self, '_scale_w', 0) != w:
            t = np.linspace(0.0, 1.0, w)
            rgb = np.stack([np.interp(t, [0, .5, 1], [.12, .95, .95]), np.interp(t, [0, .5, 1], [.75, .85, .12]), np.interp(t, [0, .5, 1], [.30, .12, .08])], axis=1)
            self._scale = pygame.Surface((w, 8))
            pygame.surfarray.blit_array(self._scale, np.repeat((rgb * 255).astype(np.uint8)[:, None, :], 8, axis=1))
            self._scale_w = w
        surf.blit(self._scale, (x0, yl + 18))
        put(fs, x0, yl + 28, 'sound', MON_DIM)
        put(fs, x0 + w // 2 - 20, yl + 28, 'cracked', MON_DIM)
        put(fs, x0 + w, yl + 28, 'give way', MON_DIM, right=True)
        data = pygame.image.tostring(surf, 'RGBA', True)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, P, H, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
        self.drawn = (self.detail, H)
        self.t_text = now

    def draw(self, W, Hs, counts_fn, fps, frame_ms):
        """Paint the panel in the top-right corner of the W x Hs picture (call after everything else is drawn)."""
        lay = self.layout()
        self.lay = lay
        P, H = self.P, lay['H']
        px, py = W - P - self.EDGE, self.EDGE
        if self.drawn != (self.detail, H) or time.perf_counter() - self.t_text >= 0.25:
            self._refresh(lay, counts_fn(), fps, frame_ms)
        if self.pending:                                                            # the new history column goes up before anything reads it
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.ring)
            for col, vals in self.pending:
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, col, 0, 1, MON_ROWS, gl.GL_RED, gl.GL_FLOAT, vals)
            self.pending = []
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glViewport(px, Hs - py - H, P, H)
        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.prog_tex)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glUniform1i(self.u_tex, 0)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glUseProgram(self.prog_gr)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ring)
        gl.glUniform1i(self.u['uRing'], 0)
        gl.glUniform1i(self.u['uHead'], self.head)
        gl.glUniform1i(self.u['uN'], MON_N)
        gl.glUniform1f(self.u['uMaxMs'], float(self.scale))
        gl.glUniform1f(self.u['uAlpha'], 0.92)
        gl.glUniform2f(self.u['uGrid'], float(self.grid[0]), float(self.grid[1]))
        for key, row0, rows, cols in (('A', 0, 10, self.cols_cpu), ('B', 10, len(MON_GPU), self.cols_gpu)):
            gx, gy, gw, gh = lay[key]
            gl.glViewport(px + gx, Hs - (py + gy + gh), gw, gh)
            gl.glUniform1i(self.u['uRow0'], row0)
            gl.glUniform1i(self.u['uRows'], rows)
            gl.glUniform3fv(self.u['uCol[0]'], 10, cols)
            gl.glUniform2f(self.u['uPx'], 1.0 / gw, 1.0 / gh)
            gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        gl.glViewport(0, 0, W, Hs)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)


# Thirst and the canteen are built but switched off for now: set SURVIVAL_ON = True to bring them back.
# (There is no water in the world at the moment, so with it on the bar would only drain.)
SURVIVAL_ON = False


class Survival:
    """Thirst and the canteen. Thirst drains slowly (faster while sprinting) and is refilled by drinking from a lake
    or from the canteen. The canteen is filled at a lake and carried. At zero thirst the player cannot sprint."""
    THIRST_SECONDS = 540.0     # a full bar lasts this long standing around
    SPRINT_DRAIN = 2.5         # sprinting uses thirst this many times faster
    DRINK_RATE = 35.0          # thirst points per second drinking from a lake
    FILL_RATE = 50.0           # canteen points per second filling it
    CANTEEN_MAX = 100.0
    CANTEEN_DRINK = 30.0       # thirst points per second drinking from the canteen

    def __init__(self):
        self.thirst = 100.0
        self.canteen = 0.0

    @property
    def parched(self):
        return self.thirst <= 0.0

    def update(self, dt, sprinting, drink, fill, at_water):
        """Advance one frame. Returns what the player did: 'drink', 'fill', 'canteen' or None."""
        act = None
        if fill and at_water and self.canteen < self.CANTEEN_MAX:
            self.canteen = min(self.CANTEEN_MAX, self.canteen + self.FILL_RATE * dt)
            act = 'fill'
        elif drink and self.thirst < 100.0:
            if at_water:
                self.thirst = min(100.0, self.thirst + self.DRINK_RATE * dt)
                act = 'drink'
            elif self.canteen > 0.0:
                take = min(self.CANTEEN_DRINK * dt, self.canteen, 100.0 - self.thirst)
                self.canteen -= take
                self.thirst += take
                act = 'canteen'
        self.thirst = max(0.0, self.thirst - 100.0 / self.THIRST_SECONDS * (self.SPRINT_DRAIN if sprinting else 1.0) * dt)
        return act


class SurvivalHud(_CachedOverlay):
    """Bottom-left: the thirst bar, the canteen, and a hint for the water keys when they apply."""
    W, H = 330, 104

    def __init__(self):
        super().__init__(self.W, self.H, 24, _E.HEIGHT - 96 - 24 - self.H - 12)
        self.f_small, self.f_hint = _mono(16), _mono(15)

    def _render(self, st):
        thirst, canteen, at_water, flash = st
        s = self.surf
        pygame.draw.rect(s, (*UI_BG[:3], 210), (0, 0, self.W, self.H), border_radius=10)
        pygame.draw.rect(s, UI_BORDER, (0, 0, self.W, self.H), 2, border_radius=10)
        low = thirst < 20
        s.blit(self.f_small.render("THIRST", True, (255, 150, 140) if low and flash else UI_DIM), (14, 10))
        if thirst <= 0:
            s.blit(self.f_small.render("PARCHED: no sprinting", True, (255, 150, 140)), (90, 10))
        bar = pygame.Rect(14, 30, self.W - 28, 12)
        pygame.draw.rect(s, UI_SEG_OFF, bar, border_radius=3)
        if thirst > 0:
            col = (225, 80, 70) if low and flash else ((90, 190, 235) if not low else (225, 150, 70))
            pygame.draw.rect(s, col, (bar.x, bar.y, max(4, int(bar.w * thirst / 100.0)), bar.h), border_radius=3)
        s.blit(self.f_small.render("CANTEEN", True, UI_DIM), (14, 50))
        bar2 = pygame.Rect(14, 70, self.W - 28, 8)
        pygame.draw.rect(s, UI_SEG_OFF, bar2, border_radius=3)
        if canteen > 0:
            pygame.draw.rect(s, (120, 215, 225), (bar2.x, bar2.y, max(4, int(bar2.w * canteen / 100.0)), bar2.h), border_radius=3)
        if at_water:
            hint = "E drink    B fill canteen"
        elif canteen > 0 and thirst < 100:
            hint = "E drink from canteen"
        else:
            hint = ""
        if hint:
            s.blit(self.f_hint.render(hint, True, (255, 220, 120)), (14, 84))



class LoadingScreen(Overlay):
    """The first thing on screen: a progress bar, the stage being worked on, and a running list of what is done.
    step() draws one frame at once and keeps the window responsive, so it is safe to call between slow jobs."""
    def __init__(self):
        super().__init__(_E.WIDTH, _E.HEIGHT, 0, 0)
        self.f_title, self.f_stage, self.f_small = _mono(64), _mono(26), _mono(18)
        self.t0 = time.perf_counter()
        self.done = []            # (stage text, seconds) of finished stages, newest last
        self.stage = None
        self.stage_t = self.t0
        self.frac = 0.0

    def step(self, text, frac, detail=""):
        """Finish the stage that was running, start `text`, and show the bar at `frac` (0..1)."""
        now = time.perf_counter()
        if self.stage is not None and text != self.stage:
            self.done.append((self.stage, now - self.stage_t))
            self.stage_t = now
        self.stage, self.frac, self.detail = text, max(0.0, min(1.0, frac)), detail
        _emit("%.3f" % self.frac, text, detail)
        self._show()

    def _render(self):
        s = self.surf
        s.fill((12, 14, 20, 255))
        cx = _E.WIDTH // 2
        t = self.f_title.render("VOIDCRAFT", True, UI_TEXT)
        s.blit(t, (cx - t.get_width() // 2, _E.HEIGHT // 2 - 190))
        bw, bh = 760, 26
        bx, by = cx - bw // 2, _E.HEIGHT // 2 - 40
        pygame.draw.rect(s, UI_TRACK, (bx, by, bw, bh), border_radius=8)
        fill = int(bw * self.frac)
        if fill > 0:
            pygame.draw.rect(s, UI_ACCENT, (bx, by, max(fill, 16), bh), border_radius=8)
        pygame.draw.rect(s, UI_BORDER, (bx, by, bw, bh), 2, border_radius=8)
        pct = self.f_stage.render("%d%%" % round(self.frac * 100), True, UI_TEXT)
        s.blit(pct, (bx + bw - pct.get_width(), by - 34))
        stage = self.f_stage.render(self.stage or "", True, UI_TEXT)
        s.blit(stage, (bx, by + bh + 10))
        if self.detail:
            s.blit(self.f_small.render(self.detail, True, UI_DIM), (bx, by + bh + 46))
        y = by + bh + 86
        for text, secs in self.done[-8:]:                       # what has finished, with how long it took
            line = self.f_small.render("done  %-46s %5.2fs" % (text[:46], secs), True, UI_DIM)
            s.blit(line, (bx, y))
            y += 22
        el = self.f_small.render("%.1fs" % (time.perf_counter() - self.t0), True, UI_DIM)
        s.blit(el, (_E.WIDTH - el.get_width() - 24, _E.HEIGHT - 36))

    def _show(self):
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit()
                raise SystemExit
        if _UI_LOADING:                       # engineui draws the loading screen: nothing to draw in this window
            return
        gl.glClearColor(0.047, 0.055, 0.078, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.draw()
        pygame.display.flip()


class _FpsMeter:
    """Frames per second, averaged over half a second so the title does not flicker."""
    def __init__(self):
        self.t = 0.0
        self.n = 0
        self.value = 0.0

    def update(self, dt):
        self.t += dt
        self.n += 1
        if self.t >= 0.5:
            self.value = self.n / self.t
            self.t, self.n = 0.0, 0
        return self.value


FPS = _FpsMeter()


# ---- first-person drawing helpers (the held TNT bundle) ---------------------------------------------------------------


def _held_axes(player):
    fwd = np.asarray(player.forward(), np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    return fwd, right


MAP_RADIUS = 15     # the whole map: 15 chunks each way from the spawn chunk (31 x 31 chunks), loaded once


def _lp(player, f, r, u):
    """A point in the player's view: f ahead, r to the right, u up from the eye."""
    fwd, right = _held_axes(player)
    up = np.cross(right, fwd)
    return np.asarray(player.eye, np.float64) + fwd * f + right * r + up * u


def _obox(player, f, r, u, hf, hr, hu, rgb, out, glow=False):
    """One box in view space (centre f/r/u, half sizes hf/hr/hu), as shaded faces appended to out for painter sorting."""
    fwd, right = _held_axes(player)
    up = np.cross(right, fwd)
    eye = np.asarray(player.eye, np.float64)
    ctr = _lp(player, f, r, u)
    A = (fwd * hf, right * hr, up * hu)
    for ax, sgn, axes in ((0, 1, (1, 2)), (0, -1, (1, 2)), (1, 1, (0, 2)), (1, -1, (0, 2)), (2, 1, (0, 1)), (2, -1, (0, 1))):
        n = A[ax] * sgn
        fc = ctr + n
        if float(np.dot(n, eye - fc)) <= 0.0:
            continue
        a1, a2 = A[axes[0]], A[axes[1]]
        P = [fc - a1 - a2, fc + a1 - a2, fc + a1 + a2, fc - a1 + a2]
        nn = n / max(float(np.linalg.norm(n)), 1e-9)
        sh = 1.0 if glow else 0.55 + 0.45 * max(0.0, float(np.dot(nn, np.array([0.3, 0.9, -0.2]))))
        col = tuple(int(min(255, c * sh)) for c in rgb) + (255,)
        out.append((float(np.linalg.norm(fc - eye)), ('poly', P, col, 0)))


def _flush(faces, prims):
    faces.sort(key=lambda x: -x[0])
    prims += [p for _d, p in faces]


def tnt_held_prims(player, t):
    """First-person TNT: a bundle of three red sticks with a paper band and a short fuse."""
    faces, prims = [], []
    f0, r0, u0 = 0.62, 0.26, -0.24
    for dr in (-0.045, 0.0, 0.045):
        _obox(player, f0, r0 + dr, u0, 0.03, 0.021, 0.12, (200, 48, 46), faces)
    _obox(player, f0 - 0.0, r0, u0 - 0.005, 0.034, 0.075, 0.03, (232, 216, 180), faces)    # paper band
    _flush(faces, prims)
    top = _lp(player, f0, r0, u0 + 0.125)
    tip = _lp(player, f0 + 0.02, r0 + 0.03, u0 + 0.2)
    prims.append(('line', top, tip, (210, 180, 100, 255), 3))
    spark = 6 + int(3 * math.sin(t * 25.0))
    prims.append(('line', tip, tip + (_lp(player, 0.0, 0.0, 0.01) - np.asarray(player.eye)), (255, 210, 80, 255), spark))
    return prims


STONE_DEPTH = 1.5   # blocks of dirt on top; the terrain turns grey below this (colour only)
BEDROCK_BLOCKS = 1.0   # thickness of the unbreakable layer at the very bottom of the map, in blocks


def _stone_mask(world, pts, *, for_colour=False):
    """Bedrock test for digging (bottom layer only), or the terrain's cliff palette for debris colour."""
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), bool)
    if not len(pts):
        return out
    C, NC = _E.CELL, _E.NC
    iy = np.floor(pts[:, 1] / C).astype(np.int64)
    if not for_colour:
        return iy < int(round(BEDROCK_BLOCKS * _E.GS))
    depth = int(round(STONE_DEPTH * _E.GS))
    ix = np.floor(pts[:, 0] / C).astype(np.int64)
    iz = np.floor(pts[:, 2] / C).astype(np.int64)
    kx, kz = ix // NC, iz // NC
    for key in set(zip(kx.tolist(), kz.tolist())):
        d = world.dens.get(key)
        if d is None:
            continue
        sel = (kx == key[0]) & (kz == key[1])
        top = d.top[ix[sel] - key[0] * NC, iz[sel] - key[1] * NC]
        height = top * C
        out[sel] = (height >= _E.STONE_ALT) | ((height < _E.DIRT_ALT) & (iy[sel] < top - depth))
    return out


def _restore_stone(world, centre, r):
    """Put back any bedrock a dig just cleared, so the bottom layer of the map can never be broken."""
    C, NC = _E.CELL, _E.NC
    depth = int(round(BEDROCK_BLOCKS * _E.GS))
    c = np.asarray(centre, np.float64)
    lo = np.floor((c - r) / C).astype(np.int64)
    hi = np.floor((c + r) / C).astype(np.int64)
    if lo[1] >= depth:
        return
    for kx in range(lo[0] // NC, hi[0] // NC + 1):
        for kz in range(lo[2] // NC, hi[2] // NC + 1):
            d = world.dens.get((kx, kz))
            if d is None or d.arr is None:
                continue
            x0, x1 = max(lo[0], kx * NC) - kx * NC, min(hi[0], kx * NC + NC - 1) - kx * NC
            z0, z1 = max(lo[2], kz * NC) - kz * NC, min(hi[2], kz * NC + NC - 1) - kz * NC
            if x0 > x1 or z0 > z1:
                continue
            d.arr[x0:x1 + 1, :depth, z0:z1 + 1] = 1
            world.dirty.add((kx, kz))


def c4_prims(position, flash=None):
    """World-space drawing for the planted charge: a red 'C4 Breaker Box' crate with black corner brackets,
    hazard stripes on the front and a blinking yellow indicator light."""
    prims = []
    if position is None:
        return prims
    centre = np.asarray(position, np.float64)
    size = C4_BOX_SIZE
    lo = centre - size * 0.5
    C = lo + _E._CUBE_UNIT * size
    top = max(_E._CUBE_FACES, key=lambda f: float(C[list(f), 1].mean()))
    for f in _E._CUBE_FACES:
        col = (232, 58, 42, 255) if f == top else (185, 34, 29, 255)
        prims.append(('poly', [C[i] for i in f], col, 0))
    for a, b in _E._CUBE_EDGES:                                   # black corner brackets
        prims.append(('line', C[a], C[b], (18, 16, 16, 255), 4))
    # hazard stripes: alternating yellow bands across the four side faces, low on the crate
    for f in _E._CUBE_FACES:
        P = C[list(f)]
        if abs(float(P[:, 1].max() - P[:, 1].min())) < 1e-6:
            continue                                              # top/bottom
        ylo, yhi = float(P[:, 1].min()), float(P[:, 1].max())
        for k in range(4):
            t0, t1 = 0.05 + k * 0.1, 0.1 + k * 0.1
            if k % 2:
                continue
            lowA, lowB = [p for p in P if abs(p[1] - ylo) < 1e-6][:2]
            hi_pts = [p for p in P if abs(p[1] - yhi) < 1e-6][:2]
            hiA = hi_pts[0] if np.linalg.norm(hi_pts[0] - lowA) < np.linalg.norm(hi_pts[1] - lowA) else hi_pts[1]
            hiB = hi_pts[1] if hiA is hi_pts[0] else hi_pts[0]
            a0 = lowA + (hiA - lowA) * t0 * 2.2
            a1 = lowA + (hiA - lowA) * t1 * 2.2
            b0 = lowB + (hiB - lowB) * t0 * 2.2
            b1 = lowB + (hiB - lowB) * t1 * 2.2
            prims.append(('poly', [a0, b0, b1, a1], (245, 200, 30, 255), 0))
    # indicator light on top, blinking
    blink = int(time.perf_counter() * 2.5) % 2
    lamp = centre + np.array([0.0, size * 0.5 + 0.03, 0.0])
    lc = (255, 220, 60, 255) if blink else (150, 110, 20, 255)
    prims.append(('line', lamp - [0, 0.025, 0], lamp + [0, 0.035, 0], lc, 7))
    return prims


C4_LABEL = "C4 Breaker Box"
C4_BOX_SIZE = 0.42   # edge length of a planted TNT box, in blocks


def c4_aimed(player, position, reach=8.0):
    """True when the crosshair is on the planted box (ray vs. a small sphere around it)."""
    if position is None:
        return False
    eye = np.asarray(player.eye, np.float64)
    fwd = np.asarray(player.forward(), np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    to = np.asarray(position, np.float64) - eye
    t = float(np.dot(to, fwd))
    return 0.0 < t < reach and float(np.linalg.norm(to - fwd * t)) < C4_BOX_SIZE * 0.8333


C4_RADIUS = 1.5        # blast radius for structures
C4_CRATER = 1.9        # radius of the ground crater, in blocks
C4_PUSH_R = 4.0       # loose cubes within this distance are shoved
C4_MAX_CUBES = 650      # leave room for other awake debris before the engine time-slices
C4_EXACT_MAX = 2600     # a blast that digs up to this many ground cubes throws every one of them as debris (scoop out = debris in); a bigger blast samples C4_MAX_CUBES of them
C4_DEBRIS_SCALE = 0.82  # clearance between a loose cube and its original voxel slot
C4_DEBRIS_SIZE = 0.5    # size of a blast's debris cubes relative to a ground cube (1.0 = full size, 0.5 = half the edge length)
C4_UP_MIN, C4_UP_MAX = 7.0, 13.0   # total launch speed of the ground fountain, blocks per second (before the multiplier)
C4_ANG_MIN, C4_ANG_MAX = 35.0, 75.0   # launch angle above the ground in degrees: low = thrown far, 90 = straight up
C4_SPEED_MUL = 1.4      # multiplies every launch speed below (ground fountain, ring and wall)
C4_RING_FRAC = 0.2     # share of cubes flung out sideways in a ring instead of up
C4_WALL_MIN, C4_WALL_MAX = 8.0, 14.0   # launch speed of wall debris (x the multiplier); it flies a parabola
C4_RING_MIN, C4_RING_MAX = 3.5, 5.5   # launch speed of the few cubes arcing out in a ring (x the multiplier)
C4_PUSH = 7.0         # speed given to a cube at the centre, blocks per second


C4_SPREAD_MIN, C4_SPREAD_MAX = 1.5, 4.5   # sideways speed of the ground fountain, so cubes scatter instead of dropping back in place
CRUMBLE_KICK = 2.0      # extra downhill speed given to blasted cubes, so they roll off a mountain
CRUMBLE_PULL = 20.0      # sustained downhill pull on moving cubes that touch the ground (blocks/s^2): rubble slides to the base
DEBRIS_STONE_DARKEN = 0.6   # debris colour x this for stone, so it matches the terrain's dark cliff colour (1.0 = the plain tile average)
DEBRIS_DIRT_DARKEN = 0.85   # same for dirt debris
C4_BURST = 3.0          # outward speed at the blast edge (x falloff): every cube is shoved away from the centre, so the packed ball of cubes expands
C4_JITTER = 1.8         # random per-cube speed in all directions, so neighbours stop moving as one block
CRUMBLE_MAX_SPEED = 3.0   # top sideways speed the downhill pull will build up to, blocks per second
CRUMBLE_REACH = 0.9     # how far (blocks) to look for the slope, wide enough to see a whole mountain face not just one step


def _crumble_dir(world, P):
    """Unit horizontal direction away from the higher ground around each point (down the mountain); zero on flat ground."""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    e = CRUMBLE_REACH
    def at(dx, dz):
        return np.stack([P[:, 0] + dx, P[:, 1], P[:, 2] + dz], axis=1)
    d = _E._density_batch(world, [at(e, 0.0), at(-e, 0.0), at(0.0, e), at(0.0, -e)])
    g = np.stack([d[0] - d[1], d[2] - d[3]], axis=1)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return np.where(n > 1e-9, -g / np.maximum(n, 1e-9), 0.0)


C4_WAVE_ANG_HI, C4_WAVE_ANG_LO = 72.0, 54.0   # launch angle (degrees) for cubes at the crater centre / at its rim: the middle goes steeper so it clears the walls
C4_WAVE_NEAR = 1.0      # blocks beyond the crater rim where the nearest cubes land
C4_WAVE_SPREAD = 7.0    # extra blocks the centre cubes are thrown, so debris lands in a spreading ring (a ripple)
C4_WAVE_AIM_JITTER = 8.0   # degrees of random sideways aim, so the ring is not perfectly regular

# ---- blast size ---------------------------------------------------------------------------------------------------
# The blast size is fixed at 0.75 (BLAST above); BLAST_LEVELS lists the sizes set_blast_scale can still be given. Everything about the blast grows with the scale s: the crater, the radius that
# damages structures and shoves cubes, how far the debris is thrown, how many cubes are lifted (up to the awake limits, which
# are raised too), and the fire, smoke and dust, which are drawn physically bigger. 1.0 is the original blast.
BLAST_LEVELS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
BLAST = dict(i=BLAST_LEVELS.index(0.75))      # locked at 0.75: nothing changes it while the game runs (the scroll wheel no longer does)
_BLAST_BASE = dict(C4_RADIUS=C4_RADIUS, C4_CRATER=C4_CRATER, C4_PUSH_R=C4_PUSH_R, C4_MAX_CUBES=C4_MAX_CUBES,
                   C4_WAVE_SPREAD=C4_WAVE_SPREAD, C4_WAVE_NEAR=C4_WAVE_NEAR, C4_BURST=C4_BURST, C4_JITTER=C4_JITTER,
                   C4_WALL_MIN=C4_WALL_MIN, C4_WALL_MAX=C4_WALL_MAX, SLAB_DEBRIS_MAX=3000,
                   MAX_AWAKE=int(getattr(_E, 'MAX_AWAKE_CUBES', 500)), HARD_AWAKE=int(getattr(_E, 'HARD_AWAKE_CUBES', 700)))


def blast_scale():
    return BLAST_LEVELS[BLAST['i']]


def set_blast_scale(i):
    """Pick a blast size (an index into BLAST_LEVELS) and rescale every blast constant from the originals."""
    global C4_RADIUS, C4_CRATER, C4_PUSH_R, C4_MAX_CUBES, C4_WAVE_SPREAD, C4_WAVE_NEAR, C4_BURST, C4_JITTER
    global C4_WALL_MIN, C4_WALL_MAX, SLAB_DEBRIS_MAX
    BLAST['i'] = max(0, min(len(BLAST_LEVELS) - 1, int(i)))
    s = blast_scale()
    r = math.sqrt(s)                                   # speeds grow with the square root of the distance thrown
    b = _BLAST_BASE
    C4_RADIUS, C4_CRATER, C4_PUSH_R = b['C4_RADIUS'] * s, b['C4_CRATER'] * s, b['C4_PUSH_R'] * s
    C4_WAVE_SPREAD, C4_WAVE_NEAR = b['C4_WAVE_SPREAD'] * s, b['C4_WAVE_NEAR'] * s
    C4_BURST, C4_JITTER = b['C4_BURST'] * r, b['C4_JITTER'] * r
    C4_WALL_MIN, C4_WALL_MAX = b['C4_WALL_MIN'] * r, b['C4_WALL_MAX'] * r
    more = max(1.0, s * s)                              # cube counts follow the crater's size: more ground, more cubes
    C4_MAX_CUBES = int(min(b['C4_MAX_CUBES'] * (s * s if s >= 1.0 else s), 11000))
    SLAB_DEBRIS_MAX = int(b['SLAB_DEBRIS_MAX'] * more)
    live = max(1.0, s ** 1.5)                           # how many cubes may be simulated at once grows more slowly, to protect the frame rate
    _E.MAX_AWAKE_CUBES = int(min(b['MAX_AWAKE'] * live, 11500))
    _E.HARD_AWAKE_CUBES = int(min(max(b['HARD_AWAKE'] * live, C4_EXACT_MAX), 12000))
    return s



def _wave_velocity(P, origin, rng):
    """Launch velocities that make a blast's cubes land in an expanding ring, like a splash ripple. Each cube is thrown straight
    out from the blast centre and aimed at a landing distance worked out from where it started: centre cubes go steeper and
    further, rim cubes lower and nearer, so everything clears the crater and the landing spreads out as a wave."""
    P = np.asarray(P, np.float64)
    n = len(P)
    g = _E.BODY_GRAVITY
    R = C4_CRATER
    h = P - origin
    h[:, 1] = 0.0
    r = np.linalg.norm(h, axis=1)
    az = np.arctan2(h[:, 2], h[:, 0])
    az = np.where(r > 1e-6, az, rng.uniform(0.0, 2.0 * np.pi, n))              # a cube dead centre goes any way
    az = az + np.radians(rng.uniform(-C4_WAVE_AIM_JITTER, C4_WAVE_AIM_JITTER, n))
    rn = np.clip(r / R, 0.0, 1.0)
    depth = np.clip((origin[1] - P[:, 1]) / R, 0.0, 1.0)                        # how far down in the crater it started
    th = np.radians(np.clip(C4_WAVE_ANG_HI + (C4_WAVE_ANG_LO - C4_WAVE_ANG_HI) * rn, 50.0, 78.0))
    reach = C4_WAVE_NEAR + C4_WAVE_SPREAD * (1.0 - rn) * rng.uniform(0.6, 1.0, n)
    land = np.maximum(R + reach, r + 1.0)                                       # distance from the centre where it should come down
    sp = np.sqrt(g * (land - r) / np.maximum(np.sin(2.0 * th), 0.3)) * (1.12 - 0.25 * depth)  # upper layers pull away from the deeper ones
    v = np.zeros((n, 3))
    v[:, 0] = np.cos(az) * sp * np.cos(th)
    v[:, 2] = np.sin(az) * sp * np.cos(th)
    v[:, 1] = sp * np.sin(th)
    return v


def update_debris(world, dt, sleep_time):
    """Step the loose cubes, but give airborne ones a true ballistic arc. The engine slows every cube's sideways motion
    constantly, even in mid-air, which bends a thrown cube's path into 'out, then straight up/down'. Here that slow-down is
    undone for cubes that are still in the air after the step, so they follow a smooth parabola; on the ground it stays."""
    n = BODIES.n
    pre = BODIES.pos.copy() if n else None
    BODIES.grounded_last = None
    BODIES.update(world, dt, sleep_time)
    if not n or BODIES.n != n:
        return
    moved = (BODIES.pos != pre).any(axis=1)
    if not moved.any():
        return
    idx = np.nonzero(moved)[0]
    gl = getattr(BODIES, 'grounded_last', None)
    if gl is not None and len(gl) == BODIES.n:             # reuse the engine's own ground test instead of sampling the ground again
        touching = gl[idx]
    else:
        touching = _E._support(world, BODIES.pos[idx], BODIES.size[idx]).any(axis=1)
    air = ~touching
    if air.any():
        k = math.exp(_E.BODY_FRICTION * min(dt, 0.05))
        sel = idx[air]
        BODIES.vel[sel, 0] *= k
        BODIES.vel[sel, 2] *= k
    if touching.any():                                     # rubble on a slope keeps sliding down it and piles up at the base
        sel = idx[touching]
        slow = np.hypot(BODIES.vel[sel, 0], BODIES.vel[sel, 2]) < CRUMBLE_MAX_SPEED   # no extra push once it is already sliding fast
        if slow.any():                                     # only look at the slope for cubes that will actually be pushed
            sel = sel[slow]
            down = _crumble_dir(world, BODIES.pos[sel])
            pull = CRUMBLE_PULL * min(dt, 0.05)
            BODIES.vel[sel, 0] += down[:, 0] * pull
            BODIES.vel[sel, 2] += down[:, 1] * pull


def _surface_normal(world, p, fwd):
    """Which way the surface the charge sits on faces: up for the ground, sideways for a wall. Found from which side of the
    hit point is solid; falls back to the view direction (snapped to an axis) when the terrain says nothing."""
    p = np.asarray(p, np.float64)
    e = max(0.6 * _E.CELL, 0.12)
    g = np.zeros(3)
    for k in range(3):
        a, b = p.copy(), p.copy()
        a[k] += e
        b[k] -= e
        sa = float(_E._density_many(world, a.reshape(1, 3))[0]) > 0.0
        sb = float(_E._density_many(world, b.reshape(1, 3))[0]) > 0.0
        g[k] = float(sa) - float(sb)
    n = -g
    if float(np.linalg.norm(n)) < 1e-6:
        n = np.zeros(3)
        k = int(np.argmax(np.abs(fwd)))
        n[k] = -np.sign(fwd[k]) if fwd[k] != 0 else 1.0
    return n / np.linalg.norm(n)


# ---- throwing TNT: hold the left button to aim, let go to throw ------------------------------------------------------------
THROW_SPEED_MIN, THROW_SPEED_MAX = 7.0, 24.0   # blocks per second: a quick click is a gentle toss, a full charge a long throw
THROW_CHARGE_S = 1.0       # seconds of holding the button to reach full power
THROW_LOFT = 0.22          # upward part of the throw as a share of its speed, so it flies a little above the crosshair
THROW_GRAVITY = 18.0       # blocks per second squared
THROW_STEP = 0.06          # seconds between the points of the path
THROW_MAX_T = 5.0          # a throw that has hit nothing after this long is lost
THROW_HAND = (0.62, 0.26, -0.24)   # where the charge leaves from: forward, right, up from the eye (where the held TNT is drawn)


def throw_velocity(player, charge):
    """Launch velocity for a throw at `charge` (0 to 1) along the view direction, lifted a little."""
    speed = THROW_SPEED_MIN + (THROW_SPEED_MAX - THROW_SPEED_MIN) * max(0.0, min(1.0, charge))
    fwd = np.asarray(player.forward(), np.float64)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
    return fwd * speed + np.array([0.0, THROW_LOFT * speed, 0.0])


def throw_path(world, start, vel):
    """The path of a thrown charge, a point every THROW_STEP seconds, ending where it first touches ground or a structure.
    Returns (points as an (N, 3) array, landing as dict(pos, normal, struct) or None when it hits nothing)."""
    p = np.asarray(start, np.float64).copy()
    v = np.asarray(vel, np.float64).copy()
    pts = [p.copy()]
    for _ in range(int(THROW_MAX_T / THROW_STEP)):
        v2 = v.copy()
        v2[1] -= THROW_GRAVITY * THROW_STEP
        q = p + (v + v2) * 0.5 * THROW_STEP
        seg = q - p
        ln = float(np.linalg.norm(seg))
        if ln > 1e-9:
            d = seg / ln
            hit = _E._first_hit(world, p, d, ln)
            if hit is not None:
                if hit[2] is not None:
                    pos, nrm = _E.structure_hit_face(p, d, hit[1], hit[2])
                    land = dict(pos=np.asarray(pos, np.float64).copy(), normal=np.asarray(nrm, np.float64), struct=True)
                else:
                    pos = np.asarray(hit[1], np.float64).copy()
                    land = dict(pos=pos, normal=_surface_normal(world, pos, d), struct=False)
                pts.append(np.asarray(hit[1], np.float64).copy())
                return np.array(pts), land
        p, v = q, v2
        pts.append(p.copy())
        if p[1] < -20.0:
            break
    return np.array(pts), None


def arc_prims(pts, land):
    """The aiming arc as dashed lines (white when it lands on something, red when it would fly off), with a cross where it lands."""
    col = (255, 244, 200, 255) if land is not None else (255, 120, 100, 255)
    prims = []
    for i in range(len(pts) - 1):
        if i % 3 == 2:
            continue
        prims.append(('line', pts[i], pts[i + 1], (0, 0, 0, 150), 6))
        prims.append(('line', pts[i], pts[i + 1], col, 3))
    if land is not None:
        c = land['pos']
        for ax in np.eye(3):
            prims.append(('line', c - ax * 0.3, c + ax * 0.3, (0, 0, 0, 150), 6))
            prims.append(('line', c - ax * 0.3, c + ax * 0.3, col, 3))
    return prims


def flight_pos(pts, t):
    """Where a thrown charge is `t` seconds after the throw (it follows the same points the arc showed)."""
    f = max(0.0, t) / THROW_STEP
    i = min(int(f), len(pts) - 2)
    a = min(1.0, f - i)
    return pts[i] + (pts[i + 1] - pts[i]) * a


# ---- the tools ---------------------------------------------------------------------------------------------------
# 1-3 pick a tool. Each one is a small chunky model built from flat-coloured boxes (the same look as the toolbar art), held
# in front of the camera and drawn through the same line/poly prims as the held TNT.
T_HAMMER, T_CANNON, T_TNT = range(3)
TOOL_NAMES = ('Impact Hammer', 'Cannon', 'Breaching Charge')
TOOL_HINTS = ('Click: swing and chip out a small crater',
              'Click: fire a cannonball that blows a crater where it lands',
              'Hold and throw TNT, right click to detonate')
TOOL_REACH = 4.5
HAMMER_R, HAMMER_SWING = 0.55, 0.42          # crater radius (blocks), seconds for one swing (it lands at 60% of it)
HAMMER_STRUCT_R, HAMMER_STRUCT_P = 0.8, 1.0
CANNON_SPEED, CANNON_GRAVITY, CANNON_LOFT, CANNON_CD = 50.0, 8.0, 0.04, 1.2   # ball speed (blocks/s), its drop, upward share of the shot, seconds to reload
CANNON_BALL_CELLS, CANNON_MAX_T = 1.5, 4.0           # ball edge in ground cubes (so it follows the cube detail), flight time before a miss is lost
CANNON_R, CANNON_STRUCT_R, CANNON_STRUCT_P, CANNON_CRACK = 1.1, 1.7, 3.2, 3.0   # crater radius, structure blast radius and power, ground weakened round it
BREACH_MULT = 1.7                                     # TNT thrown against a structure hits this much harder than before
TS = dict(swing=None, hit_done=False, balls=[], recoil=0.0, cool=0.0)


def tool_reset():
    TS.update(swing=None, hit_done=False, balls=[], recoil=0.0, cool=0.0)


def _rot_r(a):
    """Tilt about the right axis, in (forward, right, up) order: the tool's up end leans forward for a > 0."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _tool_boxes(player, anchor, rot, boxes, faces):
    """Boxes given in the tool's own frame (off, half, rgb, glow), placed at `anchor` (forward, right, up from the eye)."""
    fwd, right = _held_axes(player)
    up = np.cross(right, fwd)
    B = np.array([fwd, right, up])
    eye = np.asarray(player.eye, np.float64)
    a0 = eye + np.asarray(anchor, np.float64) @ B
    for off, half, rgb, glow in boxes:
        c = a0 + (rot @ np.asarray(off, np.float64)) @ B
        A = [(rot[:, k] * half[k]) @ B for k in range(3)]
        for ax, sgn, axes in ((0, 1, (1, 2)), (0, -1, (1, 2)), (1, 1, (0, 2)), (1, -1, (0, 2)), (2, 1, (0, 1)), (2, -1, (0, 1))):
            n = A[ax] * sgn
            fc = c + n
            if float(np.dot(n, eye - fc)) <= 0.0:
                continue
            a1, a2 = A[axes[0]], A[axes[1]]
            P = [fc - a1 - a2, fc + a1 - a2, fc + a1 + a2, fc - a1 + a2]
            nn = n / max(float(np.linalg.norm(n)), 1e-9)
            sh = 1.0 if glow else 0.55 + 0.45 * max(0.0, float(np.dot(nn, np.array([0.3, 0.9, -0.2]))))
            col = tuple(int(min(255, v * sh)) for v in rgb) + (255,)
            faces.append((float(np.linalg.norm(fc - eye)), ('poly', P, col, 0)))
    return a0, B


def _hammer_angle(p):
    """Wind up, strike, recover. p runs 0..1 over the swing; the hit lands at p = 0.6."""
    if p < 0.3:
        return -0.5 - 0.5 * (p / 0.3)
    if p < 0.6:
        return -1.0 + 2.1 * ((p - 0.3) / 0.3)
    return 1.1 - 1.6 * ((p - 0.6) / 0.4)


def tool_held_prims(player, tool, t):
    """The held model for the chosen tool (TNT has its own)."""
    if tool == T_TNT:
        return tnt_held_prims(player, t)
    faces, prims = [], []
    if tool == T_HAMMER:
        th = -0.5 if TS['swing'] is None else _hammer_angle(min(1.0, TS['swing'] / HAMMER_SWING))
        boxes = [((0, 0, 0), (0.014, 0.014, 0.13), (150, 104, 58), False),
                 ((0, 0, -0.09), (0.018, 0.018, 0.04), (58, 60, 68), False),
                 ((0, 0, 0.15), (0.04, 0.075, 0.04), (112, 118, 130), False),
                 ((0, 0.09, 0.15), (0.02, 0.018, 0.05), (236, 150, 40), False),
                 ((0, -0.09, 0.15), (0.02, 0.018, 0.05), (236, 150, 40), False)]
        _tool_boxes(player, (0.55, 0.26, -0.3), _rot_r(th), boxes, faces)
    else:
        boxes = [((0.08, 0, 0), (0.15, 0.035, 0.035), (72, 76, 86), False),
                 ((0.235, 0, 0), (0.02, 0.046, 0.046), (46, 48, 56), False),
                 ((-0.075, 0, 0), (0.04, 0.046, 0.046), (46, 48, 56), False),
                 ((-0.12, 0, 0.0), (0.012, 0.012, 0.012), (236, 150, 40), False),
                 ((0.0, 0, -0.055), (0.11, 0.05, 0.02), (150, 104, 58), False),
                 ((0.02, 0.062, -0.075), (0.032, 0.012, 0.032), (110, 76, 44), False),
                 ((0.02, -0.062, -0.075), (0.032, 0.012, 0.032), (110, 76, 44), False)]
        _tool_boxes(player, (0.55 - 0.12 * TS['recoil'], 0.2, -0.2), np.eye(3), boxes, faces)
    _flush(faces, prims)
    return prims


def tool_ray(world, player, reach):
    d = np.asarray(player.forward(), np.float64)
    d = d / np.linalg.norm(d)
    p = np.asarray(player.eye, np.float64)
    hit = _E._first_hit(world, p, d, reach)
    if hit is None:
        return None
    if hit[2] is not None:
        pos, nrm = _E.structure_hit_face(p, d, hit[1], hit[2])
        return dict(pos=np.asarray(pos, np.float64).copy(), normal=np.asarray(nrm, np.float64), obj=hit[2], dist=float(hit[0]))
    pos = np.asarray(hit[1], np.float64).copy()
    return dict(pos=pos, normal=_surface_normal(world, pos, d), obj=None, dist=float(hit[0]))


def tool_crater(world, hit, R, struct_r, struct_p, shatter, kick, crack_r):
    """A small blast for the hand tools: damages a structure if one was hit, otherwise digs a crater, weakens the ground
    round it and throws the dug cubes. Returns (structure cells, ground cubes, debris)."""
    P, nrm = hit['pos'], hit['normal']
    n0 = BODIES.n
    broken = STRUCTS.blast(world, P, radius=struct_r, power=struct_p, shatter=shatter, debris_normal=nrm)
    dug = 0
    if hit['obj'] is None:
        world.dig(P - nrm * 0.1 * R, R)
        _restore_stone(world, P, R)
        world.add_damage(P, crack_r)
        groups = [g[~_stone_mask(world, g)] for g in world.take_dug_cells()]
        fallen = _drop_groups(world, world.settle_islands(P, R, margin=SLAB_SCAN_MARGIN), P)
        if len(fallen):
            groups.append(fallen)
        _schedule_collapse(world, P)
        STRUCTS.on_terrain_changed(world, P, R + 0.5)
        if groups:
            cells = np.concatenate(groups)
            dug = len(cells)
            if len(cells) > 500:
                cells = cells[np.random.choice(len(cells), 500, replace=False)]
            stone = _stone_mask(world, cells, for_colour=True)
            for mask, material, shade in ((~stone, _E.DIRT, DEBRIS_DIRT_DARKEN), (stone, _E.STONE, DEBRIS_STONE_DARKEN)):
                if mask.any():
                    BODIES.spawn(world, [cells[mask]], 1, size_scale=C4_DEBRIS_SIZE, rgb=_E._S_RGB[material] * shade, material=material)
    n1 = BODIES.n
    if n1 > n0:
        rng = np.random.default_rng()
        k = n1 - n0
        v = nrm[None, :] * (kick * rng.uniform(0.5, 1.0, k))[:, None] + rng.normal(0.0, kick * 0.35, (k, 3))
        v[:, 1] += kick * 0.4
        BODIES.vel[n0:n1] += v
        BODIES.awake[n0:n1] = True
        BODIES.still[n0:n1] = 0.0
    return len(broken), dug, max(0, n1 - n0)


FX_REF = [None]      # the BlastFX of the running game (set in run), so the tools can nudge the camera shake


def _bump_shake(amount):
    fx = FX_REF[0]
    if fx is not None:
        fx.shake = min(fx.SHAKE, fx.shake + amount)


def ball_prims(position):
    """A cannonball: a dark cube (everything is cubes), shaded by face."""
    edge = CANNON_BALL_CELLS * float(_E.CELL)
    C = np.asarray(position, np.float64) - edge * 0.5 + _E._CUBE_UNIT * edge
    top = max(_E._CUBE_FACES, key=lambda f: float(C[list(f), 1].mean()))
    prims = [('poly', [C[i] for i in f], (92, 96, 108, 255) if f == top else (50, 52, 60, 255), 0) for f in _E._CUBE_FACES]
    prims += [('line', C[a], C[b], (20, 20, 24, 255), 2) for a, b in _E._CUBE_EDGES]
    return prims


def tool_step(world, player, tool, dt, press, release, held, det):
    """One frame of the hand tools (not TNT). press/release are this frame's left-button events, held is the button's state,
    det is a right click. Returns the world-space prims this adds."""
    TS['cool'] = max(0.0, TS['cool'] - dt)
    extra = []
    if tool != T_HAMMER:
        TS['swing'] = None
    if tool == T_HAMMER:
        if TS['swing'] is None and press and TS['cool'] <= 0.0:
            TS['swing'], TS['hit_done'] = 0.0, False
            SFX.play('jump', 0.35, gap=0.05)
        if TS['swing'] is not None:
            TS['swing'] += dt
            if not TS['hit_done'] and TS['swing'] >= HAMMER_SWING * 0.6:
                TS['hit_done'] = True
                hit = tool_ray(world, player, TOOL_REACH)
                if hit is not None:
                    s, g, d = tool_crater(world, hit, HAMMER_R, HAMMER_STRUCT_R, HAMMER_STRUCT_P, False, 3.0, HAMMER_R * 1.6)
                    SFX.play('explosion', 0.35, gap=0.05)
                    _bump_shake(0.008)
            if TS['swing'] >= HAMMER_SWING:
                TS['swing'] = None
                TS['cool'] = 0.05
    elif tool == T_CANNON:
        if press and TS['cool'] <= 0.0:
            fwd = np.asarray(player.forward(), np.float64)
            fwd = fwd / np.linalg.norm(fwd)
            start = _lp(player, 0.85, 0.2, -0.2)
            aim = np.asarray(player.eye, np.float64) + fwd * 30.0
            d = aim - start
            d = d / np.linalg.norm(d)
            TS['balls'].append(dict(p=start, v=d * CANNON_SPEED + np.array([0.0, CANNON_LOFT * CANNON_SPEED, 0.0]), age=0.0))
            TS['cool'] = CANNON_CD
            TS['recoil'] = 1.0
            SFX.play('explosion', 0.6, gap=0.05)
            _bump_shake(0.01)
    TS['recoil'] = max(0.0, TS['recoil'] - dt * 5.0)
    for b in TS['balls'][:]:
        b['age'] += dt
        v2 = b['v'].copy()
        v2[1] -= CANNON_GRAVITY * dt
        q = b['p'] + (b['v'] + v2) * 0.5 * dt
        seg = q - b['p']
        ln = float(np.linalg.norm(seg))
        hit = None
        if ln > 1e-9:
            d = seg / ln
            h = _E._first_hit(world, b['p'], d, ln)
            if h is not None:
                if h[2] is not None:
                    pos, nrm = _E.structure_hit_face(b['p'], d, h[1], h[2])
                    hit = dict(pos=np.asarray(pos, np.float64).copy(), normal=np.asarray(nrm, np.float64), obj=h[2])
                else:
                    pos = np.asarray(h[1], np.float64).copy()
                    hit = dict(pos=pos, normal=_surface_normal(world, pos, d), obj=None)
        if hit is not None:
            TS['balls'][:] = [o for o in TS['balls'] if o is not b]     # by identity: == on dicts holding arrays raises
            s_, g_, d_ = tool_crater(world, hit, CANNON_R, CANNON_STRUCT_R, CANNON_STRUCT_P, False, 7.0, CANNON_CRACK)
            SFX.play('explosion', 1.0, gap=0.0)
            _bump_shake(0.02 / (1.0 + float(np.linalg.norm(hit['pos'] - player.eye)) / 10.0))
            inv_note(f'Cannonball: {s_} structure + {g_} ground cubes blown out', 3.0)
            continue
        b['p'], b['v'] = q, v2
        if b['age'] > CANNON_MAX_T or q[1] < -20.0:
            TS['balls'][:] = [o for o in TS['balls'] if o is not b]     # by identity: == on dicts holding arrays raises
    extra = [q for b in TS['balls'] for q in ball_prims(b['p'])]
    return extra


SLAB_SCAN_MARGIN = 8.0       # blocks beyond the crater that are checked for cut-off ground: a ledge has to fit inside this to be found whole
SLAB_MIN_CELLS = 1500        # a cut-off group smaller than this (about 12 blocks of ground) crumbles into cubes; bigger falls as one rigid slab
SLAB_MAX_CELLS = 600_000     # larger than this stays put (the physics could not carry it)
SLAB_DEBRIS_MAX = 3000       # most cubes made from the small groups of one blast


def _drop_groups(world, groups, centre):
    """Remove each cut-off group of ground cubes. A group that is big enough becomes one rigid slab (it falls, tips and lands as a
    piece); a small one is removed and its cube centres are returned for the debris."""
    out = []
    budget = SLAB_DEBRIS_MAX
    for idx in groups:
        idx = np.asarray(idx, np.int64)
        n = len(idx)
        if n > SLAB_MAX_CELLS:
            continue
        pts = (idx + 0.5) * _E.CELL
        if n >= SLAB_MIN_CELLS:
            mats = np.where(_stone_mask(world, pts, for_colour=True), _E.STONE, _E.DIRT)
            world.clear_cells(idx)
            if STRUCTS.add_ground_body(world, idx, mats, centre) is not None:
                continue
        elif n > budget:
            continue
        else:
            world.clear_cells(idx)
        budget -= n
        out.append(pts)
    return np.concatenate(out) if out else np.zeros((0, 3))


def _drop_cut_ground(world, centre):
    """Ground the crater cut away from the rest of the map (see _drop_groups)."""
    return _drop_groups(world, world.settle_islands(centre, C4_CRATER, margin=SLAB_SCAN_MARGIN), centre)


# ---- ground that damage has weakened gives way after a moment --------------------------------------
COLLAPSE_HANG = 1.5          # seconds a ledge just over its limit hangs; the further over, the shorter (never under COLLAPSE_MIN)
COLLAPSE_MIN = 0.3
COLLAPSE = []                # pending failures: dict(t seconds left, com of the group, centre of the blast, idx of the group)


def _schedule_collapse(world, centre):
    """After a blast: find ground that damage has cut off or overloaded, and set a timer on each such group."""
    for isl in world.weak_islands(centre, C4_CRATER, margin=SLAB_SCAN_MARGIN):
        if len(isl['idx']) > SLAB_MAX_CELLS:
            continue
        t = float(np.clip(COLLAPSE_HANG / isl['ratio'], COLLAPSE_MIN, COLLAPSE_HANG)) * float(np.random.uniform(0.85, 1.15))
        for e in COLLAPSE:
            if float(np.linalg.norm(e['com'] - isl['com'])) < 3.0:        # the same group again (a second blast): the sooner timer wins
                e['t'] = min(e['t'], t)
                e['idx'] = isl['idx']
                break
        else:
            COLLAPSE.append(dict(t=t, com=isl['com'], centre=np.asarray(centre, np.float64), idx=isl['idx']))


def _spawn_ground_debris(world, pts):
    """Loose cubes for ground that crumbled (no push: they just fall)."""
    if not len(pts):
        return
    stone = _stone_mask(world, pts, for_colour=True)
    for mask, material, shade in ((~stone, _E.DIRT, DEBRIS_DIRT_DARKEN), (stone, _E.STONE, DEBRIS_STONE_DARKEN)):
        if mask.any():
            BODIES.spawn(world, [pts[mask]], 1, size_scale=1.0, rgb=_E._S_RGB[material] * shade, material=material)


def update_collapse(world, dt):
    """Count the timers down. When one runs out the ground is checked again (a later blast or a fall may have changed things) and
    any group still over its limit lets go: a big one as a slab, a small one as cubes. The fall can overload more ground, so
    the blast site is scheduled again."""
    if not COLLAPSE:
        return
    due = []
    for e in COLLAPSE:
        e['t'] -= dt
        if e['t'] <= 0.0:
            due.append(e)
    if not due:
        return
    for e in due:
        COLLAPSE.remove(e)
    for e in due:
        groups = [isl['idx'] for isl in world.weak_islands(e['centre'], C4_CRATER, margin=SLAB_SCAN_MARGIN)
                  if float(np.linalg.norm(isl['com'] - e['com'])) < 4.0]
        if not groups:
            continue                                                  # it was saved (or already fell)
        pts = _drop_groups(world, groups, e['centre'])
        more = _drop_cut_ground(world, e['centre'])                   # damaged scraps that were only hanging on the slab
        _spawn_ground_debris(world, np.concatenate([pts, more]) if len(more) else pts)
        STRUCTS.on_terrain_changed(world, e['centre'], C4_CRATER + SLAB_SCAN_MARGIN * 0.5)
        _schedule_collapse(world, e['centre'])


def c4_explode(world, centre, normal=None, structure_hit=False):
    """Detonate the charge: damage structures, dig a crater, and throw every loose cube near it. On the ground the debris goes
    straight up in a fountain (with a few in a ring); on a wall it is ejected out from the wall along the surface normal."""
    centre = np.asarray(centre, np.float64)
    normal = np.array([0.0, 1.0, 0.0]) if normal is None else np.asarray(normal, np.float64)
    n0 = BODIES.n
    if structure_hit:
        broken = STRUCTS.blast(world, centre, radius=C4_RADIUS, power=BREACH_MULT * 1.3 * math.sqrt(blast_scale()), debris_normal=normal)
        return dict(struct=len(broken), ground=0, debris=max(0, BODIES.n - n0))
    broken = STRUCTS.blast(world, centre, radius=C4_RADIUS, power=1.3 * math.sqrt(blast_scale()), shatter=False)
    # the crater: a ball of ground cubes is removed and turned into loose cubes
    world.dig(centre, C4_CRATER)
    _restore_stone(world, centre, C4_CRATER)                       # the stone under the dirt is unbreakable
    world.add_damage(centre, C4_CRATER)                            # the ground round the crater is weakened (until R)
    groups = [g[~_stone_mask(world, g)] for g in world.take_dug_cells()]
    fallen = _drop_cut_ground(world, centre)               # ground cut loose by the crater falls too: big slabs as one piece, the rest as cubes
    if len(fallen):
        groups.append(fallen)
    _schedule_collapse(world, centre)                      # ground the damage has cut off or overloaded will give way after a moment
    STRUCTS.on_terrain_changed(world, centre, C4_CRATER + 0.5)
    STRUCTS._wake_region(centre, C4_PUSH_R, column_up=C4_CRATER * 1.5)
    dug = sum(len(g) for g in groups)                      # ground cubes scooped out (the scoop), to compare with the debris thrown (the plate)
    rng = np.random.default_rng()
    on_ground = float(normal[1]) > 0.7
    if groups:
        cells = np.concatenate(groups)
        # Sample a fixed number of slots before spawning. The debris setting may
        # thin this sample further, but cannot exceed the awake budget.
        wake = BODIES.awake | (np.linalg.norm(BODIES.pos - centre, axis=1) < C4_PUSH_R + 0.15)
        budget = min(C4_MAX_CUBES, max(0, _E.HARD_AWAKE_CUBES - int(wake.sum())))
        if len(cells) <= C4_EXACT_MAX:
            budget = len(cells)                                    # small enough: every cube scooped out comes out as debris
        if len(cells) > budget:
            cells = cells[rng.choice(len(cells), budget, replace=False)]
        stone = _stone_mask(world, cells, for_colour=True)
        if on_ground:
            cells[:, 1] += 0.08 * _E.CELL
        for mask, material, shade in ((~stone, _E.DIRT, DEBRIS_DIRT_DARKEN),
                                      (stone, _E.STONE, DEBRIS_STONE_DARKEN)):
            BODIES.spawn(world, [cells[mask]], 1, size_scale=(C4_DEBRIS_SCALE if on_ground else 1.0) * C4_DEBRIS_SIZE,
                         rgb=_E._S_RGB[material] * shade, material=material)
    # push: debris leaves the surface along its normal. Ground (normal up): a fountain with a few cubes flung out in a ring.
    # Wall: everything is ejected out from the wall, like a real shaped blast, with a little lift.
    if BODIES.n:
        origin = centre - normal * 0.15
        d = BODIES.pos - origin
        dist = np.linalg.norm(d, axis=1)
        idx = np.nonzero(dist < C4_PUSH_R)[0]
        if len(idx):
            n = len(idx)
            helper = np.array([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.9 else np.array([0.0, 0.0, 1.0])
            t1 = np.cross(normal, helper)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            ang = rng.uniform(0.0, 2.0 * np.pi, n)
            tang = t1[None, :] * np.cos(ang)[:, None] + t2[None, :] * np.sin(ang)[:, None]   # random direction along the surface
            falloff = 1.0 - 0.5 * np.clip(dist[idx] / C4_PUSH_R, 0.0, 1.0)
            if on_ground:
                v = _wave_velocity(BODIES.pos[idx], origin, rng)               # a ripple: cubes land in an expanding ring
            else:
                # a wall: everything leaves along the wall's normal on a rising-then-falling arc (launch angle 25-45 degrees)
                sp = rng.uniform(C4_WALL_MIN, C4_WALL_MAX, n) * falloff
                th = np.radians(rng.uniform(20.0, 60.0, n))
                spread = rng.uniform(0.25, 0.8, n)                                     # a wide cone around the normal, so the cloud fans out instead of leaving as one slab
                horiz = normal[None, :] * (sp * np.cos(th))[:, None] + tang * (spread * sp * np.cos(th))[:, None]
                v = horiz
                v[:, 1] += sp * np.sin(th)
            if on_ground:
                outward = d[idx].copy()
                outward[:, 1] = np.maximum(outward[:, 1], 0.35 * dist[idx] + 1e-6)
                outward /= np.linalg.norm(outward, axis=1, keepdims=True)
                v += outward * (C4_BURST * rng.uniform(0.5, 1.0, n) * falloff)[:, None]
                v += rng.normal(0.0, C4_JITTER * 0.5, (n, 3))
                BODIES.vel[idx] += v
            else:
                # The dug ball of cubes starts packed edge to edge, and the engine's contact friction bleeds sideways speed from
                # every touching cube. Each cube gets its own outward + random velocity, so neighbours pull apart and stop touching.
                out = d[idx] / np.maximum(dist[idx], 1e-6)[:, None]
                v += out * (C4_BURST * rng.uniform(0.4, 1.0, n) * falloff)[:, None]
                v += rng.normal(0.0, C4_JITTER, (n, 3))
                down = _crumble_dir(world, BODIES.pos[idx])                        # roll off the mountain, away from the high ground
                v[:, 0] += down[:, 0] * CRUMBLE_KICK
                v[:, 2] += down[:, 1] * CRUMBLE_KICK
                BODIES.vel[idx] += v * C4_SPEED_MUL
            BODIES.awake[idx] = True
            BODIES.still[idx] = 0.0
    return dict(struct=len(broken), ground=dug, debris=max(0, BODIES.n - n0))


FX_COUNT_MAX = 8.0    # most the effect's cube count is multiplied by at fine cube detail


class BlastFX:
    """Blast effect cubes, drawn the same way as the debris. They are purely visual (no physics bodies, nothing that stays awake) but
    they test the terrain every frame, so they behave in the map they are set off in. One blast is:
      * a white-hot flash and a fireball that bursts out and cools from white to red,
      * a fast jet of burning cubes shot up (the stem), then a wide cap of glowing cubes that rolls out at the top of it,
      * billows of smoke that start glowing and go dark, rise, and gather under any roof, spread along it, linger and thin out,
      * a low ring of dust racing outward over the ground (ground blasts only), and a spray of sparks.
    Fire and dust stop at walls, sparks bounce off floors and walls, smoke cannot pass a ceiling. The cap forms lower where the
    headroom is low (a cave). It looks like a DirtCubes set to the renderer: it has `n` and `instances()`."""
    FIRE, SMOKE, SPARK, DUST = 0, 1, 2, 3
    MAX = 4000             # most effect cubes alive at once (finer cube detail makes more, smaller ones)
    FIREBALL_LIFE = 1.4    # how long the fireball burns, x the original (1.0 = the old, very brief one)
    FIREBALL_LEAN = 1.6    # how far the fireball's cubes lean upward when they burst out (0.9 was the old value)
    FIRE_RISE = 5.5        # upward acceleration of burning cubes in blocks per second squared (2.0 was the old value): bigger lifts the fireball higher
    SMOKE_CLOUD = False    # True brings back the grey smoke cloud after the fireball; False keeps just the flash, fireball, dust ring and sparks
    CLOUD_SCALE = 0.85     # size of the smoke cloud (stem, cap and billows) relative to the full size; 1.0 is the old look
    SHAKE = 0.045          # strongest camera shake (radians) from a blast right next to you; 0 switches it off

    def __init__(self):
        self.shake = 0.0
        self.clear()

    def clear(self):
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.size = np.zeros(0)
        self.age = np.zeros(0)          # negative while waiting for its start delay
        self.life = np.zeros(0)
        self.glow = np.zeros(0)         # seconds a smoke cube stays hot and glowing before it goes grey
        self.kind = np.zeros(0, np.int64)
        self.shake = 0.0

    @property
    def n(self):
        return int((self.age >= 0).sum())

    def spawn(self, world, centre, normal, delay=0.0, viewer=None):
        """Start the effect at a blast. `normal` is the way the surface faces (up for the ground, out from a wall)."""
        rng = np.random.default_rng()
        unit = float(_E.CELL)                                        # every effect cube is sized in ground cubes, so it follows the cube detail
        mult = float(np.clip((0.2 / unit) ** 2, 0.25, FX_COUNT_MAX))   # finer cubes: more of them (to keep the look), up to a cap

        def N(k):
            return max(1, int(round(k * mult)))
        up = np.array([0.0, 1.0, 0.0])
        normal = np.asarray(normal, np.float64)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-9)
        origin = np.asarray(centre, np.float64) + normal * 0.3
        for _ in range(10):                                         # the open air in front of the surface, not inside the ground
            if float(_E._density_many(world, origin.reshape(1, 3))[0]) <= 0.0:
                break
            origin = origin + normal * float(_E.CELL)
        else:
            origin = np.asarray(centre, np.float64) + np.array([0.0, 0.6, 0.0])
        # headroom: how high the cap can form before it meets a roof
        ys = origin[None, :] + up[None, :] * (np.arange(1, 41) * 0.2)[:, None]
        solid = _E._density_many(world, ys) > 0.0
        head = 8.0 if not solid.any() else float(np.argmax(solid)) * 0.2
        cap_h = float(np.clip(head * 0.85, 0.8, 7.0))
        on_ground = float(normal[1]) > 0.7

        def dirs(n, lean, axis=None):                               # random directions on the open side of the surface
            ax = normal if axis is None else axis
            v = rng.normal(size=(n, 3))
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
            d = v @ ax
            v = np.where((d < 0)[:, None], v - 2.0 * d[:, None] * ax, v)
            v = v + ax * lean
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

        def ring(n):                                                # random horizontal directions
            a = rng.uniform(0.0, 2.0 * np.pi, n)
            return np.stack([np.cos(a), np.zeros(n), np.sin(a)], axis=1)

        jet = normal * 0.5 + up
        jet /= np.linalg.norm(jet)
        U = rng.uniform
        parts = []

        bs = blast_scale()                                          # a bigger blast is physically bigger: offsets, sizes and speeds all grow

        def add(kind, p, v, size, life, glow=0.0, wait=0.0):
            n = len(p)
            p = np.asarray(p, np.float64)
            if bs != 1.0:
                p = origin + (p - origin) * bs
                v = np.asarray(v, np.float64) * bs
                size = np.asarray(size, np.float64) * bs
            if kind == self.SMOKE and self.CLOUD_SCALE != 1.0:      # the mushroom cloud (stem, cap, billows) is a bit smaller: heights, spread, speeds and cube sizes
                p = origin + (p - origin) * self.CLOUD_SCALE
                v = np.asarray(v, np.float64) * self.CLOUD_SCALE
                size = np.asarray(size, np.float64) * self.CLOUD_SCALE
            bad = _E._density_many(world, p) > 0.0                  # a start inside rock goes back to the origin
            p[bad] = origin
            parts.append((p, np.asarray(v, np.float64), size, life, np.full(n, float(glow)) if np.isscalar(glow) else glow,
                          np.full(n, kind, np.int64), -(np.full(n, float(delay)) + wait)))

        # white-hot flash: a few huge cubes that pop and are gone
        n = N(4)
        add(self.FIRE, origin + rng.normal(0, 0.2, (n, 3)), np.zeros((n, 3)), unit * U(7.0, 10.0, n), U(0.10, 0.16, n))
        # fireball: big burning cubes bursting out, leaning up
        n = N(30)
        add(self.FIRE, origin + rng.normal(0, 0.35, (n, 3)), dirs(n, self.FIREBALL_LEAN) * U(3.0, 8.0, n)[:, None],
            unit * U(3.0, 6.5, n), U(0.50, 0.95, n) * self.FIREBALL_LIFE)
        if self.SMOKE_CLOUD:                                        # the grey smoke that follows the fireball: stem, cap and billows (off)
            # stem: a violent jet of burning cubes shot straight up from the blast, cooling into smoke as it climbs
            n = N(26)
            p = origin + np.stack([rng.normal(0, 0.25, n), U(0.0, 1.2, n), rng.normal(0, 0.25, n)], axis=1)
            v = jet[None, :] * U(9.0, 16.0, n)[:, None] + rng.normal(0, 0.9, (n, 3))
            add(self.SMOKE, p, v, unit * U(2.1, 4.0, n), U(2.4, 3.6, n), U(0.8, 1.5, n), U(0.0, 0.12, n))
            # cap: glowing cubes rolling outward at the top of the stem, a little later
            n = N(32)
            r = ring(n)
            p = origin + up * cap_h + r * U(0.0, 0.5, n)[:, None]
            v = r * U(3.0, 6.5, n)[:, None] + up * U(0.4, 2.5, n)[:, None]
            add(self.SMOKE, p, v, unit * U(3.6, 6.4, n), U(3.2, 4.8, n), U(0.8, 1.4, n), U(0.28, 0.55, n))
            # billows: the body of the cloud, glowing at first and then dark, filling the column and gathering under a roof
            n = N(36)
            p = origin + np.stack([rng.normal(0, 0.7, n), U(0.4, max(0.5, cap_h * 0.8), n), rng.normal(0, 0.7, n)], axis=1)
            v = ring(n) * U(1.0, 3.0, n)[:, None] + up * U(1.0, 3.5, n)[:, None]
            glow = np.where(rng.random(n) < 0.5, U(0.4, 0.9, n), 0.0)
            add(self.SMOKE, p, v, unit * U(2.8, 5.6, n), U(3.0, 4.6, n), glow, U(0.05, 0.35, n))
        if on_ground:                                               # shockwave: a low ring of dust racing out over the ground
            n = N(28)
            r = ring(n)
            p = origin + up * 0.1 + r * 0.4
            add(self.DUST, p, r * U(7.0, 12.0, n)[:, None], unit * U(1.5, 3.2, n), U(0.45, 0.85, n))
        # sparks: a fast spray
        n = N(60)
        add(self.SPARK, origin + np.zeros((n, 3)), dirs(n, 1.0) * U(6.0, 16.0, n)[:, None], unit * U(0.35, 0.6, n) * 1.0, U(0.8, 1.7, n))

        P, V, S, L, G, K, A = (np.concatenate([a[i] for a in parts]) for i in range(7))
        room = max(0, self.MAX - len(self.age))
        if room <= 0:
            return
        P, V, S, L, G, K, A = P[:room], V[:room], S[:room], L[:room], G[:room], K[:room], A[:room]
        self.pos = np.concatenate([self.pos, P]); self.vel = np.concatenate([self.vel, V])
        self.size = np.concatenate([self.size, S]); self.life = np.concatenate([self.life, L])
        self.glow = np.concatenate([self.glow, G]); self.kind = np.concatenate([self.kind, K])
        self.age = np.concatenate([self.age, A])
        if viewer is not None:
            d = float(np.linalg.norm(np.asarray(viewer, np.float64) - np.asarray(centre, np.float64)))
            self.shake = min(self.SHAKE, self.shake + self.SHAKE * math.sqrt(blast_scale()) / (1.0 + d / (8.0 * blast_scale())))

    def _move(self, world, pos, vel, size, k, dt, step_len):
        """Move cubes one axis at a time in small steps, testing the ground ahead. Returns the smoke that is pressed against a ceiling."""
        fire, smoke, spark, dust = (k == self.FIRE), (k == self.SMOKE), (k == self.SPARK), (k == self.DUST)
        vmax = float(np.abs(vel).max()) if len(vel) else 0.0
        steps = int(min(6, max(1, math.ceil(vmax * dt / step_len))))
        sdt = dt / steps
        roof = np.zeros(len(pos), bool)
        for _ in range(steps):
            for axis in (1, 0, 2):
                new = pos.copy()
                new[:, axis] += vel[:, axis] * sdt
                probe = new.copy()
                probe[:, axis] += np.sign(vel[:, axis]) * size * 0.5
                hit = _E._density_many(world, probe) > 0.0
                pos[~hit] = new[~hit]
                if hit.any():
                    vb = vel[:, axis].copy()
                    vel[hit & spark, axis] = -vb[hit & spark] * 0.4           # sparks bounce
                    vel[hit & smoke, axis] = -vb[hit & smoke] * 0.15          # smoke barely
                    vel[hit & (fire | dust), axis] = 0.0                      # fire and dust stop dead
                    if axis == 1:
                        floor = hit & spark & (vb < 0.0)
                        vel[floor, 0] *= 0.7
                        vel[floor, 2] *= 0.7
                        vel[floor & (np.abs(vel[:, 1]) < 0.6), 1] = 0.0       # a spark that has stopped bouncing lies still
                        up = hit & smoke & (vb > 0.0)
                        roof |= up
                        vel[up, 1] = 0.0                                      # smoke cannot go through the roof
        return pos, vel, roof

    def update(self, world, dt):
        self.shake *= math.exp(-5.0 * min(float(dt), 0.05))
        if not len(self.age):
            return
        dt = min(float(dt), 0.05)
        rng = np.random.default_rng()
        waiting = self.age < 0
        act = np.nonzero(~waiting)[0]
        self.age[waiting] += dt
        if len(act):
            pos, vel, size, k = self.pos[act], self.vel[act], self.size[act], self.kind[act]
            fire, smoke, spark, dust = k == self.FIRE, k == self.SMOKE, k == self.SPARK, k == self.DUST
            # forces: fire slows quickly and drifts up; smoke is thrown fast, slows to a steady rise, and spreads out;
            # sparks fall; dust skims the ground
            vel[fire] *= math.exp(-2.2 * dt)
            vel[fire, 1] += self.FIRE_RISE * dt
            vel[smoke, 1] += 2.0 * dt
            over = smoke & (vel[:, 1] > 1.1)
            vel[over, 1] = 1.1 + (vel[over, 1] - 1.1) * math.exp(-1.6 * dt)     # a fast jet slows down to the steady rise
            vel[smoke, 0] *= math.exp(-0.9 * dt)
            vel[smoke, 2] *= math.exp(-0.9 * dt)
            vel[spark, 1] -= _E.BODY_GRAVITY * dt
            vel[spark] *= math.exp(-0.4 * dt)
            vel[dust, 1] -= 6.0 * dt
            vel[dust, 0] *= math.exp(-2.0 * dt)
            vel[dust, 2] *= math.exp(-2.0 * dt)
            # sparks are tiny and fast, so they take finer steps than the big cubes
            roof = np.zeros(len(act), bool)
            fine = spark
            for grp, step_len in ((fine, 0.12), (~fine, 0.3)):
                if grp.any():
                    pg, vg, rg = self._move(world, pos[grp], vel[grp], size[grp], k[grp], dt, step_len)
                    pos[grp], vel[grp], roof[grp] = pg, vg, rg
            if roof.any():                                          # smoke that reached a ceiling wanders sideways along it
                m = int(roof.sum())
                vel[roof, 0] += rng.normal(0.0, 1.5, m) * dt * 5.0
                vel[roof, 2] += rng.normal(0.0, 1.5, m) * dt * 5.0
            rate = np.where(roof, 0.4, 1.0)                          # trapped smoke lingers: it thins out about 2.5x slower
            self.pos[act], self.vel[act] = pos, vel
            self.age[act] += dt * rate
        keep = self.age < self.life
        if not keep.all():
            for name in ('pos', 'vel', 'size', 'age', 'life', 'glow', 'kind'):
                setattr(self, name, getattr(self, name)[keep])

    def instances(self):
        """Per cube: centre x, y, z, colour r, g, b, and side length. Colours step in bands, like the game's flat shading."""
        act = self.age >= 0
        a, L, k, gl = self.age[act], self.life[act], self.kind[act], self.glow[act]
        t = np.clip(a / np.maximum(L, 1e-6), 0.0, 1.0)
        out = np.empty((len(t), 7), np.float32)
        out[:, :3] = self.pos[act]
        sz = self.size[act]
        def smooth(lo, hi):
            x = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)
        pick = lambda pal, idx: np.array(pal)[idx]
        fire_pal = pick([(3.2, 3.0, 2.0), (3.2, 2.0, 0.55), (2.6, 0.85, 0.18), (0.9, 0.28, 0.12), (0.30, 0.20, 0.18)],
                        np.select([t < 0.18, t < 0.40, t < 0.65, t < 0.85], [0, 1, 2, 3], 4))
        grey_pal = pick([(0.17, 0.16, 0.15), (0.26, 0.25, 0.24), (0.36, 0.35, 0.34)], np.select([t < 0.25, t < 0.60], [0, 1], 2))
        u = np.clip(a / np.maximum(gl, 1e-6), 0.0, 1.5)                  # how far through its glow a smoke cube is
        hot_pal = pick([(3.2, 2.4, 0.9), (2.6, 1.0, 0.22), (1.2, 0.32, 0.10)], np.select([u < 0.30, u < 0.65], [0, 1], 2))
        smoke_pal = np.where(((gl > 0) & (u < 1.0))[:, None], hot_pal, grey_pal)
        spark_pal = pick([(3.2, 2.8, 1.4), (2.8, 1.2, 0.30), (1.6, 0.30, 0.08)], np.select([t < 0.30, t < 0.65], [0, 1], 2))
        dust_pal = pick([(0.62, 0.46, 0.30), (0.50, 0.36, 0.24)], np.select([t < 0.5], [0], 1))
        col = np.where((k == self.FIRE)[:, None], fire_pal,
              np.where((k == self.SMOKE)[:, None], smoke_pal,
              np.where((k == self.SPARK)[:, None], spark_pal, dust_pal)))
        f_fire = (0.6 + 0.4 * np.clip(t * 6.0, 0.0, 1.0)) * (1.0 - 0.85 * smooth(0.35, 1.0))
        f_smoke = (0.45 + 0.7 * np.clip(t * 2.2, 0.0, 1.0)) * (1.0 - smooth(0.75, 1.0))
        f_spark = 1.0 - 0.75 * t
        f_dust = (0.5 + 0.5 * np.clip(t * 5.0, 0.0, 1.0)) * (1.0 - smooth(0.5, 1.0))
        fac = np.where(k == self.FIRE, f_fire, np.where(k == self.SMOKE, f_smoke, np.where(k == self.SPARK, f_spark, f_dust)))
        out[:, 3:6] = col
        out[:, 6] = sz * fac
        return out


class ToolBar(_CachedOverlay):
    """The tools at the bottom of the screen (keys 1-3), the chosen one lit, with its name and status underneath."""
    SLOT = 58
    GAP = 8
    W = 3 * 58 + 2 * 8 + 24
    H = 8 + 58 + 6 + 20 + 18

    def __init__(self):
        super().__init__(self.W, self.H, (_E.WIDTH - self.W) // 2, _E.HEIGHT - self.H - 14)
        self.f_n = _mono(16)
        self.f_s = _mono(14)

    def _icon(self, s, i, x, y):
        cx = x + self.SLOT // 2
        R = lambda col, r: pygame.draw.rect(s, col + (255,), r)
        if i == T_HAMMER:
            R((150, 104, 58), (cx - 3, y + 16, 6, 36))
            R((112, 118, 130), (cx - 15, y + 9, 30, 14))
            R((236, 150, 40), (cx - 19, y + 10, 6, 12))
            R((236, 150, 40), (cx + 13, y + 10, 6, 12))
        elif i == T_CANNON:
            R((72, 76, 86), (cx - 18, y + 20, 34, 12))
            R((46, 48, 56), (cx + 14, y + 17, 6, 18))
            R((46, 48, 56), (cx - 22, y + 18, 8, 16))
            R((150, 104, 58), (cx - 16, y + 34, 28, 6))
            R((110, 76, 44), (cx - 12, y + 38, 12, 12))
            R((110, 76, 44), (cx + 4, y + 38, 12, 12))
        else:
            xx = x - 5
            for dx in (-12, 0, 12):
                bx = xx + 24 + dx
                pygame.draw.rect(s, (120, 22, 26, 255), (bx - 1, y + 10, 17, 40), border_radius=4)
                pygame.draw.rect(s, (214, 52, 52, 255), (bx, y + 10, 13, 40), border_radius=4)
                pygame.draw.rect(s, (245, 110, 100, 255), (bx + 2, y + 12, 3, 36))
            pygame.draw.rect(s, (235, 220, 190, 255), (xx + 8, y + 24, 52, 13))
            tt = self.f_s.render("TNT", True, (30, 22, 20))
            s.blit(tt, (xx + 34 - tt.get_width() // 2, y + 30 - tt.get_height() // 2))

    def _render(self, state):
        tool, placed, bscale, ext = state[0], state[1], state[2], state[3]
        s = self.surf
        pygame.draw.rect(s, (16, 18, 26, 190), (0, 0, self.W, self.H), border_radius=8)
        for i in range(3):
            x, y = 12 + i * (self.SLOT + self.GAP), 8
            on = i == tool
            pygame.draw.rect(s, (30, 34, 46, 230 if on else 160), (x, y, self.SLOT, self.SLOT), border_radius=4)
            pygame.draw.rect(s, UI_ACCENT if on else (70, 76, 92, 255), (x - (2 if on else 0), y - (2 if on else 0),
                             self.SLOT + (4 if on else 0), self.SLOT + (4 if on else 0)), 3 if on else 1, border_radius=5)
            self._icon(s, i, x, y)
            n = self.f_s.render(str(i + 1), True, (255, 255, 255) if on else (170, 176, 190))
            s.blit(n, (x + 4, y + 2))
        name = self.f_n.render(TOOL_NAMES[tool], True, (255, 255, 255))
        s.blit(name, ((self.W - name.get_width()) // 2, 8 + self.SLOT + 6))
        info = f"{placed}/{TNT_MAX}   Blast {bscale:g}x" if tool == T_TNT else ext
        if info:
            lab = self.f_s.render(info, True, (255, 214, 120))
            s.blit(lab, ((self.W - lab.get_width()) // 2, 8 + self.SLOT + 6 + 20))


# ---- the starter hut and material storage ---------------------------------------------------------------------------
# A prebuilt wooden shelter (planks, bark posts, brick foundation and chimney, steel sill) with two open storage bays
# beside it, one for dirt and one for wood, and a crafting bench in the yard. It is a normal fixed Structure, so it
# can be cut, pulled apart and falls like any other. It is rebuilt with the map (start, R, cube detail change).
HUT_X0, HUT_Z0 = -4.0, -12.0        # world corner (lowest x, lowest z) of the hut's grid; its door faces +z, towards the start
HUT_W, HUT_H, HUT_D = 14, 8, 6      # grid size in blocks
HUT_OX, HUT_OZ = 1, 1               # design coordinates start at -1 (the roof overhang), so index = coordinate + 1
HUT = dict(st=None, bench=None)
try:
    _E.TREE_AVOID = tuple(_E.TREE_AVOID) + ((3.0, -8.5, 10.0),)      # no tree grows through the hut
except Exception:
    pass


def _hut_design():
    """The hut as a grid of materials, one entry per block (it is scaled up to the cube detail when built)."""
    PL, WD, BR, ST, DT = _E.PLANKS, _E.WOOD, _E.BRICK, _E.STEEL, _E.DIRT
    g = np.zeros((HUT_W, HUT_H, HUT_D), np.int16)

    def box(x0, x1, y0, y1, z0, z1, m):
        g[x0 + HUT_OX:x1 + HUT_OX + 1, y0:y1 + 1, z0 + HUT_OZ:z1 + HUT_OZ + 1] = m

    def ring(x0, x1, z0, z1, y0, y1, m):                         # four walls, hollow inside
        box(x0, x1, y0, y1, z0, z1, m)
        box(x0 + 1, x1 - 1, y0, y1, z0 + 1, z1 - 1, 0)

    # the hut: x 0..4, z 0..3, door on the +z face
    ring(0, 4, 0, 3, 0, 0, BR)                                   # brick foundation
    ring(0, 4, 0, 3, 1, 3, PL)                                   # plank walls, three blocks high
    for cx in (0, 4):
        for cz in (0, 3):
            box(cx, cx, 1, 3, cz, cz, WD)                        # bark corner posts
    box(1, 1, 0, 2, 3, 3, 0)                                     # the doorway: one wide, two high, down to the ground
    box(3, 3, 2, 2, 3, 3, 0)                                     # a one-block window
    box(3, 3, 1, 1, 3, 3, ST)                                    # steel sill under it
    # the two storage bays: x 6..7 (dirt) and x 9..10 (wood), open to the front
    box(5, 11, 1, 3, 0, 0, PL)                                   # back wall
    box(8, 8, 1, 3, 1, 2, PL)                                    # divider between the bays
    box(8, 8, 0, 0, 0, 3, BR)                                    # brick footing under the divider
    for px in (5, 8, 11):
        for pz in (0, 3):
            box(px, px, 0, 0, pz, pz, BR)
            box(px, px, 1, 3, pz, pz, WD)                        # bark posts on brick feet
    box(6, 7, 0, 0, 0, 2, DT)                                    # the dirt bay's floor is dirt
    box(9, 10, 0, 0, 0, 2, WD)                                   # the wood bay's floor is bark
    box(6, 7, 3, 3, 3, 3, ST)                                    # a steel beam over each opening
    box(9, 10, 3, 3, 3, 3, ST)
    # one stepped roof over everything: a full deck with a plank overhang, then two steps, and a brick chimney
    box(-1, 12, 4, 4, -1, 4, PL)
    box(0, 4, 5, 5, 0, 3, PL)
    box(1, 3, 6, 6, 1, 2, PL)
    box(4, 4, 5, 7, 1, 1, BR)
    return g


def _scaled(g, k):
    return np.repeat(np.repeat(np.repeat(g, k, axis=0), k, axis=1), k, axis=2)


def _stand_structure(world, mat, x, z, cs, wide_x, wide_z, rng):
    """Put a grid on the ground with its lowest corner at (x, z); it rests on the lowest ground under its footprint.
    Returns the structure, or None while the ground there is not loaded."""
    ys = []
    for sx in np.arange(x, x + wide_x + 0.01, 0.5):
        for sz in np.arange(z, z + wide_z + 0.01, 0.5):
            gy = STRUCTS.ground_y(world, float(sx), float(sz))
            if gy is None:
                return None
            ys.append(gy)
    tint = np.ones(mat.shape, np.float32)
    bark = mat == _E.WOOD
    tint[bark] = (0.85 + 0.25 * rng.random(int(bark.sum()))).astype(np.float32)
    st = Structure(mat, np.array([x, min(ys), z]), cs, tint=tint)
    st.refresh_anchors(world)
    STRUCTS.structs.append(st)
    return st


def build_hut(world, res):
    """Stand the hut, its two bays and the bench. Returns True when done, False if the ground is not loaded yet."""
    res = max(1, int(res))
    cs = 1.0 / res
    rng = np.random.default_rng(11)
    st = _stand_structure(world, _scaled(_hut_design(), res), HUT_X0, HUT_Z0, cs, HUT_W, HUT_D, rng)
    if st is None:
        return False
    # the bench: steel legs, a plank top, about a block high, standing in the yard in front of the door
    k = max(1, int(round(res / 2.0)))
    b = np.zeros((4, 2, 2), np.int16)
    b[:, 1, :] = _E.PLANKS
    b[0, 0, :] = b[3, 0, :] = _E.STEEL
    bx, bz = HUT_X0 + 3.0, HUT_Z0 + 6.0
    bench = _stand_structure(world, _scaled(b, k), bx, bz, cs, 4 * k * cs, 2 * k * cs, rng)
    y0 = float(st.frame_pos[1])
    HUT['st'], HUT['bench'] = st, bench
    return True


# ---- scanned objects: .cells files exported by the phone scanner ----------------------------------------------------------
# The SCANS tab of the Tab menu lists them; picking one stands it in front of you. It is an ordinary fixed Structure made of the engine's own cells
# (cell size follows cube detail), so it can be cut, cracked, blasted and falls like the trees and the hut.
_HERE = os.path.dirname(os.path.abspath(globals().get('__file__') or sys.argv[0] or '.'))
SCAN_DIRS = [_HERE, os.getcwd(), os.path.join(_HERE, 'scans'), os.path.join(os.path.expanduser('~'), 'Downloads', 'scan_exports')]   # the game's own folder first, so a .cells file dropped beside voidcraft.py is found
SCAN_MAT = {'wood': 'WOOD', 'dirt': 'DIRT', 'stone': 'STONE', 'leaf': 'PLANKS', 'concrete': 'CONCRETE_WHITE'}   # leaf has no engine material: light planks


def newest_scan():
    best = None
    for d in SCAN_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith('.cells'):
                    q = os.path.join(d, f)
                    if best is None or os.path.getmtime(q) > os.path.getmtime(best):
                        best = q
    return best


def load_scan(path, res):
    """Read a voidcraft.cells.v1 file into a material grid (x, y up, z depth) at `res` cells per block, cropped to the object."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    cells = np.asarray(d['cells'], np.int64).reshape(-1, 4)
    if not len(cells):
        raise ValueError('the file has no cells')
    src = max(1, int(d.get('cubes_per_block', 5)))
    lut = np.array([getattr(_E, SCAN_MAT.get(str(n), 'STONE')) for n in d['materials']], np.int16)
    x0, y0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    tmax = max(1, int(cells[:, 2].max()))
    g = np.zeros((int(cells[:, 0].max()) - x0 + 1, int(cells[:, 1].max()) - y0 + 1, 2 * tmax), np.int16)
    for x, y, t, m in cells.tolist():
        t = max(1, t)
        g[x - x0, y - y0, tmax - t:tmax + t] = lut[m]          # a column is a stack of cells centred on the middle of the depth
    if res != src:                                                # other cube detail: resample to that cell size
        ix, iy, iz = (np.minimum((np.arange(max(1, round(n * res / src))) * src / res).astype(np.int64), n - 1) for n in g.shape)
        g = g[np.ix_(ix, iy, iz)]
    return np.ascontiguousarray(g)


def spawn_scan(world, player, res, path=None):
    """Stand a scan (the newest one unless `path` is given) 4 blocks ahead of the player, facing them. Returns the message to show."""
    if path is None:
        path = newest_scan()
    if path is None:
        return 'No .cells file found: put one in the same folder as voidcraft.py'
    res = max(1, int(res))
    g = load_scan(path, res)
    fx, fz = math.sin(player.yaw), -math.cos(player.yaw)
    if abs(fx) > abs(fz):                                         # looking along x: turn the object a quarter so its face points back at you
        g = g.transpose(2, 1, 0)
        if fx < 0:
            g = g[:, :, ::-1]
        d = (1.0 if fx > 0 else -1.0, 0.0)
    else:
        if fz > 0:
            g = g[::-1]
        d = (0.0, 1.0 if fz > 0 else -1.0)
    g = np.ascontiguousarray(g)
    cs = 1.0 / res
    wx, wz = g.shape[0] * cs, g.shape[2] * cs
    cx, cz = float(player.pos[0]) + d[0] * 4.0, float(player.pos[2]) + d[1] * 4.0
    st = _stand_structure(world, g, cx - wx / 2, cz - wz / 2, cs, wx, wz, np.random.default_rng())
    if st is None:
        return 'The ground there is not loaded yet'
    return 'Scan placed: %s (%d cells, %.1f x %.1f blocks)' % (os.path.basename(path), int((g > 0).sum()), wx, g.shape[1] * cs)


class ScanLibrary:
    """Every .cells file in SCAN_DIRS, newest first. poll() looks again about once a second, so a scan exported while the game
    is running shows up in the import menu by itself. A file is read once and kept until its size or time changes."""
    def __init__(self):
        self.items = []
        self.opened = set()          # paths that were there when the menu was opened: anything else is NEW
        self._meta = {}
        self._t = -99.0
        self._sig = None

    def _read(self, path, st):
        key = (st.st_mtime_ns, st.st_size)
        m = self._meta.get(path)
        if m is not None and m['key'] == key:
            return m
        m = dict(key=key, ok=False, err='')
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            cells = np.asarray(d['cells'], np.int64).reshape(-1, 4)
            if not len(cells):
                raise ValueError('the file has no cells')
            cpb = max(1, int(d.get('cubes_per_block', 5)))
            tmax = max(1, int(cells[:, 2].max()))
            m.update(ok=True, n=int(2 * np.maximum(cells[:, 2], 1).sum()),
                     w=float(cells[:, 0].max() - cells[:, 0].min() + 1) / cpb,
                     h=float(cells[:, 1].max() - cells[:, 1].min() + 1) / cpb, dp=2.0 * tmax / cpb)
        except Exception as err:
            m['err'] = (str(err) or err.__class__.__name__)[:70]
        self._meta[path] = m
        return m

    def poll(self, force=False):
        """Look at the folders again (at most once a second). Returns True when the list changed."""
        now = time.perf_counter()
        if not force and now - self._t < 1.0:
            return False
        self._t = now
        found = {}
        for d in SCAN_DIRS:
            try:
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    if f.lower().endswith('.cells'):
                        q = os.path.realpath(os.path.join(d, f))
                        if q not in found:
                            try:
                                found[q] = os.stat(q)
                            except OSError:
                                pass
            except OSError:
                pass
        items = []
        for q, st in found.items():
            m = self._read(q, st)
            items.append(dict(path=q, name=os.path.basename(q), folder=os.path.basename(os.path.dirname(q)) or q, mtime=st.st_mtime,
                              ok=m['ok'], err=m['err'], n=m.get('n', 0), w=m.get('w', 0.0), h=m.get('h', 0.0), dp=m.get('dp', 0.0)))
        items.sort(key=lambda i: -i['mtime'])
        sig = [(i['path'], i['mtime'], i['ok']) for i in items]
        changed = sig != self._sig
        self._sig, self.items = sig, items
        return changed


def _age_text(secs):
    secs = max(0, int(secs))
    if secs < 5:
        return 'just now'
    if secs < 90:
        return '%d s ago' % secs
    if secs < 5400:
        return '%d min ago' % round(secs / 60.0)
    if secs < 129600:
        return '%d h ago' % round(secs / 3600.0)
    return '%d days ago' % round(secs / 86400.0)


SCAN_LIB = ScanLibrary()       # the scanned objects the SCANS tab of the Tab menu lists


CUBE_DETAIL_MAX = 50      # the top of the Cube detail slider


def apply_launch_args(settings, argv=None):
    """Launch commands typed in the launcher (right-click a game in Quick launch). They arrive as words in sys.argv:
        cubedetail-max   the finest ground cubes the slider allows
        cubedetail-N     N cubes per block (1 to 50)
    Anything else is ignored with a note in the log."""
    for word in (sys.argv[1:] if argv is None else argv):
        cmd = word.strip().lower()
        if cmd == 'cubedetail-max':
            settings['dirt_cubes'] = CUBE_DETAIL_MAX
        elif cmd.startswith('cubedetail-'):
            try:
                settings['dirt_cubes'] = max(1, min(CUBE_DETAIL_MAX, int(cmd[len('cubedetail-'):])))
            except ValueError:
                print("launch command not understood (cubedetail-max or cubedetail-1..50): " + word)
        elif cmd:
            print("unknown launch command: " + word)


def run():
    set_blast_scale(BLAST['i'])                       # apply the locked blast size (0.75) to every blast constant before anything uses them
    # Open the window centred on the primary monitor instead of wherever the OS puts it.
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.mixer.pre_init(44100, -16, 2, 512)         # small buffer: sounds start without lag
    pygame.init()
    _fit_window_to_display()
    pygame.display.set_mode((_E.WIDTH, _E.HEIGHT), DOUBLEBUF | OPENGL | (getattr(pygame, "HIDDEN", 0) if _UI_LOADING else 0))
    pygame.display.set_caption("VoidCraft")
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    loader = LoadingScreen()
    loader.step("Synthesising sounds", 0.01, "Every sound effect is built from tones and noise: no sound files")
    SFX.init()
    loader.step("Starting GPU compute (CuPy)", 0.02, "Running a one-kernel self-test; falls back to the CPU if CUDA is missing")
    gpu_status = _gpu_init()
    print(gpu_status)
    loader.step("Compiling shaders: menus and settings panel", 0.08, gpu_status)
    settings = default_settings()
    apply_launch_args(settings)                   # launch commands from the launcher, e.g. cubedetail-max
    settings['struct_res'] = settings['dirt_cubes']   # structures, trees, scans and debris all use the same cube detail as the ground
    menu = TabMenu(settings)
    settings_panel = SettingsPanel()
    reset_panel = ResetPanel()
    reset_open = False           # R: the small reset menu is up (world frozen behind it, like the Tab menu)

    def place_scan(i):
        """Tab menu, SCANS tab: stand scan number i in front of the player and close the menu so it can be seen."""
        it = SCAN_LIB.items[i] if 0 <= i < len(SCAN_LIB.items) else None
        if it is None:
            return
        SFX.play('ui_click')
        if not it['ok']:
            inv_note('Cannot read %s: %s' % (it['name'], it['err']), 5.0)
            return
        try:
            inv_note(spawn_scan(world, player, settings['struct_res'], it['path']), 5.0)
        except Exception as err:
            inv_note('Scan failed: %s' % err, 6.0)
            print('scan failed:', err)
        menu.open = False
        menu.dirty = True
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
    loader.step("Compiling shaders: terrain, and starting mesh workers", 0.22, "%d background processes build chunk meshes" % MESH_WORKERS)
    set_ground_res(settings['dirt_cubes'])        # the ground is made of the same cubes as the structures
    world = World()
    renderer = Renderer()
    SPAWN_POS = (0.5, float(terrain_height(0.0, 0.0)) + 2.0, 0.5)
    player = PlayerController(SPAWN_POS)
    last_centre = None
    last_caption = None
    last_cpu_ms = None
    paused = False
    show_grid = False            # X: cube grid lines on the cubes
    show_dmg = False             # G: colour the damaged ground (and structures) by how close it is to giving way
    dmg_views, dmg_key = [], None
    debris_tint = DebrisTint()
    loader.step("Loading tools", 0.30)
    arsenal = Arsenal()
    load_ms = draw_ms = 0.0      # per-frame timings for the benchmark breakdown
    loader.step("Compiling shaders: sky, chunk grid, debris", 0.40)
    backdrop = Backdrop()
    scene = SceneTarget()
    post = PostFX()
    frame = FrameTarget()
    applied_fs = False           # whether the window is fullscreen right now
    applied_vsync = None         # what the screen refresh wait was last set to, so it is only set when the option changes
    grid = ChunkGrid()
    body_gl = DirtBodyRenderer()
    fx_gl = DirtBodyRenderer()     # draws the blast effect cubes the same way
    blast_fx = BlastFX()
    shake_rng = np.random.default_rng()
    loader.step("Compiling shaders: structures and shadow map", 0.55)
    struct_gl = StructureRenderer()
    shadow = ShadowMap()
    renderer.shadow_tex = shadow.tex
    struct_gl.shadow_tex = shadow.tex
    gfx_last = quality_snapshot(settings)
    gfx_sig = tuple(settings[k] for k in QUALITY_KEYS)
    loader.step("Building HUD: crosshair", 0.66)
    crosshair = Crosshair()
    target_label = TargetLabel()
    survival = Survival()
    survival_hud = SurvivalHud()
    toolbar = ToolBar()
    pending_hut = True           # the hut is stood on the ground once that ground has loaded
    c4_charges = []             # planted TNT, up to TNT_MAX: dict(pos, normal, struct). Left click plants one, right click sets them all off
    tool = T_TNT                # 1-5: the chosen tool
    lmb_held = False            # the left button is down (menus off)
    tool_extra = []             # world prims the hand tools add this frame
    FX_REF[0] = blast_fx
    charging = False            # left button held: the throw arc is showing
    charge_t = 0.0              # seconds the button has been held
    tnt_flights = []            # thrown charges still in the air: dict(pts, land, t)
    aim_arc = []                # the arc's lines for this frame
    c4_flash = None             # (centre, seconds remaining) for a short blast-ring effect
    inv_hud = InvHud()
    perf = PerfOverlay()                # G: the monitor panel (H: compact / detailed)

    def mon_counts():
        _soon = min((e['t'] for e in COLLAPSE), default=0.0)
        return dict(cubes_awake=int(BODIES.awake.sum()) if BODIES.n else 0, cubes=int(BODIES.n),
                    slabs_awake=sum(1 for b in STRUCTS.bodies if not b.asleep), slabs=len(STRUCTS.bodies),
                    structs=len(STRUCTS.structs), ledges=len(COLLAPSE), next_t=max(0.0, _soon), dmg_chunks=len(world.gdmg))
    pending_trees = True
    tree_t = 0.0
    built_res = settings['dirt_cubes']   # cube detail the layout was last built at
    last_strength = 100
    menu_frozen = False          # True once the world has been frozen behind the open menu
    counted = False              # True when the last pass was a real game frame (not the menu)
    clock = pygame.time.Clock()
    # What was on screen last frame, so a menu opening, a pause or a toggle can make its sound once.
    sfx_seen = dict(menu=False, paused=False, grid=False,
                    hud=settings['show_hud'])

    # Generate the terrain around the player now, nearest chunks first, so the game opens on a finished world.
    # The map is a fixed square of chunks around the spawn: it is generated once here and never grows or unloads.
    settings['render_distance'] = MAP_RADIUS
    settings['cache_chunks'] = True
    rd0 = MAP_RADIUS
    pcx, pcz = math.floor(player.pos[0] / CHUNK_SIZE), math.floor(player.pos[2] / CHUNK_SIZE)
    MAP_X, MAP_Z = (pcx + 0.5) * CHUNK_SIZE, (pcz + 0.5) * CHUNK_SIZE
    todo = sorted(((cx, cz) for cx in range(pcx - rd0, pcx + rd0 + 1) for cz in range(pcz - rd0, pcz + rd0 + 1)),
                  key=lambda c: (c[0] - pcx) ** 2 + (c[1] - pcz) ** 2)
    CACHE.terrain_on, CACHE.mesh_on = bool(settings['cache_terrain']), bool(settings['cache_meshes'])
    _c0 = [getattr(CACHE, k, 0) for k in ('t_hit', 't_new', 'm_hit', 'm_new')]      # the cache's counters now, to report what this load used
    for i, (cx, cz) in enumerate(todo):
        if i % 12 == 0:
            loader.step("Generating terrain", 0.70 + 0.15 * i / len(todo), "chunk %d of %d" % (i, len(todo)))
        world.ensure_chunk(cx, cz)
    if getattr(CACHE, 'terrain_on', False):                       # the stage's summary line in the launcher's log
        _th = getattr(CACHE, 't_hit', 0) - _c0[0]
        loader.step("Generating terrain", 0.85, "> cache: %d of %d chunks loaded from disk, %d generated%s. Cubes: %d per block (asked for %d)"
                    % (_th, len(todo), len(todo) - _th, " and saved" if len(todo) - _th else "", int(_E.GS), int(settings['dirt_cubes'])))
    else:
        loader.step("Generating terrain", 0.85, "> cache off: all %d chunks generated. Cubes: %d per block (asked for %d)" % (len(todo), int(_E.GS), int(settings['dirt_cubes'])))
    # Generating the chunks only makes the cube data. Meshing them and uploading the meshes to the GPU used to happen in the
    # first frames after the window opened, so the map popped in while you watched. Do that work here, behind the loading
    # screen: draw the map (into the hidden window) until every chunk has its mesh on the GPU.
    renderer.lod = int(settings['lod'])                               # the detail the game loop will ask for, so nothing is rebuilt later
    t_pre = time.perf_counter()
    ring_n = max(1, (2 * (MAP_RADIUS - 1) + 1) ** 2)
    while time.perf_counter() - t_pre < 180.0:                        # safety limit: never hang the loading screen
        pygame.event.pump()
        renderer.draw(world, player.eye, player.yaw, player.pitch, radius=MAP_RADIUS - 1, fov=settings['fov'])
        got = len(renderer.gpu)
        loader.step("Building terrain meshes", 0.85 + 0.14 * min(1.0, got / ring_n), "chunk %d of %d" % (min(got, ring_n), ring_n))
        if not renderer.mesher.pending and not renderer._waiting and not renderer._touched:
            break
        time.sleep(0.005)
    if getattr(CACHE, 'mesh_on', False):
        time.sleep(0.2)                                               # let the last cache files finish writing so they are counted
        _mh, _mn = getattr(CACHE, 'm_hit', 0) - _c0[2], getattr(CACHE, 'm_new', 0) - _c0[3]
        loader.step("Building terrain meshes", 0.99, "> cache: %d meshes loaded from disk, %d built%s" % (_mh, _mn, " and saved" if _mn else ""))
    else:
        loader.step("Building terrain meshes", 0.99, "> cache: mesh cache is off (or this engineui.py has none), all meshes built")
    loader.step("Starting the game", 1.0)
    SFX.play('level_up', 0.8, gap=0.0)                              # loading finished
    gl.glClearColor(SKY_RGB[0], SKY_RGB[1], SKY_RGB[2], 1.0)        # the loading screen changed the clear colour: put the sky back
    _window_ready()                                                 # engineui did the loading screen: show the window now

    running = True
    while running:
        # With the menu open the world is frozen, so idle at a low rate instead of the FPS limit.
        dt = clock.tick(30 if (menu.open or reset_open) else settings['fps_limit']) / 1000.0    # 0 = uncapped
        work_start = time.perf_counter()                     # everything after the cap's sleep
        if counted:
            PROF.push(dt * 1000.0, last_cpu_ms or 0.0)
        else:
            PROF.cur = {}
        counted = False
        t_in = time.perf_counter()
        settings['render_distance'] = MAP_RADIUS                     # fixed map: the whole thing stays loaded
        settings['cache_chunks'] = True
        rd = MAP_RADIUS
        if settings['fullscreen'] != applied_fs:
            want = bool(settings['fullscreen'])
            ok = True
            try:
                if want:
                    ok = frame.enable()                                  # the frame is drawn offscreen first, then scaled to the screen
                if ok:
                    ok = pygame.display.toggle_fullscreen() != 0
                    if not ok and want:
                        frame.disable()
                if ok and not want:
                    frame.disable()
            except Exception as err:
                print(f"Fullscreen could not be switched: {err}")
                ok = False
                if frame.active:
                    frame.disable()
            if ok:
                applied_fs = want
            else:
                settings['fullscreen'] = applied_fs
                print("Fullscreen could not be switched")
            menu_frozen = False                                          # the picture behind the menu is drawn again at the new size
            menu.dirty = True
        if settings['vsync'] != applied_vsync:
            applied_vsync = settings['vsync']
            if not _set_vsync(applied_vsync) and applied_vsync:
                print("VSync could not be switched on here")

        SFX.tick()
        SFX.set_listener(player.eye, player.yaw)
        events = pygame.event.get()
        pending_reset = False
        press_click = release_click = det_click = False
        for e in events:
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE and not menu.open and not reset_open):
                running = False
            elif reset_open:                                         # the R menu swallows every other event while it is up
                close_reset = False
                if e.type == KEYDOWN and e.key in (K_r, K_ESCAPE):
                    close_reset = True
                elif e.type == MOUSEBUTTONDOWN and e.button == 1:
                    act = reset_panel.mouse_down(e.pos)
                    if act:
                        SFX.play('ui_click')
                    if act == 'damage':
                        pending_reset = True
                    elif act == 'spawn':
                        player.pos[:] = SPAWN_POS
                        player.vel[:] = 0.0
                        player.on_ground = False
                        player.yaw = player.pitch = 0.0
                        inv_note('Back at spawn', 1.5)
                        close_reset = True
                    elif act == 'close':
                        close_reset = True
                if close_reset:
                    reset_open = False
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
            elif e.type == KEYDOWN and (e.key == K_TAB or (e.key == K_ESCAPE and menu.open)):
                menu.open = not menu.open
                menu.dirty = True
                if menu.open:                                        # the SCANS tab starts from a fresh look at the folders
                    SCAN_LIB.poll(force=True)
                    CACHE.refresh()
                    SCAN_LIB.opened = {i['path'] for i in SCAN_LIB.items}
                    menu.scan_sel = menu.scan_scroll = 0
                pygame.mouse.set_visible(menu.open)
                pygame.event.set_grab(not menu.open)
            elif menu.open and e.type == MOUSEBUTTONDOWN and e.button in (4, 5):
                if menu.on_scans:
                    menu.scan_wheel(-1 if e.button == 4 else 1)
            elif menu.open and e.type == MOUSEBUTTONDOWN and e.button == 1:
                act = settings_panel.mouse_down(menu, e.pos)
                SFX.play('ui_click')
                if act == 'exit':
                    running = False
                elif act == 'reset_damage':
                    pending_reset = True
                elif act == 'clear_cache':
                    CACHE.clear()
                    menu.dirty = True
                    inv_note('Cache deleted', 2.5)
                elif isinstance(act, tuple) and act[0] == 'place':
                    place_scan(act[1])
            elif menu.open and e.type == MOUSEBUTTONUP and e.button == 1:
                menu.drag = None
            elif menu.open and e.type == MOUSEMOTION and menu.drag is not None:
                settings_panel.mouse_drag(menu, e.pos)
                SFX.play('ui_click', 0.4, gap=0.07)                 # a slider being dragged ticks
            elif menu.open and e.type == MOUSEMOTION and menu.on_scans:
                settings_panel.mouse_move(menu, e.pos)
            elif not menu.open and e.type == MOUSEBUTTONDOWN and e.button == 1:
                press_click = True
            elif not menu.open and e.type == MOUSEBUTTONUP and e.button == 1:
                release_click = True
            elif not menu.open and e.type == MOUSEBUTTONDOWN and e.button == 3:
                det_click = True
            # (scroll wheel tool switching is switched off for now: TNT is the only tool)
            elif e.type == KEYDOWN and e.key in (K_1, K_2, K_3) and not menu.open and not reset_open:
                _nt = e.key - K_1
                if _nt != tool:
                    tool = _nt
                    charging = False
                    TS['swing'] = None
                    inv_note(TOOL_NAMES[tool] + ': ' + TOOL_HINTS[tool], 3.0)
            elif e.type == KEYDOWN and e.key == K_x:
                show_grid = not show_grid
            elif e.type == KEYDOWN and e.key == K_v and not menu.open:
                player.toggle_fly()                                  # V: fly (Space up, Ctrl down)
                inv_note('Flying on: Space up, Ctrl down' if player.flying else 'Flying off', 2.0)
            elif e.type == KEYDOWN and e.key == K_g and not menu.open:
                show_dmg = not show_dmg
                inv_note('Damage overlay ' + ('on' if show_dmg else 'off'), 1.5)
            elif e.type == KEYDOWN and e.key == K_h and not menu.open and show_dmg:
                perf.detail = not perf.detail                                # H: compact or detailed monitor
            elif menu.open:
                if e.type == KEYDOWN and e.key in (K_UP, K_DOWN, K_LEFT, K_RIGHT, K_LEFTBRACKET, K_RIGHTBRACKET, K_RETURN, K_KP_ENTER):
                    SFX.play('ui_click')
                if e.type == KEYDOWN and e.key == K_UP:
                    menu.scan_move(-1) if menu.on_scans else menu.move(-1)
                elif e.type == KEYDOWN and e.key == K_DOWN:
                    menu.scan_move(1) if menu.on_scans else menu.move(1)
                elif e.type == KEYDOWN and e.key == K_LEFTBRACKET:
                    menu.switch_tab(-1)
                elif e.type == KEYDOWN and e.key == K_RIGHTBRACKET:
                    menu.switch_tab(1)
                elif e.type == KEYDOWN and e.key == K_LEFT:
                    menu.adjust(-1)
                elif e.type == KEYDOWN and e.key == K_RIGHT:
                    menu.adjust(1)
                elif e.type == KEYDOWN and e.key in (K_RETURN, K_KP_ENTER):
                    if menu.on_scans:
                        place_scan(menu.scan_sel)
                    else:
                        menu.activate()
            elif e.type == KEYDOWN and e.key == K_r:
                reset_open = True                                    # R: open the small reset menu (damage / spawn)
                SFX.play('ui_click')
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
            elif e.type == MOUSEMOTION:           # look around works while paused too
                player.yaw += e.rel[0] * 0.002
                player.pitch = max(-1.5, min(1.5, player.pitch - e.rel[1] * 0.002))

        PROF.add('input', (time.perf_counter() - t_in) * 1000.0)

        for _tab, _rows in SETTINGS_TABS:                            # the slider is live only while 'Change cube detail' is on
            for _r in _rows:
                if _r.get('key') == 'dirt_cubes':
                    _r['locked'] = not settings['cube_unlock']
        CACHE.terrain_on, CACHE.mesh_on = bool(settings['cache_terrain']), bool(settings['cache_meshes'])
        if not settings['cube_unlock']:
            settings['dirt_cubes'] = built_res                       # locked: a stray drag can never change it
        settings['struct_res'] = settings['dirt_cubes']              # the cube detail sets how fine the structures are cut
        if settings['struct_res'] != built_res:                      # cube detail changed: rebuild the ground and everything on it
            built_res = settings['struct_res']
            STRUCTS.clear()
            BODIES.clear()
            pending_trees = pending_hut = True
            HUT['st'] = HUT['bench'] = None
            TREES.clear()
            set_ground_res(built_res)                                # the ground follows: new cube size, map regenerated
            for _k, (_v, _f, _s) in list(renderer.mesher.pending.items()):
                _f.cancel()
            renderer.mesher.pending.clear()
            renderer._waiting.clear() if hasattr(renderer._waiting, "clear") else None
            for _k in list(renderer.gpu):
                renderer.evict(_k)
            with CACHE.lock:
                CACHE.mesh.clear()
                CACHE.mesh_bytes = 0
            world.reset_all()
            COLLAPSE.clear()
            last_centre = None
        act_kb = menu.take_action()                                      # a button pressed with Enter
        if act_kb == 'clear_cache':
            CACHE.clear()
            menu.dirty = True
            inv_note('Cache deleted', 2.5)
        if pending_reset or act_kb == 'reset_damage':
            SFX.play('totem', 0.9, gap=0.0)
            world.reset_damage()
            COLLAPSE.clear()
            arsenal.reset()
            c4_charges.clear()
            tnt_flights.clear()
            charging = False
            tool_reset()
            blast_fx.clear()
            c4_flash = None
            STRUCTS.clear()
            pending_trees = pending_hut = True
            HUT['st'] = HUT['bench'] = None
            TREES.clear()
            if not reset_open:                                       # the R menu stays up after a reset so the other button is still there
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
            BODIES.clear()
            world.take_dug_cells()

        # Sounds for things that just changed: windows opening, pausing, toggles.
        if menu.open != sfx_seen['menu']:
            SFX.play('door_open' if menu.open else 'door_close', gap=0.0)
        if paused != sfx_seen['paused']:
            SFX.play('bucket_fill' if paused else 'door_close', gap=0.0)
        if show_grid != sfx_seen['grid'] or settings['show_hud'] != sfx_seen['hud']:
            SFX.play('ui_click')
        sfx_seen.update(menu=menu.open, paused=paused,
                        grid=show_grid, hud=settings['show_hud'])

        modal = menu.open or reset_open   # a menu is up: world frozen
        if modal:
            charging = False                                          # a menu opened while the button was held: the throw is cancelled
            lmb_held = False

        # Graphics tab: a preset copies its values over the settings, and a hand change switches the preset.
        gfx_last = sync_quality(settings, gfx_last)
        renderer.lod = int(settings['lod'])
        shadow.set_quality(settings['shadow_quality'], settings['shadow_update'])
        renderer.shadow_texel = struct_gl.shadow_texel = 1.0 / shadow.res
        sig_now = tuple(settings[k] for k in QUALITY_KEYS)
        gfx_tab_open = modal and menu.tab == GFX_TAB
        if gfx_tab_open and (sig_now != gfx_sig or renderer.mesher.pending or renderer._waiting):
            menu_frozen = False          # live preview: the picture behind the panel is drawn again until the new detail has arrived
        gfx_sig = sig_now

        # Menu open and nothing changed in it: keep the frozen frame on screen and skip the work.
        scans_live = menu.open and menu.on_scans and SCAN_LIB.poll()    # the SCANS tab looks for new scan files about once a second
        menu_changed = scans_live or any(e.type != MOUSEMOTION or menu.drag is not None or reset_open or (menu.open and menu.on_scans) for e in events)   # the R menu redraws on mouse moves for its hover
        if modal and menu_frozen and not menu_changed:
            last_cpu_ms = None
            continue

        apply_wobble(settings['wobble'])
        STRENGTH['k'] = settings['strength'] / 100.0
        if settings['strength'] != last_strength:                    # new strength: look at every structure again
            last_strength = settings['strength']
            for st_ in STRUCTS.structs:
                st_.stress_dirty = True
                st_.dirty_all = True
                st_.wait = 0.0
        GFX['physics_device'] = 'gpu' if (settings['gpu_physics'] and _E.GPU_OK) else 'cpu'
        CONFIG['debris_on'] = settings['debris_on']
        CONFIG['debris_amount'] = settings['debris_amount'] / 100.0
        CONFIG['dirt_per_block'] = settings['dirt_cubes']
        if not settings['debris_on']:
            BODIES.clear()
        GT.enabled = show_dmg                                        # GPU timer queries only while the G monitor is open
        GT.poll()
        keys = pygame.key.get_pressed()
        t_load = time.perf_counter()
        if not modal:
            with Timed('chunks', gpu=False):
                load_around(world, MAP_X, MAP_Z, radius=MAP_RADIUS, budget=32)
                if SPAWN_TREES and pending_trees and STRUCTS.ground_y(world, 0.0, 0.0) is not None:
                    build_trees(world, settings['struct_res'])
                    pending_trees = not TREES
                if SPAWN_HUT and pending_hut and STRUCTS.ground_y(world, 0.0, 0.0) is not None:
                    pending_hut = not build_hut(world, settings['struct_res'])
        elif gfx_tab_open:                                           # live preview: a longer render distance fills in while the panel is up
            load_around(world, MAP_X, MAP_Z, radius=MAP_RADIUS, budget=32)
        load_ms = (time.perf_counter() - t_load) * 1000.0

        if not modal:
            yaw = player.yaw
            fwd = np.array([math.sin(yaw), 0.0, -math.cos(yaw)])
            right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
            wish = np.zeros(3)
            if keys[K_w]: wish += fwd
            if keys[K_s]: wish -= fwd
            if keys[K_d]: wish += right
            if keys[K_a]: wish -= right
            if player.flying:
                if keys[K_SPACE]: wish[1] += 1.0
                if keys[K_LCTRL]: wish[1] -= 1.0
            n = np.linalg.norm(wish)
            if n > 0:
                wish /= n                                            # keys always walk at full speed
            with Timed('player', gpu=False):
                player.update(world, dt, wish, keys[K_SPACE], keys[K_LSHIFT] and not (SURVIVAL_ON and survival.parched))
            if SURVIVAL_ON:
                survival.update(dt, bool(keys[K_LSHIFT]) and n > 0 and not player.flying,
                                bool(keys[K_e]), bool(keys[K_b]), False)
            if not paused:                                           # the simulation freezes; the player keeps moving
                with Timed('structs', gpu=False):
                    STRUCTS.update(world, dt)
                    update_collapse(world, dt)
                with Timed('debris', gpu=False):
                    update_debris(world, dt, settings['cube_sleep_time'])
            with Timed('weapon', gpu=False):
                tree_t += dt
                if tree_t > 0.3:
                    n_trees = len(TREES)
                    check_chopped(tree_t)
                    if len(TREES) < n_trees:                             # a tree fell: drop the seeds and apples
                        INV['seeds'] = INV['apples'] = 0
                    tree_t = 0.0
                INV['msg_t'] = max(0.0, INV['msg_t'] - dt)
                arsenal.update(world, player, dt)
                if press_click:
                    lmb_held = True
                if release_click:
                    lmb_held = False
                if tool != T_TNT:
                    charging = False
                tool_extra = tool_step(world, player, tool, dt, press_click and tool != T_TNT, release_click, lmb_held, det_click)
                if tool == T_TNT and press_click:
                    if len(c4_charges) + len(tnt_flights) >= TNT_MAX:
                        inv_note(f'All {TNT_MAX} TNT used: right click to detonate', 2.0)
                    else:
                        charging, charge_t = True, 0.0                   # hold: the arc shows and the throw gets stronger
                if charging:
                    charge_t += dt
                    pts, land = throw_path(world, _lp(player, *THROW_HAND), throw_velocity(player, charge_t / THROW_CHARGE_S))
                    if release_click:                                    # let go: throw it along the path the arc showed
                        charging = False
                        tnt_flights.append(dict(pts=pts, land=land, t=0.0))
                        SFX.play('ender_pearl', 0.5, gap=0.1)
                        aim_arc = []
                    else:
                        aim_arc = arc_prims(pts, land)
                else:
                    aim_arc = []
                for fl in tnt_flights[:]:
                    fl['t'] += dt
                    if fl['t'] >= (len(fl['pts']) - 1) * THROW_STEP:     # it has landed
                        tnt_flights.remove(fl)
                        if fl['land'] is None:
                            inv_note('The TNT flew off and was lost', 2.5)
                        else:
                            c4_charges.append(dict(pos=fl['land']['pos'].copy(), normal=fl['land']['normal'], struct=fl['land']['struct']))
                            inv_note(f'TNT landed ({len(c4_charges)}/{TNT_MAX}): right click to detonate', 3.0)
                            SFX.play('ui_click', 0.8)
                if det_click and c4_charges:
                    n_boom = len(c4_charges)
                    boom, c4_charges[:] = list(c4_charges), []
                    tot = dict(struct=0, ground=0, debris=0)
                    for i, c in enumerate(boom):
                        got = c4_explode(world, c['pos'].copy(), c['normal'], structure_hit=c['struct'])
                        for k in tot:
                            tot[k] += got[k]
                        blast_fx.spawn(world, c['pos'], c['normal'], delay=0.07 * i, viewer=player.eye)   # fireball, smoke cloud and sparks that react to the terrain
                    c4_flash = None
                    SFX.play('explosion', 1.0)
                    inv_note(f"TNT x{n_boom}: removed {tot['struct']} structure + {tot['ground']} ground cubes, threw {tot['debris']} debris cubes", 6.0)
                blast_fx.update(world, dt)
                if c4_flash is not None:
                    centre_flash, flash_left = c4_flash
                    flash_left -= dt
                    c4_flash = (centre_flash, flash_left) if flash_left > 0 else None
            with Timed('eject', gpu=False):
                STRUCTS.eject_player(world, player)                  # keep the player out of falling pieces

        # Unload distant chunks only when caching is off, and only when the player
        # enters a new chunk, so the check doesn't run every frame.
        centre = (math.floor(player.pos[0] / CHUNK_SIZE), math.floor(player.pos[2] / CHUNK_SIZE))
        if not settings['cache_chunks'] and centre != last_centre:
            with Timed('unload', gpu=False):
                unload_far_chunks(world, renderer, centre[0], centre[1], keep=rd + 1)
        last_centre = centre

        # Heading and position go in the window title, since the 3D view has no HUD.
        state = "PAUSED   " if paused else ""
        caption = f"VoidCraft   {state}{FPS.update(dt):.0f} fps"
        if caption != last_caption:
            pygame.display.set_caption(caption)
            last_caption = caption

        if frame.active:
            frame.bind()                                                 # fullscreen: this frame is drawn offscreen, then scaled to the screen
        # Sun shadows: the shadow map is redrawn from the sun each frame, centred on the player.
        with Timed('shadow'):
            lvp = shadow.update(renderer, struct_gl, player.eye)
        renderer.light_vp = lvp
        renderer.grid = show_grid                                    # X: grid lines on the ground too
        struct_gl.light_vp = lvp
        struct_gl.view = (centre[0], centre[1], rd)                  # same square of chunks as the terrain, plus one for objects that straddle the edge

        if modal:
            arsenal.release()
            if not menu_frozen:
                # First frame with the menu open: draw the world once and keep it as the backdrop.
                post_on = settings['bloom'] > 0 or settings['contrast'] != 100 or settings['sharpness'] > 0
                scene_on = scene.begin(settings['render_scale'] / 100.0, force=post_on)
                renderer.draw(world, player.eye, player.yaw, player.pitch,
                              radius=rd - 1, fov=settings['fov'])
                if scene_on:
                    post.apply(scene, settings['bloom'] / 100.0, settings['contrast'] / 100.0, settings['sharpness'] / 100.0)
                backdrop.capture()
                menu_frozen = True
            else:
                backdrop.draw()
            if reset_open:
                reset_panel.draw(pygame.mouse.get_pos())
            else:
                settings_panel.draw(menu)
        else:
            menu_frozen = False
            t_draw = time.perf_counter()
            post_on = settings['bloom'] > 0 or settings['contrast'] != 100 or settings['sharpness'] > 0
            scene_on = scene.begin(settings['render_scale'] / 100.0, force=post_on)    # below or above 100%: the world goes into the offscreen target first
            with Timed('terrain'):
                sh = blast_fx.shake * settings['shake'] / 100.0                                      # a blast near you shakes the view for a moment
                renderer.draw(world, player.eye,
                              player.yaw + (shake_rng.uniform(-sh, sh) if sh > 1e-4 else 0.0),
                              player.pitch + (shake_rng.uniform(-sh, sh) if sh > 1e-4 else 0.0),
                              radius=rd - 1, fov=settings['fov'])
            draw_ms = (time.perf_counter() - t_draw) * 1000.0
            # renderer.draw times its own parts; they sit inside 'terrain' and are listed under it.
            PROF.add('terrain.mesh', renderer.mesh_s * 1000.0)
            PROF.add('terrain.upload', renderer.upload_s * 1000.0)
            PROF.add('terrain.list', renderer.list_s * 1000.0)
            PROF.add('terrain.collect', renderer.collect_s * 1000.0)
            PROF.add('terrain.call', renderer.call_s * 1000.0)
            with Timed('struct_draw'):
                extra = []
                if show_dmg:                                            # every damaged chunk within reach, rebuilt only when the ground changes, a ledge is scheduled, or you cross into another chunk
                    key = (world.edit_ver, len(COLLAPSE), math.floor(player.eye[0] / _E.CHUNK_SIZE), math.floor(player.eye[2] / _E.CHUNK_SIZE))
                    if key != dmg_key:
                        dmg_views = _E.make_ground_dmg_views(world, player.eye, hot=[e['idx'] for e in COLLAPSE])
                        dmg_key = key
                    extra.extend(dmg_views)
                struct_gl.render(renderer.last_vp, 1.0, overlay=settings['show_hud'] or show_dmg, grid=show_grid, extra=extra)
            with Timed('debris_draw'):
                body_gl.render(renderer.last_vp, 1.0, player.yaw, player.pitch, bodies=debris_tint if show_dmg else None)
                fx_gl.render(renderer.last_vp, 1.0, player.yaw, player.pitch, bodies=blast_fx)
            with Timed('overlay'):
                if scene_on:
                    post.apply(scene, settings['bloom'] / 100.0, settings['contrast'] / 100.0, settings['sharpness'] / 100.0)   # scale to the window with bloom, contrast and sharpness; everything after is drawn at full size
                tg = arsenal.target
                target_label.draw((C4_LABEL, 255) if any(c4_aimed(player, c['pos']) for c in c4_charges) else ((tg['label'], 255) if tg is not None else ('', 0)))
                _ext = ''
                if tool == T_CANNON:
                    _ext = 'Reloading' if TS['cool'] > 0.0 else 'Click to fire'
                elif tool == T_HAMMER:
                    _ext = 'Click to swing'
                toolbar.draw((tool, len(c4_charges) + len(tnt_flights), blast_scale(), _ext))
                inv_hud.draw((INV['msg'] if INV['msg_t'] > 0 else '',))
                if SURVIVAL_ON:
                    survival_hud.draw((int(survival.thirst), int(survival.canteen), False,
                                       int(time.perf_counter() * 2) % 2 if survival.thirst < 20 else 0))
                _tn = time.perf_counter()
                arsenal.extra = (tool_held_prims(player, tool, _tn) + tool_extra + [q for c in c4_charges for q in c4_prims(c['pos'])]
                                 + [q for fl in tnt_flights for q in c4_prims(flight_pos(fl['pts'], fl['t']))] + aim_arc)
                arsenal.draw(renderer.last_vp, settings['fov'])
            if show_dmg:                                                 # the monitor goes on last, over everything, then this frame's numbers are recorded
                with Timed('panel', gpu=False):
                    perf.draw(_E.WIDTH, _E.HEIGHT, mon_counts, FPS.value, dt * 1000.0)
                perf.sample(PROF.cur, (time.perf_counter() - work_start) * 1000.0, GT.latest)
        # Measure before the flip, since flip can wait on vsync and that isn't game work.
        last_cpu_ms = (time.perf_counter() - work_start) * 1000.0
        if frame.active:
            frame.present()
        t_flip = time.perf_counter()
        pygame.display.flip()
        PROF.add(PROF_FLIP, (time.perf_counter() - t_flip) * 1000.0)
        counted = not modal

    renderer.mesher.pool.shutdown(wait=False, cancel_futures=True)
    pygame.quit()


def main():
    run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()


