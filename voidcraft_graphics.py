# voidcraft_graphics.py
import sys, math, random, os, struct, time, json, ctypes, functools, subprocess
import numpy as np
import pygame
from pygame.locals import *
import csv, re, logging, threading, heapq, collections, bisect
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
from OpenGL import GL as gl

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import wgpu
    HAS_WGPU = True
except ImportError as _wgpu_err:
    HAS_WGPU = False
    _WGPU_ERR = str(_wgpu_err)

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
TARGET_FPS        = 60
RENDER_DIST       = 15
MESH_WORKERS      = 2     # background processes that build chunk meshes (separate from the GIL)
MESH_MAX_PENDING  = 6     # chunks queued for meshing at once; the rest wait for a later frame
VERT_STRIDE       = 9 * 4  # bytes per vertex: position (3) + colour (3) + texture coords and flag (3)
ARENA_VERTS       = 1_000_000   # starting size of the shared chunk-mesh buffer (grows as needed)
FOV               = 90
DAY_LENGTH        = 240.0
MOVE_SPEED        = 5
SPRINT_MULT       = 2.5
FLY_SPEED         = 10.0
ALWAYS_DAY        = False
DAY_ONLY          = True
DAY_CYCLE_RANGE   = (0.40, 0.84)
STORM_START_ON    = True
STORM_STRENGTH    = 0.85
LIGHTNING_MIN     = 3.0
LIGHTNING_MAX     = 9.0
LIGHT_REF         = 0.70
HARDNESS_SCALE    = 1.0
STRUCT_SPAN_ON    = True

HOUSE_SPAWN_FREQUENCY = 12
HOUSE_SPAWN_CHANCE = 0.0

# Missing Globals required by the original script
GFX = {
    'chunk_size': 15, 'water_flow': True, 'smooth': True, 'debris': 2,
    'phys_steps': 0, 'sleep_time': 10.0, 'soft_max': 5, 'blast_grains': 0,
    'grain_n': 6, 'blast_cubes': 5000, 'molten_n': 3, 'blast_light': True,
    'sun_strength': 0.80, 'ambient_floor': 0.35, 'detail_strength': 1.0,
    'water_fresnel': 1.0, 'clouds': 0.55, 'horizon_haze': 0.6,
    'foam_width': 0.85, 'foam_strength': 0.50, 'masonry_detail': 1.0, 'physics_device': 'cpu'
}

class PerfStub:
    def mark(self, t, name): pass
PERF = PerfStub()

class StormStub:
    k = 0.0
STORM = StormStub()

STREAMER = None
_DEBRIS_RNG = np.random.default_rng()

# ═══════════════════════════════════════════════
#  MISSING HELPER FUNCTIONS
# ═══════════════════════════════════════════════
def _deb_n(blk: int) -> int:
    """Returns the number of debris subdivisions for a given block type."""
    return 2 if blk < 100 else 3

def _mini_offsets(n: int) -> np.ndarray:
    """Generates local offsets for mini-cube subdivisions."""
    g = (np.arange(n) + 0.5) / n - 0.5
    return np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)

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

SOLID       = set(range(1, 66)) - {WATER, TORCH}
TRANSPARENT = {AIR, WATER, GLASS, LEAVES, TORCH}

COLORS = {
    GRASS:  (0.30, 0.80, 0.15), DIRT:   (0.65, 0.40, 0.15), STONE:  (0.60, 0.60, 0.60),
    SAND:   (0.95, 0.90, 0.50), WOOD:   (0.55, 0.35, 0.12), LEAVES: (0.20, 0.70, 0.15),
    WATER:  (0.20, 0.50, 0.95), SNOW:   (0.98, 1.00, 1.00), GRAVEL: (0.70, 0.68, 0.65),
    PLANKS: (0.80, 0.65, 0.30), GLASS:  (0.75, 0.90, 0.98), TORCH:  (1.00, 0.85, 0.25),
    BRICK:  (0.85, 0.40, 0.30), SLAB:   (0.65, 0.65, 0.65), STEEL:  (0.44, 0.52, 0.64)
}

FACES = [
    (( 0,  1,  0), 1.00), (( 0, -1,  0), 0.45),
    (( 0,  0,  1), 0.75), (( 0,  0, -1), 0.75),
    (( 1,  0,  0), 0.60), ((-1,  0,  0), 0.60),
]

FACE_VERTS = {
    ( 0,  1,  0): [(0,1,0),(0,1,1),(1,1,1),(1,1,0)],
    ( 0, -1,  0): [(0,0,0),(1,0,0),(1,0,1),(0,0,1)],
    ( 0,  0,  1): [(0,0,1),(1,0,1),(1,1,1),(0,1,1)],
    ( 0,  0, -1): [(1,0,0),(0,0,0),(0,1,0),(1,1,0)],
    ( 1,  0,  0): [(1,0,1),(1,0,0),(1,1,0),(1,1,1)],
    (-1,  0,  0): [(0,0,0),(0,0,1),(0,1,1),(0,1,0)],
}

WIDTH, HEIGHT  = 1920, 1080
NEAR, FAR      = 0.05, 300.0
GRAVITY        = -28.0
JUMP_VEL       = 9.0
WORLD_H        = 64
TERRAIN_MOUNTAIN_COVER = 0.56
MOUNTAIN_HEIGHT = 34.0
RIVER_WIDTH    = 0.035
CAVES_ON       = True
CAVE_TUNNEL_W  = 0.035
CAVE_ROOM_THR  = 0.64
TERRAIN_RELIEF = 8.0
CHUNK_SIZE     = 15
CHUNK_SIZE_BASE = 3
REACH          = 20.0
WATER_LEVEL    = 11
PLAYER_HEIGHT  = 1.8
PLAYER_RADIUS  = 0.3
STEP_HEIGHT    = 0.6       # the player climbs slopes and small steps up to this height

# ═══════════════════════════════════════════════
#  FLAT WORLD GENERATION
# ═══════════════════════════════════════════════
FLAT_SURFACE_Y  = 10   # y of the grass layer (must stay below WATER_LEVEL)
FLAT_DIRT_DEPTH = 3    # dirt layers beneath the grass

# ═══════════════════════════════════════════════
#  SMOOTH TERRAIN GENERATION
# ═══════════════════════════════════════════════
# Every block position also has a density. Above zero is solid, below zero is air, and the
# surface sits where the density crosses zero. Terrain density is height minus y.
DENS_SOLID = 1.0      # density given to a block placed or removed by hand
DIG_RADIUS   = 2.5    # blocks: radius of the crater each dig makes
DIG_STRENGTH = 3.0    # density removed from the crater's centre per dig tick
DIG_INTERVAL = 0.15   # seconds between dig ticks while the left button is held
DEBRIS_PER_DUG = 20.0       # dust particles made per unit of dirt dug (DEBUG: raised from 0.5)
DEBRIS_MAX    = 2000         # most dust particles alive at once
DEBRIS_GRAVITY = -12.0       # how fast dust falls

FLAT_GROUND_H = FLAT_SURFACE_Y + 1.0    # top of the grass layer: the ground is flat everywhere

def terrain_height(gx, gz):
    """Ground height at world column (gx, gz). Flat. Works on numpy arrays or scalars."""
    return np.full(np.shape(gx), FLAT_GROUND_H, dtype=np.float64)

def generate_chunk(cx, cz):
    """Build one chunk of ground. Returns (block types, densities), each shaped
    (CHUNK_SIZE, WORLD_H, CHUNK_SIZE). Blocks are solid where the density is above zero, and all are dirt."""
    S, H = CHUNK_SIZE, WORLD_H
    gx, gz = np.meshgrid(cx * S + np.arange(S), cz * S + np.arange(S), indexing='ij')
    h = terrain_height(gx, gz)[:, None, :]                  # (S, 1, S)
    ys = np.arange(H, dtype=np.float64)[None, :, None]      # (1, H, 1)
    depth = h - ys                                          # (S, H, S)
    dens = depth.astype(np.float32)
    blocks = np.zeros((S, H, S), dtype=np.uint8)
    solid = depth > 0
    blocks[solid] = DIRT
    return blocks, dens

# ═══════════════════════════════════════════════
#  CORE CLASSES & GENERATION
# ═══════════════════════════════════════════════
class Player:
    def __init__(self):
        self.health = 10.0
        self.hunger = 10.0
        self.dead = False
        self.invincible = 0.0

    def damage(self, amt, _=None): pass
    def heal(self, amt): pass
    def eat(self, food_val): pass
    def respawn(self):
        self.health = 10.0
        self.hunger = 10.0
        self.dead = False
        self.invincible = 0.0

PLAYER = Player()

class _DirtySet(set):
    def __init__(self, *a):
        set.__init__(self, *a); self.ver = {}; self.bg = set(); self.gen = 0
    def add(self, k):
        self.ver[k] = self.ver.get(k, 0) + 1; self.gen += 1; set.add(self, k)
        if threading.current_thread() is threading.main_thread(): self.bg.discard(k)
        else: self.bg.add(k)
    def update(self, it):
        for k in it: self.add(k)
    def clear(self):
        set.clear(self); self.ver.clear(); self.bg.clear()

class ChunkCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = collections.OrderedDict()
        self.mesh = collections.OrderedDict()
        self.data_bytes = 0; self.mesh_bytes = 0
        self.hit_d = self.miss_d = self.hit_m = self.miss_m = 0

    def clear(self):
        with self.lock:
            self.data.clear(); self.mesh.clear(); self.data_bytes = self.mesh_bytes = 0
            self.hit_d = self.miss_d = self.hit_m = self.miss_m = 0

    def put_data(self, k, blocks, hard):
        n = blocks.nbytes + (hard.nbytes if hard is not None else 0)
        with self.lock:
            old = self.data.pop(k, None)
            if old: self.data_bytes -= old[2]
            self.data[k] = (blocks, hard, n); self.data_bytes += n

    def take_data(self, k):
        with self.lock:
            v = self.data.pop(k, None)
            if v is None: self.miss_d += 1; return None
            self.data_bytes -= v[2]; self.hit_d += 1
            return v[0], v[1]

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

class World:
    def __init__(self):
        self.chunks = {}
        self.dirty = _DirtySet()
        self.torches = set()
        self.orig = {}
        self._glock = threading.RLock()
        self._building = set()
        self._gen_owner = None
        self._gd = 0
        self._gd_tid = None
        self.dmg = {}
        self.modified_chunks = set()
        self.touched = set()          # chunk keys whose draw-list entry may need a fresh look
        self.hard = {}
        self.flow = {}
        self.water_wake = set()
        self.saved = {}         # edited chunks kept after being unloaded
        self.dens = {}          # density per chunk: above zero is solid
        self.saved_dens = {}
        self.dug_cells = []     # (n,3) arrays of cells dug out since the last take_dug_cells()
        self.stamps = []        # solid dirt cubes added by stamp_block: ((x, y, z) lowest corner, side)

    @property
    def _gen_depth(self):
        return self._gd if threading.get_ident() == self._gd_tid else 0

    @_gen_depth.setter
    def _gen_depth(self, v):
        self._gd = v; self._gd_tid = threading.get_ident()

    def density(self, x, y, z):
        """Density at one sample, or None if its chunk isn't loaded. Below the world is solid, above it is air."""
        if y < 0: return DENS_SOLID
        if y >= WORLD_H: return -DENS_SOLID
        cx = math.floor(x / CHUNK_SIZE)
        cz = math.floor(z / CHUNK_SIZE)
        d = self.dens.get((cx, cz))
        if d is None: return None
        return float(d[int(x - cx * CHUNK_SIZE), y, int(z - cz * CHUNK_SIZE)])

    def dig(self, center, radius, strength):
        """Carve a smooth crater: density drops by `strength` at the centre and fades to nothing at
        `radius`. Carved chunks are saved and remeshed. Neighbours whose border samples changed are
        remeshed too, since the mesh reads one sample across each chunk border."""
        cx, cy, cz = center
        lo = (math.floor(cx - radius), max(0, math.floor(cy - radius)), math.floor(cz - radius))
        hi = (math.ceil(cx + radius), min(WORLD_H - 1, math.ceil(cy + radius)), math.ceil(cz + radius))
        if lo[1] > hi[1]: return 0.0
        S = CHUNK_SIZE
        carved = set()
        removed = 0.0
        for kx in range(lo[0] // S, hi[0] // S + 1):
            for kz in range(lo[2] // S, hi[2] // S + 1):
                d = self.dens.get((kx, kz))
                if d is None: continue
                x0, x1 = max(lo[0], kx * S), min(hi[0], kx * S + S - 1)
                z0, z1 = max(lo[2], kz * S), min(hi[2], kz * S + S - 1)
                if x0 > x1 or z0 > z1: continue
                gx = np.arange(x0, x1 + 1, dtype=np.float32)[:, None, None]
                gy = np.arange(lo[1], hi[1] + 1, dtype=np.float32)[None, :, None]
                gz = np.arange(z0, z1 + 1, dtype=np.float32)[None, None, :]
                dist = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2)
                fade = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 2
                if not fade.any(): continue
                sl = (slice(x0 - kx * S, x1 - kx * S + 1), slice(lo[1], hi[1] + 1),
                      slice(z0 - kz * S, z1 - kz * S + 1))
                old = d[sl]
                new = old - strength * fade
                removed += float((np.maximum(old, 0.0) - np.maximum(new, 0.0)).sum())
                flip = (old > 0) & (new < 0.5)                # dirt that is mostly gone comes out whole
                new = np.where(flip, -DENS_SOLID, new)
                if flip.any():
                    ii, jj, kk = np.nonzero(flip)
                    self.dug_cells.append(np.stack([ii + x0, jj + lo[1], kk + z0], axis=1))
                d[sl] = new
                blk = self.chunks.get((kx, kz))
                if blk is not None:
                    blk[sl][new <= 0] = AIR          # carved samples are air in the block grid too
                carved.add((kx, kz))
        if not carved: return 0.0
        self.modified_chunks |= carved
        # Remesh every loaded chunk within one sample of the carved box: the mesh reads across borders.
        for kx in range((lo[0] - 1) // S, (hi[0] + 1) // S + 1):
            for kz in range((lo[2] - 1) // S, (hi[2] + 1) // S + 1):
                if (kx, kz) in self.dens:
                    self.dirty.add((kx, kz))
                    self.touched.add((kx, kz))
        return removed

    def reset_damage(self):
        """Put back every dug block: edited chunks are regenerated and saved edits are dropped."""
        for key in list(self.modified_chunks):
            self.saved.pop(key, None)
            self.saved_dens.pop(key, None)
            if key in self.chunks:
                self.chunks[key], self.dens[key] = generate_chunk(*key)
                for lo, n in self.stamps:
                    self._stamp_into(key, lo, n)
            for dx, dz in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (key[0] + dx, key[1] + dz)
                if nb in self.chunks:
                    self.dirty.add(nb)
                    self.touched.add(nb)
        self.modified_chunks.clear()
        self.orig.clear()
        self.hard.clear()
        self.dmg.clear()
        self.dug_cells = []

    def _stamp_keys(self, lo, size):
        """Chunk keys that a cube with lowest corner lo and side size overlaps."""
        S = CHUNK_SIZE
        return [(kx, kz) for kx in range(lo[0] // S, (lo[0] + size - 1) // S + 1)
                         for kz in range(lo[2] // S, (lo[2] + size - 1) // S + 1)]

    def _stamp_into(self, key, lo, size):
        """Write one solid cube into a loaded chunk, density and block type. True if the chunk was touched."""
        d = self.dens.get(key)
        if d is None:
            return False
        S = CHUNK_SIZE
        kx, kz = key
        x0, y0, z0 = lo
        x1, y1, z1 = x0 + size - 1, min(WORLD_H - 1, lo[1] + size - 1), z0 + size - 1
        a0, a1 = max(x0, kx * S), min(x1, kx * S + S - 1)
        b0, b1 = max(z0, kz * S), min(z1, kz * S + S - 1)
        if a0 > a1 or b0 > b1 or y0 > y1:
            return False
        sl = (slice(a0 - kx * S, a1 - kx * S + 1), slice(y0, y1 + 1), slice(b0 - kz * S, b1 - kz * S + 1))
        d[sl] = DENS_SOLID
        blk = self.chunks.get(key)
        if blk is not None:
            blk[sl] = DIRT
        return True

    def stamp_block(self, lo, size):
        """Add a solid cube of dirt with its lowest sample corner at lo. From then on it is ordinary ground:
        it deforms when dug, and it survives reset damage. Chunks not loaded yet get it when they load."""
        lo = tuple(int(v) for v in lo)
        size = int(size)
        self.stamps.append((lo, size))
        hits = [self._stamp_into(k, lo, size) for k in self._stamp_keys(lo, size)]
        if not any(hits):
            return False
        self.modified_chunks |= {k for k, h in zip(self._stamp_keys(lo, size), hits) if h}
        S = CHUNK_SIZE
        x0, z0 = lo[0], lo[2]
        x1, z1 = x0 + size - 1, z0 + size - 1
        for kx in range((x0 - 1) // S, (x1 + 1) // S + 1):          # remesh, since the mesh reads across borders
            for kz in range((z0 - 1) // S, (z1 + 1) // S + 1):
                if (kx, kz) in self.dens:
                    self.dirty.add((kx, kz))
                    self.touched.add((kx, kz))
        return True

    def clear_stamps(self):
        """Remove every stamped cube. Also clears the damage, since the chunks they sat in are regenerated."""
        keys = set()
        for lo, n in self.stamps:
            keys |= set(self._stamp_keys(lo, n))
        self.stamps = []
        self.modified_chunks |= keys
        self.reset_damage()

    def take_dug_cells(self):
        out = self.dug_cells
        self.dug_cells = []
        return out

    def get(self, x, y, z):
        if y < 0 or y >= WORLD_H: return AIR
        cx = math.floor(x / CHUNK_SIZE)
        cz = math.floor(z / CHUNK_SIZE)
        lx = int(x - cx * CHUNK_SIZE)
        lz = int(z - cz * CHUNK_SIZE)
        ch = self.chunks.get((cx, cz))
        if ch is None: return AIR
        if self.dens[(cx, cz)][lx, y, lz] <= 0: return AIR     # the density decides what is solid
        return int(ch[lx, y, lz])

    def set(self, x, y, z, b):
        if y < 0 or y >= WORLD_H: return
        cx = math.floor(x / CHUNK_SIZE)
        cz = math.floor(z / CHUNK_SIZE)
        lx = int(x - cx * CHUNK_SIZE)
        lz = int(z - cz * CHUNK_SIZE)
        ch = self.chunks.get((cx, cz))
        if ch is None: return

        if self._gen_depth == 0:
            _old = int(ch[lx, y, lz])
            if _old != b: self.orig.setdefault((int(x), int(y), int(z)), _old)

        ch[lx, y, lz] = b
        self.dens[(cx, cz)][lx, y, lz] = DENS_SOLID if b != AIR else -DENS_SOLID

        if self._gen_depth > 0:
            _hm = self.hard.setdefault((cx, cz), np.zeros(ch.shape, dtype=bool))
            _hm[lx, y, lz] = True

        if self.dmg: self.dmg.pop((int(x), int(y), int(z)), None)

        p3 = (int(x), int(y), int(z))
        if b == TORCH: self.torches.add(p3)
        else: self.torches.discard(p3)

        self.dirty.add((cx, cz))
        self.modified_chunks.add((cx, cz))

        if lx == 0: self.dirty.add((cx - 1, cz))
        if lx == CHUNK_SIZE - 1: self.dirty.add((cx + 1, cz))
        if lz == 0: self.dirty.add((cx, cz - 1))
        if lz == CHUNK_SIZE - 1: self.dirty.add((cx, cz + 1))

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
            if any([self._stamp_into((cx, cz), lo, n) for lo, n in self.stamps]):
                self.modified_chunks.add((cx, cz))
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

def load_around(world, px, pz, radius=RENDER_DIST):
    """Make sure every chunk within `radius` of world position (px, pz) exists.
    Call this each frame with the player's position."""
    pcx = math.floor(px / CHUNK_SIZE)
    pcz = math.floor(pz / CHUNK_SIZE)
    for cx in range(pcx - radius, pcx + radius + 1):
        for cz in range(pcz - radius, pcz + radius + 1):
            world.ensure_chunk(cx, cz)

# ═══════════════════════════════════════════════
#  CHUNK MESHER & CACHE LOOKUP
# ═══════════════════════════════════════════════
# Blocks that hide the faces next to them. Anything else (air, water,
# glass, torches) lets neighbouring faces show through.
_OPAQUE = np.zeros(256, dtype=bool)
_OPAQUE[list(SOLID)] = True

# Colour per block ID. Unknown IDs render magenta so gaps are easy to spot.
_COLOR_LUT = np.tile(np.array([1.0, 0.0, 1.0], dtype=np.float32), (256, 1))
for _blk, _rgb in COLORS.items():
    _COLOR_LUT[_blk] = _rgb

def _padded_chunk(world, cx, cz):
    """Copy a chunk into an array with a one-block border taken from its
    four side neighbours, so faces on chunk edges can see across the boundary.
    Shape: (CHUNK_SIZE+2, WORLD_H+2, CHUNK_SIZE+2). Local block (x, y, z)
    sits at [x+1, y+1, z+1]. The top and bottom borders stay AIR."""
    S, H = CHUNK_SIZE, WORLD_H
    pad = np.zeros((S + 2, H + 2, S + 2), dtype=np.uint8)
    pad[1:S + 1, 1:H + 1, 1:S + 1] = world.chunks[(cx, cz)]
    left  = world.chunks.get((cx - 1, cz))
    right = world.chunks.get((cx + 1, cz))
    back  = world.chunks.get((cx, cz - 1))
    front = world.chunks.get((cx, cz + 1))
    if left is not None:  pad[0,     1:H + 1, 1:S + 1] = left[S - 1, :, :]
    if right is not None: pad[S + 1, 1:H + 1, 1:S + 1] = right[0, :, :]
    if back is not None:  pad[1:S + 1, 1:H + 1, 0]     = back[:, :, S - 1]
    if front is not None: pad[1:S + 1, 1:H + 1, S + 1] = front[:, :, 0]
    return pad

# ═══════════════════════════════════════════════
#  SMOOTH MESHER (surface nets over the density field)
# ═══════════════════════════════════════════════
def _crossings(d0, d1):
    """Where the surface crosses each grid edge between samples d0 and d1: a mask, and the
    fraction along the edge (from d0 towards d1) where the density is zero."""
    m = (d0 > 0) != (d1 > 0)
    denom = np.where(m, d0 - d1, 1.0)
    t = np.where(m, d0 / denom, 0.0)
    return m, t.astype(np.float32)

def _padded_density(world, cx, cz):
    """Densities for one chunk plus a border of one sample on every side, taken from the
    neighbouring chunks. Shape (CHUNK_SIZE+3, WORLD_H+3, CHUNK_SIZE+3). The chunk's own sample
    (x, y, z) sits at [x+1, y+1, z+1]. Below the world is solid rock and above it is air."""
    S, H = CHUNK_SIZE, WORLD_H
    pad = np.full((S + 3, H + 3, S + 3), -DENS_SOLID, dtype=np.float32)
    pad[:, 0, :] = DENS_SOLID
    # For each neighbour offset: where its samples go in the border, and which samples to copy.
    spans = {-1: (slice(0, 1), slice(S - 1, S)),
             0: (slice(1, S + 1), slice(0, S)),
             1: (slice(S + 1, S + 3), slice(0, 2))}
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            tx, sx = spans[dx]
            tz, sz = spans[dz]
            src = world.dens[(cx + dx, cz + dz)]
            pad[tx, 1:H + 1, tz] = src[sx, :, sz]
    return pad

def _mesh_density(D, cx, cz):
    """Build smooth terrain geometry for one chunk with surface nets, from its padded density array D.
    Reads only D and constants, so it can run on a worker thread. Each grid cell the surface
    passes through gets one vertex, and each grid edge the surface crosses gets a quad joining the
    four vertices around it. A chunk emits only the quads for edges that start inside it, so
    neighbouring chunks join without gaps.
    Returns (positions, colours, uv_flags) float32 arrays of shape (N, 3) in world space."""
    S, H = CHUNK_SIZE, WORLD_H
    A, B, C = D.shape
    origin = np.array([cx * S - 1, -1, cz * S - 1], dtype=np.float32)

    mx, tx = _crossings(D[:-1], D[1:])              # edges running along x
    my, ty = _crossings(D[:, :-1], D[:, 1:])        # along y
    mz, tz = _crossings(D[:, :, :-1], D[:, :, 1:])  # along z

    # Each cell averages the crossing points on its 12 edges to get its vertex.
    I, J, K = np.indices((A - 1, B - 1, C - 1), dtype=np.float32)
    cnt = np.zeros(I.shape, np.float32)
    sx = np.zeros(I.shape, np.float32)
    sy = np.zeros(I.shape, np.float32)
    sz = np.zeros(I.shape, np.float32)
    for dj in (0, 1):
        for dk in (0, 1):
            m = mx[:, dj:dj + B - 1, dk:dk + C - 1].astype(np.float32)
            t = tx[:, dj:dj + B - 1, dk:dk + C - 1]
            cnt += m; sx += m * (I + t); sy += m * (J + dj); sz += m * (K + dk)
    for di in (0, 1):
        for dk in (0, 1):
            m = my[di:di + A - 1, :, dk:dk + C - 1].astype(np.float32)
            t = ty[di:di + A - 1, :, dk:dk + C - 1]
            cnt += m; sx += m * (I + di); sy += m * (J + t); sz += m * (K + dk)
    for di in (0, 1):
        for dj in (0, 1):
            m = mz[di:di + A - 1, dj:dj + B - 1, :].astype(np.float32)
            t = tz[di:di + A - 1, dj:dj + B - 1, :]
            cnt += m; sx += m * (I + di); sy += m * (J + dj); sz += m * (K + t)
    n = np.maximum(cnt, 1.0)
    V = np.stack([sx / n, sy / n, sz / n], axis=-1)     # local vertex for each cell
    PW = V + origin                                     # world-space vertex for each cell

    # Surface normal per cell from the density gradient, pointing out of the solid.
    g0, g1, g2 = np.gradient(D)
    N = -np.stack([g0[:-1, :-1, :-1], g1[:-1, :-1, :-1], g2[:-1, :-1, :-1]], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-6

    # All ground is dirt: one colour, shaded by how much each surface points up.
    col = _COLOR_LUT[DIRT] * np.clip(0.55 + 0.45 * N[..., 1], 0.4, 1.0)[..., None]

    pos_parts, col_parts, uv_parts = [], [], []

    def emit(owned, offs):
        """Quads for the owned edges in `owned`, a boolean mask whose index 0 is sample 1.
        offs lists the four cells around each edge, relative to the edge's own cell, in cycle order."""
        ii, jj, kk = np.nonzero(owned)
        if ii.size == 0:
            return
        cells = [(ii + 1 + di, jj + 1 + dj, kk + 1 + dk) for di, dj, dk in offs]
        order = [cells[0], cells[1], cells[2], cells[0], cells[2], cells[3]]
        pts = np.stack([PW[c] for c in order], axis=1)              # (M, 6, 3) world positions
        pos_parts.append(pts.reshape(-1, 3))
        col_parts.append(np.stack([col[c] for c in order], axis=1).reshape(-1, 3))
        # Texture coordinates come from the axes the quad faces least. Steep faces are not squashed.
        nq = np.mean([N[c] for c in cells], axis=0)                 # (M, 3) average normal of the quad
        ax = np.argmax(np.abs(nq), axis=1)[:, None]
        u = np.where(ax == 0, pts[..., 2], pts[..., 0])
        v = np.where(ax == 1, pts[..., 2], pts[..., 1])
        uv_parts.append(np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).astype(np.float32))

    emit(mx[1:S + 1, 1:H + 1, 1:S + 1], [(0, 0, 0), (0, -1, 0), (0, -1, -1), (0, 0, -1)])
    emit(my[1:S + 1, 1:H + 1, 1:S + 1], [(0, 0, 0), (-1, 0, 0), (-1, 0, -1), (0, 0, -1)])
    emit(mz[1:S + 1, 1:H + 1, 1:S + 1], [(0, 0, 0), (-1, 0, 0), (-1, -1, 0), (0, -1, 0)])

    if not pos_parts:
        empty = np.zeros((0, 3), np.float32)
        return empty, empty.copy(), empty.copy()
    return (np.concatenate(pos_parts).astype(np.float32),
            np.concatenate(col_parts).astype(np.float32),
            np.concatenate(uv_parts).astype(np.float32))

def vk_mesh_chunk(world, cx, cz):
    """Build one chunk's mesh on the calling thread. Returns None if the chunk isn't loaded."""
    if (cx, cz) not in world.chunks:
        return None
    return _mesh_density(_padded_density(world, cx, cz), cx, cz)

class MeshWorker:
    """Builds chunk meshes in background processes, so they never hold up the main thread's
    Python lock. The main thread copies each chunk's density when it queues the job, so the
    workers never read live world data. It also collects finished meshes, which the renderer uploads."""
    def __init__(self):
        self.pool = ProcessPoolExecutor(max_workers=MESH_WORKERS)
        self.pending = {}     # chunk key -> (version it was built from, future)

    def submit(self, world, key):
        cx, cz = key
        ver = world.dirty.ver.get(key, 0)
        snap = _padded_density(world, cx, cz)
        self.pending[key] = (ver, self.pool.submit(_mesh_density, snap, cx, cz))

    def collect(self, world):
        """Finished meshes as (key, version, payload). Jobs for chunks that have unloaded are dropped."""
        done = []
        for key, (ver, fut) in list(self.pending.items()):
            if key not in world.chunks:
                fut.cancel()
                del self.pending[key]
            elif fut.done():
                del self.pending[key]
                done.append((key, ver, fut.result()))
        return done

@functools.lru_cache(maxsize=None)
def _ring_offsets(radius):
    """(dx, dz) offsets for a square of chunks around the player, nearest first."""
    offs = [(dx, dz) for dx in range(-radius, radius + 1) for dz in range(-radius, radius + 1)]
    offs.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])
    return tuple(offs)

# ═══════════════════════════════════════════════
#  RENDERER (pygame window + OpenGL 3.3 core)
# ═══════════════════════════════════════════════
SKY_RGB = (0.55, 0.75, 0.95)

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
void main() {
    vCol = aCol;
    vUVF = aUVF;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

_FRAG_SRC = """
#version 330 core
in vec3 vCol;
in vec3 vUVF;
out vec4 fragColor;
uniform sampler2D uTex;
uniform float uLight;     // scene light level, set per frame (floored at GFX['ambient_floor'])
void main() {
    vec3 c = vCol;
    if (vUVF.z > 0.5) c *= texture(uTex, vUVF.xy).rgb;
    fragColor = vec4(c * uLight, 1.0);
}
"""

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

def _dirt_texture_bytes(size=16, seed=7):
    """A small noisy dirt tile (RGB bytes). Values stay near 1.0 so block colour still shows."""
    rng = np.random.default_rng(seed)
    shade = 1.0 + (rng.random((size, size, 1)) - 0.5) * 0.35
    rgb = np.clip(np.repeat(shade, 3, axis=2) * 255.0, 0, 255).astype(np.uint8)
    return rgb.tobytes()

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
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_tex = gl.glGetUniformLocation(self.program, "uTex")
        self.u_light = gl.glGetUniformLocation(self.program, "uLight")
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 16, 16, 0,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, _dirt_texture_bytes())
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        # Two GPU timer queries used alternately, so reading one never stalls the pipeline.
        self.queries = list(gl.glGenQueries(2))
        self.q_index = 0
        self.q_started = False
        self.last_gpu_ms = None
        self.mesh_s = self.upload_s = 0.0      # seconds spent meshing / uploading in the last draw
        self.list_s = self.call_s = self.collect_s = 0.0   # visible-list rebuild, draw call, mesh collection
        self.gen = 0                           # bumped on every GPU upload or eviction
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

    def _upload(self, key, payload):
        self.gen += 1
        self._touched.add(key)
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
        self.gen += 1
        self._touched.add(key)
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
            payload = CACHE.get_mesh(k, CHUNK_SIZE, ver)
            if payload is None or k in world.dirty:
                pend = self.mesher.pending.get(k)
                if pend is None or pend[0] != ver:
                    if len(self.mesher.pending) < MESH_MAX_PENDING:
                        t_s = time.perf_counter()
                        self.mesher.submit(world, k)
                        self.mesh_s += time.perf_counter() - t_s
                    else:
                        self._wait(k)
            else:
                entry = self.gpu.get(k)
                if entry is None or entry['src'] is not payload:
                    t_c = time.perf_counter()
                    self._upload(k, payload)
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

    def draw(self, world, eye, yaw, pitch, radius, fov, ambient=1.0):
        eye = np.asarray(eye, dtype=np.float32)

        # Read the GPU timer from last frame. It was issued a frame ago, so it is normally ready.
        if self.q_started:
            prev = self.queries[1 - self.q_index]
            if gl.glGetQueryObjectuiv(prev, gl.GL_QUERY_RESULT_AVAILABLE):
                self.last_gpu_ms = gl.glGetQueryObjectuiv(prev, gl.GL_QUERY_RESULT) / 1e6

        gl.glBeginQuery(gl.GL_TIME_ELAPSED, self.queries[self.q_index])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        vp = _perspective(fov, WIDTH / HEIGHT, NEAR, FAR) @ _view(eye, yaw, pitch)
        self.last_vp = vp     # kept for the debug chunk grid, which is drawn over this frame
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glUniform1i(self.u_tex, 0)
        gl.glUniform1f(self.u_light, max(float(ambient), GFX['ambient_floor']))

        pcx = math.floor(eye[0] / CHUNK_SIZE)
        pcz = math.floor(eye[2] / CHUNK_SIZE)
        # Finished meshes from the worker threads go up first. Then each chunk is checked: a chunk
        # that needs a new mesh is queued for a worker, and its old mesh keeps drawing until the new
        # one arrives. At most MESH_MAX_PENDING chunks are queued at once.
        self.mesh_s = self.upload_s = self.list_s = self.call_s = 0.0
        t_col = time.perf_counter()
        completed = self.mesher.collect(world)
        self.collect_s = time.perf_counter() - t_col
        for k, ver, payload in completed:
            t_c = time.perf_counter()
            self._upload(k, payload)
            self.upload_s += time.perf_counter() - t_c
            if world.dirty.ver.get(k, 0) == ver:      # no edit happened while it was being built
                CACHE.put_mesh(k, CHUNK_SIZE, ver, payload)
                world.dirty.discard(k)

        # Update the draw list incrementally: only chunks that changed since last frame are looked at.
        t_list = time.perf_counter()
        mesh0, up0 = self.mesh_s, self.upload_s
        if (pcx, pcz, radius) != self._ring_key:
            self._change_ring(pcx, pcz, radius)
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

        gl.glEndQuery(gl.GL_TIME_ELAPSED)
        self.q_index = 1 - self.q_index
        self.q_started = True

# ═══════════════════════════════════════════════
#  GPU & FRAME STATS
# ═══════════════════════════════════════════════
class GpuPoller:
    """Polls nvidia-smi on a background thread so it never stalls a frame.
    `stats` is None until the first reading, or if nvidia-smi isn't available."""
    QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"

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
                self.stats = {'util': f[0], 'mem_used': f[1], 'mem_total': f[2],
                              'temp': f[3], 'power': f[4]}
            except Exception:
                self.available = False
                self.stats = None
                return
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

class PerfMonitor:
    """Rolling history of frame time, CPU work time and GPU timer readings, for the graphs."""
    def __init__(self, history=180):
        self.frame_ms = collections.deque(maxlen=history)   # wall time between frames (includes the FPS-cap sleep)
        self.cpu_ms = collections.deque(maxlen=history)     # game work per frame, sleep excluded
        self.gpu_ms = collections.deque(maxlen=history)     # GPU time for the terrain draw

    def record(self, frame_ms, gpu_ms, cpu_ms):
        self.frame_ms.append(frame_ms)
        if cpu_ms is not None:
            self.cpu_ms.append(cpu_ms)
        if gpu_ms is not None:
            self.gpu_ms.append(gpu_ms)

# ═══════════════════════════════════════════════
#  GRAPHICS SETTINGS MENU (Tab)
# ═══════════════════════════════════════════════
# Tabs of the settings panel. Each row is a header, a slider, a toggle or an info line.
# Sliders change with Left/Right within [min, max] in steps of `step`.
# Toggles flip with Left/Right or Enter. Headers and info lines are skipped by the cursor.
SETTINGS_TABS = [
    ("DISPLAY", [
        {'type': 'header', 'text': 'VIEW'},
        {'type': 'slider', 'key': 'render_distance', 'label': 'Render distance', 'min': 4, 'max': 24, 'step': 1,
         'locked': True,
         'fmt': lambda v: f"{v} chunks (locked)",
         'help': 'Locked at 15 chunks. How far the world is loaded and drawn around you.'},
        {'type': 'slider', 'key': 'fov', 'label': 'Field of view', 'min': 60, 'max': 110, 'step': 5,
         'fmt': lambda v: f"{v} deg",
         'help': 'Horizontal camera field of view in degrees.'},
        {'type': 'header', 'text': 'FRAME RATE'},
        {'type': 'slider', 'key': 'fps_limit', 'label': 'FPS limit', 'min': 0, 'max': 240, 'step': 30,
         'fmt': lambda v: 'uncapped' if v == 0 else f"{v} fps",
         'help': 'Frame rate cap. Uncapped shows the true frame time but uses a full CPU core.'},
    ]),
    ("WORLD", [
        {'type': 'header', 'text': 'CHUNKS'},
        {'type': 'toggle', 'key': 'cache_chunks', 'label': 'Cache chunks',
         'help': 'Keep chunks you walk away from in memory, so walking back to them is instant.'},
        {'type': 'header', 'text': 'DAMAGE'},
        {'type': 'button', 'key': 'reset_damage', 'label': 'Reset damage', 'action': 'reset_damage',
         'help': 'Put back every dug block and clear the loose dirt and dust.'},
    ]),
    ("PERFORMANCE", [
        {'type': 'header', 'text': 'OVERLAY'},
        {'type': 'toggle', 'key': 'show_hud', 'label': 'Performance graphs',
         'help': 'Frame time and GPU graphs across the top of the screen. Also toggled with G.'},
        {'type': 'header', 'text': 'COMPUTE'},
        {'type': 'toggle', 'key': 'gpu_physics', 'label': 'GPU physics',
         'help': 'Structure support checks run on the GPU (needs CuPy). Falls back to the CPU on its own.'},
    ]),
    ("DEBRIS", [
        {'type': 'header', 'text': 'DEBRIS'},
        {'type': 'toggle', 'key': 'debris_on', 'label': 'Debris',
         'help': 'Dirt cubes and dust from digging and blasts. Turning it off removes what is there.'},
        {'type': 'slider', 'key': 'debris_amount', 'label': 'Amount', 'min': 0, 'max': 100, 'step': 10,
         'when': 'debris_on', 'fmt': lambda v: f"{v}%",
         'help': 'How much debris is made and kept. Lowering it deletes the oldest cubes.'},
        {'type': 'slider', 'key': 'struct_res', 'label': 'Cube detail', 'min': 1, 'max': 10, 'step': 1,
         'fmt': lambda v: f"{int(v)} cells per block ({int(v) ** 3} per block)",
         'help': 'Test cubes keep their size but are built from smaller pieces. Rebuilds the cubes.'},
    ]),
    ("CONTROLS", [
        {'type': 'header', 'text': 'WEAPONS (number keys, left click fires)                  KEYS'},
        {'type': 'info', 'a': ('1', 'Drill', 'hold: chips and digs'), 'b': ('W A S D', 'move')},
        {'type': 'info', 'a': ('2', 'Rifle', 'hitscan, small hole'), 'b': ('Space', 'jump')},
        {'type': 'info', 'a': ('3', 'Grenade launcher', 'arcing shell, medium blast'), 'b': ('V', 'fly')},
        {'type': 'info', 'a': ('4', 'Rocket launcher', 'fast, big blast'), 'b': ('Mouse', 'look')},
        {'type': 'info', 'a': ('5', 'Melting laser', 'hold: melts what it touches'), 'b': ('Left click', 'fire weapon')},
        {'type': 'info', 'a': ('6', 'Demo charge', 'sticks to what it hits'), 'b': ('Right click', 'detonate charges')},
        {'type': 'info', 'a': None, 'b': ('R', 'reset map and damage')},
        {'type': 'info', 'a': None, 'b': ('F', 'pause')},
        {'type': 'info', 'a': None, 'b': ('G', 'debug + integrity (V hides text)')},
        {'type': 'info', 'a': None, 'b': ('Tab', 'settings')},
        {'type': 'info', 'a': None, 'b': ('F5', 'benchmark')},
    ]),
]

def default_settings():
    return {'cache_chunks': True, 'render_distance': RENDER_DIST, 'fov': FOV, 'show_hud': False,
            'fps_limit': TARGET_FPS, 'debris_yield': 1.0, 'debris_size': 1.0, 'dirt_cubes': 10,
            'dirt_cube_size': 1.0, 'cube_sleep_time': 1.0, 'debris_on': True, 'debris_amount': 100,
            'struct_res': 2, 'gpu_physics': True}

class TabMenu:
    """Cursor state for the settings panel: which tab, and which row in it."""
    def __init__(self, settings):
        self.settings = settings
        self.open = False
        self.tab = 0
        self.row = self.selectable()[0]
        self.drag = None     # row index of a slider being dragged with the mouse

    def rows(self):
        """Rows of the current tab that are shown now. A row with 'when' only shows while that setting is on."""
        return [r for r in SETTINGS_TABS[self.tab][1] if r.get('when') is None or self.settings[r['when']]]

    def selectable(self):
        return [i for i, r in enumerate(self.rows())
                if r['type'] in ('slider', 'toggle', 'button') and not r.get('locked')]

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
        if r['type'] == 'slider':
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

PANEL_W, PANEL_H = 1000, 560   # settings panel (Tab); tall enough for the PARTICLES tab
HUD_W, HUD_H = 760, 230        # performance HUD across the top of the view (G)
HUD_REBUILD_S = 0.1           # the HUD's vertices are rebuilt this often, not every frame

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
        x0, x1 = x / WIDTH * 2 - 1, (x + w) / WIDTH * 2 - 1
        y0, y1 = 1 - (y + h) / HEIGHT * 2, 1 - y / HEIGHT * 2
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

    def _text(self, x, y, text, color, font=None):
        self.surf.blit((font or self.font).render(text, True, color), (x, y))

    def _graph(self, rect, values, max_val, color, label, ref=None):
        s = self.surf
        pygame.draw.rect(s, (12, 12, 18, 235), rect)
        pygame.draw.rect(s, (90, 90, 110, 255), rect, 1)
        if ref is not None:   # reference line, e.g. the 60 fps budget
            y = rect.bottom - min(ref / max_val, 1.0) * rect.h
            pygame.draw.line(s, (60, 110, 60, 255), (rect.left, y), (rect.right, y), 1)
        if len(values) >= 2:
            n = len(values)
            pts = [(rect.left + i * rect.w / (n - 1),
                    rect.bottom - min(v / max_val, 1.0) * rect.h)
                   for i, v in enumerate(values)]
            pygame.draw.lines(s, color, False, pts, 2)
        shown = f"{label}   {values[-1]:.1f}" if values else label
        s.blit(self.font_small.render(shown, True, color), (rect.left + 8, rect.top + 4))

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

class SettingsPanel(Overlay):
    """The Tab menu: tabbed settings with sliders and toggles, centred on screen."""
    def __init__(self):
        super().__init__(PANEL_W, PANEL_H, (WIDTH - PANEL_W) // 2, (HEIGHT - PANEL_H) // 2)
        self.f_title = _mono(34)
        self.f_tab = _mono(22)
        self.f_head = _mono(20)
        self.f_label = _mono(22)
        self.f_value = _mono(20)
        self.f_help = _mono(16)
        self.f_exit = _mono(20)
        self.exit_rect = pygame.Rect(self.w - 24 - 170, 74, 170, 44)   # EXIT GAME, right of the tab bar
        self.x0 = (WIDTH - PANEL_W) // 2     # panel's top-left on screen, for mouse coordinates
        self.y0 = (HEIGHT - PANEL_H) // 2

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
        y = tab_y + 66
        for i, row in enumerate(menu.rows()):
            if row['type'] == 'header':
                y += 44
                continue
            if row['type'] == 'info':
                y += 30
                continue
            if row.get('locked'):          # locked rows have no click targets at all
                y += 70
                continue
            regions.append(('row', i, pygame.Rect(16, y - 8, self.w - 32, 66)))
            if row['type'] == 'slider':
                regions.append(('slider', i, pygame.Rect(360, y + 10, 320, 8).inflate(0, 24)))
            else:
                if row['type'] == 'button':
                    regions.append(('button', i, pygame.Rect(360, y, 220, 34)))
                else:
                    regions.append(('off', i, pygame.Rect(360, y, 108, 34)))
                    regions.append(('on', i, pygame.Rect(476, y, 108, 34)))
            y += 70
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
            if kind == 'button' and rect.collidepoint(p):
                menu.select_row(idx)
                return 'reset_damage'
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
            if kind == 'tab' and rect.collidepoint(p):
                menu.set_tab(idx)
                return
        for kind, idx, rect in regions:
            if kind == 'row' and rect.collidepoint(p):
                menu.select_row(idx)
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

    def _render(self, menu):
        s = self.surf
        s.fill((0, 0, 0, 0))
        pygame.draw.rect(s, UI_BG, (0, 0, self.w, self.h), border_radius=12)
        pygame.draw.rect(s, UI_BORDER, (0, 0, self.w, self.h), 2, border_radius=12)

        title = self.f_title.render("SETTINGS", True, UI_TEXT)
        s.blit(title, ((self.w - title.get_width()) // 2, 16))
        close = self.f_help.render("Tab / Esc to close", True, UI_DIM)
        s.blit(close, (self.w - close.get_width() - 24, 24))

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

        # Rows for the current tab.
        y = tab_y + 66
        for i, row in enumerate(menu.rows()):
            if row['type'] == 'header':
                head = self.f_head.render(row['text'], True, UI_ACCENT)
                s.blit(head, (24, y))
                ly = y + head.get_height() // 2
                pygame.draw.line(s, UI_LINE, (24 + head.get_width() + 14, ly), (self.w - 24, ly), 1)
                y += 44
                continue

            if row['type'] == 'info':
                if row['a']:
                    k, name, note = row['a']
                    s.blit(self.f_label.render(k, True, (255, 220, 120)), (32, y))
                    s.blit(self.f_label.render(name, True, UI_TEXT), (62, y))
                    s.blit(self.f_help.render(note, True, UI_DIM), (270, y + 4))
                k, d = row['b']
                s.blit(self.f_value.render(k, True, (255, 220, 120)), (610, y + 1))
                s.blit(self.f_value.render(d, True, UI_TEXT), (750, y + 1))
                y += 30
                continue
            if i == menu.row:
                pygame.draw.rect(s, UI_ROW_SEL, (16, y - 8, self.w - 32, 66), border_radius=6)
            locked = bool(row.get('locked'))
            s.blit(self.f_label.render(row['label'], True, UI_DIM if locked else UI_TEXT), (32, y))
            s.blit(self.f_help.render(row['help'], True, UI_DIM), (32, y + 34))
            if row['type'] == 'slider':
                self._slider(360, y + 2, row, menu.settings[row['key']], locked)
            elif row['type'] == 'button':
                btn = pygame.Rect(360, y, 220, 34)
                pygame.draw.rect(s, UI_SEG_ON, btn, border_radius=5)
                t = self.f_value.render("RESET", True, UI_TEXT)
                s.blit(t, t.get_rect(center=btn.center))
            else:
                self._toggle(360, y, menu.settings[row['key']])
            y += 70

        foot = self.f_help.render("[ ] switch tab   Up/Down select   Left/Right change   Enter toggle",
                                  True, UI_DIM)
        s.blit(foot, (24, self.h - 34))

# GPU HUD: panels, graph lines and text are all built as triangles and drawn in one call.
# Vertices are (x, y, u, v, r, g, b, a). A u below zero marks a solid colour; otherwise the
# colour is multiplied by the glyph atlas alpha.
_HUD_VERT = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aCol;
out vec2 vUV;
out vec4 vCol;
void main() {
    vUV = aUV;
    vCol = aCol;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_HUD_FRAG = """
#version 330 core
in vec2 vUV;
in vec4 vCol;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    if (vUV.x < 0.0) fragColor = vCol;
    else fragColor = vec4(vCol.rgb, vCol.a * texture(uTex, vUV).a);
}
"""

_ATLAS_COLS = 16
_ATLAS_CHARS = [chr(c) for c in range(32, 127)]    # printable ASCII

class PerfHud:
    """In-game frame and GPU graphs across the top of the view, drawn on the GPU. Toggle with G.

    Everything is built as triangles each frame and sent in one draw call. Text comes from a
    glyph atlas rendered once at start-up, so no text is rasterised per frame."""
    def __init__(self):
        self.x = (WIDTH - HUD_W) // 2
        self.y = 12
        self.font = pygame.font.Font(None, 30)
        self._verts_count = 0
        self._built_at = 0.0

        vs = _compile_shader(_HUD_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_HUD_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_tex = gl.glGetUniformLocation(self.program, "uTex")

        # Glyph atlas: white glyphs with alpha, laid out in a grid.
        glyphs = [self.font.render(c, True, (255, 255, 255)) for c in _ATLAS_CHARS]
        cw = max(g.get_width() for g in glyphs) + 2
        ch = max(g.get_height() for g in glyphs) + 2
        rows = (len(glyphs) + _ATLAS_COLS - 1) // _ATLAS_COLS
        self.aw, self.ah = cw * _ATLAS_COLS, ch * rows
        atlas = pygame.Surface((self.aw, self.ah), pygame.SRCALPHA)
        self.glyph_uv = {}
        for i, (c, g) in enumerate(zip(_ATLAS_CHARS, glyphs)):
            ox, oy = (i % _ATLAS_COLS) * cw, (i // _ATLAS_COLS) * ch
            atlas.blit(g, (ox, oy))
            self.glyph_uv[c] = (ox, oy, g.get_width(), g.get_height())

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.aw, self.ah, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pygame.image.tostring(atlas, "RGBA", True))

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        stride = 8 * 4
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(16))
        gl.glBindVertexArray(0)

    # --- vertex builders. Positions are panel-relative pixels until _ndc converts them. ---

    def _ndc(self, xy):
        x = (xy[:, 0] + self.x) / WIDTH * 2.0 - 1.0
        y = 1.0 - (xy[:, 1] + self.y) / HEIGHT * 2.0
        return np.stack([x, y], axis=1)

    @staticmethod
    def _verts(pos, uv, col):
        n = len(pos)
        c = np.broadcast_to(np.asarray(col, dtype=np.float32) / 255.0, (n, 4))
        return np.hstack([pos.astype(np.float32), uv.astype(np.float32), c]).astype(np.float32)

    def _rect(self, x, y, w, h, col):
        corners = np.array([[x, y], [x + w, y], [x + w, y + h],
                            [x, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        return self._verts(self._ndc(corners), np.full((6, 2), -1.0), col)

    def _outline(self, x, y, w, h, col):
        return np.vstack([self._rect(x, y, w, 1, col), self._rect(x, y + h - 1, w, 1, col),
                          self._rect(x, y, 1, h, col), self._rect(x + w - 1, y, 1, h, col)])

    def _polyline(self, pts, thick, col):
        """Each segment becomes a thin quad, since wide GL lines aren't available in core profile."""
        a, b = pts[:-1], pts[1:]
        d = b - a
        length = np.hypot(d[:, 0], d[:, 1])
        length[length == 0] = 1e-6
        n = np.stack([-d[:, 1], d[:, 0]], axis=1) / length[:, None] * (thick / 2.0)
        corners = np.stack([a + n, a - n, b - n, b + n], axis=1)          # (m, 4, 2)
        tris = corners[:, [0, 1, 2, 0, 2, 3], :].reshape(-1, 2)           # (m*6, 2)
        return self._verts(self._ndc(tris), np.full((len(tris), 2), -1.0), col)

    def _text(self, x, y, text, col):
        pos, uv = [], []
        cx = x
        for c in text:
            if c not in self.glyph_uv:
                c = "?"
            ox, oy, gw, gh = self.glyph_uv[c]
            x0, x1, y0, y1 = cx, cx + gw, y, y + gh
            u0, u1 = ox / self.aw, (ox + gw) / self.aw
            vt, vb = 1.0 - oy / self.ah, 1.0 - (oy + gh) / self.ah
            tl, tr, br, bl = ((x0, y0), (u0, vt)), ((x1, y0), (u1, vt)), ((x1, y1), (u1, vb)), ((x0, y1), (u0, vb))
            for p in (tl, tr, br, tl, br, bl):
                pos.append(p[0])
                uv.append(p[1])
            cx += gw
        if not pos:
            return np.zeros((0, 8), dtype=np.float32)
        return self._verts(self._ndc(np.array(pos, dtype=np.float64)),
                           np.array(uv, dtype=np.float64), col)

    def _graph(self, parts, rect, values, max_val, color, label, ref=None):
        x, y, w, h = rect
        parts.append(self._rect(x, y, w, h, (12, 12, 18, 235)))
        parts.append(self._outline(x, y, w, h, (90, 90, 110, 255)))
        if ref is not None:   # reference line, e.g. the 60 fps budget
            ry = y + h - min(ref / max_val, 1.0) * h
            parts.append(self._rect(x, ry, w, 1, (60, 110, 60, 255)))
        if len(values) >= 2:
            n = len(values)
            v = np.minimum(np.asarray(values, dtype=np.float64) / max_val, 1.0)
            xs = x + np.arange(n) * (w / (n - 1))
            ys = y + h - v * h
            parts.append(self._polyline(np.stack([xs, ys], axis=1), 2.0, color))
        shown = f"{label}   {values[-1]:.1f}" if values else label
        parts.append(self._text(x + 8, y + 4, shown, color))

    def _build(self, perf, poller, show_text=True):
        """Build the HUD's triangles from the current stats. This is the expensive Python part,
        so it only runs every HUD_REBUILD_S seconds."""
        box_h = HUD_H if show_text else 136                  # graphs only: the box shrinks to fit them
        parts = [self._rect(0, 0, HUD_W, box_h, (20, 20, 28, 200)),
                 self._outline(0, 0, HUD_W, box_h, (120, 120, 140, 255))]
        self._graph(parts, (8, 8, 244, 120), list(perf.frame_ms), 40.0,
                    (110, 200, 255, 255), "Frame ms")
        self._graph(parts, (258, 8, 244, 120), list(perf.cpu_ms), 40.0,
                    (150, 230, 140, 255), "CPU work ms", ref=1000.0 / 60.0)
        self._graph(parts, (508, 8, 244, 120), list(perf.gpu_ms), 40.0,
                    (255, 170, 90, 255), "GPU compute ms")

        if not show_text:
            return np.vstack(parts).astype(np.float32)

        def fmt(v, unit=""):
            return "n/a" if v is None else f"{v:.0f}{unit}"

        last_ms = perf.frame_ms[-1] if perf.frame_ms else None
        st = poller.stats if poller is not None else None
        fps = f"{1000.0 / last_ms:.0f}" if last_ms else "n/a"
        frame_txt = "n/a" if last_ms is None else f"{last_ms:.1f} ms"
        usage = fmt(st['util'] if st else None, ' %')
        cpu_ms = perf.cpu_ms[-1] if perf.cpu_ms else None
        row1 = f"Frame {frame_txt} ({fps} fps)   CPU work {fmt(cpu_ms, ' ms')}   GPU usage {usage}"
        vram = f"{st['mem_used'] / 1024:.1f}/{st['mem_total'] / 1024:.1f} GB" if st and st['mem_total'] else "n/a"
        row2 = f"VRAM {vram}   Temp {fmt(st['temp'] if st else None, ' C')}   Power {fmt(st['power'] if st else None, ' W')}"
        parts.append(self._text(12, 140, row1, (220, 220, 220, 255)))
        parts.append(self._text(12, 170, row2, (220, 220, 220, 255)))

        # Average and 1% low over the rolling window: the 1% low is the 99th percentile frame time.
        frames = np.asarray(perf.frame_ms, dtype=np.float64)
        if frames.size:
            avg = frames.mean()
            low = np.percentile(frames, 99)
            stats_txt = (f"Frame avg {avg:.1f} ms ({1000.0 / avg:.0f} fps)   "
                         f"1% low {low:.1f} ms ({1000.0 / low:.0f} fps)")
        else:
            stats_txt = "Frame avg n/a   1% low n/a"
        parts.append(self._text(12, 198, stats_txt, (220, 220, 220, 255)))

        if poller is not None and not poller.available:
            parts.append(self._text(12, 218, "nvidia-smi not found: GPU usage, VRAM, temp and power unavailable",
                                    (160, 160, 170, 255)))
        return np.vstack(parts).astype(np.float32)

    def draw(self, perf, poller, show_text=True):
        # Rebuild the vertices only every HUD_REBUILD_S seconds; in between, the stored ones are drawn.
        now = time.perf_counter()
        if self._verts_count == 0 or now - self._built_at >= HUD_REBUILD_S or show_text != getattr(self, '_shown_text', True):
            self._shown_text = show_text
            verts = self._build(perf, poller, show_text)
            self._verts_count = len(verts)
            self._built_at = now
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STREAM_DRAW)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glUniform1i(self.u_tex, 0)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._verts_count)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)


_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

def _heading(yaw):
    """Compass name and bearing in degrees. Facing -Z (yaw 0) is North, +X (yaw 90) is East."""
    deg = math.degrees(yaw) % 360.0
    return _COMPASS[int((deg + 22.5) // 45) % 8], deg

def raycast_terrain(world, origin, direction, reach, step=0.1):
    """Walk along the look direction from `origin`. Returns the last air sample before the first
    solid one (the sample a crater is centred on), or None if nothing solid is in reach."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    last_air = None
    t = 0.0
    while t <= reach:
        p = o + d * t
        x, y, z = (math.floor(v + 0.5) for v in p)
        v = world.density(x, y, z)
        if v is None:
            return None              # ran into chunks that aren't loaded
        if v > 0:
            return last_air
        last_air = (x, y, z)
        t += step
    return None

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
    """Density at a continuous position, interpolated between the eight nearest samples.
    Unloaded samples count as air. Positive means solid, and the value crosses zero on the surface."""
    x0, y0, z0 = math.floor(x), math.floor(y), math.floor(z)
    fx, fy, fz = x - x0, y - y0, z - z0
    total = 0.0
    for dx in (0, 1):
        wx = fx if dx else 1.0 - fx
        for dy in (0, 1):
            wy = fy if dy else 1.0 - fy
            for dz in (0, 1):
                w = wx * wy * (fz if dz else 1.0 - fz)
                if w == 0.0:
                    continue
                v = world.density(x0 + dx, y0 + dy, z0 + dz)
                total += w * (-DENS_SOLID if v is None else v)
    return total

def _box_hits_solid(world, x, y, z):
    """True if the player's body at feet position (x, y, z) is inside the solid surface.
    Samples a ring of points at the feet and at the head, so slopes and crater edges are felt."""
    r, h = PLAYER_RADIUS, PLAYER_HEIGHT
    ring = ((-r, -r), (-r, 0.0), (-r, r), (0.0, -r), (0.0, 0.0), (0.0, r), (r, -r), (r, 0.0), (r, r))
    for y_s in (y + 0.05, y + h - 0.05):
        for dx, dz in ring:
            if _density_at(world, x + dx, y_s, z + dz) > 0.0:
                return True
    return STRUCTS.hits_box(x, y, z, r, h)          # structures and loose bodies are solid too

class PlayerController:
    EPS = 1e-4

    def __init__(self, feet_pos):
        self.pos = np.array(feet_pos, dtype=np.float64)   # feet position
        self.vel = np.zeros(3)
        self.on_ground = False
        self.flying = False
        self.yaw = 0.0
        self.pitch = 0.0

    @property
    def eye(self):
        return self.pos + np.array([0.0, PLAYER_HEIGHT * 0.9, 0.0])

    def forward(self):
        cp = math.cos(self.pitch)
        return np.array([cp * math.sin(self.yaw), math.sin(self.pitch), -cp * math.cos(self.yaw)])

    def toggle_fly(self):
        self.flying = not self.flying
        self.vel[:] = 0.0

    def update(self, world, dt, wish, jump, sprint):
        """wish: desired direction (3-vector, length <= 1). Y is only used when flying."""
        dt = min(dt, 0.05)   # avoid tunnelling after a long frame
        mult = SPRINT_MULT if sprint else 1.0
        if self.flying:
            self.pos += wish * FLY_SPEED * mult * dt
            return

        self.vel[0] = wish[0] * MOVE_SPEED * mult
        self.vel[2] = wish[2] * MOVE_SPEED * mult
        if jump and self.on_ground:
            self.vel[1] = JUMP_VEL
        self.vel[1] = max(self.vel[1] + GRAVITY * dt, -60.0)

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

def _cell_overlaps_player(cell, pos):
    """True if the unit block at `cell` overlaps the player's box."""
    r, h = PLAYER_RADIUS, PLAYER_HEIGHT
    cx, cy, cz = cell
    return (cx < pos[0] + r and cx + 1 > pos[0] - r and
            cy < pos[1] + h and cy + 1 > pos[1] and
            cz < pos[2] + r and cz + 1 > pos[2] - r)

# Blocks the player can place, selected with keys 1-4.
HOTBAR = [STONE, DIRT, PLANKS, BRICK]

# ═══════════════════════════════════════════════
#  MAIN LOOP (WASD + mouse, Space jump, F pause, V fly, 1 benchmark (autowalk), 2-5 hotbar, LMB break, RMB place, Tab menu)
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
#  PARTICLES (dust, smoke, fire): from particles.py
# ═══════════════════════════════════════════════
CONFIG = {'enabled': True, 'particle_cap': DEBRIS_MAX, 'debris_on': True, 'debris_amount': 1.0}
BLOCK_RGB = np.full((256, 3), 0.6, dtype=np.float32)

FX_CAP = 9000
FX_FIRE, FX_SMOKE, FX_DUST = 0, 1, 2
_rng = np.random.default_rng()
_FX_ACC = [0.0, 0.0]      # laser: fractional emission carried between frames
_CHAIN_ACC = [0.0, 0.0]   # chainsaw: same, for chips and sparks


class ParticlePool:
    _F1 = ('age', 'life', 's0', 's1', 'a0', 'drag', 'buoy', 'rot', 'spin')
    _ALL = ('pos', 'vel', 'col', 'kind') + _F1

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = 0
        self.pos = np.zeros((FX_CAP, 3), np.float32)
        self.vel = np.zeros((FX_CAP, 3), np.float32)
        self.col = np.zeros((FX_CAP, 3), np.float32)
        self.kind = np.zeros(FX_CAP, np.int8)
        for k in self._F1:
            setattr(self, k, np.zeros(FX_CAP, np.float32))

    def remove_kind(self, kind):
        """Delete every particle of one kind (used to clear dust when debris is turned off)."""
        if self.n == 0:
            return
        keep = np.nonzero(self.kind[:self.n] != kind)[0]
        m = len(keep)
        for nm in self._ALL:
            a = getattr(self, nm)
            a[:m] = a[keep]
        self.n = m

    def add(self, kind, P, V, life, s0, s1, col, a0, drag, buoy):
        """Spawn particles at P (k,3). V, life, s0, s1, col, a0, drag, buoy broadcast to k."""
        if not CONFIG['enabled'] or kind in (FX_FIRE, FX_SMOKE):
            return 0                      # fire and smoke are switched off; only dust remains
        P = np.asarray(P, np.float32).reshape(-1, 3)
        k0 = len(P)
        sel = None
        if kind == FX_DUST:
            if not CONFIG['debris_on']:
                return 0
            if CONFIG['debris_amount'] < 1.0:
                sel = np.nonzero(_rng.random(k0) < CONFIG['debris_amount'])[0]
                P = P[sel]
        k = min(len(P), min(FX_CAP, int(CONFIG['particle_cap'])) - self.n)
        if k <= 0:
            return 0

        def f(x, extra=()):
            a = np.broadcast_to(np.asarray(x, dtype=np.float32), (k0,) + extra)
            if sel is not None:
                a = a[sel]
            return a[:k]

        s = self.n
        e = s + k
        self.pos[s:e] = P[:k]
        self.vel[s:e] = f(V, (3,))
        self.col[s:e] = f(col, (3,))
        self.kind[s:e] = kind
        self.age[s:e] = 0.0
        self.life[s:e] = f(life)
        self.s0[s:e] = f(s0)
        self.s1[s:e] = f(s1)
        self.a0[s:e] = f(a0)
        self.drag[s:e] = f(drag)
        self.buoy[s:e] = f(buoy)
        self.rot[s:e] = _rng.uniform(0.0, 6.283, k)
        self.spin[s:e] = _rng.uniform(-1.5, 1.5, k)
        self.n = e
        return k

    def update(self, dt):
        n = self.n
        if n == 0 or dt <= 0.0:
            return
        self.age[:n] += dt
        alive = self.age[:n] < self.life[:n]
        if not alive.all():
            idx = np.nonzero(alive)[0]
            m = len(idx)
            for nm in self._ALL:
                a = getattr(self, nm)
                a[:m] = a[idx]
            self.n = n = m
            if n == 0:
                return
        v = self.vel[:n]
        v *= np.exp(-self.drag[:n] * dt)[:, None]
        v[:, 1] += self.buoy[:n] * dt
        self.pos[:n] += v * dt
        self.rot[:n] += self.spin[:n] * dt


PARTS = ParticlePool()


def update(dt):
    PARTS.update(dt)


def draw(eye, yaw, pitch, ambient, submit):
    """Build camera-facing quads and hand them to submit(positions, rgba, uv, blend).

    positions: (4*Q, 3) float32, four corners per quad in order
    rgba:      (4*Q, 4) float32
    uv:        (4*Q, 2) float32, the corners of a 0..1 soft round sprite
    blend:     'add' for fire, 'alpha' for smoke and dust
    Fire is drawn first, additive. Smoke and dust follow, alpha blended, back to front.
    """
    P = PARTS
    n = P.n
    if n == 0:
        return
    cp, sp = math.cos(pitch), math.sin(pitch)
    fwd = np.array([cp * math.sin(yaw), sp, -cp * math.cos(yaw)])          # same convention as _view()
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    light = 0.25 + 0.75 * max(0.0, min(1.0, float(ambient)))
    pos = P.pos[:n].astype(np.float64)
    kind = P.kind[:n]
    u = np.clip(P.age[:n] / np.maximum(P.life[:n], 1e-3), 0.0, 1.0)
    size = P.s0[:n] + (P.s1[:n] - P.s0[:n]) * (1.0 - (1.0 - u) ** 2)     # grows fast, then slowly
    alpha = P.a0[:n] * np.minimum(1.0, u * 12.0) * (1.0 - u) ** 1.4       # quick fade in, long fade out
    col = P.col[:n].copy()
    fire = kind == FX_FIRE
    if fire.any():                                                         # white-yellow -> orange -> dark red
        uf = u[fire]
        col[fire, 0] = np.interp(uf, [0.0, 0.35, 1.0], [1.00, 1.00, 0.55])
        col[fire, 1] = np.interp(uf, [0.0, 0.35, 1.0], [0.95, 0.50, 0.08])
        col[fire, 2] = np.interp(uf, [0.0, 0.35, 1.0], [0.65, 0.10, 0.02])
    col[~fire] *= light                                                    # smoke and dust are lit by the sky, fire is not
    rot = P.rot[:n]

    def quads(idx):
        p = pos[idx]
        h = (size[idx] * 0.5)[:, None]
        cs = np.cos(rot[idx])[:, None]
        sn = np.sin(rot[idx])[:, None]
        r = (right * cs + up * sn) * h
        w = (-right * sn + up * cs) * h
        v = np.stack([p - r - w, p + r - w, p + r + w, p - r + w], axis=1).astype(np.float32)
        c = np.empty((len(idx), 4, 4), np.float32)
        c[:, :, :3] = col[idx][:, None, :]
        c[:, :, 3] = alpha[idx][:, None]
        t = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), (len(idx), 1))
        return (np.ascontiguousarray(v.reshape(-1, 3)),
                np.ascontiguousarray(c.reshape(-1, 4)),
                t)

    for additive in (False, True):
        idx = np.nonzero(fire if additive else ~fire)[0]
        if len(idx) == 0:
            continue
        if not additive:
            d2 = ((pos[idx] - np.asarray(eye, np.float64)) ** 2).sum(axis=1)
            idx = idx[np.argsort(-d2)]
        v, c, t = quads(idx)
        submit(v, c, t, 'add' if additive else 'alpha')


def _unit_dirs(k):
    d = _rng.normal(size=(k, 3))
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)


def fx_fireball(c, r):
    """Glowing core of an explosion: a few large white-hot additive blobs, plus fast sparks."""
    R = _rng
    c = np.asarray(c, np.float32)
    nc = 5
    P = c + R.normal(0, 0.12 * r, (nc, 3)).astype(np.float32)
    V = R.normal(0, 0.6, (nc, 3))
    V[:, 1] += 0.8
    s0 = R.uniform(0.9, 1.3, nc) * r
    PARTS.add(FX_FIRE, P, V, R.uniform(0.45, 0.75, nc), s0, s0 * 2.6, 1.0, 1.0, 1.6, 0.8)
    ns = int(10 + 6 * r)
    d = _unit_dirs(ns)
    s1 = R.uniform(0.18, 0.35, ns) * (0.6 + 0.2 * r)
    PARTS.add(FX_FIRE, c + d * (0.3 * r),
              d * R.uniform(6.0, 16.0, (ns, 1)) * (0.5 + 0.1 * r) + np.array([0.0, 2.0, 0.0]),
              R.uniform(0.3, 0.8, ns), s1, s1 * 0.5, 1.0, 1.0, 1.4, -4.0)


def fx_explosion(centre, radius, broken, block_rgb=None):
    """Dust off every broken block (no fire or smoke).
    broken: list of (power, x, y, z, block, dx, dy, dz); pass [] for nothing.
    block_rgb: optional (256, 3) colour table, defaults to BLOCK_RGB."""
    table = BLOCK_RGB if block_rgb is None else block_rgb
    c = np.asarray(centre, np.float32)
    r = float(radius)
    R = _rng
    if broken:
        m = min(len(broken), int(10 * r) + 8)
        sel = R.choice(len(broken), m, replace=False) if len(broken) > m else range(len(broken))
        b = [broken[i] for i in sel]
        m = len(b)
        pts = np.array([(t[1] + 0.5, t[2] + 0.5, t[3] + 0.5) for t in b], np.float32)
        ids = np.array([t[4] for t in b], np.int64)
        dirs = np.array([(t[5], t[6], t[7]) for t in b], np.float64)
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 0.1)
        V = dirs * R.uniform(1.5, 5.0, (m, 1))
        V[:, 1] += R.uniform(0.0, 0.6, m)                         # puffs no longer drift upward
        PARTS.add(FX_DUST, pts + R.uniform(-0.4, 0.4, (m, 3)), V, R.uniform(0.9, 2.0, m),
                  R.uniform(0.6, 1.0, m), R.uniform(2.0, 3.2, m),
                  table[ids] * 0.55 + 0.30, 0.38, 2.0, 0.0)


def fx_laser(end, d, dt, heat, burning):
    """Sparks and smoke where a beam is cutting. Emission is per second, not per frame."""
    R = _rng
    _FX_ACC[0] += dt * (40.0 + 130.0 * heat)
    _FX_ACC[1] += dt * (14.0 + 20.0 * heat + (25.0 if burning else 0.0))
    nf = int(_FX_ACC[0])
    _FX_ACC[0] -= nf
    ns = int(_FX_ACC[1])
    _FX_ACC[1] -= ns
    back = -np.asarray(d, np.float64)
    e = np.asarray(end, np.float64) + back * 0.12
    if nf > 0:
        V = back * R.uniform(1.0, 4.0, (nf, 1)) + R.normal(0, 1.6, (nf, 3))
        V[:, 1] += R.uniform(0.5, 2.5, nf)
        s0 = R.uniform(0.12, 0.25, nf)
        PARTS.add(FX_FIRE, e + R.normal(0, 0.12, (nf, 3)), V, R.uniform(0.2, 0.5, nf),
                  s0, s0 * 2.4, 1.0, 1.0, 2.5, 2.0)
    if ns > 0:
        V = R.normal(0, 0.3, (ns, 3))
        V[:, 1] += R.uniform(0.8, 2.0, ns)
        s0 = R.uniform(0.25, 0.45, ns)
        PARTS.add(FX_SMOKE, e + R.normal(0, 0.15, (ns, 3)), V, R.uniform(1.5, 3.0, ns),
                  s0, s0 * 3.2, R.uniform(0.12, 0.30, (ns, 1)), 0.38, 0.8, 1.2)


def fx_muzzle(m, d):
    """Short muzzle flash, with the odd wisp of smoke, at the gun."""
    R = _rng
    d = np.asarray(d, np.float64)
    m = np.asarray(m, np.float64)
    V = d * R.uniform(5.0, 9.0, (2, 1)) + R.normal(0, 0.6, (2, 3))
    s0 = R.uniform(0.12, 0.2, 2)
    PARTS.add(FX_FIRE, np.tile(m + d * 0.15, (2, 1)), V, R.uniform(0.05, 0.1, 2),
              s0, s0 * 2.2, 1.0, 1.0, 4.0, 0.0)
    if R.random() < 0.3:
        s = R.uniform(0.1, 0.16)
        PARTS.add(FX_SMOKE, [m + d * 0.3], [d * 1.5 + np.array([0.0, 0.8, 0.0])],
                  0.9, s, s * 4.0, 0.3, 0.3, 1.5, 0.6)


def fx_chainsaw(end, d, dt, block_rgb, sparks):
    """Chips in the block's colour, plus sparks when the blade bites stone or metal.
    block_rgb: (3,) colour of the block being cut. sparks: True for stone or metal."""
    R = _rng
    _CHAIN_ACC[0] += dt * 70.0
    _CHAIN_ACC[1] += dt * (45.0 if sparks else 0.0)
    nc = int(_CHAIN_ACC[0])
    _CHAIN_ACC[0] -= nc
    nf = int(_CHAIN_ACC[1])
    _CHAIN_ACC[1] -= nf
    back = -np.asarray(d, np.float64)
    e = np.asarray(end, np.float64) + back * 0.1
    if nc > 0:
        V = back * R.uniform(1.5, 4.5, (nc, 1)) + R.normal(0, 1.4, (nc, 3))
        V[:, 1] += R.uniform(0.5, 2.5, nc)
        col = np.tile(np.asarray(block_rgb, np.float32) * 0.55 + 0.30, (nc, 1))
        PARTS.add(FX_DUST, e + R.normal(0, 0.1, (nc, 3)), V, R.uniform(0.5, 1.2, nc),
                  R.uniform(0.10, 0.2, nc), R.uniform(0.25, 0.5, nc), col, 0.45, 2.0, 0.1)
    if nf > 0:
        V = back * R.uniform(2.0, 6.0, (nf, 1)) + R.normal(0, 2.0, (nf, 3))
        V[:, 1] += R.uniform(0.5, 2.0, nf)
        s0 = R.uniform(0.08, 0.16, nf)
        PARTS.add(FX_FIRE, e + R.normal(0, 0.08, (nf, 3)), V, R.uniform(0.15, 0.35, nf),
                  s0, s0 * 1.8, 1.0, 1.0, 2.5, 1.5)


_DIG_ACC = [0.0]    # dig dust: fractional particles carried between dig ticks

def dig_dust(centre, dug, yield_mult=1.0, size_mult=1.0):
    """Dust thrown up by digging: DEBRIS_PER_DUG particles per unit of density removed.
    yield_mult and size_mult come from the Tab menu's Particles sliders."""
    if not CONFIG['enabled'] or dug <= 0.0:
        return
    _DIG_ACC[0] += dug * DEBRIS_PER_DUG * yield_mult
    n = int(_DIG_ACC[0])
    if n <= 0:
        return
    _DIG_ACC[0] -= n
    R = _rng
    c = np.asarray(centre, np.float32)
    d = _unit_dirs(n)
    V = d * R.uniform(1.0, 3.0, (n, 1))
    V[:, 1] += R.uniform(2.5, 5.5, n)
    s0 = R.uniform(0.22, 0.40, n) * size_mult
    col = np.array(COLORS[DIRT], np.float32) * 0.55 + 0.30       # pale dust, not the dirt colour
    PARTS.add(FX_DUST, c + R.uniform(-0.4, 0.4, (n, 3)), V, R.uniform(0.9, 2.0, n),
              s0, s0 * 2.5, col, 0.38, 2.0, 0.2)


_PART_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aCol;
layout(location = 2) in vec2 aUV;
uniform mat4 uVP;
out vec4 vCol;
out vec2 vUV;
void main() {
    vCol = aCol;
    vUV = aUV;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

_PART_FRAG = """
#version 330 core
in vec4 vCol;
in vec2 vUV;
out vec4 fragColor;
void main() {
    float r = length(vUV - vec2(0.5)) * 2.0;
    if (r > 1.0) discard;                                   // round sprite
    float a = vCol.a * (1.0 - smoothstep(0.5, 1.0, r));
    fragColor = vec4(vCol.rgb, a);
}
"""


class ParticleRenderer:
    """Draws draw()'s quads. Each call is split into triangles and sent in one buffer upload."""
    STRIDE = 9 * 4    # position 3, colour 4, uv 2

    def __init__(self):
        vs = _compile_shader(_PART_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_PART_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.STRIDE, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, self.STRIDE, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, self.STRIDE, ctypes.c_void_p(28))
        gl.glBindVertexArray(0)

    def render(self, eye, yaw, pitch, ambient, vp):
        if PARTS.n == 0:
            return
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glEnable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_FALSE)                          # particles are tested, not written
        draw(eye, yaw, pitch, ambient, self._submit)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)
        gl.glBindVertexArray(0)

    def _submit(self, pos, col, uv, blend):
        q = len(pos) // 4
        tri = ((np.arange(q) * 4)[:, None] + np.array([0, 1, 2, 0, 2, 3])).reshape(-1)
        data = np.hstack([pos[tri], col[tri], uv[tri]]).astype(np.float32)
        if blend == 'add':
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
        else:
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(data))


# ═══════════════════════════════════════════════
#  DUG DIRT: each dug block splits into mini cubes that fly out and land
# ═══════════════════════════════════════════════
BODY_MAX       = 12000               # cubes alive at once; the oldest go first
BODY_GRAVITY   = 14.0                # blocks per second squared
BODY_MAX_FALL  = 9.0                 # terminal speed, blocks per second
BODY_FRICTION  = 6.0                 # sideways slow-down

def _sample_many(world, X, Y, Z):
    """Raw density at integer samples, for arrays. Below the world is solid, above it is air, unloaded is air."""
    vals = np.full(len(X), -DENS_SOLID, np.float64)
    vals[Y < 0] = DENS_SOLID
    inside = (Y >= 0) & (Y < WORLD_H)
    cx = np.floor(X / CHUNK_SIZE).astype(np.int64)
    cz = np.floor(Z / CHUNK_SIZE).astype(np.int64)
    keys = np.unique((cx + 100000) * 200000 + (cz + 100000))     # 1-D keys: much faster than unique(axis=0)
    for kk in keys.tolist():
        kx, kz = kk // 200000 - 100000, kk % 200000 - 100000
        d = world.dens.get((kx, kz))
        if d is None:
            continue
        m = inside & (cx == kx) & (cz == kz)
        if not m.any():
            continue
        vals[m] = d[X[m] - kx * CHUNK_SIZE, Y[m], Z[m] - kz * CHUNK_SIZE]
    return vals

def _density_many(world, P):
    """Interpolated density at many points: the same smooth field the player and the mesh use."""
    x0 = np.floor(P[:, 0]).astype(np.int64)
    y0 = np.floor(P[:, 1]).astype(np.int64)
    z0 = np.floor(P[:, 2]).astype(np.int64)
    fx = P[:, 0] - x0
    fy = P[:, 1] - y0
    fz = P[:, 2] - z0
    out = np.zeros(len(P), np.float64)
    for dx in (0, 1):
        wx = fx if dx else 1.0 - fx
        for dy in (0, 1):
            wy = fy if dy else 1.0 - fy
            for dz in (0, 1):
                wz = fz if dz else 1.0 - fz
                out += wx * wy * wz * _sample_many(world, x0 + dx, y0 + dy, z0 + dz)
    return out

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
    gx = _density_many(world, at(e, 0.0)) - _density_many(world, at(-e, 0.0))
    gz = _density_many(world, at(0.0, e)) - _density_many(world, at(0.0, -e))
    g = np.stack([gx, gz], axis=1)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return np.where(n > 1e-9, -g / np.maximum(n, 1e-9), 0.0)

def _grid_for(count):
    """Smallest k with k^3 >= count. A block is cut into a k x k x k grid of slots, and `count` of them are used."""
    k = 1
    while k ** 3 < count:
        k += 1
    return k


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
    """For cubes with the underside middle or any bottom corner inside the ground: a move along the density gradient
    at the deepest of those points. Catches cubes wedged against a wall by one corner. Returns (push (m,3), hit (m,))."""
    m = len(P)
    off = np.array([[0.0, 0.0], [0.4, 0.4], [0.4, -0.4], [-0.4, 0.4], [-0.4, -0.4]])
    pts = np.repeat(P, 5, axis=0)
    s5 = np.repeat(size, 5)
    pts[:, 0] += np.tile(off[:, 0], m) * s5
    pts[:, 2] += np.tile(off[:, 1], m) * s5
    pts[:, 1] -= s5 * 0.5
    d = _density_many(world, pts).reshape(m, 5)
    k = np.argmax(d, axis=1)
    dm = d[np.arange(m), k]
    hit = dm > 0.0
    push = np.zeros_like(P)
    if not hit.any():
        return push, hit
    Q = pts.reshape(m, 5, 3)[np.arange(m), k][hit]
    e = 0.25
    g = np.stack([(_density_many(world, Q + np.array([e, 0, 0])) - _density_many(world, Q - np.array([e, 0, 0]))),
                  (_density_many(world, Q + np.array([0, e, 0])) - _density_many(world, Q - np.array([0, e, 0]))),
                  (_density_many(world, Q + np.array([0, 0, e])) - _density_many(world, Q - np.array([0, 0, e])))],
                 axis=1) / (2 * e)
    gn = np.linalg.norm(g, axis=1)
    ok = gn > 1e-3
    dirn = np.where(ok[:, None], -g / np.maximum(gn, 1e-9)[:, None], np.array([0.0, 1.0, 0.0]))
    dist = np.where(ok, dm[hit] / np.maximum(gn, 1e-9), 0.3) + 0.005
    push[hit] = dirn * dist[:, None]
    return push, hit

class DirtCubes:
    """Mini dirt cubes. Simulated as numpy arrays: gravity, collision with the smooth terrain, and sleep
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
        self.col = np.zeros((0, 3), np.float64)         # colour of each cube

    @property
    def n(self):
        return len(self.pos)

    def wake_all(self):
        """Wake every cube so it re-checks what it rests on (called when the ground or a structure changes)."""
        self.awake[:] = True
        self.still[:] = 0.0

    def spawn(self, world, cell_groups, per_block, size_scale=1.0):
        """Split each dug block into `per_block` cubes. Nothing is thrown: the cubes start at rest in the
        dug space and fall. Cubes whose slot overlaps the ground are skipped, so none starts inside it.
        size_scale shrinks each cube inside its slot (1.0 fills the slot)."""
        groups = [np.asarray(g, np.float64).reshape(-1, 3) for g in cell_groups if len(g)]
        per_block = int(max(0, min(100, round(per_block))))
        if not CONFIG['debris_on']:
            return
        per_block = int(round(per_block * CONFIG['debris_amount']))
        if not groups or per_block == 0:
            return
        k = _grid_for(per_block)
        slots = np.stack(np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing='ij'), -1).reshape(-1, 3)
        slot_c = (slots + 0.5) / k - 0.5                        # slot centres inside one block
        P = []
        for block in np.concatenate(groups):
            pick = _rng.permutation(len(slots))[:per_block]    # which slots get a cube
            P.append(block + slot_c[pick])
        P = np.concatenate(P)
        side = float(np.clip(size_scale, 0.1, 1.0)) / k
        sizes = np.full(len(P), side)
        free = ((_density_many(world, P) <= 0.0) & ~_corners_touching(world, P, sizes).any(axis=1)
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
        self.col = np.concatenate([self.col, np.tile(np.array(COLORS[DIRT], np.float64), (len(P), 1))])
        if self.n > BODY_MAX:
            cut = self.n - BODY_MAX
            self.pos, self.vel = self.pos[cut:], self.vel[cut:]
            self.size, self.shade, self.awake = self.size[cut:], self.shade[cut:], self.awake[cut:]
            self.still = self.still[cut:]
            self.col = self.col[cut:]

    def spawn_chunks(self, P, V, sizes, rgb, world=None):
        """Chunks thrown from a broken structure: positions P, velocities V, sizes (disc diameters) and colours."""
        if not CONFIG['debris_on']:
            return
        P = np.asarray(P, np.float64).reshape(-1, 3)
        V = np.asarray(V, np.float64).reshape(-1, 3)
        sizes = np.broadcast_to(np.asarray(sizes, np.float64), (len(P),))
        rgb = np.broadcast_to(np.asarray(rgb, np.float64), (len(P), 3))
        if CONFIG['debris_amount'] < 1.0:
            sel = _rng.random(len(P)) < CONFIG['debris_amount']
            P, V, sizes, rgb = P[sel], V[sel], sizes[sel], rgb[sel]
        if world is not None and len(P):
            # A chunk that starts inside the ground or inside what is left of the structure was the source of the
            # "float up through the pillar" cubes. Nudge it up once, and drop it if it still starts inside something.
            def clear(Q, S):
                return (_density_many(world, Q) <= 0.0) & ~_corners_touching(world, Q, S).any(axis=1) \
                       & ~_struct_corners(Q, S).any(axis=1)
            ok = clear(P, sizes)
            if not ok.all():
                P = P.copy()
                P[~ok, 1] += 0.15
                ok = clear(P, sizes)
            P, V, sizes, rgb = P[ok], V[ok], sizes[ok], rgb[ok]
        k = len(P)
        if k == 0:
            return
        self.pos = np.concatenate([self.pos, P])
        self.vel = np.concatenate([self.vel, np.asarray(V, np.float64).reshape(-1, 3)])
        self.size = np.concatenate([self.size, np.broadcast_to(np.asarray(sizes, np.float64), (k,))])
        self.shade = np.concatenate([self.shade, np.ones(k)])
        self.awake = np.concatenate([self.awake, np.ones(k, bool)])
        self.still = np.concatenate([self.still, np.zeros(k)])
        self.col = np.concatenate([self.col, np.broadcast_to(np.asarray(rgb, np.float64), (k, 3))])
        if self.n > BODY_MAX:
            cut = self.n - BODY_MAX
            self.pos, self.vel = self.pos[cut:], self.vel[cut:]
            self.size, self.shade, self.awake = self.size[cut:], self.shade[cut:], self.awake[cut:]
            self.still = self.still[cut:]
            self.col = self.col[cut:]

    def _separate(self, passes=3, dt=1.0 / 60.0):
        """Round cube against round cube. Each cube is treated as a sphere with radius half its size.
        Overlapping pairs are pushed apart along the line between their centres. Sleeping cubes never
        move; when one of a pair is asleep, the awake one takes the whole correction. Motion that drives
        the pair into each other is removed, so piles settle instead of jittering."""
        self._contact = np.zeros(self.n, bool)                    # cubes that touched another cube this frame
        if self.n < 2:
            return
        M = 1 << 20
        OFF = 1 << 19
        SEP_SLOP = 0.003
        k = max(1.0, dt * 60.0)          # slow frames fall further, so they are allowed to correct further
        SEP_SOFT = min(1.0, 0.35 * k)    # share of an overlap fixed per pass
        SEP_MAX_STEP = 0.02 * k          # most a pair is moved per pass, in blocks
        SEP_CUBE_CAP = 0.03 * k          # most one cube is moved per pass, however many neighbours push it
        for _ in range(passes):
            cell = float(self.size.max())
            ci = np.floor(self.pos / cell).astype(np.int64) + OFF
            key = (ci[:, 0] * M + ci[:, 1]) * M + ci[:, 2]
            order = np.argsort(key, kind='stable')
            skey = key[order]
            I, J = [], []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = key + (dx * M * M + dy * M + dz)
                        lo = np.searchsorted(skey, nk, 'left')
                        hi = np.searchsorted(skey, nk, 'right')
                        cnt = hi - lo
                        total = int(cnt.sum())
                        if total == 0:
                            continue
                        i = np.repeat(np.arange(self.n), cnt)
                        starts = np.repeat(lo, cnt)
                        within = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                        j = order[starts + within]
                        keep = i < j
                        I.append(i[keep])
                        J.append(j[keep])
            if not I:
                return
            I = np.concatenate(I)
            J = np.concatenate(J)

            d = self.pos[J] - self.pos[I]
            dist = np.linalg.norm(d, axis=1)
            radius = (self.size[I] + self.size[J]) * 0.5            # sum of the two radii
            depth = radius - dist - SEP_SLOP
            aI = self.awake[I]
            aJ = self.awake[J]
            hit = (depth > 0.0) & (aI | aJ)
            if not hit.any():
                return
            I, J, d, dist, depth = I[hit], J[hit], d[hit], dist[hit], depth[hit]
            aI, aJ = aI[hit], aJ[hit]

            n = d / np.maximum(dist, 1e-9)[:, None]                 # unit vector from I towards J
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

            strong = depth > 0.02
            self.still[np.concatenate([I[strong], J[strong]])] = 0.0
            touched = np.unique(np.concatenate([I[aI], J[aJ]]))
            self._contact[touched] = True
            self.vel[touched] *= 0.85                                  # contact friction: bleeds off the energy that makes hops

            # Remove only the velocity that drives the pair into each other.
            vI = np.einsum('ij,ij->i', self.vel[I], n)
            vJ = np.einsum('ij,ij->i', self.vel[J], n)
            # Every contact used the same pre-correction velocity, so a cube resting on two cubes had its fall
            # cancelled twice and came out moving UP. Each contact now takes only its share (1 / contacts), which
            # can never remove more than the approach speed.
            deg = (np.bincount(I, minlength=self.n) + np.bincount(J, minlength=self.n)).astype(np.float64)
            kill_I = aI & (vI > 0)
            kill_J = aJ & (vJ < 0)
            np.add.at(self.vel, I[kill_I], -(vI[kill_I] / deg[I[kill_I]])[:, None] * n[kill_I])
            np.add.at(self.vel, J[kill_J], -(vJ[kill_J] / deg[J[kill_J]])[:, None] * n[kill_J])

    def update(self, world, dt, sleep_time=1.0):
        cap = int(BODY_MAX * CONFIG['debris_amount']) if CONFIG['debris_on'] else 0
        if self.n > cap:
            cut = self.n - cap
            self.pos, self.vel = self.pos[cut:], self.vel[cut:]
            self.size, self.shade, self.awake = self.size[cut:], self.shade[cut:], self.awake[cut:]
            self.still = self.still[cut:]
            self.col = self.col[cut:]
        if self.n == 0:
            return
        dt = min(dt, 0.05)
        aw = np.nonzero(self.awake)[0]
        if len(aw):
            V = self.vel[aw]
            P = self.pos[aw]
            P_start = P.copy()
            V[:, 1] = np.maximum(V[:, 1] - BODY_GRAVITY * dt, -BODY_MAX_FALL)
            kf = math.exp(-BODY_FRICTION * dt)
            V[:, 0] *= kf
            V[:, 2] *= kf
            top = max(float(np.abs(V).max()), 1e-6)
            steps = int(min(16, max(1, math.ceil(top * dt * 16))))
            sdt = dt / steps
            size = self.size[aw]
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
            # A cube whose centre is inside the ground (a push from a neighbour, or a blast that moved the surface)
            # is moved back out along the density gradient. It used to stay stuck there for good.
            for _ in range(2):
                tpush, tin = _terrain_push(world, P, size)
                if not tin.any():
                    break
                L = np.linalg.norm(tpush[tin], axis=1)
                P[tin] += tpush[tin] * (np.minimum(L, 0.08) / np.maximum(L, 1e-9))[:, None]
                V[tin, 1] = np.maximum(V[tin, 1], 0.0)                                   # stop it sinking back in
            for _ in range(steps):
                for axis in (1, 0, 2):
                    new = P.copy()
                    new[:, axis] += V[:, axis] * sdt
                    if axis == 1:
                        under = new.copy()
                        under[:, 1] -= size * 0.5
                        blocked = (_corners_touching(world, new, size)
                                   | _struct_corners(new, size)).any(axis=1)   # any bottom corner in terrain or a structure
                        blocked |= _density_many(world, under) > 0.0            # ...or the middle of its underside (a bump between the corners)
                    else:
                        blocked = (_density_many(world, new) > 0.0) | _struct_corners(new, size).any(axis=1)
                    go = ~blocked
                    P[go] = new[go]
                    V[blocked, axis] = 0.0
            # A cube stops on its highest bottom corner, so its low side sinks slightly into the ground.
            # Where only some corners are supported, the cube slides downhill until its base is level.
            hits = _support(world, P, size)
            partial = ~hits.all(axis=1) & (V[:, 1] > -0.5)     # not fully on the ground, and not falling
            if partial.any():
                down = _downhill(world, P[partial], size[partial])
                V[partial, 0] += down[:, 0] * BODY_GRAVITY * 0.6 * dt
                V[partial, 2] += down[:, 1] * BODY_GRAVITY * 0.6 * dt
            # Hard sleep timer: a cube that stays still for sleep_time seconds goes to sleep for good.
            self.pos[aw] = P
            self.vel[aw] = V
            self._separate(dt=dt)
            # Separation knows nothing about the ground, so a pile pressing down can push its bottom cubes into it.
            # Bring any cube whose underside went into the surface back out.
            for _ in range(2):
                tp, tin = _terrain_push(world, self.pos[aw], size)
                if not tin.any():
                    break
                L = np.linalg.norm(tp[tin], axis=1)
                self.pos[aw[tin]] += tp[tin] * (np.minimum(L, 0.1) / np.maximum(L, 1e-9))[:, None]
                self.vel[aw[tin], 1] = np.maximum(self.vel[aw[tin], 1], 0.0)
            # Static friction: a cube that is touching others and already moving slowly stops sliding sideways, so a
            # pile of round-ish cubes stops creeping and can go to sleep.
            ct = self._contact[aw]
            crawl = np.linalg.norm(self.vel[aw], axis=1) < 0.35
            stick = aw[ct & crawl]
            self.vel[stick, 0] = 0.0
            self.vel[stick, 2] = 0.0
            moved_speed = np.linalg.norm(self.pos[aw] - P_start, axis=1) / max(dt, 1e-6)   # real movement this frame, pushes included                                       # pushes cubes apart, stops motion into each other
            Vn = self.vel[aw]
            still_now = (np.abs(Vn).max(axis=1) < 0.12) & (moved_speed < 0.25)   # crawling AND not being shoved around
            self.vel[aw[still_now]] = 0.0
            self.still[aw] = np.where(still_now, self.still[aw] + dt, 0.0)
            ready = np.nonzero(still_now & (self.still[aw] >= sleep_time))[0]          # about to fall asleep
            if len(ready):
                held = _support(world, self.pos[aw[ready]], size[ready]).any(axis=1) | ct[ready]
                lost = ready[~held]                                                    # nothing under it: keep it awake
                self.still[aw[lost]] = 0.0
            asleep = self.still[aw] >= sleep_time
            self.awake[aw[asleep]] = False
        keep = self.pos[:, 1] > -4.0                         # drop cubes that fell out of the world
        if not keep.all():
            self.pos, self.vel = self.pos[keep], self.vel[keep]
            self.size, self.shade, self.awake = self.size[keep], self.shade[keep], self.awake[keep]
            self.still = self.still[keep]
            self.col = self.col[keep]

    def instances(self):
        """Per cube: centre x, y, z, colour r, g, b, and side length."""
        out = np.empty((self.n, 7), np.float32)
        out[:, :3] = self.pos
        out[:, 3:6] = self.col.astype(np.float32) * self.shade[:, None].astype(np.float32)
        out[:, 6] = self.size
        return out

BODIES = DirtCubes()

CUBE_SIDE = 10                       # blocks across a stamped dirt cube
CUBE_ROW_STEP = 14                   # blocks between stamped cubes along a row

def cube_slot_lo(k):
    """Lowest corner of the k-th stamped cube: a row in front of the start point, sitting on the ground."""
    return (-4 + CUBE_ROW_STEP * (k % 4), int(FLAT_GROUND_H), -15 - CUBE_ROW_STEP * (k // 4))

_DISC_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;        // unit cube vertex, -0.5..0.5
layout(location = 1) in vec3 aInstPos;    // cube centre
layout(location = 2) in vec3 aInstCol;    // cube colour
layout(location = 3) in float aInstSize;  // cube side length
layout(location = 4) in vec3 aNrm;        // face normal
uniform mat4 uVP;
uniform float uLight;
out vec3 vCol;
out vec3 vNrm;
void main() {
    vec3 p = aInstPos + aPos * aInstSize;
    vCol = aInstCol * uLight;
    vNrm = aNrm;
    gl_Position = uVP * vec4(p, 1.0);
}
"""

_DISC_FRAG = """
#version 330 core
in vec3 vCol;
in vec3 vNrm;
out vec4 fragColor;
void main() {
    vec3 n = normalize(vNrm);
    vec3 sun = normalize(vec3(0.45, 0.80, 0.35));
    float shade = 0.38 + 0.30 * (0.5 + 0.5 * n.y) + 0.42 * max(dot(n, sun), 0.0);
    fragColor = vec4(vCol * shade, 1.0);
}
"""


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
    """Draws every dirt cube as a camera-facing disc with one instanced call."""
    def __init__(self):
        vs = _compile_shader(_DISC_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_DISC_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_right = gl.glGetUniformLocation(self.program, "uRight")
        self.u_up = gl.glGetUniformLocation(self.program, "uUp")
        self.u_light = gl.glGetUniformLocation(self.program, "uLight")
        self.vao = gl.glGenVertexArrays(1)
        self.vbo_c = gl.glGenBuffers(1)
        self.vbo_i = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_c)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, _DISC_CORNERS.nbytes, _DISC_CORNERS, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(4)
        gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_i)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 28, ctypes.c_void_p(0))
        gl.glVertexAttribDivisor(1, 1)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 28, ctypes.c_void_p(12))
        gl.glVertexAttribDivisor(2, 1)
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, 28, ctypes.c_void_p(24))
        gl.glVertexAttribDivisor(3, 1)
        gl.glBindVertexArray(0)

    def render(self, vp, ambient, yaw, pitch):
        n = BODIES.n
        if n == 0:
            return
        # Camera right and up, using the same basis as _view() and the particle code.
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = np.array([cp * math.sin(yaw), sp, -cp * math.cos(yaw)], dtype=np.float32)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        inst = BODIES.instances()
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glUniform3f(self.u_right, float(right[0]), float(right[1]), float(right[2]))
        gl.glUniform3f(self.u_up, float(up[0]), float(up[1]), float(up[2]))
        gl.glUniform1f(self.u_light, max(float(ambient), GFX['ambient_floor']))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_i)
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
STRUCT_PUSHOUT      = 0.25     # fraction of a penetration removed per step
STRUCT_MAX_PUSH     = 0.5      # fastest push-out speed, blocks per second
STRUCT_MAX_CONTACTS = 10       # contacts kept per body per step
STRUCT_SLEEP_LIN    = 0.15
STRUCT_SLEEP_ANG    = 0.20
STRUCT_SLEEP_TIME   = 0.6
STRUCT_SHATTER      = 0.25     # blast strength (after falloff) that destroys a cell of hardness 1
STRUCT_BLAST_IMPULSE = 140.0   # converts blast pressure on a body's surface to momentum
STRUCT_CRACK_STEP   = 0.05     # seconds between rounds of bond failures, so cracks spread visibly
STRUCT_BUDGET_S     = 0.0025   # most time per frame spent on the support checks
STRUCT_BLAST_DIGS   = True     # a blast also carves the terrain
STRUCT_MIN_BODY_CELLS = 6    # a loose group smaller than this crumbles into chunks instead of becoming a body

# id: mass per block of volume, compressive, tensile and shear strength (force per face area), blast hardness.
# The numbers are tuned so a 2 x 2 pillar stands whole but cracks and topples once about a third of its
# section is gone. Blocks that are not listed behave like stone.
STRUCT_MATERIALS = {
    DIRT:   dict(rho=1.6, comp=1200.0,  tens=25.0,    shear=60.0,   hard=0.5),    # crumbly: weak in tension and shear
    STONE:  dict(rho=2.4, comp=3000.0,  tens=300.0,   shear=450.0,  hard=1.0),
    BRICK:  dict(rho=1.9, comp=1500.0,  tens=120.0,   shear=200.0,  hard=0.7),
    PLANKS: dict(rho=0.6, comp=500.0,   tens=400.0,   shear=100.0,  hard=0.4),
    STEEL:  dict(rho=7.8, comp=20000.0, tens=20000.0, shear=12000.0, hard=3.0),
}
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


def _label_cells(alive, bonds):
    """Connected groups of live cells (see _label_cells_xp). Large grids are labelled on the GPU when it is
    available and switched on; any GPU error turns it off and the CPU does the work instead."""
    global GPU_OK
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
        self.heat = np.zeros(self.mat.shape, np.float32)     # 0 cold .. 1 melted; set by the laser
        self.stress = np.zeros(self.mat.shape, np.float32)   # stress / strength per cell from the last support pass (1 = about to break)
        self.stress_ver = 0
        self.heated = False                                  # any cell warm? (Structure already uses 'hot' for the last blast)
        self.frame_pos = np.zeros(3)
        self.R = np.eye(3)
        self.corner = np.zeros(3)
        self.mesh_ver = 0
        self._mesh_cache = None
        self._aabb = None
        self._geometry()

    # ---- geometry, rebuilt after every change to the cells -------------------------------------------
    def _geometry(self):
        self.alive = self.mat > 0
        nx, ny, nz = self.alive.shape
        pad = np.pad(self.alive, 1)
        expo = np.zeros((nx, ny, nz, 6), bool)
        for f, (dx, dy, dz) in enumerate(_FACE_DIRS):
            nb = pad[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz]
            expo[..., f] = self.alive & ~nb
        self.expo = expo
        self.cell_mass = np.where(self.alive, _S_RHO[self.mat] * self.cs ** 3, 0.0)
        pts, nrm = [], []
        for f in range(6):
            idx = np.argwhere(expo[..., f])
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
            uniq, inv = np.unique(P, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            sumn = np.zeros((len(uniq), 3))
            np.add.at(sumn, inv, N)
            ln = np.linalg.norm(sumn, axis=1, keepdims=True)
            self.surf_g = uniq / 2.0                          # grid units
            self.surf_n = np.where(ln > 1e-9, sumn / np.maximum(ln, 1e-9), 0.0)
            self.surf_corner = (uniq % 2 == 0).all(axis=1)    # cell corners; the rest are face centres
        else:
            self.surf_g = np.zeros((0, 3))
            self.surf_n = np.zeros((0, 3))
            self.surf_corner = np.zeros(0, bool)
        idx = np.argwhere(self.alive)
        if len(idx):
            self.cell_lo, self.cell_hi = idx.min(0), idx.max(0) + 1
        else:
            self.cell_lo = self.cell_hi = np.zeros(3, np.int64)
        self.mesh_ver += 1
        self._mesh_cache = None
        self._aabb = None

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
        """(n, 9) float32 vertices, 6 per exposed face: position in the frame, normal, colour."""
        if mode is None and self._mesh_cache is not None:
            return self._mesh_cache
        out = []
        cs = self.cs
        for f in range(6):
            idx = np.argwhere(self.expo[..., f])
            if not len(idx):
                continue
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
            col = _S_RGB[self.mat[ii]] * self.tint[ii][:, None]
            dm = np.clip(self.cdmg[ii], 0.0, 1.0)[:, None]
            col = col * (1.0 - 0.6 * dm) + np.array([0.20, 0.18, 0.17], np.float32) * (0.6 * dm)
            hm = np.clip(self.heat[ii], 0.0, 1.0)[:, None]
            col = col * (1.0 - hm) + np.array([1.6, 0.55, 0.12], np.float32) * hm      # glows as it heats up
            if mode == 'integrity':
                col = self._integrity_rgb(ii)
            nrm = np.zeros(3)
            nrm[ax] = sg
            v9 = np.empty((len(idx), 6, 9), np.float32)
            v9[:, :, 0:3] = pos
            v9[:, :, 3:6] = nrm
            v9[:, :, 6:9] = col[:, None, :]
            out.append(v9.reshape(-1, 9))
        res = np.concatenate(out) if out else np.zeros((0, 9), np.float32)
        if mode is None:
            self._mesh_cache = res
        return res

    def load_of(self, ii):
        """How close each cell is to failing, 0..1: the worst of its stress, its cracks and its heat."""
        return np.clip(np.maximum(np.maximum(self.stress[ii], self.cdmg[ii] / 0.85), self.heat[ii]), 0.0, 1.0)

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
            tint = _rng.uniform(0.93, 1.07, mat.shape).astype(np.float32)
        super().__init__(mat, cdmg, tint, cs)
        self.frame_pos[:] = origin
        self.bond = [self.alive[:-1] & self.alive[1:],
                     self.alive[:, :-1] & self.alive[:, 1:],
                     self.alive[:, :, :-1] & self.alive[:, :, 1:]]
        self.anch = np.zeros(self.alive.shape, bool)
        self.hot = np.asarray(origin, np.float64).copy()
        self.todo = []                    # (axis, plane) cuts still to be checked in this pass
        self.pass_broke = False
        self.stress_dirty = True          # a new structure is checked once before it is trusted
        self.detach_dirty = False
        self.wait = 0.0
        self.max_ratio = 0.0              # highest stress / strength seen in the last pass (1.0 = about to break)
        self.crack_pts = []               # where bonds just broke, for dust
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
        if not np.array_equal(new, self.anch):
            self.anch = new
            self.detach_dirty = True
            self.stress_dirty = True

    # ---- damage ---------------------------------------------------------------------------------------
    def _remove(self, kill):
        """Take the cells in `kill` out of the structure."""
        if not kill.any():
            return
        self.mat[kill] = 0
        self.cdmg[kill] = 0
        self.stress[kill] = 0
        self.stress_new[kill] = 0
        a = self.mat > 0
        for ax in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(0, -1), slice(1, None)
            self.bond[ax] &= a[tuple(lo)] & a[tuple(hi)]
        self.anch &= a
        self._geometry()
        self.detach_dirty = True
        self.stress_dirty = True

    def blast(self, centre, radius, power):
        """Blast at a world point. Cells close enough are destroyed; cells further out are cracked, which
        weakens their bonds. Returns (centres, materials) of the destroyed cells."""
        idx = np.argwhere(self.alive)
        if not len(idx):
            return np.zeros((0, 3)), np.zeros(0, np.int64)
        cc = self.frame_pos + (idx + 0.5) * self.cs
        dist = np.linalg.norm(cc - centre, axis=1)
        near = dist < radius
        if not near.any():
            return np.zeros((0, 3)), np.zeros(0, np.int64)
        idx, cc, dist = idx[near], cc[near], dist[near]
        f = power * (1.0 - dist / radius) ** 2
        ii = (idx[:, 0], idx[:, 1], idx[:, 2])
        m = self.mat[ii]
        thr = _S_HARD[m] * STRUCT_SHATTER
        kill = f >= thr
        hurt = ~kill
        if hurt.any():
            jj = (idx[hurt, 0], idx[hurt, 1], idx[hurt, 2])
            self.cdmg[jj] = np.minimum(0.85, self.cdmg[jj] + 0.7 * f[hurt] / thr[hurt])
        killed_c, killed_m = cc[kill], m[kill]
        self.hot = np.asarray(centre, np.float64).copy()
        if kill.any():
            mask = np.zeros(self.alive.shape, bool)
            mask[idx[kill, 0], idx[kill, 1], idx[kill, 2]] = True
            self._remove(mask)
        else:
            self._geometry()
        self.stress_dirty = True
        return killed_c, killed_m

    # ---- does it still stand? -------------------------------------------------------------------------
    def _queue_planes(self):
        """Every cut plane between two layers of cells, nearest the last blast first."""
        planes = []
        for a in range(3):
            for p in range(self.alive.shape[a] - 1):
                if self.bond[a][tuple(p if k == a else slice(None) for k in range(3))].any():
                    mid = self.frame_pos[a] + (p + 1) * self.cs + (self.corner[a])
                    planes.append((abs(mid - self.hot[a]), a, p))
        planes.sort()
        self.todo = [(a, p) for _, a, p in planes]
        self.pass_broke = False
        self.stress_new = np.zeros(self.mat.shape, np.float32)
        self._pass_open = True

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
        bonds = [b.copy() for b in self.bond]
        bonds[a][sl] = False
        lab = _label_cells(self.alive, bonds)
        anchored = np.unique(lab[self.anch & self.alive])
        ij = np.argwhere(cross)
        lo = np.insert(ij, a, p, axis=1)
        hi = np.insert(ij, a, p + 1, axis=1)
        l_lo = lab[lo[:, 0], lo[:, 1], lo[:, 2]]
        l_hi = lab[hi[:, 0], hi[:, 1], hi[:, 2]]
        a_lo, a_hi = np.isin(l_lo, anchored), np.isin(l_hi, anchored)
        broke = False
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
            comp = np.minimum(_S_COMP[ml], _S_COMP[mh]) * fac
            tens = np.minimum(_S_TENS[ml], _S_TENS[mh]) * fac
            shear = np.minimum(_S_SHEAR[ml], _S_SHEAR[mh]) * fac
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
                if len(self.crack_pts) < 12:
                    self.crack_pts.append((pos[w] + self.frame_pos, int(ml[w])))
                broke = True
        return broke

    def update(self, world, dt, budget=STRUCT_BUDGET_S):
        """Run the support checks within a time budget. Returns the rigid bodies that came loose."""
        new = []
        if self.detach_dirty:
            new += self._detach()
        self.wait = max(0.0, self.wait - dt)
        if not self.todo and self.stress_dirty and self.wait <= 0.0:
            self._queue_planes()
            self.stress_dirty = False
        if self.todo:
            t0 = time.perf_counter()
            while self.todo and time.perf_counter() - t0 < budget:
                a, p = self.todo.pop(0)
                if self.check_plane(a, p):
                    self.pass_broke = True
                    self.detach_dirty = True
                    self._mesh_cache = None
                    self.mesh_ver += 1
            if not self.todo and self._pass_open:
                self._pass_open = False
                self.stress = self.stress_new
                self.stress_ver += 1                  # the integrity overlay redraws from this
            if not self.todo and self.pass_broke:
                self.stress_dirty = True              # something broke, so look at everything again
                self.wait = STRUCT_CRACK_STEP
                self.pass_broke = False
            if self.detach_dirty:
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
                bodies.append(RigidBody.from_cells(self, mask))
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
        self._top = float(WORLD_H)
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
            d = _density_many(world, Pc)
            near = d > -0.3
            if near.any():
                Q = Pc[near]
                dq = d[near]
                e = 0.2
                g = np.stack([(_density_many(world, Q + np.array([e, 0, 0])) - _density_many(world, Q - np.array([e, 0, 0]))),
                              (_density_many(world, Q + np.array([0, e, 0])) - _density_many(world, Q - np.array([0, e, 0]))),
                              (_density_many(world, Q + np.array([0, 0, e])) - _density_many(world, Q - np.array([0, 0, e])))],
                             axis=1) / (2 * e)
                gn = np.linalg.norm(g, axis=1)
                ok = gn > 1e-3
                depth = np.where(ok, dq / np.maximum(gn, 1e-9), -1.0)
                ok &= depth > -0.06
                if ok.any():
                    pts.append(Q[ok])
                    nrm.append(-g[ok] / gn[ok][:, None])
                    dep.append(depth[ok])
                    vob.append(np.zeros((int(ok.sum()), 3)))
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
        """Highest solid terrain sample under the body's footprint, so points above it skip the terrain test."""
        lo, hi = self.aabb()
        x0, x1 = int(math.floor(lo[0])) - 1, int(math.ceil(hi[0])) + 1
        z0, z1 = int(math.floor(lo[2])) - 1, int(math.ceil(hi[2])) + 1
        n = (x1 - x0 + 1) * (z1 - z0 + 1)
        if n > 900:
            return float(WORLD_H)
        X, Z = np.meshgrid(np.arange(x0, x1 + 1), np.arange(z0, z1 + 1), indexing='ij')
        XX = np.repeat(X.ravel(), WORLD_H)
        ZZ = np.repeat(Z.ravel(), WORLD_H)
        YY = np.tile(np.arange(WORLD_H), n)
        d = _sample_many(world, XX, YY, ZZ)
        hit = YY[d > 0]
        return float(hit.max()) + 1.0 if len(hit) else 0.0

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
        c = np.where(c > 0, np.minimum(c * STRUCT_PUSHOUT, STRUCT_MAX_PUSH), c)      # push out of a penetration; a gap allows approach
        bounce = vn0 < -1.5
        c = np.where(bounce, np.maximum(c, -STRUCT_RESTITUTION * vn0), c)
        lam = [0.0] * (3 * m)
        cl = c.tolist()
        dg = diag.tolist()
        for _ in range(STRUCT_SOLVER_ITERS):
            moved = 0.0
            for i in range(m):
                new = max(0.0, lam[i] - (jv[i] - cl[i]) / dg[i])
                dl = new - lam[i]
                if dl:
                    lam[i] = new
                    jv += K[:, i] * dl
                    moved += abs(dl)
                lim = STRUCT_FRICTION * lam[i]
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
        self._top = self._ground_top(world) + 0.5 * float(np.linalg.norm(self.v)) * dt
        contact_steps = 0
        support = False
        for _ in range(STRUCT_SUBSTEPS):
            self.v[1] -= STRUCT_GRAVITY * h
            self.w *= math.exp(-0.25 * h)
            con = self._contacts(world, obstacles)
            if con is not None:
                pts, nrm, dep, vob = self._reduce(*con)
                self._solve(pts, nrm, dep, vob, h)
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
        slow = np.linalg.norm(self.v) < STRUCT_SLEEP_LIN and np.linalg.norm(self.w) < STRUCT_SLEEP_ANG
        if slow and support:
            self.still += dt
            if self.still >= STRUCT_SLEEP_TIME:
                self.asleep = True
                self.v[:] = 0.0
                self.w[:] = 0.0
        else:
            self.still = 0.0


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
        ys = np.arange(WORLD_H - 1.0, 0.0, -0.25)
        P = np.stack([np.full(len(ys), x), ys, np.full(len(ys), z)], axis=1)
        d = _density_many(world, P)
        hit = np.nonzero(d > 0)[0]
        if not len(hit):
            return None
        k = int(hit[0])
        if k == 0:
            return float(ys[0])
        d0, d1 = d[k - 1], d[k]
        return float(ys[k - 1] + (ys[k] - ys[k - 1]) * (0.0 - d0) / (d1 - d0))

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

    def spawn_pillar(self, world, x, z, cells=(4, 16, 4), cs=0.5, mat=STONE):
        """A pillar standing on the ground at (x, z)."""
        y = self.ground_y(world, x, z)
        if y is None:
            return None
        nx, ny, nz = cells
        origin = np.array([x - nx * cs * 0.5, y, z - nz * cs * 0.5])
        st = Structure(np.full(cells, mat, np.int16), origin, cs)
        st.refresh_anchors(world)
        self.structs.append(st)
        return st

    # ---- events -------------------------------------------------------------------------------------
    def wake_all(self):
        for b in self.bodies:
            b.wake()

    def on_terrain_changed(self, world):
        """The ground changed: structures re-check what they stand on and loose bodies wake up."""
        for s in self.structs:
            s.refresh_anchors(world)
        self.wake_all()
        BODIES.wake_all()

    def blast(self, world, centre, radius, power=1.0):
        """Damage structures, push bodies, throw chunks. The caller draws the fireball."""
        centre = np.asarray(centre, np.float64)
        BODIES.wake_all()
        broken = []
        for s in self.structs:
            lo, hi = s.aabb()
            if (centre < lo - radius).any() or (centre > hi + radius).any():
                continue
            kc, km = s.blast(centre, radius, power)
            for c, m in zip(kc, km):
                d = c - centre
                broken.append((1.0, c[0] - 0.5, c[1] - 0.5, c[2] - 0.5, int(m), d[0], d[1], d[2]))
            if len(kc):
                self._throw_chunks(kc, km, centre, radius, world=world, cs=s.cs)
        for b in self.bodies:
            b.blast(centre, radius, power)
            if np.linalg.norm(b.frame_pos - centre) < radius * 3:
                b.wake()
        return broken

    def _dust(self, pos, mats, per=1):
        """Puffs of dust in the colour of the material, where bonds broke or scraps fell away."""
        P = np.repeat(np.asarray(pos, np.float64).reshape(-1, 3), per, axis=0)
        n = len(P)
        col = _S_RGB[np.repeat(np.asarray(mats, np.int64), per)] * 0.55 + 0.30
        V = _rng.normal(0, 0.6, (n, 3))
        PARTS.add(FX_DUST, P + _rng.normal(0, 0.08, (n, 3)), V, _rng.uniform(0.6, 1.4, n),
                  _rng.uniform(0.10, 0.20, n), _rng.uniform(0.3, 0.6, n), col, 0.4, 2.0, 0.0)

    def _throw_chunks(self, pos, mats, centre, radius, slow=False, world=None, cs=0.5):
        n = len(pos)
        scale = cs / 0.5                                          # debris is sized to the cells it came from
        k = min(2, max(1, 40 // max(n, 1)))
        P = np.repeat(pos, k, axis=0) + _rng.uniform(-0.12, 0.12, (n * k, 3))
        d = P - centre
        dist = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 0.2)
        speed = _rng.uniform(0.5, 2.0, (n * k, 1)) if slow else _rng.uniform(3.0, 9.0, (n * k, 1))
        V = d / dist * speed * (1.2 - 0.5 * np.minimum(dist / radius, 1.0))
        V[:, 1] = np.minimum(V[:, 1], 1.0)                      # no extra lift: pieces drop and pile up instead of floating
        col = _S_RGB[np.repeat(mats, k)] * _rng.uniform(0.8, 1.1, (n * k, 1)).astype(np.float32)
        BODIES.spawn_chunks(P, V, _rng.uniform(0.12, 0.26, n * k) * scale, col, world)

    def update(self, world, dt):
        dt = min(dt, 0.05)
        for s in list(self.structs):
            new = s.update(world, dt)
            if new:
                self.bodies.extend(new)
                self.wake_all()
                BODIES.wake_all()                     # a piece broke away: cubes resting on it must re-check
            if s.crack_pts:
                self._dust([p for p, _ in s.crack_pts], [m for _, m in s.crack_pts])
                s.crack_pts = []
            for cc, mm in s.crumbs:
                self._dust(cc, mm, per=2)
                self._throw_chunks(cc, mm, cc.mean(0) - np.array([0.0, 0.4, 0.0]), 1.5, slow=True, world=world, cs=s.cs)
            s.crumbs = []
            if not s.alive.any():
                self.structs.remove(s)
        for o in self.objects():
            o.cool(dt)                                    # heat from the laser drains away
        awake = [b for b in self.bodies if not b.asleep]
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
        """First point along a ray that is inside a structure or body, or None. Returns (t, last point outside)."""
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
        return float(t[k]), P[max(k - 1, 0)]


STRUCTS = StructureManager()


BLAST_RADIUS = 3.0       # blocks: how far a blast reaches
BLAST_POWER  = 1.0       # 1.0 shatters concrete cells inside about half the radius and cracks the ring beyond


def _structure_chip(self, world, point):
    """The drill on a structure: a small blast that bites out a few cells."""
    broken = self.blast(world, point, 1.0, 1.6)
    fx_explosion(point, 0.5, broken, _S_RGB)


StructureManager.chip = _structure_chip


# ---- the demo layout: one solid test cube per material, beside the dirt cube ------------------------------
def _solid_cube(mat):
    return lambda n: np.full((n, n, n), mat, np.int16)           # always 10 x 10 x 10 blocks; n = 10 * cells per block


DEMO_LAYOUT = [(0, _solid_cube(DIRT)), (1, _solid_cube(STONE)), (2, _solid_cube(BRICK)), (3, _solid_cube(PLANKS)), (4, _solid_cube(STEEL))]


def _layout_ready(self, world):
    """True when every chunk under the layout is loaded."""
    for cx in range(math.floor(-8 / CHUNK_SIZE), math.floor(70 / CHUNK_SIZE) + 1):
        for cz in range(math.floor(-50 / CHUNK_SIZE), math.floor(0 / CHUNK_SIZE) + 1):
            if (cx, cz) not in world.dens:
                return False
    return True


def _build_layout(self, world, res=2):
    """Stand the demo shapes in a row beside the dirt cube (slot 0), on the same grid as the cubes: a slot is
    14 blocks across, four to a row, rows going back along -z. `res` is cells per block along each side: the
    cubes keep their size, and a higher value builds them from smaller pieces."""
    res = max(1, int(res))
    cs = 1.0 / res
    for k, make in DEMO_LAYOUT:
        m = make(10 * res)
        x0, _, z0 = cube_slot_lo(min(k, 3))
        x0 += 14.0 * max(k - 3, 0)                                 # a fifth slot continues the row instead of starting a new one
        x0 -= 0.5                                                  # the cube's faces sit half a block outside its samples
        z0 -= 0.5
        w, d = m.shape[0] * cs, m.shape[2] * cs                    # centre each shape in its 10 x 10 slot
        self.add(world, m, x0 + (10.0 - w) * 0.5, z0 + (10.0 - d) * 0.5, cs)


StructureManager.layout_ready = _layout_ready
StructureManager.build_layout = _build_layout


_SOLID_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aCol;
uniform mat4 uVP;
uniform mat4 uModel;       // the object's pose: rotation and position
out vec3 vCol;
out vec3 vNrm;
void main() {
    vec4 w = uModel * vec4(aPos, 1.0);
    vCol = aCol;
    vNrm = mat3(uModel) * aNrm;
    gl_Position = uVP * w;
}
"""

_SOLID_FRAG = """
#version 330 core
in vec3 vCol;
in vec3 vNrm;
out vec4 fragColor;
uniform float uLight;
uniform float uAlpha;
void main() {
    vec3 n = normalize(vNrm);
    vec3 sun = normalize(vec3(0.45, 0.80, 0.35));
    float diff = max(dot(n, sun), 0.0);
    float sky = 0.5 + 0.5 * n.y;
    float shade = 0.38 + 0.30 * sky + 0.42 * diff;
    fragColor = vec4(vCol * shade * uLight, uAlpha);
}
"""


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
        self.icache = {}     # integrity colouring: id(object) -> [vao, vbo, count, (mesh ver, stress ver), object]
        self.cache = {}      # id(object) -> [vao, vbo, vertex count, mesh version, object]

    def _make(self):
        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        for loc, off in ((0, 0), (1, 12), (2, 24)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 36, ctypes.c_void_p(off))
        gl.glBindVertexArray(0)
        return vao, vbo

    def render(self, vp, ambient, overlay=False):
        objs = STRUCTS.objects()
        live = {id(o) for o in objs}
        for k in [k for k in self.icache if k not in live]:
            vao, vbo = self.icache.pop(k)[:2]
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteBuffers(1, [vbo])
        for k in [k for k in self.cache if k not in live]:
            vao, vbo = self.cache.pop(k)[:2]
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteBuffers(1, [vbo])
        if not objs:
            return
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glUniform1f(self.u_light, max(float(ambient), GFX['ambient_floor']))
        gl.glUniform1f(self.u_alpha, 1.0)
        model = np.eye(4, dtype=np.float32)
        for o in objs:
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
            for o in objs:
                ent = self.icache.get(id(o))
                if ent is None:
                    vao, vbo = self._make()
                    ent = self.icache[id(o)] = [vao, vbo, 0, None, o]
                key = (o.mesh_ver, o.stress_ver, o.heated)
                if ent[3] != key or o.heated:
                    data = np.ascontiguousarray(o.mesh('integrity'), np.float32)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, ent[1])
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
                    ent[2], ent[3] = len(data), key
                if not ent[2]:
                    continue
                model[:3, :3] = o.R
                model[:3, 3] = o.frame_pos
                gl.glUniformMatrix4fv(self.u_model, 1, gl.GL_TRUE, model)
                gl.glBindVertexArray(ent[0])
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, ent[2])
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glDisable(gl.GL_BLEND)
            gl.glUniform1f(self.u_alpha, 1.0)
        gl.glBindVertexArray(0)





# ---- weapons: keys 1-6 pick one, the left mouse button fires it -------------------------------------------
def explode(world, centre, radius, power, settings, dig=True):
    """A blast at a point: structures and loose pieces take the damage and the push, the terrain is carved."""
    broken = STRUCTS.blast(world, centre, radius, power)
    fx_explosion(centre, radius, broken, _S_RGB)
    if dig and STRUCT_BLAST_DIGS:
        dug = world.dig(centre, radius * 0.9, DIG_STRENGTH * 1.6)
        BODIES.spawn(world, world.take_dug_cells(), int(settings['dirt_cubes']), settings['dirt_cube_size'])
        if dug > 0.0:
            STRUCTS.on_terrain_changed(world)


def _first_hit(world, origin, d, reach):
    """Nearest solid thing along a ray, terrain or structure: the last free point before it, or None."""
    origin = np.asarray(origin, np.float64)
    th = raycast_terrain(world, origin, d, reach)
    sh = STRUCTS.raycast(origin, d, reach)
    t_ter = float(np.linalg.norm(np.array(th, np.float64) - origin)) if th is not None else 1e9
    if sh is not None and sh[0] < t_ter:
        return np.asarray(sh[1], np.float64)
    return None if th is None else np.array(th, np.float64)


WEAPONS = [
    dict(name='Drill',            note='hold: chips and digs',          rate=DIG_INTERVAL),
    dict(name='Rifle',            note='hitscan, small hole',           rate=0.12),
    dict(name='Grenade launcher', note='arcing shell, medium blast',    rate=0.8),
    dict(name='Rocket launcher',  note='fast, big blast',               rate=1.2),
    dict(name='Melting laser',    note='hold: melts what it touches',   rate=0.0),
    dict(name='Demo charge',      note='sticks; right click detonates', rate=0.4),
]
_CHARGE_MAX = 8


class Projectiles:
    """Grenades, rockets and sticky charges in flight or placed."""
    GRAV = {'grenade': 18.0, 'rocket': 0.0, 'charge': 24.0}

    def __init__(self):
        self.items = []

    def clear(self):
        self.items = []

    def launch(self, kind, pos, vel):
        if kind == 'charge' and sum(p['kind'] == 'charge' for p in self.items) >= _CHARGE_MAX:
            for p in self.items:                       # the oldest charge makes room
                if p['kind'] == 'charge':
                    self.items.remove(p)
                    break
        self.items.append(dict(kind=kind, pos=np.array(pos, np.float64), vel=np.array(vel, np.float64),
                               age=0.0, stuck=False))

    @property
    def n_charges(self):
        return sum(p['kind'] == 'charge' for p in self.items)

    def detonate(self, world, settings):
        keep = []
        for p in self.items:
            if p['kind'] == 'charge':
                explode(world, p['pos'], 5.0, 3.2, settings)
            else:
                keep.append(p)
        self.items = keep

    def update(self, world, dt, settings):
        keep = []
        for p in self.items:
            if p['stuck']:
                p['age'] += dt
                keep.append(p)
                continue
            p['age'] += dt
            p['vel'][1] -= self.GRAV[p['kind']] * dt
            step = p['vel'] * dt
            L = float(np.linalg.norm(step))
            hit = _first_hit(world, p['pos'], step / L, L + 0.05) if L > 1e-9 else None
            if hit is not None:
                if p['kind'] == 'charge':
                    p['pos'], p['vel'], p['stuck'] = hit, np.zeros(3), True
                    keep.append(p)
                elif p['kind'] == 'rocket':
                    explode(world, hit, 4.2, 2.4, settings)
                else:
                    explode(world, hit, 3.2, 1.2, settings)
                continue
            p['pos'] = p['pos'] + step
            if p['kind'] == 'grenade' and p['age'] > 3.5:
                explode(world, p['pos'], 3.2, 1.2, settings)
            elif p['age'] > 8.0 or p['pos'][1] < -50.0:
                pass                                     # flew off: gone
            else:
                keep.append(p)
        self.items = keep


PROJ = Projectiles()
_last_shot = {}


def fire_weapon(world, player, settings, idx):
    """Called every frame the left button is held: fires weapon idx when its own rate allows."""
    w = WEAPONS[idx]
    now = time.perf_counter()
    if now - _last_shot.get(idx, 0.0) < w['rate']:
        return
    prev = _last_shot.get(idx)
    _last_shot[idx] = now
    held_dt = 1.0 / 60.0 if prev is None or now - prev > 0.12 else min(now - prev, 0.05)   # seconds the button has been held this frame
    eye = np.asarray(player.eye, np.float64)
    fwd = np.asarray(player.forward(), np.float64)
    if idx == 0:                                         # drill: the original dig, and a chip on structures
        hit = raycast_terrain(world, eye, fwd, REACH)
        sh = STRUCTS.raycast(eye, fwd, REACH)
        t_hit = float(np.linalg.norm(np.array(hit, np.float64) - eye)) if hit is not None else 1e9
        if sh is not None and sh[0] < t_hit:
            STRUCTS.chip(world, sh[1])
        elif hit is not None:
            dug = world.dig(hit, DIG_RADIUS, DIG_STRENGTH)
            BODIES.spawn(world, world.take_dug_cells(), int(settings['dirt_cubes']), settings['dirt_cube_size'])
            if dug > 0.0:
                STRUCTS.on_terrain_changed(world)
    elif idx == 1:                                       # rifle
        p = _first_hit(world, eye, fwd, 120.0)
        if p is not None:
            explode(world, p, 1.0, 1.5, settings, dig=False)
    elif idx == 2:                                       # grenade launcher
        PROJ.launch('grenade', eye + fwd * 0.8, fwd * 30.0 + np.array([0.0, 4.0, 0.0]))
    elif idx == 3:                                       # rocket launcher
        PROJ.launch('rocket', eye + fwd * 0.8, fwd * 55.0)
    elif idx == 4:                                       # melting laser
        laser_fire(world, eye, fwd, held_dt)
    elif idx == 5:                                       # demolition charge
        PROJ.launch('charge', eye + fwd * 0.8, fwd * 16.0 + np.array([0.0, 2.0, 0.0]))


_PROJ_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aCol;
uniform mat4 uVP;
out vec4 vCol;
void main() {
    vCol = aCol;
    gl_Position = uVP * vec4(aPos, 1.0);
    gl_PointSize = clamp(260.0 / max(gl_Position.w, 0.5), 5.0, 28.0);
}
"""
_PROJ_FRAG = """
#version 330 core
in vec4 vCol;
out vec4 fragColor;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    fragColor = vCol;
}
"""


# ---- melting laser ------------------------------------------------------------------------------------------
LASER_REACH        = 45.0    # blocks
LASER_MELT_RADIUS  = 1.5     # blocks: width of the tunnel it bores into the ground
LASER_MELT_RATE    = 16.0    # density removed per second at the centre of the spot
LASER_CELL_RADIUS  = 0.9     # blocks: how far from the spot a structure cell is heated
LASER_MELT_TIME    = 0.35    # seconds of full beam to melt a cell of hardness 1 (stone); steel takes 3x longer
LASER_COOL         = 1.2     # heat lost per second once the beam moves away
_MOLTEN_RGB = np.array([1.0, 0.42, 0.08], np.float32)
LASER = dict(t=-10.0, start=np.zeros(3), end=np.zeros(3), hit=False, ter_t=0.0)


def _grid_melt(self, point, radius, dt):
    """Heat the cells around a world point. A cell melts when its heat reaches 1, which takes longer for harder
    materials. Heated cells also crack, which weakens the bonds that hold a structure up.
    Returns (centres, materials) of the cells that melted this call."""
    none = (np.zeros((0, 3)), np.zeros(0, np.int64))
    q = ((np.asarray(point, np.float64) - self.frame_pos) @ self.R - self.corner) / self.cs      # the point in grid units
    r = radius / self.cs
    lo = np.maximum(np.floor(q - r).astype(np.int64), 0)
    hi = np.minimum(np.ceil(q + r).astype(np.int64), np.array(self.alive.shape))
    if (hi <= lo).any():
        return none
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    idx = np.argwhere(self.alive[sl]) + lo
    if not len(idx):
        return none
    dist = np.linalg.norm(idx + 0.5 - q, axis=1) * self.cs
    near = dist < radius
    if not near.any():
        return none
    idx, dist = idx[near], dist[near]
    ii = (idx[:, 0], idx[:, 1], idx[:, 2])
    falloff = 1.0 - dist / radius
    self.heat[ii] += dt * falloff / (_S_HARD[self.mat[ii]] * LASER_MELT_TIME)
    self.cdmg[ii] = np.maximum(self.cdmg[ii], np.minimum(0.85, 0.85 * self.heat[ii]))
    self.heated = True
    self._mesh_cache = None
    self.mesh_ver += 1
    gone = self.heat[ii] >= 1.0
    if not gone.any():
        if hasattr(self, 'stress_dirty'):
            self.stress_dirty = True                  # cracked cells carry less: look at the supports again
        return none
    g = idx[gone]
    mask = np.zeros(self.alive.shape, bool)
    mask[g[:, 0], g[:, 1], g[:, 2]] = True
    centres = self.frame_pos + ((g + 0.5) * self.cs + self.corner) @ self.R.T
    mats = self.mat[mask].copy()
    self._remove(mask)
    self.heat[mask] = 0.0
    return centres, mats


def _grid_cool(self, dt):
    """Heat drains away once the beam has moved off a cell."""
    if not self.heated:
        return
    self.heat *= math.exp(-LASER_COOL * dt)
    self.heat[self.heat < 0.03] = 0.0
    self.heated = bool(self.heat.any())
    self._mesh_cache = None
    self.mesh_ver += 1


def _body_remove(self, kill):
    """A loose body loses cells: its mass and spin properties are worked out again for what is left."""
    self.mat[kill] = 0
    self.cdmg[kill] = 0
    self.heat[kill] = 0
    self._geometry()
    if self.alive.any():
        self._mass_props()
        self.wake()


GridObject.melt = _grid_melt
GridObject.cool = _grid_cool
RigidBody._remove = _body_remove


def _raycast_obj(self, origin, direction, reach, step=0.05):
    """Like raycast, but also says which object was hit. Returns (t, last free point, object) or None."""
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


def _molten_drips(world, centres, mats, per=2, cap=40, cs=0.5):
    """Small glowing cubes where something melted. They drop and pile up like any other debris."""
    C = np.asarray(centres, np.float64).reshape(-1, 3)
    M = np.asarray(mats, np.int64).reshape(-1)
    if not len(C):
        return
    if len(C) * per > cap:
        pick = _rng.choice(len(C), max(1, cap // per), replace=False)
        C, M = C[pick], M[pick]
    P = np.repeat(C, per, axis=0) + _rng.uniform(-0.2, 0.2, (len(C) * per, 3))
    n = len(P)
    V = _rng.normal(0.0, 0.5, (n, 3))
    V[:, 1] = _rng.uniform(-0.5, 0.3, n)
    base = 0.3 * _S_RGB[np.repeat(M, per)] + 0.7 * _MOLTEN_RGB
    BODIES.spawn_chunks(P, V, _rng.uniform(0.08, 0.17, n) * (cs / 0.5), base *_rng.uniform(0.85, 1.1, (n, 1)).astype(np.float32), world)


def _manager_melt(self, world, point, radius, dt):
    """Heat every structure and loose body near the point; whatever melts drips."""
    point = np.asarray(point, np.float64)
    out = False
    for o in self.objects():
        lo, hi = o.aabb()
        if (point < lo - radius).any() or (point > hi + radius).any():
            continue
        c, m = o.melt(point, radius, dt)
        if len(c):
            _molten_drips(world, c, m, cs=o.cs)
            out = True
    if out:
        BODIES.wake_all()                      # cubes asleep on the melted cells must fall
    return out


StructureManager.raycast_obj = _raycast_obj
StructureManager.melt = _manager_melt


def _terrain_hit_t(world, eye, fwd, reach, step=0.15):
    """Distance along the ray to the ground surface, from the smooth density field, or None."""
    ts = np.arange(0.0, reach, step)
    P = eye[None, :] + ts[:, None] * fwd[None, :]
    d = _density_many(world, P)
    hit = np.nonzero(d > 0.0)[0]
    if not len(hit):
        return None
    k = int(hit[0])
    if k == 0:
        return 0.0
    d0, d1 = d[k - 1], d[k]
    return float(ts[k - 1] + step * (0.0 - d0) / max(d1 - d0, 1e-9))


def laser_fire(world, eye, fwd, dt):
    """One frame of the beam: find what it touches, heat it, and remember where it ended so it can be drawn."""
    eye = np.asarray(eye, np.float64)
    fwd = np.asarray(fwd, np.float64)
    t_ter = _terrain_hit_t(world, eye, fwd, LASER_REACH)
    so = STRUCTS.raycast_obj(eye, fwd, LASER_REACH)
    use_struct = so is not None and (t_ter is None or so[0] <= t_ter)
    if so is None and t_ter is None:
        end, hit = eye + fwd * LASER_REACH, False
    elif use_struct:
        end, hit = np.asarray(so[1], np.float64), True
        STRUCTS.melt(world, end + fwd * 0.1, LASER_CELL_RADIUS, dt)
    else:
        end, hit = eye + fwd * t_ter, True
        spot = end + fwd * 0.7
        # The ground gets denser the deeper it goes, so a fixed rate would slow to a crawl as the hole deepens.
        # Scale the rate by how dense the spot is, which keeps the bore at a steady speed.
        deep = max(0.0, float(_density_many(world, spot[None, :])[0]))
        dug = world.dig(spot, LASER_MELT_RADIUS, LASER_MELT_RATE * dt * (1.0 + 1.5 * deep))
        cells = world.take_dug_cells()
        if cells:
            _molten_drips(world, np.concatenate(cells), np.full(sum(len(c) for c in cells), DIRT), per=1)
        LASER['ter_t'] += dt
        if dug > 0.0 and LASER['ter_t'] > 0.12:           # tell structures and loose pieces the ground changed
            LASER['ter_t'] = 0.0
            STRUCTS.on_terrain_changed(world)
    LASER.update(t=time.perf_counter(), start=eye.copy(), end=end, hit=hit)


_LASER_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aCol;
layout(location = 2) in vec2 aUV;
uniform mat4 uVP;
out vec4 vCol;
out vec2 vUV;
void main() {
    vCol = aCol;
    vUV = aUV;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

_LASER_FRAG = """
#version 330 core
in vec4 vCol;
in vec2 vUV;
out vec4 fragColor;
uniform int uMode;      // 0: beam ribbon, bright along its middle   1: round glow where it lands
void main() {
    float a;
    if (uMode == 0) {
        float w = 1.0 - abs(vUV.y * 2.0 - 1.0);
        a = w * w;
    } else {
        float r = length(vUV - vec2(0.5)) * 2.0;
        if (r > 1.0) discard;
        a = (1.0 - r) * (1.0 - r);
    }
    fragColor = vec4(vCol.rgb, vCol.a * a);
}
"""


class LaserRenderer:
    """Draws the beam as three additive ribbons (red halo, orange body, white core) and a glow where it lands."""
    STRIDE = 9 * 4

    def __init__(self):
        vs = _compile_shader(_LASER_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_LASER_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_mode = gl.glGetUniformLocation(self.program, "uMode")
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.STRIDE, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, self.STRIDE, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, self.STRIDE, ctypes.c_void_p(28))
        gl.glBindVertexArray(0)

    def _draw(self, quads, mode):
        """quads: list of (4 corners (4,3), rgba (4,)). Each is sent as two triangles."""
        if not quads:
            return
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        rows = []
        for corners, rgba in quads:
            for i in (0, 1, 2, 0, 2, 3):
                rows.append(np.concatenate([corners[i], rgba, uv[i]]))
        data = np.asarray(rows, np.float32)
        gl.glUniform1i(self.u_mode, mode)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(data))

    def render(self, vp, eye, yaw, pitch):
        now = time.perf_counter()
        if now - LASER['t'] > 0.12:                    # not fired lately: nothing to draw
            return
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = np.array([cp * math.sin(yaw), sp, -cp * math.cos(yaw)])
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        eye = np.asarray(eye, np.float64)
        start = eye + fwd * 0.6 + right * 0.30 - up * 0.24          # the beam leaves from the lower right of the view
        end = np.asarray(LASER['end'], np.float64)
        d = end - start
        L = float(np.linalg.norm(d))
        if L < 1e-3:
            return
        d /= L
        flick = 1.0 + 0.12 * math.sin(now * 90.0) + 0.06 * math.sin(now * 37.0)
        side = np.cross(d, (start + end) * 0.5 - eye)
        sn = float(np.linalg.norm(side))
        if sn < 1e-6:
            return
        side /= sn
        ribbons = []
        for half, rgba in ((0.11, (1.0, 0.12, 0.04, 0.55)), (0.05, (1.0, 0.55, 0.15, 0.85)), (0.018, (1.0, 1.0, 0.9, 1.0))):
            w = side * half * flick
            ribbons.append((np.array([start - w, end - w, end + w, start + w]), np.array(rgba, np.float32)))
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)                     # additive: the beam glows
        gl.glDepthMask(gl.GL_FALSE)
        gl.glBindVertexArray(self.vao)
        self._draw(ribbons, 0)
        if LASER['hit']:
            glow = []
            for size, rgba in ((0.55, (1.0, 0.25, 0.05, 0.9)), (0.28, (1.0, 0.75, 0.3, 1.0)), (0.1, (1.0, 1.0, 1.0, 1.0))):
                h = size * flick
                c = end + (eye - end) * 0.02
                glow.append((np.array([c - right * h - up * h, c + right * h - up * h,
                                       c + right * h + up * h, c - right * h + up * h]), np.array(rgba, np.float32)))
            self._draw(glow, 1)
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)



class ProjectileRenderer:
    """Draws projectiles and placed charges as round points."""
    def __init__(self):
        vs = _compile_shader(_PROJ_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_PROJ_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 28, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 28, ctypes.c_void_p(12))
        gl.glBindVertexArray(0)

    def render(self, vp):
        if not PROJ.items:
            return
        rows = []
        for p in PROJ.items:
            if p['kind'] == 'rocket':
                c = (1.0, 0.55, 0.1, 1.0)
            elif p['kind'] == 'grenade':
                c = (0.45, 0.85, 0.3, 1.0)
            else:
                c = (1.0, 0.15, 0.1, 1.0) if int(p['age'] * 4) % 2 == 0 else (1.0, 0.9, 0.2, 1.0)
            rows.append([*p['pos'], *c])
        data = np.asarray(rows, np.float32)
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(data))
        gl.glBindVertexArray(0)


class _CachedOverlay(Overlay):
    """An overlay that redraws its texture only when its state changes."""
    def draw(self, state):
        if state != getattr(self, '_state', None):
            self._state = state
            self.surf.fill((0, 0, 0, 0))
            self._render(state)
            self._data = pygame.image.tostring(self.surf, "RGBA", True)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self._data)
        gl.glUniform1i(self.u_tex, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)


_MAT_NAMES = {b: n for b, n in ((DIRT, 'Dirt'), (STONE, 'Stone'), (BRICK, 'Brick'), (PLANKS, 'Planks'), (STEEL, 'Steel'))}


def aimed_cell(player):
    """The structure cell under the crosshair as a dict of facts, or None."""
    eye = np.asarray(player.eye, np.float64)
    fwd = np.asarray(player.forward(), np.float64)
    so = STRUCTS.raycast_obj(eye, fwd, 40.0)
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
                stress=float(o.stress[c]), cracks=float(o.cdmg[c]), heat=float(o.heat[c]),
                hard=float(_S_HARD[m]), mass=float(o.cell_mass[c]),
                grounded=bool(o.anch[c]) if isinstance(o, Structure) else None)


def integrity_state(player):
    """Everything the panel shows as a tuple of strings, so it is redrawn only when something changes."""
    worst = 0.0
    for o in STRUCTS.objects():
        if o.alive.any():
            worst = max(worst, float(np.where(o.alive, np.maximum(o.stress, o.cdmg / 0.85), 0.0).max()))
    a = aimed_cell(player)
    lines = [f"{len(STRUCTS.structs)} fixed   {len(STRUCTS.bodies)} loose   worst cell {min(worst, 1.0) * 100:.0f}%"]
    if a is None:
        lines.append("crosshair: no structure")
    else:
        lines += [f"crosshair: {a['mat']}, {'fixed' if a['fixed'] else 'loose piece'}",
                  f"stress {min(a['stress'], 1.5) * 100:.0f}% of limit",
                  f"cracks {a['cracks'] * 100:.0f}%   heat {a['heat'] * 100:.0f}%",
                  f"hardness {a['hard']:.1f}   mass {a['mass']:.2f}" + ("" if a['grounded'] is None else f"   grounded {'yes' if a['grounded'] else 'no'}")]
    return tuple(lines)


class IntegrityPanel(_CachedOverlay):
    """Top left while G is on: colour legend, material strengths, and the cell under the crosshair."""
    W, H = 440, 392

    def __init__(self):
        super().__init__(self.W, self.H, 16, 16)
        self.f_t = _mono(24)
        self.f_s = _mono(17)

    def _render(self, state):
        s = self.surf
        pygame.draw.rect(s, (14, 16, 24, 225), (0, 0, self.W, self.H), border_radius=10)
        pygame.draw.rect(s, (90, 98, 120, 255), (0, 0, self.W, self.H), 2, border_radius=10)
        s.blit(self.f_t.render("STRUCTURE INTEGRITY", True, (226, 230, 238)), (14, 10))
        bx, by, bw, bh = 14, 46, self.W - 28, 16
        for i in range(bw):
            x = i / (bw - 1)
            col = (int(np.interp(x, [0, .5, 1], [.12, .95, .95]) * 255), int(np.interp(x, [0, .5, 1], [.75, .85, .12]) * 255),
                   int(np.interp(x, [0, .5, 1], [.30, .12, .08]) * 255))
            pygame.draw.line(s, col, (bx + i, by), (bx + i, by + bh))
        s.blit(self.f_s.render("safe", True, (180, 188, 204)), (bx, by + 20))
        end = self.f_s.render("breaks", True, (180, 188, 204))
        s.blit(end, (bx + bw - end.get_width(), by + 20))
        s.blit(self.f_s.render("colour = worst of stress, cracks, heat", True, (132, 140, 158)), (14, by + 42))
        y = by + 72
        s.blit(self.f_s.render("MATERIAL  compress tension shear hard", True, (70, 190, 230)), (14, y))
        y += 22
        for b, name in _MAT_NAMES.items():
            m = STRUCT_MATERIALS.get(b) if isinstance(STRUCT_MATERIALS, dict) else STRUCT_MATERIALS[b]
            if m is None:
                continue
            vals = m if isinstance(m, dict) else dict(zip(('rho', 'comp', 'tens', 'shear', 'hard'), m))
            s.blit(self.f_s.render(f"{name:<9} {vals['comp']:8.0f} {vals['tens']:7.0f} {vals['shear']:6.0f} {vals['hard']:5.1f}", True, (226, 230, 238)), (14, y))
            y += 21
        y += 12
        for i, line in enumerate(state):
            s.blit(self.f_s.render(line, True, (255, 220, 120) if i == 1 else (226, 230, 238)), (14, y))
            y += 22


class Crosshair(_CachedOverlay):
    SIZE = 44
    def __init__(self):
        super().__init__(self.SIZE, self.SIZE, (WIDTH - self.SIZE) // 2, (HEIGHT - self.SIZE) // 2)

    def _render(self, state):
        c, s = self.SIZE // 2, self.SIZE
        for col, wd in (((0, 0, 0, 200), 4), ((255, 255, 255, 255), 2)):     # dark outline, then white
            for a, b in (((c - 14, c), (c - 5, c)), ((c + 5, c), (c + 14, c)),
                         ((c, c - 14), (c, c - 5)), ((c, c + 5), (c, c + 14))):
                pygame.draw.line(self.surf, col, a, b, wd)
        pygame.draw.circle(self.surf, (0, 0, 0, 200), (c, c), 3)
        pygame.draw.circle(self.surf, (255, 255, 255, 255), (c, c), 1)


class WeaponBar(_CachedOverlay):
    """The six weapons as a row of boxes along the bottom of the screen; the picked one is lit."""
    BOX_W, BOX_H, GAP = 128, 76, 10
    SHORT = ['Drill', 'Rifle', 'Grenade', 'Rocket', 'Laser', 'Charge']

    def __init__(self):
        n = len(WEAPONS)
        w = n * self.BOX_W + (n - 1) * self.GAP
        super().__init__(w, self.BOX_H, (WIDTH - w) // 2, HEIGHT - self.BOX_H - 26)
        self.f_num = pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", 24, bold=True)
        self.f_name = pygame.font.SysFont("consolas,dejavusansmono,couriernew,monospace", 21, bold=True)

    def _render(self, state):
        weapon, charges = state
        for i in range(len(WEAPONS)):
            r = pygame.Rect(i * (self.BOX_W + self.GAP), 0, self.BOX_W, self.BOX_H)
            on = i == weapon
            pygame.draw.rect(self.surf, (50, 92, 150, 235) if on else (10, 12, 18, 190), r, border_radius=8)
            pygame.draw.rect(self.surf, (150, 205, 255, 255) if on else (80, 88, 108, 255), r, 3 if on else 2,
                             border_radius=8)
            self.surf.blit(self.f_num.render(str(i + 1), True, (255, 220, 120)), (r.x + 10, r.y + 6))
            name = self.f_name.render(self.SHORT[i], True, (255, 255, 255) if on else (190, 190, 200))
            self.surf.blit(name, name.get_rect(center=(r.centerx, r.bottom - 22)))
            if i == 5 and charges:
                c = self.f_num.render(str(charges), True, (255, 120, 100))
                self.surf.blit(c, (r.right - c.get_width() - 10, r.y + 6))


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
#  BENCHMARK (1 starts, 1 or Esc ends, results screen after)
# ═══════════════════════════════════════════════
BENCH_WARMUP_S  = 1.0              # the first second is skipped: chunks are still loading
BENCH_CLIMB    = 5.0              # the benchmark flies straight up this many blocks, then forward
BENCH_SLOW_MS   = 1000.0 / 30.0    # a frame slower than this counts as a slow frame

def _f(v):
    return float('nan') if v is None else float(v)

def summarize_benchmark(samples, duration, snapshot):
    """Turn the recorded samples into the figures shown on the results screen.
    Each sample is (time, frame_ms, cpu_ms, gpu_ms, gpu_util, gpu_temp, gpu_power,
    vram_used, vram_total, load_ms, mesh_ms, upload_ms, draw_ms, list_ms, call_ms, collect_ms)."""
    kept = [s for s in samples if s[0] >= BENCH_WARMUP_S and s[2] is not None and s[3] is not None]
    if len(kept) < 10:
        return {'too_short': True, 'duration': duration, 'frames': len(kept)}
    frame = np.maximum(np.array([s[1] for s in kept], dtype=float), 0.01)
    cpu = np.array([s[2] for s in kept], dtype=float)
    gpu = np.array([s[3] for s in kept], dtype=float)
    load = np.array([s[9] for s in kept], dtype=float)
    mesh = np.array([s[10] for s in kept], dtype=float)
    upload = np.array([s[11] for s in kept], dtype=float)
    other = np.array([s[12] for s in kept], dtype=float) - mesh - upload   # draw time not spent meshing or uploading
    list_ms = np.array([s[13] for s in kept], dtype=float)
    call_ms = np.array([s[14] for s in kept], dtype=float)
    collect_ms = np.array([s[15] for s in kept], dtype=float)
    util = [s[4] for s in kept if s[4] is not None]
    temp = [s[5] for s in kept if s[5] is not None]
    power = [s[6] for s in kept if s[6] is not None]
    used = [s[7] for s in kept if s[7] is not None]
    total = next((s[8] for s in reversed(kept) if s[8]), None)

    worst = frame >= np.percentile(frame, 99)     # the slowest 1% of frames

    def wmean(a): return float(a[worst].mean())
    def avg(a): return float(np.mean(a)) if len(a) else None
    def top(a): return float(np.max(a)) if len(a) else None

    frame_avg = float(frame.mean())
    p99 = float(np.percentile(frame, 99))
    return {
        'too_short': False, 'duration': duration, 'frames': len(kept),
        'avg_fps': 1000.0 / frame_avg,
        'low1': 1000.0 / p99,
        'low01': 1000.0 / float(np.percentile(frame, 99.9)),
        'frame_avg': frame_avg, 'frame_p99': p99, 'frame_worst': float(frame.max()),
        'slow_frames': int((frame > BENCH_SLOW_MS).sum()),
        'cpu_avg': float(cpu.mean()), 'cpu_worst': wmean(cpu),
        'gpu_avg': float(gpu.mean()), 'gpu_worst': wmean(gpu),
        'load_avg': float(load.mean()), 'load_worst': wmean(load),
        'mesh_avg': float(mesh.mean()), 'mesh_worst': wmean(mesh),
        'upload_avg': float(upload.mean()), 'upload_worst': wmean(upload),
        'other_avg': float(other.mean()), 'other_worst': wmean(other),
        'list_avg': float(list_ms.mean()), 'list_worst': wmean(list_ms),
        'call_avg': float(call_ms.mean()), 'call_worst': wmean(call_ms),
        'collect_avg': float(collect_ms.mean()), 'collect_worst': wmean(collect_ms),
        'util_avg': avg(util), 'util_max': top(util),
        'temp_max': top(temp),
        'power_avg': avg(power), 'power_max': top(power),
        'vram_peak': top(used), 'vram_total': total,
        'snapshot': snapshot,
    }

class BenchmarkSession:
    """Records every frame between starting and ending a benchmark. The FPS cap is lifted
    while it runs so the numbers show what the game really costs, and restored afterwards."""
    def __init__(self):
        self.active = False
        self.t0 = 0.0
        self.samples = []
        self.saved_fps = None
        self.snapshot = None
        self.start_y = 0.0
        self.saved_flying = False

    def start(self, settings, player):
        self.active = True
        self.t0 = time.perf_counter()
        self.samples = []
        self.saved_fps = settings['fps_limit']
        self.snapshot = dict(settings)
        settings['fps_limit'] = 0        # uncapped
        self.start_y = float(player.pos[1])
        self.saved_flying = player.flying
        if not player.flying:
            player.toggle_fly()          # the benchmark flies: climb, then forward

    def sample(self, frame_ms, cpu_ms, gpu_ms, stats, parts):
        st = stats or {}
        self.samples.append((time.perf_counter() - self.t0, frame_ms, cpu_ms, gpu_ms,
                             st.get('util'), st.get('temp'), st.get('power'),
                             st.get('mem_used'), st.get('mem_total'), *parts))

    def stop(self, settings, player):
        self.active = False
        settings['fps_limit'] = self.saved_fps
        if player.flying != self.saved_flying:
            player.toggle_fly()
        return summarize_benchmark(self.samples, time.perf_counter() - self.t0, self.snapshot)

RES_W, RES_H = 900, 740
class BenchmarkResults(Overlay):
    """Results screen shown when a benchmark ends. Closed with the button, Enter or Esc."""
    def __init__(self):
        super().__init__(RES_W, RES_H, (WIDTH - RES_W) // 2, (HEIGHT - RES_H) // 2)
        self.f_title = _mono(34)
        self.f_head = _mono(20)
        self.f_row = _mono(20)
        self.f_btn = _mono(24)
        self.x0 = (WIDTH - RES_W) // 2
        self.y0 = (HEIGHT - RES_H) // 2
        self.close_rect = pygame.Rect(RES_W // 2 - 110, RES_H - 76, 220, 52)

    def hit_close(self, screen_pos):
        return self.close_rect.collidepoint(screen_pos[0] - self.x0, screen_pos[1] - self.y0)

    def _section(self, x, y, title, rows, width=390):
        s = self.surf
        s.blit(self.f_head.render(title, True, UI_ACCENT), (x, y))
        pygame.draw.line(s, UI_LINE, (x, y + 30), (x + width, y + 30), 1)
        y += 40
        for label, value in rows:
            s.blit(self.f_row.render(label, True, UI_DIM), (x, y))
            v = self.f_row.render(value, True, UI_TEXT)
            s.blit(v, (x + width - v.get_width(), y))
            y += 30
        return y + 14

    def _stats(self, r):
        def ms(v): return 'n/a' if v is None else f"{v:.1f} ms"
        def num(v, unit=''): return 'n/a' if v is None else f"{v:.0f}{unit}"
        def pair(a, b, unit): return 'n/a' if a is None else f"{a:.0f}{unit} / {b:.0f}{unit}"
        def ms_pair(a, b): return f"{a:.1f} / {b:.1f} ms"
        snap = r['snapshot']
        frames = max(r['frames'], 1)
        vram = ('n/a' if r['vram_peak'] is None or not r['vram_total']
                else f"{r['vram_peak'] / 1024:.1f} / {r['vram_total'] / 1024:.1f} GB")
        y = self._section(36, 84, "FRAME RATE", [
            ("Average FPS", num(r['avg_fps'])),
            ("1% low FPS", num(r['low1'])),
            ("0.1% low FPS", num(r['low01'])),
            ("Slow frames (<30 fps)", f"{r['slow_frames']} ({r['slow_frames'] / frames * 100:.1f}%)"),
            ("Average frame", ms(r['frame_avg'])),
            ("99th percentile frame", ms(r['frame_p99'])),
            ("Worst frame", ms(r['frame_worst'])),
        ])
        self._section(36, y, "TIMINGS  (avg / worst 1%)", [
            ("CPU work", ms_pair(r['cpu_avg'], r['cpu_worst'])),
            ("Chunk loading", ms_pair(r['load_avg'], r['load_worst'])),
            ("Meshing", ms_pair(r['mesh_avg'], r['mesh_worst'])),
            ("GPU upload", ms_pair(r['upload_avg'], r['upload_worst'])),
            ("Draw and checks", ms_pair(r['other_avg'], r['other_worst'])),
            ("GPU", ms_pair(r['gpu_avg'], r['gpu_worst'])),
        ])
        y = self._section(474, 84, "GPU", [
            ("Utilisation avg / max", pair(r['util_avg'], r['util_max'], '%')),
            ("Temperature max", num(r['temp_max'], ' C')),
            ("Power avg / max", pair(r['power_avg'], r['power_max'], ' W')),
            ("VRAM peak / total", vram),
        ])
        y = self._section(474, y, "RUN", [
            ("Run length", f"{r['duration']:.1f} s"),
            ("Frames measured", str(r['frames'])),
            ("Field of view", f"{snap['fov']} deg"),
            ("Render distance", f"{snap['render_distance']} chunks"),
            ("Cache chunks", "On" if snap['cache_chunks'] else "Off"),
            ("Performance graphs", "On" if snap['show_hud'] else "Off"),
        ])
        self._section(474, y, "DRAW PARTS  (avg / worst 1%)", [
            ("List rebuild", ms_pair(r['list_avg'], r['list_worst'])),
            ("Draw call", ms_pair(r['call_avg'], r['call_worst'])),
            ("Mesh collection", ms_pair(r['collect_avg'], r['collect_worst'])),
        ])

    def _render(self, res):
        s = self.surf
        s.fill((0, 0, 0, 0))
        pygame.draw.rect(s, UI_BG, (0, 0, self.w, self.h), border_radius=12)
        pygame.draw.rect(s, UI_BORDER, (0, 0, self.w, self.h), 2, border_radius=12)
        title = self.f_title.render("BENCHMARK RESULTS", True, UI_TEXT)
        s.blit(title, ((self.w - title.get_width()) // 2, 18))

        if res['too_short']:
            msg = self.f_row.render("Run too short to measure. Let it run for a few seconds.", True, UI_DIM)
            s.blit(msg, ((self.w - msg.get_width()) // 2, 200))
        else:
            self._stats(res)

        btn = self.close_rect
        pygame.draw.rect(s, UI_SEG_ON, btn, border_radius=6)
        label = self.f_btn.render("CLOSE", True, UI_TEXT)
        s.blit(label, label.get_rect(center=btn.center))

def run():
    # Open the window centred on the primary monitor instead of wherever the OS puts it.
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    _fit_window_to_display()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("VoidCraft")
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    print(_gpu_init())
    settings = default_settings()
    menu = TabMenu(settings)
    settings_panel = SettingsPanel()
    hud = PerfHud()
    poller = GpuPoller()
    perf = PerfMonitor()
    world = World()
    renderer = Renderer()
    player = PlayerController((0.5, float(terrain_height(0.0, 0.0)) + 2.0, 0.5))
    selected = 0
    last_centre = None
    last_caption = None
    last_cpu_ms = None
    paused = False
    auto_walk = False            # on while benchmarking: walk forward without holding W
    last_dig = 0.0               # when the last dig tick happened
    bench = BenchmarkSession()
    bench_results = None         # summary shown when a benchmark ends; None = no results screen
    results_panel = BenchmarkResults()
    load_ms = draw_ms = 0.0      # per-frame timings for the benchmark breakdown
    backdrop = Backdrop()
    grid = ChunkGrid()
    particles_gl = ParticleRenderer()
    body_gl = DirtBodyRenderer()
    struct_gl = StructureRenderer()
    weapon = 0                   # the weapon picked with keys 1-6
    crosshair = Crosshair()
    weapon_bar = WeaponBar()
    proj_gl = ProjectileRenderer()
    laser_gl = LaserRenderer()
    integrity_panel = IntegrityPanel()
    show_integrity = True
    built_res = settings['struct_res']   # cube detail the layout was last built at
    pending_layout = True        # the shapes beside the dirt cube are built once the chunks under them are loaded
    menu_frozen = False          # True once the world has been frozen behind the open menu
    clock = pygame.time.Clock()

    running = True
    while running:
        # With the menu open the world is frozen, so idle at a low rate instead of the FPS limit.
        dt = clock.tick(30 if (menu.open or bench_results is not None) else settings['fps_limit']) / 1000.0    # 0 = uncapped
        work_start = time.perf_counter()                     # everything after the cap's sleep
        perf.record(dt * 1000.0, renderer.last_gpu_ms, last_cpu_ms)
        if bench.active:
            bench.sample(dt * 1000.0, last_cpu_ms, renderer.last_gpu_ms, poller.stats,
                         (load_ms, renderer.mesh_s * 1000.0, renderer.upload_s * 1000.0, draw_ms,
                          renderer.list_s * 1000.0, renderer.call_s * 1000.0, renderer.collect_s * 1000.0))
        rd = settings['render_distance']

        events = pygame.event.get()
        bench_toggle = bench_stop = False
        pending_reset = False
        for e in events:
            if bench_results is not None:
                # The results screen takes all input until it is closed.
                if e.type == QUIT:
                    running = False
                elif ((e.type == KEYDOWN and e.key in (K_ESCAPE, K_RETURN, K_KP_ENTER))
                      or (e.type == MOUSEBUTTONDOWN and e.button == 1 and results_panel.hit_close(e.pos))):
                    bench_results = None
                continue
            if e.type == KEYDOWN and e.key == K_ESCAPE and bench.active:
                bench_stop = True           # Esc ends a benchmark rather than quitting
                continue
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE and not menu.open):
                running = False
            elif e.type == KEYDOWN and ((e.key == K_TAB and not bench.active) or (e.key == K_ESCAPE and menu.open)):
                menu.open = not menu.open
                menu.dirty = True
                pygame.mouse.set_visible(menu.open)
                pygame.event.set_grab(not menu.open)
            elif menu.open and e.type == MOUSEBUTTONDOWN and e.button == 1:
                act = settings_panel.mouse_down(menu, e.pos)
                if act == 'exit':
                    running = False
                elif act == 'reset_damage':
                    pending_reset = True
            elif menu.open and e.type == MOUSEBUTTONUP and e.button == 1:
                menu.drag = None
            elif menu.open and e.type == MOUSEMOTION and menu.drag is not None:
                settings_panel.mouse_drag(menu, e.pos)
            elif e.type == KEYDOWN and e.key == K_g and not bench.active:
                settings['show_hud'] = not settings['show_hud']
            elif menu.open:
                if e.type == KEYDOWN and e.key == K_UP:
                    menu.move(-1)
                elif e.type == KEYDOWN and e.key == K_DOWN:
                    menu.move(1)
                elif e.type == KEYDOWN and e.key == K_LEFTBRACKET:
                    menu.switch_tab(-1)
                elif e.type == KEYDOWN and e.key == K_RIGHTBRACKET:
                    menu.switch_tab(1)
                elif e.type == KEYDOWN and e.key == K_LEFT:
                    menu.adjust(-1)
                elif e.type == KEYDOWN and e.key == K_RIGHT:
                    menu.adjust(1)
                elif e.type == KEYDOWN and e.key in (K_RETURN, K_KP_ENTER):
                    menu.activate()
            elif e.type == KEYDOWN and e.key == K_r and not bench.active:
                pending_reset = True                                 # R: put the whole map back and clear all damage
                last_centre = None
            elif e.type == KEYDOWN and K_1 <= e.key <= K_6 and not bench.active:
                weapon = e.key - K_1                                 # 1-6: pick a weapon
            elif e.type == MOUSEBUTTONDOWN and e.button == 3 and not menu.open and not bench.active:
                PROJ.detonate(world, settings)                       # right click: set off the placed charges
            elif e.type == KEYDOWN and e.key == K_f and not bench.active:
                paused = not paused
            elif e.type == KEYDOWN and e.key == K_v and not bench.active:
                if settings['show_hud']:
                    show_integrity = not show_integrity      # in G mode V shows/hides the text panels
                else:
                    player.toggle_fly()
            elif e.type == KEYDOWN and e.key == K_F5:
                bench_toggle = True         # F5 starts a benchmark (autowalk on); F5 again ends it
            elif e.type == MOUSEMOTION and not paused:
                player.yaw += e.rel[0] * 0.002
                player.pitch = max(-1.5, min(1.5, player.pitch - e.rel[1] * 0.002))

        if settings['struct_res'] != built_res:                      # cube detail changed: rebuild the cubes at the new detail
            built_res = settings['struct_res']
            STRUCTS.clear()
            BODIES.clear()
            PROJ.clear()
            pending_layout = True
        if pending_reset or menu.take_action() == 'reset_damage':
            world.reset_damage()
            STRUCTS.clear()
            PROJ.clear()
            pending_layout = True                                    # the whole map comes back whole
            BODIES.clear()
            world.take_dug_cells()
            PARTS.clear()

        if bench_toggle or bench_stop:
            if bench.active:
                bench_results = bench.stop(settings, player)
                auto_walk = False
            else:
                bench.start(settings, player)
                auto_walk = True

        modal = menu.open or bench_results is not None   # menu or results screen is up: world frozen

        # Menu open and nothing changed in it: keep the frozen frame on screen and skip the work.
        menu_changed = any(e.type != MOUSEMOTION or menu.drag is not None for e in events)
        if modal and menu_frozen and not menu_changed:
            last_cpu_ms = None
            continue

        GFX['physics_device'] = 'gpu' if (settings['gpu_physics'] and GPU_OK) else 'cpu'
        CONFIG['debris_on'] = settings['debris_on']
        CONFIG['debris_amount'] = settings['debris_amount'] / 100.0
        if not settings['debris_on']:
            BODIES.clear()
            PARTS.remove_kind(FX_DUST)
        keys = pygame.key.get_pressed()
        t_load = time.perf_counter()
        if not paused and not modal:
            load_around(world, player.pos[0], player.pos[2], radius=rd)
            if pending_layout and STRUCTS.layout_ready(world):
                STRUCTS.build_layout(world, settings['struct_res'])
                pending_layout = False
        load_ms = (time.perf_counter() - t_load) * 1000.0

        if not modal and not paused:
            yaw = player.yaw
            fwd = np.array([math.sin(yaw), 0.0, -math.cos(yaw)])
            right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
            wish = np.zeros(3)
            if keys[K_w] or auto_walk: wish += fwd
            if keys[K_s]: wish -= fwd
            if keys[K_d]: wish += right
            if keys[K_a]: wish -= right
            if player.flying:
                if keys[K_SPACE]: wish[1] += 1.0
                if keys[K_LCTRL]: wish[1] -= 1.0
            if bench.active and player.pos[1] < bench.start_y + BENCH_CLIMB:
                wish = np.array([0.0, 1.0, 0.0])     # climb straight up first
            n = np.linalg.norm(wish)
            if n > 0:
                wish /= n
            player.update(world, dt, wish, keys[K_SPACE], keys[K_LSHIFT])
            if pygame.mouse.get_pressed()[0] and not bench.active:
                fire_weapon(world, player, settings, weapon)         # the picked weapon, at its own rate
            PROJ.update(world, dt, settings)
            PARTS.update(dt)
            BODIES.update(world, dt, settings['cube_sleep_time'])
            STRUCTS.update(world, dt)
            STRUCTS.eject_player(world, player)                      # a falling piece does not trap the player

        # Unload distant chunks only when caching is off, and only when the player
        # enters a new chunk, so the check doesn't run every frame.
        centre = (math.floor(player.pos[0] / CHUNK_SIZE), math.floor(player.pos[2] / CHUNK_SIZE))
        if not settings['cache_chunks'] and centre != last_centre:
            unload_far_chunks(world, renderer, centre[0], centre[1], keep=rd + 1)
        last_centre = centre

        # Heading and position go in the window title, since the 3D view has no HUD.
        state = ("BENCHMARK   " if bench.active else "") + ("PAUSED   " if paused else "") + ("AUTO-WALK   " if auto_walk else "")
        caption = f"VoidCraft   {state}{FPS.update(dt):.0f} fps"
        if caption != last_caption:
            pygame.display.set_caption(caption)
            last_caption = caption

        if modal:
            if not menu_frozen:
                # First frame with the menu open: draw the world once and keep it as the backdrop.
                renderer.draw(world, player.eye, player.yaw, player.pitch,
                              radius=rd - 1, fov=settings['fov'])
                if settings['show_hud']:
                    grid.draw(renderer.last_vp, centre[0], centre[1], rd - 1)
                    hud.draw(perf, poller)
                backdrop.capture()
                menu_frozen = True
            else:
                backdrop.draw()
            if bench_results is not None:
                results_panel.draw(bench_results)
            else:
                settings_panel.draw(menu)
        else:
            menu_frozen = False
            t_draw = time.perf_counter()
            renderer.draw(world, player.eye, player.yaw, player.pitch,
                          radius=rd - 1, fov=settings['fov'])
            draw_ms = (time.perf_counter() - t_draw) * 1000.0
            struct_gl.render(renderer.last_vp, 1.0, overlay=settings['show_hud'])
            body_gl.render(renderer.last_vp, 1.0, player.yaw, player.pitch)
            proj_gl.render(renderer.last_vp)
            laser_gl.render(renderer.last_vp, player.eye, player.yaw, player.pitch)
            particles_gl.render(player.eye, player.yaw, player.pitch, 1.0, renderer.last_vp)
            if settings['show_hud']:
                grid.draw(renderer.last_vp, centre[0], centre[1], rd - 1)
                hud.draw(perf, poller, show_integrity)
                if show_integrity:
                    integrity_panel.draw(integrity_state(player))
            crosshair.draw(0)
            weapon_bar.draw((weapon, PROJ.n_charges))
        # Measure before the flip, since flip can wait on vsync and that isn't game work.
        last_cpu_ms = (time.perf_counter() - work_start) * 1000.0
        pygame.display.flip()

    poller.stop()
    renderer.mesher.pool.shutdown(wait=False, cancel_futures=True)
    pygame.quit()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
