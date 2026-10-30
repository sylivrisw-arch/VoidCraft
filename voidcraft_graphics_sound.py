# voidcraft_graphics.py
import math, os, time, ctypes, functools, subprocess, threading, heapq, collections, bisect
import logging, multiprocessing, textwrap
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
TARGET_FPS        = 60
RENDER_DIST       = 15
MESH_WORKERS      = 2     # background processes that build chunk meshes (separate from the GIL)
MESH_MAX_PENDING  = 6     # chunks queued for meshing at once; the rest wait for a later frame
VERT_STRIDE       = 9 * 4  # bytes per vertex: position (3) + colour (3) + texture coords and flag (3)
ARENA_VERTS       = 1_000_000   # starting size of the shared chunk-mesh buffer (grows as needed)
FOV               = 90
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


COLORS = {
    GRASS:  (0.30, 0.80, 0.15), DIRT:   (0.65, 0.40, 0.15), STONE:  (0.60, 0.60, 0.60),
    SAND:   (0.95, 0.90, 0.50), WOOD:   (0.55, 0.35, 0.12), LEAVES: (0.20, 0.70, 0.15),
    WATER:  (0.20, 0.50, 0.95), SNOW:   (0.98, 1.00, 1.00), GRAVEL: (0.70, 0.68, 0.65),
    PLANKS: (0.80, 0.65, 0.30), GLASS:  (0.75, 0.90, 0.98), TORCH:  (1.00, 0.85, 0.25),
    BRICK:  (0.85, 0.40, 0.30), SLAB:   (0.65, 0.65, 0.65), STEEL:  (0.44, 0.52, 0.64)
}

WIDTH, HEIGHT  = 1920, 1080
NEAR, FAR      = 0.05, 300.0
GRAVITY        = -28.0
JUMP_VEL       = 9.0
WORLD_H        = 64
CHUNK_SIZE     = 15
PLAYER_HEIGHT  = 1.8
PLAYER_RADIUS  = 0.3
STEP_HEIGHT    = 0.6       # the player climbs slopes and small steps up to this height

# ═══════════════════════════════════════════════
#  FLAT WORLD GENERATION
# ═══════════════════════════════════════════════
FLAT_SURFACE_Y  = 10   # y of the grass layer (keep it below the water line)

# ═══════════════════════════════════════════════
#  SMOOTH TERRAIN GENERATION
# ═══════════════════════════════════════════════
# Every block position also has a density. Above zero is solid, below zero is air, and the
# surface sits where the density crosses zero. Terrain density is height minus y.
DENS_SOLID = 1.0      # density given to a block placed or removed by hand

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
            for dx, dz in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (key[0] + dx, key[1] + dz)
                if nb in self.chunks:
                    self.dirty.add(nb)
                    self.touched.add(nb)
        self.modified_chunks.clear()
        self.dug_cells = []

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
# Colour per block ID. Unknown IDs render magenta so gaps are easy to spot.
_COLOR_LUT = np.tile(np.array([1.0, 0.0, 1.0], dtype=np.float32), (256, 1))
for _blk, _rgb in COLORS.items():
    _COLOR_LUT[_blk] = _rgb

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
SKY_RGB = (0.64, 0.73, 0.84)   # neutral daylight; matches the shaders' fog colour

# Sun and shadows. The shaders use the same direction (normalize(0.45, 0.80, 0.35)).
SUN_DIR = np.array([0.45, 0.80, 0.35])
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

def light_matrix(centre):
    """Sun's view-projection, centred on the player. The centre is snapped to shadow-map texels so the shadows
    do not shimmer as the player walks."""
    texel = 2.0 * SHADOW_HALF / SHADOW_RES
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

# Shadow lookup shared by the terrain and structure shaders: 3x3 percentage-closer filtering of the depth map.
_SHADOW_GLSL = """
uniform mat4 uLVP;
uniform sampler2D uShadow;
float sunShadow(vec3 wp, vec3 nrm) {
    vec4 ls = uLVP * vec4(wp + nrm * 0.06, 1.0);     // nudged along the normal to avoid self-shadowing
    vec3 s = ls.xyz / ls.w * 0.5 + 0.5;
    if (s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0 || s.z > 1.0) return 1.0;
    float lit = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            float d = texture(uShadow, s.xy + vec2(i, j) * (1.2 / 2048.0)).r;
            lit += (s.z - 0.0015 > d) ? 0.0 : 1.0;
        }
    }
    return lit / 9.0;
}
"""

# One daylight model for terrain, structures and debris: the same sun, sky fill and fog colour everywhere.
_LIGHT_GLSL = """
const vec3 SUN_DIR  = vec3(0.4581, 0.8144, 0.3563);   // normalize(0.45, 0.80, 0.35)
const vec3 SUN_COL  = vec3(1.00, 0.98, 0.94);          // near-white sun
const vec3 SKY_TINT = vec3(0.92, 0.97, 1.04);          // sky fill, slightly cool
const vec3 FOG_COL  = vec3(0.64, 0.73, 0.84);          // same as the sky clear colour, so the horizon blends
// Light reaching a surface with normal n; sh is the sun visibility (0 in shadow, 1 in sun).
vec3 dayLight(vec3 n, float sh) {
    float sky = 0.5 + 0.5 * n.y;
    float diff = max(dot(n, SUN_DIR), 0.0);
    return SKY_TINT * (0.44 + 0.22 * sky) + SUN_COL * (0.62 * diff * sh);
}
"""

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
in vec3 vWorld;      // world position; the tile mapping and the depth below the surface both come from it
in float vDepth;
out vec4 fragColor;
uniform sampler2DArray uTiles;   // the material tiles, bound to unit 2; layer 0 is dirt, layer 1 is stone
uniform float uLight;     // scene light level, set per frame (floored at GFX['ambient_floor'])
uniform vec3 uEye;        // camera position, for orienting face normals
SHADOW_GLSL_PLACEHOLDER
LIGHT_GLSL_PLACEHOLDER
void main() {
    // Face normal from the screen-space derivatives of the world position, turned to face the camera.
    // That makes one-sided diffuse correct on every visible face, including the sides of a cut.
    vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    if (dot(n, uEye - vWorld) < 0.0) n = -n;
    float sh = sunShadow(vWorld, n);
    vec3 light = dayLight(n, sh);
    // One dirt tile per block, mapped along the face's dominant axis in world space.
    vec2 uv = abs(n.y) > 0.5 ? vWorld.xz : (abs(n.x) > 0.5 ? vWorld.zy : vWorld.xy);
    // Topsoil is the dirt tile. Blocks more than about a block below the surface blend toward the stone
    // tile and darken, so a cut or crater reads as layered ground.
    float depth = SURF_Y_PLACEHOLDER - vWorld.y;
    vec3 topsoil = texture(uTiles, vec3(uv, 0.0)).rgb;
    vec3 subsoil = mix(topsoil, texture(uTiles, vec3(uv, 1.0)).rgb, 0.45) * 0.84;
    vec3 soil = mix(topsoil, subsoil, smoothstep(0.5, 2.0, depth));
    // Keep the mesher's shading: vCol divided by the dirt colour it was built from.
    vec3 c = min(soil * (vCol / vec3(0.65, 0.40, 0.15)), vec3(1.0));
    c *= light * uLight;
    // Distance haze: far ground fades toward the sky colour.
    float fog = 1.0 - exp(-vDepth * 0.014);
    fragColor = vec4(mix(c, FOG_COL, fog), 1.0);
}
"""
_FRAG_SRC = (_FRAG_SRC.replace("SHADOW_GLSL_PLACEHOLDER", _SHADOW_GLSL)
             .replace("LIGHT_GLSL_PLACEHOLDER", _LIGHT_GLSL)
             .replace("SURF_Y_PLACEHOLDER", repr(float(FLAT_GROUND_H))))

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
        self.u_vp = gl.glGetUniformLocation(self.program, "uVP")
        self.u_tiles = gl.glGetUniformLocation(self.program, "uTiles")
        self.u_light = gl.glGetUniformLocation(self.program, "uLight")
        self.u_eye = gl.glGetUniformLocation(self.program, "uEye")
        self.u_lvp = gl.glGetUniformLocation(self.program, "uLVP")
        self.u_shadow = gl.glGetUniformLocation(self.program, "uShadow")
        self.light_vp = np.eye(4, dtype=np.float32)   # set each frame by the main loop
        self.shadow_tex = 0                            # the shadow map's depth texture, set by the main loop
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

    def _upload(self, key, payload):
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

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        vp = _perspective(fov, WIDTH / HEIGHT, NEAR, FAR) @ _view(eye, yaw, pitch)
        self.last_vp = vp     # kept for the debug chunk grid, which is drawn over this frame
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_vp, 1, gl.GL_TRUE, vp.astype(np.float32))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniformMatrix4fv(self.u_lvp, 1, gl.GL_TRUE, self.light_vp.astype(np.float32))
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.shadow_tex)
        gl.glUniform1i(self.u_shadow, 1)
        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tile_texture())
        gl.glUniform1i(self.u_tiles, 2)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1f(self.u_light, max(float(ambient), GFX['ambient_floor']))
        gl.glUniform3f(self.u_eye, float(eye[0]), float(eye[1]), float(eye[2]))

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

# ═══════════════════════════════════════════════
#  GRAPHICS SETTINGS MENU (Tab)
# ═══════════════════════════════════════════════
# Tabs of the settings panel. Each row is a header, a slider, a toggle or an info line.
# Sliders change with Left/Right within [min, max] in steps of `step`.
# Toggles flip with Left/Right or Enter. Headers and info lines are skipped by the cursor.
SETTINGS_TABS = [
    ("DISPLAY", [
        {'type': 'header', 'text': 'VIEW'},
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
        {'type': 'header', 'text': 'DEBRIS'},
        {'type': 'toggle', 'key': 'debris_on', 'label': 'Debris',
         'help': 'Dirt cubes from digging and blasts. Turning it off removes what is there.'},
        {'type': 'slider', 'key': 'dirt_cubes', 'label': 'Cube detail', 'min': 1, 'max': 50, 'step': 1,
         'fmt': lambda v: f"{v} ({v ** 3}/block)",
         'help': 'Higher = each block is split into more, smaller cubes per side, and the X grid lines get finer to match. 1 = one cube per block, 10 = 1,000, 50 = 125,000. Above about 20 every hit gets slow and memory use climbs fast.'},
    ]),
    ("PERFORMANCE", [
        {'type': 'header', 'text': 'OVERLAY'},
        {'type': 'toggle', 'key': 'show_hud', 'label': 'Performance graphs',
         'help': 'Full monitor overlay: frame graphs, stage timings, GPU, counts, integrity. Also toggled with G.'},
        {'type': 'header', 'text': 'COMPUTE'},
        {'type': 'toggle', 'key': 'gpu_physics', 'label': 'GPU physics',
         'help': 'Structure support checks run on the GPU (needs CuPy). Falls back to the CPU on its own.'},
    ]),
    ("PC CONTROLS", [
        {'type': 'header', 'text': 'KEYS', 'width': 508},
        {'type': 'info', 'a': None, 'b': ('W A S D', 'move')},
        {'type': 'info', 'a': None, 'b': ('Space', 'jump')},
        {'type': 'info', 'a': None, 'b': ('V', 'fly (monitor on: stats on / off)')},
        {'type': 'info', 'a': None, 'b': ('Mouse', 'look')},
        {'type': 'info', 'a': None, 'b': ('Left click', 'use the weapon (hold for the laser)')},
        {'type': 'info', 'a': None, 'b': ('R', 'reset map and damage')},
        {'type': 'info', 'a': None, 'b': ('F', 'pause')},
        {'type': 'info', 'a': None, 'b': ('G', 'monitor overlay on / off')},
        {'type': 'info', 'a': None, 'b': ('X', 'cube grid lines on / off')},
        {'type': 'info', 'a': None, 'b': ('Q / E', 'material / shape')},
        {'type': 'info', 'a': None, 'b': ('T', 'switch weapon: axe / melting laser')},
        {'type': 'info', 'a': None, 'b': ('Tab / F5', 'settings / benchmark')},
    ]),
    ("CONTROLLER", []),
]

DEBRIS_AMOUNT = 70   # percent of debris made and kept: fixed, no longer a menu setting
WOBBLE_PCT = 50      # loose-piece wobble: fixed at the original 50%, no longer a menu setting
STRENGTH_PCT = 100   # material strength: fixed at 100%, no longer a menu setting
STRUCT_RES = 2       # cells per block along each side of a test cube: fixed at 2, no longer a menu setting
STRUCT_HIT = {'pos': None, 't0': 0.0}   # the last structure hit: where, and when (for the dent)
STRUCT_SHARDS = True   # broken cells crumble off as loose cubes (capped per hit, see _throw_chunks)


def default_settings():
    return {'cache_chunks': True, 'render_distance': RENDER_DIST, 'fov': FOV, 'show_hud': False,
            'fps_limit': TARGET_FPS, 'debris_yield': 1.0, 'debris_size': 1.0, 'dirt_cubes': 5,
            'dirt_cube_size': 1.0, 'cube_sleep_time': 0.3, 'debris_on': True, 'debris_amount': DEBRIS_AMOUNT,
            'struct_res': STRUCT_RES, 'gpu_physics': True, 'wobble': WOBBLE_PCT, 'strength': STRENGTH_PCT}

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


# ═══════════════════════════════════════════════
#  CONTROLLER (Xbox One / any XInput or SDL game controller)
# ═══════════════════════════════════════════════
PAD_DEADZONE   = 0.18      # stick travel ignored around the centre
PAD_LOOK_SPEED = 2.8       # radians per second at full right-stick push
PAD_TRIGGER_ON = 0.5       # how far a trigger is pulled before it counts as pressed
PAD_EVENTS = tuple(getattr(pygame, n) for n in ('JOYDEVICEADDED', 'JOYDEVICEREMOVED',
                                                'CONTROLLERDEVICEADDED', 'CONTROLLERDEVICEREMOVED')
                   if hasattr(pygame, n))


class Gamepad:
    """First connected controller. Uses SDL's game-controller mapping when pygame has it, so an Xbox pad
    reports the same buttons everywhere, and falls back to the raw joystick layout otherwise.
    Call poll() once per frame, then ask pressed() (this frame only) or held()."""
    NAMES = ('A', 'B', 'X', 'Y', 'LB', 'RB', 'BACK', 'START', 'L3', 'R3', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'LT', 'RT')

    def __init__(self):
        self.ctrl = None
        self.joy = None
        self.name = ''
        self.btn = dict.fromkeys(self.NAMES, False)
        self.prev = dict(self.btn)
        self.lx = self.ly = self.rx = self.ry = self.lt = self.rt = 0.0
        self.changed = False          # anything moved or pressed since the last poll (the settings panel redraws on it)
        self._sig = None
        self._next_scan = 0.0
        self._sdl = None
        try:
            pygame.joystick.init()
        except Exception:
            pass
        try:
            from pygame._sdl2 import controller as _ctl
            _ctl.init()
            self._sdl = _ctl
        except Exception:
            self._sdl = None
        self.rescan()

    @property
    def connected(self):
        return self.ctrl is not None or self.joy is not None

    def close(self):
        self.ctrl = None
        self.joy = None
        self.name = ''
        for n in self.NAMES:
            self.btn[n] = False
        self.lx = self.ly = self.rx = self.ry = self.lt = self.rt = 0.0

    def rescan(self):
        """Look for a controller again (called when one is plugged in or removed)."""
        self.close()
        if self._sdl is not None:
            try:
                for i in range(self._sdl.get_count()):
                    if self._sdl.is_controller(i):
                        self.ctrl = self._sdl.Controller(i)
                        try:
                            nm = self._sdl.name_forindex(i)
                            self.name = nm.decode(errors='replace') if isinstance(nm, bytes) else str(nm or 'Controller')
                        except Exception:
                            self.name = 'Controller'
                        return
            except Exception:
                self.ctrl = None
        try:
            if pygame.joystick.get_count() > 0:
                self.joy = pygame.joystick.Joystick(0)
                self.joy.init()
                self.name = self.joy.get_name()
        except Exception:
            self.joy = None

    def _read_controller(self):
        P, b = pygame, self.btn
        if not self.ctrl.attached():
            self.close()
            return
        g = self.ctrl.get_button
        b['A'], b['B'], b['X'], b['Y'] = (bool(g(P.CONTROLLER_BUTTON_A)), bool(g(P.CONTROLLER_BUTTON_B)),
                                           bool(g(P.CONTROLLER_BUTTON_X)), bool(g(P.CONTROLLER_BUTTON_Y)))
        b['LB'], b['RB'] = bool(g(P.CONTROLLER_BUTTON_LEFTSHOULDER)), bool(g(P.CONTROLLER_BUTTON_RIGHTSHOULDER))
        b['BACK'], b['START'] = bool(g(P.CONTROLLER_BUTTON_BACK)), bool(g(P.CONTROLLER_BUTTON_START))
        b['L3'], b['R3'] = bool(g(P.CONTROLLER_BUTTON_LEFTSTICK)), bool(g(P.CONTROLLER_BUTTON_RIGHTSTICK))
        b['UP'], b['DOWN'] = bool(g(P.CONTROLLER_BUTTON_DPAD_UP)), bool(g(P.CONTROLLER_BUTTON_DPAD_DOWN))
        b['LEFT'], b['RIGHT'] = bool(g(P.CONTROLLER_BUTTON_DPAD_LEFT)), bool(g(P.CONTROLLER_BUTTON_DPAD_RIGHT))
        a = self.ctrl.get_axis
        self.lx, self.ly = a(P.CONTROLLER_AXIS_LEFTX) / 32767.0, a(P.CONTROLLER_AXIS_LEFTY) / 32767.0
        self.rx, self.ry = a(P.CONTROLLER_AXIS_RIGHTX) / 32767.0, a(P.CONTROLLER_AXIS_RIGHTY) / 32767.0
        self.lt = a(P.CONTROLLER_AXIS_TRIGGERLEFT) / 32767.0
        self.rt = a(P.CONTROLLER_AXIS_TRIGGERRIGHT) / 32767.0

    def _read_joystick(self):
        """Raw layout of an Xbox pad: buttons 0-9 are A B X Y LB RB Back Start L3 R3, axes 0-5 are
        left stick, right stick, left trigger, right trigger, and the D-pad is hat 0."""
        j, b = self.joy, self.btn
        nb, na = j.get_numbuttons(), j.get_numaxes()
        gb = lambda i: bool(j.get_button(i)) if i < nb else False
        ga = lambda i: float(j.get_axis(i)) if i < na else 0.0
        for n, i in (('A', 0), ('B', 1), ('X', 2), ('Y', 3), ('LB', 4), ('RB', 5),
                     ('BACK', 6), ('START', 7), ('L3', 8), ('R3', 9)):
            b[n] = gb(i)
        hx, hy = j.get_hat(0) if j.get_numhats() else (0, 0)
        b['LEFT'], b['RIGHT'], b['UP'], b['DOWN'] = hx < 0, hx > 0, hy > 0, hy < 0
        self.lx, self.ly, self.rx, self.ry = ga(0), ga(1), ga(2), ga(3)
        self.lt, self.rt = (ga(4) + 1.0) * 0.5, (ga(5) + 1.0) * 0.5

    def poll(self):
        self.prev = dict(self.btn)
        if not self.connected:
            now = time.perf_counter()
            if now >= self._next_scan:                # a missed plug-in event is picked up within two seconds
                self._next_scan = now + 2.0
                self.rescan()
        try:
            if self.ctrl is not None:
                self._read_controller()
            elif self.joy is not None:
                self._read_joystick()
        except Exception:
            self.close()
        self.lt = max(0.0, min(1.0, self.lt))
        self.rt = max(0.0, min(1.0, self.rt))
        self.btn['LT'] = self.lt > PAD_TRIGGER_ON
        self.btn['RT'] = self.rt > PAD_TRIGGER_ON
        sig = (tuple(self.btn.values()), round(self.lx, 2), round(self.ly, 2), round(self.rx, 2),
               round(self.ry, 2), round(self.lt, 2), round(self.rt, 2), self.connected, self.name)
        self.changed = sig != self._sig
        self._sig = sig

    def pressed(self, name):
        return self.btn[name] and not self.prev[name]

    def held(self, name):
        return self.btn[name]

    @staticmethod
    def _shape(x, y):
        """Dead zone around the centre, then a gentle curve so small pushes give fine control."""
        m = math.hypot(x, y)
        if m < PAD_DEADZONE:
            return 0.0, 0.0
        m = min(m, 1.0)
        k = ((m - PAD_DEADZONE) / (1.0 - PAD_DEADZONE)) ** 1.4 / m
        return x * k, y * k

    def move(self):
        """Left stick as (right, back): up on the stick is a negative y."""
        return self._shape(self.lx, self.ly)

    def look(self):
        return self._shape(self.rx, self.ry)

PANEL_W, PANEL_H = 1000, 560   # settings panel (Tab); tall enough for every tab

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
        self.exit_rect = pygame.Rect(self.w - 24 - 170, 16, 170, 40)   # EXIT GAME, above the tab bar on the right
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

    def _render(self, menu, pad=None):
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

        # Rows for the current tab.
        y = tab_y + 66
        for i, row in enumerate(menu.rows()):
            if row['type'] == 'header':
                head = self.f_head.render(row['text'], True, UI_ACCENT)
                s.blit(head, (24, y))
                ly = y + head.get_height() // 2
                pygame.draw.line(s, UI_LINE, (24 + head.get_width() + 14, ly), (row.get('width', self.w - 24), ly), 1)
                y += 44
                continue

            if row['type'] == 'info':
                if row['a']:
                    k, name, note = row['a']
                    s.blit(self.f_label.render(k, True, (255, 220, 120)), (32, y))
                    s.blit(self.f_label.render(name, True, UI_TEXT), (62, y))
                    s.blit(self.f_help.render(note, True, UI_DIM), (270, y + 4))
                k, d = row['b']
                s.blit(self.f_value.render(k, True, (255, 220, 120)), (32, y + 1))
                s.blit(self.f_value.render(d, True, UI_TEXT), (172, y + 1))
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

        if pad is not None and SETTINGS_TABS[menu.tab][0] == 'CONTROLLER':
            self._draw_pad(pad)

        foot = self.f_help.render("[ ] switch tab   Up/Down select   Left/Right change   Enter toggle",
                                  True, UI_DIM)
        s.blit(foot, (24, self.h - 34))


    # ---- controller view (CONTROLS tab): the pad drawn live, each button lit while it is held ----
    PAD_X0 = 548
    PAD_SHIFT = -520     # the whole controller block moves this far left, onto the free side of the tab

    def _pad_round(self, cx, cy, r, on, col, letter):
        s = self.surf
        pygame.draw.circle(s, col if on else (24, 27, 38), (cx, cy), r)
        pygame.draw.circle(s, col, (cx, cy), r, 2)
        t = self.f_value.render(letter, True, (10, 10, 14) if on else col)
        s.blit(t, t.get_rect(center=(cx, cy)))

    def _pad_rect(self, rect, on, label=None, radius=5):
        s = self.surf
        pygame.draw.rect(s, UI_ACCENT if on else (24, 27, 38), rect, border_radius=radius)
        pygame.draw.rect(s, UI_ACCENT if on else (84, 92, 114), rect, 2, border_radius=radius)
        if label:
            t = self.f_help.render(label, True, (10, 10, 14) if on else UI_DIM)
            s.blit(t, t.get_rect(center=rect.center))

    def _pad_stick(self, cx, cy, vx, vy, pressed):
        s = self.surf
        m = math.hypot(vx, vy)
        if m > 1.0:
            vx, vy = vx / m, vy / m
        active = m > PAD_DEADZONE
        pygame.draw.circle(s, (20, 23, 32), (cx, cy), 27)
        pygame.draw.circle(s, UI_TEXT if pressed else (84, 92, 114), (cx, cy), 27, 3 if pressed else 2)
        pygame.draw.circle(s, UI_ACCENT if (active or pressed) else (70, 78, 98),
                           (int(cx + vx * 13), int(cy + vy * 13)), 13)
        if pressed:
            pygame.draw.circle(s, UI_TEXT, (int(cx + vx * 13), int(cy + vy * 13)), 13, 2)

    def _pad_trigger(self, rect, amount, label):
        s = self.surf
        pygame.draw.rect(s, (24, 27, 38), rect, border_radius=4)
        fill = rect.copy()
        fill.w = int(rect.w * amount)
        if fill.w > 0:
            pygame.draw.rect(s, UI_ACCENT, fill, border_radius=4)
        pygame.draw.rect(s, UI_ACCENT if amount > PAD_TRIGGER_ON else (84, 92, 114), rect, 2, border_radius=4)
        t = self.f_help.render(f"{label} {int(amount * 100):3d}%", True, UI_TEXT if amount > 0.3 else UI_DIM)
        s.blit(t, t.get_rect(center=rect.center))

    def _draw_pad(self, pad):
        # Drawn on a separate layer, then moved left as one block.
        real = self.surf
        layer = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.surf = layer
        try:
            self._draw_pad_layer(pad)
        finally:
            self.surf = real
        real.blit(layer, (self.PAD_SHIFT, 0))

    def _draw_pad_layer(self, pad):
        s = self.surf
        X0 = self.PAD_X0
        b = pad.btn
        # Header with the connection state on the right.
        head = self.f_head.render("XBOX ONE CONTROLLER", True, UI_ACCENT)
        s.blit(head, (X0, 140))
        status = (pad.name[:18] or "controller") if pad.connected else "not connected"
        st = self.f_help.render(status, True, (110, 220, 130) if pad.connected else (230, 170, 80))
        tx = self.w - 24 - st.get_width()
        s.blit(st, (tx, 142))
        ly = 140 + head.get_height() // 2
        if tx - 12 > X0 + head.get_width() + 14:
            pygame.draw.line(s, UI_LINE, (X0 + head.get_width() + 14, ly), (tx - 12, ly), 1)

        # Body, triggers and bumpers.
        pygame.draw.rect(s, (30, 34, 46), (584, 202, 364, 132), border_radius=58)
        for gx in (620, 912):
            pygame.draw.circle(s, (30, 34, 46), (gx, 314), 32)
        pygame.draw.rect(s, UI_BORDER, (584, 202, 364, 132), 2, border_radius=58)
        self._pad_trigger(pygame.Rect(596, 168, 96, 16), pad.lt, "LT")
        self._pad_trigger(pygame.Rect(840, 168, 96, 16), pad.rt, "RT")
        self._pad_rect(pygame.Rect(596, 187, 96, 14), b['LB'], radius=6)
        self._pad_rect(pygame.Rect(840, 187, 96, 14), b['RB'], radius=6)
        for r, name in ((pygame.Rect(596, 187, 96, 14), "LB"), (pygame.Rect(840, 187, 96, 14), "RB")):
            t = self.f_help.render(name, True, (10, 10, 14) if b[name] else UI_DIM)
            s.blit(t, t.get_rect(center=r.center))

        # Sticks (the dot follows the stick; the ring lights on a click), D-pad, face buttons, centre buttons.
        self._pad_stick(640, 244, pad.lx, pad.ly, b['L3'])
        self._pad_stick(832, 296, pad.rx, pad.ry, b['R3'])
        dx, dy = 700, 296
        self._pad_rect(pygame.Rect(dx - 9, dy - 25, 18, 24), b['UP'], radius=3)
        self._pad_rect(pygame.Rect(dx - 9, dy + 1, 18, 24), b['DOWN'], radius=3)
        self._pad_rect(pygame.Rect(dx - 25, dy - 9, 24, 18), b['LEFT'], radius=3)
        self._pad_rect(pygame.Rect(dx + 1, dy - 9, 24, 18), b['RIGHT'], radius=3)
        pygame.draw.rect(s, (30, 34, 46), (dx - 8, dy - 8, 16, 16))
        fx, fy = 892, 244
        self._pad_round(fx, fy - 24, 13, b['Y'], (240, 200, 60), "Y")
        self._pad_round(fx, fy + 24, 13, b['A'], (90, 200, 90), "A")
        self._pad_round(fx - 24, fy, 13, b['X'], (80, 140, 240), "X")
        self._pad_round(fx + 24, fy, 13, b['B'], (230, 80, 80), "B")
        self._pad_rect(pygame.Rect(727, 236, 24, 14), b['BACK'], radius=7)
        self._pad_rect(pygame.Rect(781, 236, 24, 14), b['START'], radius=7)
        pygame.draw.circle(s, (46, 52, 68), (766, 218), 10)          # Guide button: not used by the game
        pygame.draw.circle(s, (84, 92, 114), (766, 218), 10, 2)

        # What each control does. A row lights while its control is held.
        ls = math.hypot(pad.lx, pad.ly) > PAD_DEADZONE
        rs = math.hypot(pad.rx, pad.ry) > PAD_DEADZONE
        cols = [
            [('LStick', 'move', ls), ('RStick', 'look', rs), ('L3', 'sprint', b['L3']),
             ('A', 'jump/fly up', b['A']), ('B', 'fly down', b['B']), ('X', 'monitor', b['X']),
             ('Y', 'fly on/off', b['Y']), ('Start', 'settings', b['START'])],
            [('D-Up', 'reset map', b['UP']),
             ('D-L/R', 'cube material', b['LEFT'] or b['RIGHT']),
             ('R3', 'pause', b['R3']), ('Back', 'benchmark', b['BACK'])],
        ]
        for ci, rows in enumerate(cols):
            x = X0 + ci * 222
            y = 360
            for key, text, on in rows:
                if on:
                    pygame.draw.rect(s, UI_SEG_ON, (x, y - 1, 212, 20), border_radius=4)
                s.blit(self.f_help.render(key, True, (255, 220, 120) if not on else UI_TEXT), (x + 6, y))
                s.blit(self.f_help.render(text, True, UI_TEXT if on else UI_DIM), (x + 78, y))
                y += 20

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

    def loop_set(self, name, on):
        """Keep a looping sound running while on is True, and fade it out when it goes False."""
        if not self.ok:
            return
        ch = self.loops.get(name)
        if on:
            if ch is None or not ch.get_busy():
                snd = self.bank.get(name)
                if snd:
                    ch = snd[0].play(loops=-1, fade_ms=60)
                    if ch is not None:
                        ch.set_volume(0.8)
                    self.loops[name] = ch
        elif ch is not None:
            ch.fadeout(140)
            self.loops[name] = None

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

    def punch(self, mat, pos):
        """A cannonball cut through a cell of this material."""
        if mat == PLANKS:
            self.play('break_wood', 0.8, pos, gap=0.06)
        elif mat == STEEL:
            self.play('glass_break', 0.45, pos, gap=0.06)
        elif mat in (STONE, CONCRETE_WHITE, BRICK, DIRT):
            self.play('break_stone', 0.8, pos, gap=0.06)
        else:
            self.play('block_hit', 0.8, pos, gap=0.06)

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

    def roll(self, pos, speed):
        self.play('step_stone', min(0.3, 0.06 + 0.03 * speed), pos, gap=0.22)


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
#  DEBRIS SETTINGS
# ═══════════════════════════════════════════════
CONFIG = {'debris_on': True, 'debris_amount': 1.0, 'dirt_per_block': 1}
_NEIGH_OFFS = np.array([-1099512676353, -1099512676352, -1099512676351, -1099511627777, -1099511627776, -1099511627775, -1099510579201, -1099510579200, -1099510579199, -1048577, -1048576, -1048575, -1, 0, 1, 1048575, 1048576, 1048577, 1099510579199, 1099510579200, 1099510579201, 1099511627775, 1099511627776, 1099511627777, 1099512676351, 1099512676352, 1099512676353], np.int64)   # the 27 cells around a cube's cell, as key offsets
_rng = np.random.default_rng()


# ═══════════════════════════════════════════════
#  DUG DIRT: each dug block splits into mini cubes that fly out and land
# ═══════════════════════════════════════════════
BODY_MAX       = 12000               # cubes alive at once; the oldest go first
MAX_AWAKE_CUBES = 500               # cubes simulated per frame; the slowest beyond this are put to sleep
HARD_AWAKE_CUBES = 700              # most cubes simulated in one frame even in flight; beyond this they take turns (see update)
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
        vals[m] = d[X[m] - kx * CHUNK_SIZE, Y[m], Z[m] - kz * CHUNK_SIZE]
    return vals

def _density_many(world, P):
    """Interpolated density at many points: the same smooth field the player and the mesh use.
    All eight corner samples of every point are fetched in one batch."""
    m = len(P)
    if m == 0:
        return np.zeros(0, np.float64)
    x0 = np.floor(P[:, 0]).astype(np.int64)
    y0 = np.floor(P[:, 1]).astype(np.int64)
    z0 = np.floor(P[:, 2]).astype(np.int64)
    fx = P[:, 0] - x0
    fy = P[:, 1] - y0
    fz = P[:, 2] - z0
    corners = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    vals = _sample_many(world,
                        np.concatenate([x0 + dx for dx, _, _ in corners]),
                        np.concatenate([y0 + dy for _, dy, _ in corners]),
                        np.concatenate([z0 + dz for _, _, dz in corners])).reshape(8, m)
    out = np.zeros(m, np.float64)
    for i, (dx, dy, dz) in enumerate(corners):
        out += (fx if dx else 1.0 - fx) * (fy if dy else 1.0 - fy) * (fz if dz else 1.0 - fz) * vals[i]
    return out

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
    ex, ey, ez = np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e])
    d = _density_batch(world, [Q + ex, Q - ex, Q + ey, Q - ey, Q + ez, Q - ez])
    g = np.stack([d[0] - d[1], d[2] - d[3], d[4] - d[5]], axis=1) / (2 * e)
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

    def spawn(self, world, cell_groups, per_block, size_scale=1.0):
        """Split each dug block into level x level x level smaller cubes (level 1 = one full-size cube per block,
        level 10 = 1,000 cubes per block). The debris amount setting keeps that share of the cubes.
        Nothing is thrown: the cubes start at rest in the dug space and fall. Cubes whose slot overlaps the
        ground are skipped, so none starts inside it. size_scale shrinks each cube inside its slot."""
        groups = [np.asarray(g, np.float64).reshape(-1, 3) for g in cell_groups if len(g)]
        level = int(max(1, min(10, round(per_block))))
        if not CONFIG['debris_on']:
            return
        if not groups:
            return
        m = level
        slots = np.stack(np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing='ij'), -1).reshape(-1, 3)
        keep = int(round(m ** 3 * CONFIG['debris_amount']))
        if keep == 0:
            return
        slot_c = (slots + 0.5) / m - 0.5                        # slot centres inside one block
        P = []
        for block in np.concatenate(groups):
            pick = _rng.permutation(len(slots))[:keep]         # which slots get a cube
            P.append(block + slot_c[pick])
        P = np.concatenate(P)
        side = float(np.clip(size_scale, 0.1, 1.0)) / m
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
        self.col = np.concatenate([self.col, np.tile(np.array(_S_RGB[DIRT], np.float64), (len(P), 1))])
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

    def _separate(self, passes=3, dt=1.0 / 60.0):
        """Round cube against round cube. Each cube is treated as a sphere with radius half its size.
        Overlapping pairs are pushed apart along the line between their centres. Sleeping cubes never
        move; when one of a pair is asleep, the awake one takes the whole correction. Motion that drives
        the pair into each other is removed, so piles settle instead of jittering."""
        self._contact = np.zeros(self.n, bool)                    # cubes that touched another cube this frame
        self._on_sleeper = np.zeros(self.n, bool)                 # cubes resting on a cube that is already asleep
        self._unstable = np.zeros(self.n, bool)                   # cubes perched off-centre on another cube: they slide off, never freeze or sleep
        if self.n < 2:
            return
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

            strong = depth > 0.02
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
            self.vel[touched, 0] *= 0.85
            self.vel[touched, 2] *= 0.85

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
            top = max(float(np.abs(V).max()), 1e-6)
            steps = int(min(6, max(1, math.ceil(top * dt * 16))))
            sdt = dt / steps
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
            for _ in range(steps):
                for axis in (1, 0, 2):
                    new = P.copy()
                    new[:, axis] += V[:, axis] * sdt
                    if axis == 1:
                        under = new.copy()
                        under[:, 1] -= size * 0.5
                        _t0 = time.perf_counter()
                        corners = _corners_touching(world, new, size)
                        _mark('debris.move.terrain', _t0)
                        _t0 = time.perf_counter()
                        sc = _struct_corners(new, size)
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
                        sc = _struct_corners(new, size).any(axis=1)
                        _mark('debris.move.struct', _t0)
                        blocked = dens | sc
                    go = ~blocked
                    P[go] = new[go]
                    V[blocked, axis] = 0.0
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
            self._separate(dt=dt)
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
            stick = aw[ct & crawl & ~self._unstable[aw]]
            self.vel[stick, 0] = 0.0
            self.vel[stick, 2] = 0.0
            moved_speed = np.linalg.norm(self.pos[aw] - P_start, axis=1) / max(dt, 1e-6)   # real movement this frame, pushes included                                       # pushes cubes apart, stops motion into each other
            Vn = self.vel[aw]
            still_now = (np.abs(Vn).max(axis=1) < 0.12) & (moved_speed < 0.25)   # crawling AND not being shoved around
            still_now &= ~self._unstable[aw]                                      # a cube that is still toppling is not at rest
            self.vel[aw[still_now]] = 0.0
            self.still[aw] = np.where(still_now, self.still[aw] + dt, 0.0)
            ready = np.nonzero(still_now & (self.still[aw] >= sleep_time))[0]          # about to fall asleep
            if len(ready):
                held = _support(world, self.pos[aw[ready]], size[ready]).any(axis=1) | self._on_sleeper[aw[ready]]
                lost = ready[~held]                                                    # nothing under it: keep it awake
                self.still[aw[lost]] = 0.0
            asleep = self.still[aw] >= sleep_time
            if asleep.any():
                SFX.settle(self.pos[aw[asleep]][0], int(asleep.sum()))
            self.awake[aw[asleep]] = False
            for k_, v_ in _tm.items():
                PROF.add(k_ if k_.startswith('debris.') else 'debris.' + k_, v_ * 1000.0)
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

TOPPLE_MIN     = 0.18                # a cube resting on another slides off once its centre is this many cube widths off-centre
TOPPLE_FULL    = 0.5                 # ...and is pushed at full strength by this offset (the edge of the cube below)
TOPPLE_ACCEL   = BODY_GRAVITY * 1.2  # sideways acceleration at full strength, blocks per second squared

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
out vec3 vLocal;     // unit-cube position, for the edge wear
out vec3 vWorld;     // world position, for the contact shading near the ground
void main() {
    vec3 p = aInstPos + aPos * aInstSize;
    vCol = aInstCol * uLight * 1.3;     // lifted to sit with the lit tiles
    vNrm = aNrm;
    vLocal = aPos;
    vWorld = p;
    gl_Position = uVP * vec4(p, 1.0);
}
"""

_DISC_FRAG = """
#version 330 core
in vec3 vCol;
in vec3 vNrm;
in vec3 vLocal;
in vec3 vWorld;
out vec4 fragColor;
LIGHT_GLSL_PLACEHOLDER
float hash3d(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}
void main() {
    vec3 n = normalize(vNrm);
    vec3 light = dayLight(n, 1.0);
    // Contact shading: near the ground, faces that look down or sideways fall into occlusion.
    float h = vWorld.y - SURF_Y_PLACEHOLDER;                 // height above the ground top, in blocks
    float contact = 1.0 - smoothstep(0.0, 0.45, h);
    light *= 1.0 - 0.5 * contact * (1.0 - 0.6 * max(n.y, 0.0));
    // Edge wear: distance from this point to the nearest cube edge. Edges chip to a lighter, dustier tone.
    vec3 d = 0.5 - abs(vLocal);
    float edge = min(d.x + abs(n.x) * 9.0, min(d.y + abs(n.y) * 9.0, d.z + abs(n.z) * 9.0));
    float wear = 1.0 - smoothstep(0.0, 0.07, edge);
    // Fine speckle per face cell, so flat faces are not perfectly uniform.
    float speck = 0.96 + 0.08 * hash3d(floor((vLocal + 0.5) * 4.0) + n * 3.0);
    vec3 col = vCol * light * speck;
    col = mix(col, col * 1.18 + vec3(0.02), wear * 0.6);
    fragColor = vec4(col, 1.0);
}
"""
_DISC_FRAG = (_DISC_FRAG.replace("LIGHT_GLSL_PLACEHOLDER", _LIGHT_GLSL)
              .replace("SURF_Y_PLACEHOLDER", repr(float(FLAT_GROUND_H))))


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
        # Camera right and up, using the same basis as _view().
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
# One 96 x 96 tile per structure material, cut from the cartoon tile sheet and embedded here so the game stays a
# single file. Layer k of the GL texture array is TILE_MATS[k]. A face is mapped one tile per block, in the
# object's own frame, so a toppled wall carries its pattern with it. Materials not listed stay plain colour.
TILE_SIZE = 96
TILE_MATS = [DIRT, STONE, BRICK, PLANKS, STEEL, CONCRETE_WHITE]
_TILE_PNG = {
    'dirt': (
        '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'
    ),
    'stone': (
        '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'
    ),
    'brick': (
        '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'
    ),
    'planks': (
        'iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAIAAABt+uBvAAAWI0lEQVR4XpVdy5JcO26sKmm8mFnbDq8cfvzNfMx8sTfeOeydY+y4HqmOg0BmIgHyVGsQN3Rb'
        '3adIEEwkHuRpPf/0x3/557///f/89vP9WPJ8PJ7P9ef1eLzf189r/fl+X+tHr+f31/P7t+fr+Xw+H9f1+Pm+fvx8//h5/Xhf//6f//uv//CH//vx/v7t+btv'
        'r2+v57fX87HGub69ntf1eF/XtYYJudb4jwe+eV1XzLDG/K///u0f//b3v/3l/Xw+ciL/E7qt0dbs72vplhM9n8/X8/Ht9Xy9ns/H49/+48//9Hd/+PNvP378'
        'XM/k+DHa42++ryGXOtfj5xoB6ryeS+fv357f1lhLyff1eNEqYRpaZ5c1fq7qWtMcRN+81jrNCJAcv/6S48XDeP5aT4QiNGOsIS2bX8uyuZc+Wq4f3+0acl9S'
        'fX3KHoqRsXdcZsorNWram0xTEAT1kdjTzWKx5Jyy5o599iUAR7FwPfwETKBuTEnw2bBmodQg1hsmss02KfhuO5yWz63CLkjrhc6F41B8QkPrxhdYdSqBj+SC'
        '982Y+x3Cx/wjY+ce14J6PinYUntfcJP8QG5WrmgtqvkEBg23x26UW2Ar8zv0fyAImsM4HQsTxfn52itumXAhZ4Et21ybcHgDeKA6eYf4y23F3saAR0NhsGUh'
        'rgWbP/Y+mRGjwEbp5unvRFMZaALluBBuOsgu9QhVXolBitvosJITfw0Hz3G1Tmjsy7qRGod+KjTB9eL7uc0yNMOD4Ur+HrsFA/siXYZvy1dyMi5mDZTfKfLB'
        'ztdgIiuQThqbQ8p0QJDZSOQzLH6wdhEQNpBbWNoCJqEE4wnhg/CiSPP43ig2w3sJ/hJBXRBuNsrFhHVgn7XQxOqBvG/cA7wZUzCupyeUOhHj8GfbwCOcSJLc'
        'ttQlwR5WoaMhCOTMCBfX+hJbF2G+kaxcpS1GeOm6xS69loFeKwGBXwgMR7Jo/p+zG3mnL6wBEZuMjM5y47TpGfgpCYIpXpLaoMjibIt4mQeB/ZsLOFIYntOl'
        'PeNbi2GSRihhkbK7hhIgEt1uv9pqsdsGQGYFBeGxtTK9MrFcHQZnUNNzCqAYkRo5NiLMp61vsSTzYEcQdLWfueHpaJH2Im2Ax1QqmBpYuiESgA5kZ6C6YvUm'
        '0+G3rIvxKKFy69rlMLJSh/33ObXlkf5kgSg8tWebRGEk6MXcTLBUYsiybqayM6GHiLNYb7FBkNnmrJMuQxSQRF9QFKsG8UwwYKLGd4tD18TfM3ot5QzyPacH'
        'eSqlDBeDysPe6ezM1eCSP99geKYYVTqkTi1QWiZNJJ25DCQz1nWtcLLUY3WSOU6FJ2qVxpTlPdVIPXN2kPRBNpJmkInKk+SPWO61AnKMiuGrFn083ulQtM47'
        'SsFW+JSrM8M7R6gmzCfBuSDaTHOU9VkQj80FZzNL2koIExar3I7ySa+iKNQmEcz6MddMQvGsF5hH/k6lo3YvGFmRT4JuGg8Gce5LteVclQpX5g01iSNE+Eri'
        'rECZ+xEzfReM09lZXt9QFmUhMB+pELU+814Ajw1Cp+D5ilFdVFjBvyJj8gynKuEhd7kVFFnYQJK8Qm2rTvBMYPYZ6Uh+KLglIvPJi5aBeglpPGUx7xPSRZb8'
        'Ihn8/XguGz0fV7ZzqKAbHtBZ6iGOKti3KW+V3wyfAaQ0KRcWS6cVtaIInIuJjlktDASgmg3xpYGfu4SwgIQoSC7bZxmzElOh4rVs9Lie78fvvkGfd+TkAb14'
        'QoOmhUpIEgbYXXaghXUWSyNbnnkO5lsJCdOEZ9benlB22cL8l5I4Ro6zBn6Fw2W18VpoYNVHGySErlX4XY93BM8ImdhpMmY+Zo2KMmDYHW2EJgYWKBWmHt7/'
        'SSJ8Z4cUG3c0kG9f61HZfMWGmXjXZxbRZKnxeiZMKEJzblHs7XomcssFsQS8iu+tYlZvIG3BKiplLx0j/4DpG5sXVZpqsk7AriVbeqyiWBXiNliLIMwPi9dj'
        'YSjEAj75n5eITzaeq52IOru1IGqojfCKt+gjE0SbK94x7l51oq+GpuSBbUXSh0JeQC2dg7nV1sCAwAY6HtHtjm1kZrx6/qv0DEfxlh1ztvTNrFJe0eFnHuGZ'
        'cLUZG10dFjUYDRSzcz26A/CyjOJzuEOi6M2aD1KVAT2uyjAV6OGKmROuw4N3fMEcKndANeTz9Yh+AHqiYiD2iZFnOjwa2uIDFXw3FB17DOAAg/OwUJL0Hsc/'
        'hlaGAH2OZWwDe+M1Fox5SvEOS5EsZKY1zKJ5FgcyDYBn5LYzELKq0UXiBpi5IhfwDgsLvqNnfopiG1LpIybO3I0KmOSMnlcUHGmjINxI2y6WRKPFISzvFFvb'
        'es83R58qHvNcSP1Ap/CQdLGeAlnbpnG+NvKAOLNOREGWWlH6oBGDLl8CJJwuH1vnf6171QF9tIAHChcrS39VSDxo1naCCwO5BtXKJlOWlGFmHJkuSh/J9VcD'
        'hK6ekqWJ22hZKs7qQHBjjQdT7eSAkcez47mx8WwaQvzhQNCG4AHOm9GbDvI1Vrks3SMNU0u/igmBvayJNshiax1seOHq0L1reKvITs4+LPnQoSlvG6Ya7Y5u'
        '9Y88/UF0xuRncBHpEMjDXjDZwUxoTlY+ZYelsdcKPQfO2vjiF4Slhs4a3WkU5qvuUvpzK21z5MPaB2SXOoFLBdCBtdwnpcClQi+7HpGXq4XLozIFzB7SutQS'
        'xp6P57g/6CX7+QFl5kFYXAnG9Ap4bJPLnphUUSqJh9jtV16N5joCM3KrZQ+3oOdNKgj9dMLYwje6LDPppGyEwYeeIGkbACcO3QqqdkEJZcdcrd+sOBHnPGFJ'
        '4aFVnouoIkNGw6gyKKyiziHsnIC0LfwGRi03KgSVP/T+ZwOnmbmVJB1eyh9d8MhXh/SJ0GwJqIUsC+q0wxv7vvAeiD9PNiAPXq4+rLY8flQI6mE9gMShuPLp'
        'QK4W/abhZ/g8+wREp8UYpgJoPLB3bXhPp2MGIKpGJ0Wi84bJN5T9+2UR76Th+WjmpGzuSz+YX7RsPj6lY4yyk42pwK8p0vxqGDVDZ1dQFy0O6ySlMsNz9Q+P'
        'fyWHnr36aYGg2q0Ooma0DzmbktoioyFo6jSjVbrOMwdtUZ1GOM+yqpKA+AoazZhlKdfv0wkSGnsjHMnF2PcfaDpK+kgRNSHW22i7wG3V7e51gfd9iCAcnEgp'
        '8Df2Av/rEWaAiBTTZQJtBukSJ2lAhmc4vuFNqhzpR0Paqpb+fgV65JU4FLI6kKEPB0o2ThvyVGx8wdKbSkrL0bExLjKSrvs8hNFXa/OZKg9kUYra1ITNly1d'
        'qiMzoLP5QnX352FdJeLU5UuVue0nDHVb59/CQH71LEGhvOV+QjXnS3RiSdu0GVtKtgVEhrbjrpRv1qWArpkcvnpInftNzUEje3YCS8V+ZhTLFkWFDyb9H2Vj'
        'IgdIednJBba2f5Jaa3psYYGZBw+UPzDu8UcHRczIGcuqIaUwPxqsihsVW7zmPMlghAy+1YftPqbDLLmkVIHBfbibhrOyhCl22Cm1GnB3HG1n7sOQBw6qRJaw'
        'R6ds1ye5wq5P4GCLxUcWA2AURSCrWyTYmEKl9bXtLIi6ZUHEsJLi1pkD75bs0lMB/zSLVbsI4UoXW5ZGzcSecEt4rkNH89DEWsogRnK01JHSD8oUPJQfdLjR'
        'Ui2I/SptA/cJqTJ9hPnZVAQt8QdVdsAcN4znmVqBKM45/QoHUz50LnTMrJUMo6sKyXn507rv0lIdkf1nitpEa/LIGFGsHRUfMk3XWxAfMqIsaUhNaHhEKZ3V'
        'pxXQ3HzUaxuDNNiTFuHd5bP9gOZL7Az2NPBInTzm+1L6IyPtBMAc7sUyS5A38BpRCTNixQ5ZsNiPdmyoZbDPxxCAjsdXJyFz/oJcdDGI33MwjbxEWKNvlFYJ'
        'nJmysCqCQzKKTVFaMPChQzTMw+lGnKbdmZHaUZcrOLdTdF4hxreMk3CLjlfwKlHSJnv2ntX0IQQ3ysQTGEu30EYRZ/pUo8Jy0ATZbsoq1ugZlUAcy33aiTwW'
        'oilPkirH2by2KO4RhU+kP6Zm6pIfLM3bH7xyYlgsrFWdXr49BTCLbpnKMC4+78scauEwJe/546DTrYl3BOKK1KCSRL0HrINcj++TX/IWSMQeHdEoJA3J0QVa'
        'UbVeGSid8iEcvDNc54czzPl/forIOQ6A1fC8L1wTWwTIjppsIbrIywr0JBlqRj+EeR1+6lyIHWLeatJm/jXi/s+iHCoVp6v2Q8LZsgqASJ52ilNAlugcDX98'
        'H7kr64g1UhgBfxwTOTrggidVKFWyf84bYwWfNN5dDrSDy5/1H7tKHTgsxzoHOVXvfRu5DiB2sA6sKtfjjtjsO2Fz3Hl5gVdmEDQzSUtjyYQ4b0964qVJX1X5'
        'Xr91dLwNoVovrxeKpnifIK6hysBbMG+k21AD7ISeZI1UGC8O5XX4+BFv4Ikuc8LruuBiRYc6F2eLHImsLsohMw7HLst7jS12P8c6VRsu2U6oy+b2s7KJFXHm'
        'c/trCtUatyOKQmLO9WVXINFNkrYtQsCgQwE5qVwgIq738vozbvcx+oyV4XrtoELYCLskh/e2bwNb4plBSnSoudDPKipmjg5KjQ9nPslSPG8yGoi4TpNYDDio'
        '6epnEzj5JNWxelSS1pi1W+dQSYjpwQtMcFimsi+Vu+d5AmFnFMgchIYi+x+bD7Iq87KqMC2mbrLC/M0Do//oa/OilFfAxa/rImvS05Z5lAiVFVRji9Xb9Ec9'
        'LHgQy6mDuaoYBqlgVTN0AKx84TDuUub11rwKfBA0zCxhOQ96Mi68NG9A8fULM58iYBLeGILBRR6W2Y+weVydLCuz6XawNY57p6hN3IxQFRKVGBIclMrjbs7R'
        'EHpFImNBrVbGXDPFdTFtTlJ87Azzt44KkoDHuYhizE90j7CtaXBFZnrr3aNyBGhobTxLG2yF+SWjoGZxhZ5pIEtMsGSlwjDGeCHD/AcJRLy8EjfvrryoapF5'
        'Clk23hT2oWqvvArAMpx88CATaCbHMUA2D0hhefhQjHlyBEsBIEwNloboB7kbN7xhdLU0EiJ+WiunAIIU+7VbrRnObVNztpaO1EELLn3NWXEJxO9ipAawC8oU'
        'uavRHL95Jhs1p5XwxXeqo+jObwGU+YK9FMZ7J7Tk1s0Q4HNM3p1K2ygHleFmyu2dI+9M+nmkEpwTJoif6mi0MY+JmIRxG4JzsdkThWvxf4y7uXImjbWwlrwT'
        '1oUAZgd1LlRvLLT7mkSBtC2esG6GrYUaVjQ0Ka1qC3PD0IRsSYNtH9AUW5inGqR9JnSauL4wovLLTrCSWnwiklQ9iEmVMKu8/u7vaPbQqammNRVopC0e30Re'
        'j81g8fX/1hqy6o42yg4nJokoVoSfL1vd4BY1sJwkHsVrfga5sf24NQHsiY+i7RRXyePlH7zzlztUiaJsIrcIrIwXS7p9kPFTLMAp6irD5rhdlMg9n09EMbnt'
        'yFSVlWZXhUZmWMt3x+6EjZhV6/ITvCoNK6/363rkFkejttHlMkj+YomJcS2UlzxdLwYyLvI1WxgaO7GDvCS3v87mZ/DQFEhMyskARZRIp03ox1lCsVab9zJ1'
        'ExhTKR5L4RwcLakWXgyyZS+3ysARsLlBZd9gtZPSUKrFqk2VP5if9Dagftofch6t6t/2pyQxJVdntpbZnU4pVM1pY/eMYacfzGi64Rk+yg12dVJ57Ug14UDS'
        'HxzlIDbYFP6Il1Ythm+1WWGUJJhVC5IBAr46Ud2zELlcdzrIKQloa5z46l7mglcRtgrsC25J8RV6tWXNo3BkZRE6YC0y9jwTr8DY8oAhZYdNOl0hdhOVzC1v'
        'N9hlAKQiib+K0Cn+4GU2yinQ8fMIAXlVHD0YpiSel7dLU/V2EBBE9XMmT3PQ69oA8aX8iptUpkaimW/7JDD04xSdBNwQInvgWhX5RQlFOhEbhgzk6tKjG41u'
        'Vq9fK0gwFUO6poLLZecXLeGzOEFZFFop26/Lza45udQ8dTdBjVmvVxwCXpp4V5iv/7QAy1QUW7e3Z3aS/jXTdDsTw7dvHA5GNHRMUX6xX+HQb+lCkZINdHYy'
        'ZFilQmiN4JdL8lC/EjCMIy9jur1t3amDOpYgQ6OnXV6dasK7YaDBL58o+iTOESmodyJm+56zYZRKZKLIP9esK2/MFxCzCfdCS4kHEUxokuXHrmNl1p//NbFq'
        'gGOo7X7jYt2cuzIlVNJPuDTNwdIqhFqRpZO49XO+gxi0Vdkj4p2AR+QeE8W/dotv6CN+AZns3r+67Zo041iTxI5EKSOvU7FZyTEkyxG0GngfGZw9Gkywi/XE'
        'arbeqfJEpGC1gYpppGr+JhNBEb9mgX0Uw3cdGuLI6U6oHRsgARy+5AOzWXtB92AUSbOB2Xj6JFUJbLn+xJb7qkUb0WIz0BYjOXzRaapTxlEGoBMblIR4pBQc'
        'qY272Ee0zvTFfHqmF+Xb7Pg4lm01O4r2YAjBb8HzFmYbtJsjHyt0M0d2AjrnJwq+ihG4lDxSrhB1dG8rgh7su0wn6m2x8dOWgvb+fEoP8zs3HyVhaeBkppcZ'
        'jQcFz78KodXTs3MFdYKs6L23UCqwZcdpfys7yq35tJmo+9dR4peOWNYglvCnhxr6q/jSTo3JrJ7BHfj+OE/iNfpwOtGliab7NyVMbI/Hdh9aFCbK5oYsDlLf'
        'IZX5MJB+ZMzPfhRO1zC+O9pnzSQ6gPCOZ+JBio0M0DHUzIEtVwPDXPyYALg3dGxamBecj0NINuKuqrpqK7r25gKfRY/z+tYXupzZhw7hG0Pq+8AZiV9uMB/M'
        '30hOv9197eDkeQ5VCZuGVPT1g7oTIe2toSaHDVL9HJD6wsx6Zn/w9FGorYrP7ZS/QJghbLp5t2mTOwbAkY+OshNJXhnJkfSFgRMa3Nwj+BXblKOdWM7tU1+T'
        'L1qWRTAlB8XcyT6zhf6F7FvScqSsTNVCYWdH3FE+NYAxxu3dSJHBUVRn3v30g9iuQbX4rc0t7yg5EMhuj5tt1RFg3nBzCugWpFRp1uYAZe4wI7wx2pl7u7fs'
        'UnzKPdtOPfN3qkXiIOxsCN+WRK1qdQ0AhVl4GM8zqHG70WNTIwMyvfMT5PyjrB/iE/fx0jdblHvGQkdruFjoCEu1diKjX//MQJZnCZ1tuIECKRqo9XpspUs8'
        '/xpQqGO/3t5wBrF9uglWLIjcow9ePDp/8TQ4SNZpYtnBXeAhsZdm+y5aZizPqOQ7Z8k98usJZRG2aG9FXnAT4XIdCcPAG0jMndljWSUu/V9FgKInOWX1KXWY'
        'ZDiXpEurkSGy81+4jGEUIsrBeB+GmbrSlE8s3SmMXU7Pb2wL9XA58cAR/ukabzIe7AoqwDTlR5Maz/EZ9kkTCBmwkQ3KO9mczSqY0L+a9Zu4zrVo6j1atNrX'
        'EbJ1exBn1bEfKDW0JbiNsokFZm6GKXUfQ0zyNjq7Prqbsv5VKnhZ3vrguxtsDLF/VjYaksofo3tv0abJu11iFVJG94jELvqHR9gkI+DPnsYJST2dIHfsUQ40'
        'pI4qezHqEFHjOgHT5U62pE/aFRM1L6sW7Z5MNYGXDZv+P4yAPecD6PT7AAAAAElFTkSuQmCC'
    ),
    'steel': (
        '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'
    ),
    'concrete': (
        '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'
    ),
}

def _decode_tiles():
    import base64, io
    out = []
    for n in ('dirt', 'stone', 'brick', 'planks', 'steel', 'concrete'):
        raw = base64.b64decode(''.join(_TILE_PNG[n]))
        surf = pygame.image.load(io.BytesIO(raw), n + '.png')
        assert surf.get_size() == (TILE_SIZE, TILE_SIZE)
        out.append(np.frombuffer(pygame.image.tostring(surf, 'RGB'), np.uint8).reshape(TILE_SIZE, TILE_SIZE, 3).copy())
    return np.stack(out)

TILE_RGB = _decode_tiles()                         # (layers, 96, 96, 3) uint8
_S_TEXLAYER = np.full(256, -1.0, np.float32)       # block id -> texture layer, -1 = plain colour
for _k, _b in enumerate(TILE_MATS):
    _S_TEXLAYER[_b] = _k
    _S_RGB[_b] = TILE_RGB[_k].reshape(-1, 3).mean(axis=0) / 255.0    # a tile's average colour: debris and chips match it

# Simplified look: each structure material is one flat colour instead of a texture. These colours are also what its debris
# and chips use. (The ground keeps its own dirt / stone tiles; they are drawn by the terrain shader, not from this table.)
MATERIAL_COLORS = {
    DIRT:           (0.66, 0.49, 0.31),    # light brown
    STONE:          (0.30, 0.30, 0.32),    # dark grey
    BRICK:          (0.70, 0.22, 0.17),    # red
    PLANKS:         (0.82, 0.65, 0.42),    # light brown (a touch lighter and warmer than dirt)
    STEEL:          (0.68, 0.70, 0.74),    # light grey
    CONCRETE_WHITE: (0.90, 0.90, 0.89),    # very light grey
}
for _b, _c in MATERIAL_COLORS.items():
    _S_TEXLAYER[_b] = -1.0                 # -1 = plain colour, no texture
    _S_RGB[_b] = _c

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
            d = _density_many(world, Pc)
            near = d > -0.3
            if near.any():
                Q = Pc[near]
                dq = d[near]
                e = 0.2
                ex, ey, ez = np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e])
                dd = _density_batch(world, [Q + ex, Q - ex, Q + ey, Q - ey, Q + ez, Q - ez])
                g = np.stack([dd[0] - dd[1], dd[2] - dd[3], dd[4] - dd[5]], axis=1) / (2 * e)
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

    def blast(self, world, centre, radius, power=1.0, shatter=False, throw=True):
        """Damage structures, push bodies, throw chunks. The caller draws the fireball. throw=False: what is destroyed is
        vaporised (the melting laser), so no chunks are thrown."""
        centre = np.asarray(centre, np.float64)
        STRUCT_HIT['pos'] = tuple(float(c) for c in centre)          # the surface dents here, see the solid shader
        STRUCT_HIT['t0'] = time.perf_counter()
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
                self._throw_chunks(kc, km, centre, radius, world=world, cs=s.cs)
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
                self._break_loose(world, b, centre, radius, power, shatter)
        return broken

    def _break_loose(self, world, body, centre, radius, power, shatter=False):
        """A blast or drill hit on a loose body: cells near the centre crack or are destroyed, then whatever is no
        longer connected breaks away."""
        kc, km = body.damage_at(centre, radius, power, shatter)
        self._finish_body(world, body, kc, km, centre, radius)

    def _finish_body(self, world, body, kc, km, centre, radius):
        """Throw out destroyed cells, split off pieces that are no longer connected, and keep the body count under
        the cap. Pieces split off a body topple away from the hit, like detached structure."""
        if len(kc):
            self._throw_chunks(kc, km, centre, radius, world=world, cs=body.cs)
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

    def _throw_chunks(self, pos, mats, centre, radius, slow=False, world=None, cs=0.5):
        if not STRUCT_SHARDS:
            return                                                # no cloud of loose cubes (see STRUCT_SHARDS)
        # A big hit at a high cube detail can break thousands of small cells at once; spawning a piece for each
        # of them bogs the frame down. Keep a random sample of them as debris and let the rest vanish.
        MAX_CHUNK_CELLS = 300
        pos, mats = np.asarray(pos), np.asarray(mats)
        if len(pos) > MAX_CHUNK_CELLS:
            sel = _rng.choice(len(pos), MAX_CHUNK_CELLS, replace=False)
            pos, mats = pos[sel], mats[sel]
        n = len(pos)
        scale = cs / 0.5                                          # debris is sized to the cells it came from
        k = min(3, max(1, 60 // max(n, 1)))
        P = np.repeat(pos, k, axis=0) + _rng.uniform(-0.12, 0.12, (n * k, 3)) * min(1.0, cs / 0.5)
        d = P - centre
        dist = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 0.2)
        speed = _rng.uniform(0.5, 2.0, (n * k, 1)) if slow else _rng.uniform(4.0, 12.0, (n * k, 1))
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



# ---- the demo layout: one solid test cube per material, beside the dirt cube ------------------------------
def _solid_cube(mat):
    return lambda n: np.full((n, n, n), mat, np.int16)           # always 10 x 10 x 10 blocks; n = 10 * cells per block


DEMO_NAMES = ['Dirt', 'Stone', 'Brick', 'Planks', 'Steel', 'Concrete']
DEMO_LAYOUT = [(0, _solid_cube(DIRT)), (1, _solid_cube(STONE)), (2, _solid_cube(BRICK)), (3, _solid_cube(PLANKS)), (4, _solid_cube(STEEL)),
               (5, _solid_cube(CONCRETE_WHITE))]
DEMO_MATS = [DIRT, STONE, BRICK, PLANKS, STEEL, CONCRETE_WHITE]
SHAPE_NAMES = ['cube']


def _shape_grid(shape, mat, res):
    """Material grid for the giant cube (10 x 10 x 10 blocks), `res` cells per block."""
    return np.full((10 * res,) * 3, mat, np.int16)


def _layout_ready(self, world):
    """True when every chunk under the layout is loaded."""
    x0, _, z0 = cube_slot_lo(0)
    for cx in range(math.floor((x0 - 1) / CHUNK_SIZE), math.floor((x0 + 11) / CHUNK_SIZE) + 1):
        for cz in range(math.floor((z0 - 1) / CHUNK_SIZE), math.floor((z0 + 11) / CHUNK_SIZE) + 1):
            if (cx, cz) not in world.dens:
                return False
    return True


def _build_layout(self, world, res=2, which=0, shape=0):
    """Stand ONE test cube, of material number `which` in DEMO_LAYOUT, in the first slot straight ahead of the
    start point. `res` is cells per block along each side: the cube keeps its size, and a higher value builds it
    from smaller pieces."""
    res = max(1, int(res))
    cs = 1.0 / res
    for k, make in [DEMO_LAYOUT[which % len(DEMO_LAYOUT)]]:
        m = _shape_grid(shape, DEMO_MATS[which % len(DEMO_MATS)], res)
        x0, _, z0 = cube_slot_lo(0)
        x0 -= 0.5                                                  # the cube's faces sit half a block outside its samples
        z0 -= 0.5
        w, d = m.shape[0] * cs, m.shape[2] * cs                    # centre each shape in its 10 x 10 slot
        pad = int(math.ceil(0.4 * res))                            # empty cells all round, so metal pressed aside by a dent has room to rise
        m = np.pad(m, ((pad, pad), (0, pad), (pad, pad)))          # not under it: the cube still stands on the ground
        self.add(world, m, x0 + (10.0 - w) * 0.5 - pad * cs, z0 + (10.0 - d) * 0.5 - pad * cs, cs)


StructureManager.layout_ready = _layout_ready
StructureManager.build_layout = _build_layout


_SOLID_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aCol;
layout(location = 3) in float aMat;   // texture layer, -1 = plain colour
uniform mat4 uVP;
uniform mat4 uModel;       // the object's pose: rotation and position
uniform vec3 uHit;         // world point of the last hit on a structure
uniform float uHitAmt;     // how strong the dent still is: 0 = none, decays after the hit
uniform float uHitT;       // seconds since the hit, for the wobble
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
        col = mix(col, FOG_COL, 1.0 - exp(-vDepth * 0.014));
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

    def render(self, renderer, struct_gl, lvp):
        """Draw the terrain and the structures' last-uploaded meshes from the sun into the depth map. Loose
        dirt cubes do not cast shadows."""
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, SHADOW_RES, SHADOW_RES)
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
            model[:3, :3] = o.R
            model[:3, 3] = o.frame_pos
            gl.glUniformMatrix4fv(self.u_model, 1, gl.GL_TRUE, model)
            gl.glBindVertexArray(ent[0])
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, ent[2])
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, WIDTH, HEIGHT)


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
        self.u_tiles = gl.glGetUniformLocation(self.program, "uTiles")
        self.u_grid = gl.glGetUniformLocation(self.program, "uGrid")
        self.u_cell = gl.glGetUniformLocation(self.program, "uCell")
        self.u_corner = gl.glGetUniformLocation(self.program, "uCorner")
        self.u_hit = gl.glGetUniformLocation(self.program, "uHit")
        self.u_hit_amt = gl.glGetUniformLocation(self.program, "uHitAmt")
        self.u_hit_t = gl.glGetUniformLocation(self.program, "uHitT")
        self.light_vp = np.eye(4, dtype=np.float32)   # set each frame by the main loop
        self.shadow_tex = 0
        self.icache = {}     # integrity colouring: id(object) -> [vao, vbo, count, (mesh ver, stress ver), object]
        self.cache = {}      # id(object) -> [vao, vbo, vertex count, mesh version, object]

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

    def render(self, vp, ambient, overlay=False, grid=False):
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
        gl.glUniform1f(self.u_grid, 1.0 if grid else 0.0)
        age = time.perf_counter() - STRUCT_HIT['t0']
        amt = math.exp(-age * 2.2) if (STRUCT_HIT['pos'] is not None and age < 4.0) else 0.0
        hit = STRUCT_HIT['pos'] if STRUCT_HIT['pos'] is not None else (0.0, 0.0, 0.0)
        gl.glUniform3f(self.u_hit, *(float(c) for c in hit))
        gl.glUniform1f(self.u_hit_amt, float(amt))
        gl.glUniform1f(self.u_hit_t, float(max(age, 0.0)))
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
            for o in objs:
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


def _split_body(body):
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


_MAT_NAMES = {b: n for b, n in ((DIRT, 'Dirt'), (STONE, 'Stone'), (BRICK, 'Brick'), (PLANKS, 'Planks'), (STEEL, 'Steel'), (CONCRETE_WHITE, 'Concrete'))}


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
        super().__init__(self.SIZE, self.SIZE, (WIDTH - self.SIZE) // 2, (HEIGHT - self.SIZE) // 2)

    def _render(self, state):
        c = self.SIZE // 2
        arms = (((c - 22, c), (c - 8, c)), ((c + 8, c), (c + 22, c)),
                ((c, c - 22), (c, c - 8)), ((c, c + 8), (c, c + 22)))
        for col, wd in (((0, 0, 0, 220), 6), ((255, 255, 255, 255), 3)):   # dark outline, then white
            for a, b in arms:
                pygame.draw.line(self.surf, col, a, b, wd)
        pygame.draw.circle(self.surf, (0, 0, 0, 220), (c, c), 5)
        pygame.draw.circle(self.surf, (255, 255, 255, 255), (c, c), 2)


class LaserMeter(_CachedOverlay):
    """Bottom-left readout shown while the melting laser is in hand: its temperature, and a bar for how long it has dwelt on one spot."""
    W, H = 330, 96
    SEGS = 20

    def __init__(self):
        super().__init__(self.W, self.H, 24, HEIGHT - self.H - 24)
        self.f_big, self.f_small = _mono(26), _mono(16)

    def _render(self, level):
        s = self.surf
        pygame.draw.rect(s, (*UI_BG[:3], 210), (0, 0, self.W, self.H), border_radius=10)
        pygame.draw.rect(s, UI_BORDER, (0, 0, self.W, self.H), 2, border_radius=10)
        s.blit(self.f_small.render("MELTING LASER", True, UI_DIM), (14, 10))
        txt = self.f_big.render("{:,} \u00b0C".format(Arsenal.TEMP_C), True, (200, 225, 255))
        s.blit(txt, (self.W - 14 - txt.get_width(), 30))
        n = self.SEGS
        seg_w, gap = (self.W - 28 - (n - 1) * 3) // n, 3
        for i in range(n):
            t = i / (n - 1)
            col = (int(90 + 165 * t), int(140 + 115 * t), 255)         # blue -> white-hot
            r = pygame.Rect(14 + i * (seg_w + gap), 66, seg_w, 10)
            pygame.draw.rect(s, col if i < level else UI_SEG_OFF, r, border_radius=2)
        s.blit(self.f_small.render("T: switch weapon", True, UI_DIM), (14, 78 - 2))


class LoadingScreen(Overlay):
    """The first thing on screen: a progress bar, the stage being worked on, and a running list of what is done.
    step() draws one frame at once and keeps the window responsive, so it is safe to call between slow jobs."""
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, 0, 0)
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
        self._show()

    def _render(self):
        s = self.surf
        s.fill((12, 14, 20, 255))
        cx = WIDTH // 2
        t = self.f_title.render("VOIDCRAFT", True, UI_TEXT)
        s.blit(t, (cx - t.get_width() // 2, HEIGHT // 2 - 190))
        bw, bh = 760, 26
        bx, by = cx - bw // 2, HEIGHT // 2 - 40
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
        s.blit(el, (WIDTH - el.get_width() - 24, HEIGHT - 36))

    def _show(self):
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit()
                raise SystemExit
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

# ═══════════════════════════════════════════════
#  MONITOR (G): every stage of the frame, the GPU, the world and the process, in one overlay
# ═══════════════════════════════════════════════
PROF_WINDOW = 240              # frames kept for every graph and average (4 s at 60 fps)
PROF_TICK_S = 0.25             # the monitor's numbers and graphs refresh this often
MON_W, MON_H = 1200, 775
MON_BANDS = 6                 # the overlay is redrawn one horizontal band per frame, spread over this many frames
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
    'input': 'input + pad', 'chunks': 'chunk load + layout', 'player': 'player',
    'debris': 'debris physics', 'structs': 'structs + bodies', 'weapon': 'axe hit (cut + dig)',
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


def _monitor_tick():
    return int(time.perf_counter() / PROF_TICK_S)


class MonitorOverlay(_CachedOverlay):
    """The G overlay. Redrawn every PROF_TICK_S rather than every frame, so its own cost stays small."""

    def __init__(self):
        super().__init__(min(MON_W, WIDTH - 16), min(MON_H, HEIGHT - 16), 8, 8)
        self.src = {}
        self.show_stats = True                       # V toggles the stat columns; the graphs always stay
        self.f_head = _mono(16)
        self.f_row = _mono(13)
        self._allocated = False                      # set once the texture has storage
        self._glyphs = {}                            # (text, colour, font) -> rendered surface
        self._tick = None                            # tick the overlay content is for
        self._stage = None                           # next piece to redraw, None when idle
        self._bar_cols = None
        self._bg_for = None                          # show_stats the panel background was drawn for
        self._integ = ()                             # integrity lines, computed once per tick
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.w, self.h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        self._allocated = True
        self.proc = None
        if HAS_PSUTIL:
            try:
                self.proc = psutil.Process()
                self.proc.cpu_percent(None)          # the first call only starts the measurement
            except Exception:
                self.proc = None

    STAGES = 4                                       # header + graphs, then the three stat columns: one per frame

    def draw(self, tick):
        # A new tick redraws the overlay one piece per frame (graphs, then each stat column), uploading just that piece, so
        # no single frame pays for the whole layout.
        if self._stage is None and tick != self._tick and self.src:
            self._tick = tick
            self._stage = 0
        if self._stage is not None:
            self._run_stage(self._stage)
            self._stage += 1
            if self._stage >= self.STAGES:
                self._stage = None
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.program)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glUniform1i(self.u_tex, 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)

    _BG = (14, 16, 24, 235)

    def _run_stage(self, i):
        W, H = self.w, self.h
        s = self.surf
        if self._bg_for != self.show_stats:                  # first draw, or V toggled: lay the panel down again
            s.fill((0, 0, 0, 0))
            if self.show_stats:
                pygame.draw.rect(s, self._BG, (0, 0, W, H), border_radius=10)
                pygame.draw.rect(s, (90, 98, 120, 255), (0, 0, W, H), 2, border_radius=10)
            self._bg_for = self.show_stats
            self._upload_rect(pygame.Rect(0, 0, W, H))
        if i > 0 and not self.show_stats:
            return                                           # graphs only: there are no columns
        rect = pygame.Rect([(4, 4, W - 8, 192), (4, 196, 400, H - 200), (404, 196, 400, H - 200), (804, 196, W - 808, H - 200)][i])
        s.fill(self._BG if self.show_stats else (0, 0, 0, 0), rect)
        if i == 0:
            self._render_top()
        elif i == 1:
            self._render_col1()
        elif i == 2:
            self._render_col2()
        else:
            self._render_col3()
        self._upload_rect(rect)

    def _upload_rect(self, r):
        """Upload one rectangle of the finished surface to the texture."""
        # The texture is uploaded flipped (row 0 at the bottom), so the rectangle's offset is counted from the bottom.
        data = pygame.image.tostring(self.surf.subsurface(r), "RGBA", True)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, r.x, self.h - r.bottom, r.w, r.h,
                           gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)

    def _t(self, x, y, text, col=UI_TEXT, f=None):
        # Glyph cache: labels never change and numbers repeat between ticks, so most
        # font.render() calls (the expensive part of each redraw) become a dict lookup.
        f = f or self.f_row
        key = (text, col, id(f))
        glyph = self._glyphs.get(key)
        if glyph is None:
            if len(self._glyphs) > 2000:
                self._glyphs.clear()
            glyph = self._glyphs[key] = f.render(text, True, col)
        self.surf.blit(glyph, (x, y))

    def _bar(self, x, y, w, h, frac, col):
        s = self.surf
        pygame.draw.rect(s, (30, 34, 46, 255), (x, y, w, h))
        fw = int(w * max(0.0, min(frac, 1.0)))
        if fw > 0:
            pygame.draw.rect(s, col, (x, y, fw, h))

    def _section(self, x, y, w, title):
        self._t(x, y, title, UI_ACCENT, self.f_head)
        pygame.draw.line(self.surf, UI_LINE, (x, y + 20), (x + w, y + 20), 1)
        return y + 26

    def _header(self, x, y):
        for dx, text in ((14, 'stage'), (178, 'now'), (220, 'avg'), (262, '1%'), (304, 'budget')):
            self._t(x + dx, y, text, UI_DIM)
        return y + 16

    def _lines(self, x, y, lines, col=UI_TEXT):
        for line in lines:
            self._t(x, y, line, col)
            y += 16
        return y

    def _nums(self, x, y, label, color, now, avg, low, indent=False, chip=True):
        s = self.surf
        if chip:
            pygame.draw.rect(s, color, (x, y + 4, 8, 8))
        self._t(x + (24 if indent else 14), y, label, UI_DIM if indent else UI_TEXT)
        self._t(x + 178, y, f"{now:5.2f}")
        self._t(x + 220, y, f"{avg:5.2f}")
        self._t(x + 262, y, f"{low:5.2f}", UI_DIM)
        self._bar(x + 304, y + 4, 60, 9, avg / FRAME_BUDGET_MS, color)
        return y + 15

    def _row(self, x, y, label, color, vals, indent=False, chip=True):
        now, avg, low = _stats(vals)
        return self._nums(x, y, label, color, now, avg, low, indent, chip)

    def _plot(self, rect, title, series, vmax, refs):
        s = self.surf
        x, y, w, h = rect
        pygame.draw.rect(s, (12, 12, 18, 235), rect)
        for val, col in refs:
            yy = int(y + h - min(val / vmax, 1.0) * h)
            pygame.draw.line(s, col, (x, yy), (x + w, yy), 1)
        for vals, col in series:
            n = len(vals)
            if n >= 2:
                pts = [(x + i * w / (n - 1), y + h - min(v / vmax, 1.0) * h) for i, v in enumerate(vals)]
                pygame.draw.lines(s, col, False, pts, 2)
        pygame.draw.rect(s, (90, 90, 110, 255), rect, 1)
        self._t(x + 6, y + 3, title, UI_DIM)

    def _stacked(self, rect, series, vmax, title):
        """One column per frame, newest on the right, each stage stacked on the one before it."""
        s = self.surf
        x, y, w, h = rect
        pygame.draw.rect(s, (12, 12, 18, 235), rect)
        n = len(series[0]) if series else 0
        cw = w / PROF_WINDOW
        for i in range(n):
            base = 0.0
            cx = int(x + w - (n - i) * cw)
            for k, vals in enumerate(series):
                ms = vals[i]
                if ms <= 0.0:
                    continue
                top = base + ms
                y_top = int(y + h - min(top / vmax, 1.0) * h)
                y_bot = int(y + h - min(base / vmax, 1.0) * h)
                pygame.draw.rect(s, PROF_COLOR[PROF_TOP[k]],
                                 (cx, y_top, max(1, int(cw + 0.999)), max(1, y_bot - y_top)))
                base = top
        pygame.draw.rect(s, (90, 90, 110, 255), rect, 1)
        self._t(x + 6, y + 3, title, UI_DIM)

    def _render_top(self):
        s = self.surf
        W, H = self.w, self.h
        src = self.src
        if not src:
            return
        world, renderer, player = src['world'], src['renderer'], src['player']
        settings, bench, poller = src['settings'], src['bench'], src['poller']

        if self.show_stats:
            self._t(12, 6, "VOIDCRAFT MONITOR", UI_ACCENT, self.f_head)
            hint = "G closes this overlay   V stats on / off"
            self._t(W - 12 - self.f_row.size(hint)[0], 8, hint, UI_DIM)

        # ---- headline numbers
        _, fr_avg, fr_p99 = _stats(list(PROF.frame_ms))
        _, wk_avg, _ = _stats(list(PROF.work_ms))
        fps = 1000.0 / fr_avg if fr_avg > 0 else 0.0
        fps_low = 1000.0 / fr_p99 if fr_p99 > 0 else 0.0
        flags = [name for name, on in (("BENCHMARK", bench.active), ("PAUSED", src.get('paused')),
                                        ("AUTO-WALK", src.get('auto_walk')), ("FLYING", player.flying)) if on]
        strip = (f"FPS {fps:5.1f}   frame {fr_avg:5.2f} ms   1% low {fps_low:5.1f} fps   "
                 f"work {wk_avg:5.2f} ms   gpu {GT.total():5.2f} ms   "
                 f"cap {settings['fps_limit'] or 'off'}   {' '.join(flags) or 'walking'}")
        if self.show_stats:
            self._t(12, 30, strip, UI_TEXT)

        # ---- graphs: frame time (top left), stacked stage time (top right). Same size with or without the stats.
        frames = list(PROF.frame_ms)
        n = len(frames)
        vmax = max(40.0, min(120.0, max(frames) * 1.05)) if frames else 40.0
        top_y = 58 if self.show_stats else 0                           # text hidden: the graphs move up to the top edge
        gx, gy, gw, gh = 10, top_y, 585, 130                           # side by side, above the stat columns
        g2x, g2y, g2w, g2h = 605, top_y, 585, 130
        self._plot((gx, gy, gw, gh), "frame time (blue)   CPU work (green)   lines at 16.7 and 33.3 ms",
                   [(frames, UI_ACCENT), (_window(PROF.work_ms, n), (150, 230, 140))], vmax,
                   [(FRAME_BUDGET_MS, (60, 120, 60)), (2 * FRAME_BUDGET_MS, (140, 120, 50))])
        top = [_window(PROF.hist[k], n) for k in PROF_TOP]
        totals = [sum(col[i] for col in top) for i in range(n)]
        vmax2 = max(20.0, min(160.0, max(totals) * 1.05)) if totals else 20.0
        self._stacked((g2x, g2y, g2w, g2h), top, vmax2, "stages stacked over the last 4 s (ms)")

    def _render_col1(self):
        s = self.surf
        W, H = self.w, self.h
        src = self.src
        y0 = 200                                                       # below the graphs; the stage list must fit above the bottom edge
        n = len(PROF.frame_ms)
        # ---- column 1: CPU stages
        x = 10
        y = self._section(x, y0, 380, "CPU PER STAGE (ms)")
        y = self._header(x, y)
        for key in PROF_TOP:
            y = self._row(x, y, PROF_LABEL.get(key, key), PROF_COLOR[key], list(PROF.hist[key]))
            if key == 'terrain':
                for sub in PROF_SUB:
                    y = self._row(x, y, PROF_LABEL[sub], PROF_COLOR[sub], list(PROF.hist[sub]), True)
            if key == 'debris':
                for sub in DEBRIS_SUB:
                    y = self._row(x, y, PROF_LABEL[sub], PROF_COLOR[sub], list(PROF.hist[sub]), True)
            if key == 'structs':
                for sub in STRUCTS_SUB:
                    y = self._row(x, y, PROF_LABEL[sub], PROF_COLOR[sub], list(PROF.hist[sub]), True)
        y += 4
        y = self._row(x, y, PROF_LABEL[PROF_FLIP], PROF_COLOR[PROF_FLIP], list(PROF.hist[PROF_FLIP]))
        idle = [f - w - fl for f, w, fl in zip(_window(PROF.frame_ms, n), _window(PROF.work_ms, n),
                                               _window(PROF.hist[PROF_FLIP], n))]
        y = self._row(x, y, "work total", UI_TEXT, list(PROF.work_ms), chip=False)
        y = self._row(x, y, "idle (fps cap)", UI_DIM, idle, chip=False)


    def _render_col2(self):
        s = self.surf
        W, H = self.w, self.h
        src = self.src
        y0 = 200                                                       # below the graphs; the stage list must fit above the bottom edge
        poller = src['poller']
        # ---- column 2: GPU stages and the device
        x = 410
        y = self._section(x, y0, 380, "GPU PER STAGE (ms, timer queries)")
        y = self._header(x, y)
        gpu_avg_sum = 0.0
        for key in GPU_STAGES:
            vals = list(GT.hist.get(key, ()))
            gpu_avg_sum += _stats(vals)[1]
            y = self._row(x, y, PROF_LABEL.get(key, key), PROF_COLOR[key], vals)
        y += 4
        y = self._nums(x, y, "sum of stages", UI_TEXT, GT.total(), gpu_avg_sum, 0.0, chip=False)
        y = self._section(x, y + 8, 380, "DEVICE")
        st = poller.stats if poller is not None else None
        if st:
            used, total = st.get('mem_used'), st.get('mem_total')
            vram = 'n/a' if used is None or not total else f"{used / 1024:.2f} / {total / 1024:.2f} GB"
            y = self._lines(x, y, [
                GPU_NAME[:46],
                f"util {_fmt(st.get('util'), '.0f', '%')}   sm {_fmt(st.get('clock_sm'), '.0f', ' MHz')}",
                f"mem clock {_fmt(st.get('clock_mem'), '.0f', ' MHz')}   fan {_fmt(st.get('fan'), '.0f', '%')}",
                f"vram {vram}",
                f"temp {_fmt(st.get('temp'), '.0f', ' C')}   power {_fmt(st.get('power'), '.0f', ' W')}",
                "timer queries lag about one frame",
            ])
        else:
            msg = "nvidia-smi: no reading" if poller is not None and poller.available else "nvidia-smi: not found"
            y = self._lines(x, y, [GPU_NAME[:46], msg])


    def _render_col3(self):
        s = self.surf
        W, H = self.w, self.h
        src = self.src
        y0 = 200                                                       # below the graphs; the stage list must fit above the bottom edge
        world, renderer, player = src['world'], src['renderer'], src['player']
        settings = src['settings']
        self._integ = tuple(integrity_state(player))
        # ---- column 3: process, world, dynamics, player, settings, integrity
        x = 810
        y = self._section(x, y0, 380, "PROCESS")
        cpu_txt, rss_txt = "n/a", "n/a"
        if self.proc is not None:
            try:
                cpu_txt = f"{self.proc.cpu_percent(None):.0f}%"
                rss_txt = f"{self.proc.memory_info().rss / 2 ** 20:.0f} MB"
            except Exception:
                pass
        y = self._lines(x, y, [
            f"cpu {cpu_txt}   rss {rss_txt}   threads {threading.active_count()}",
            f"mesh {MESH_WORKERS}w   jobs {len(renderer.mesher.pending)}/{MESH_MAX_PENDING}   wait {len(renderer._waiting)}",
        ])
        y = self._section(x, y + 6, 380, "WORLD")
        used_mb = (renderer.capacity - sum(length for _, length in renderer.free)) * VERT_STRIDE / 2 ** 20
        cap_mb = renderer.capacity * VERT_STRIDE / 2 ** 20
        hit = 100.0 * CACHE.hit_m / max(1, CACHE.hit_m + CACHE.miss_m)
        y = self._lines(x, y, [
            f"chunks {len(world.chunks)}   edited {len(world.saved)}   dist {settings['render_distance']}",
            f"draw ranges {renderer._n}   free runs {len(renderer.free)}",
            f"arena {used_mb:.1f} / {cap_mb:.1f} MB   cache hit {hit:.0f}%",
            f"mesh cache {CACHE.mesh_bytes / 2 ** 20:.1f} MB",
        ])
        awake_debris = int(BODIES.awake.sum())
        bodies_awake = sum(1 for b in STRUCTS.bodies if not b.asleep)
        cells = sum(o.live_count for o in STRUCTS.objects())
        y = self._section(x, y + 6, 380, "DYNAMICS")
        y = self._lines(x, y, [
            f"debris {BODIES.n}   awake {awake_debris}   cap {BODY_MAX}",
            f"fixed {len(STRUCTS.structs)}   loose {len(STRUCTS.bodies)}   loose awake {bodies_awake}",
            f"live cells {cells}",
        ])
        yaw_d, pitch_d = math.degrees(player.yaw) % 360.0, math.degrees(player.pitch)
        y = self._section(x, y + 6, 380, "PLAYER")
        y = self._lines(x, y, [
            f"pos {player.pos[0]:.1f}  {player.pos[1]:.1f}  {player.pos[2]:.1f}",
            f"yaw {yaw_d:.0f}  pitch {pitch_d:.0f}  ground {'y' if player.on_ground else 'n'}  fly {'y' if player.flying else 'n'}",
        ])
        y = self._section(x, y + 6, 380, "SETTINGS")
        phys = 'gpu' if (settings['gpu_physics'] and GPU_OK) else 'cpu'
        y = self._lines(x, y, [
            f"fov {settings['fov']}   fps cap {settings['fps_limit'] or 'off'}   cache {'on' if settings['cache_chunks'] else 'off'}",
            f"debris {'on' if settings['debris_on'] else 'off'} {settings['debris_amount']}%   wobble {settings['wobble']}%   strength {settings['strength']}%",
            f"physics {phys}   gpu module {'ready' if GPU_OK else 'off'}",
        ])
        y = self._section(x, y + 6, 380, "INTEGRITY")
        y = self._lines(x, y, list(self._integ))
        if self._bar_cols is None:                                     # the safe-to-break colour bar never changes
            fs = np.arange(200) / 199.0
            r = np.interp(fs, [0, .5, 1], [.12, .95, .95]) * 255
            g = np.interp(fs, [0, .5, 1], [.75, .85, .12]) * 255
            b = np.interp(fs, [0, .5, 1], [.30, .12, .08]) * 255
            self._bar_cols = [(int(r[k]), int(g[k]), int(b[k])) for k in range(200)]
        for i, col in enumerate(self._bar_cols):
            pygame.draw.line(s, col, (x + i, y + 2), (x + i, y + 10))
        self._t(x + 208, y, "safe to breaks", UI_DIM)




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
        v = world.density(*(math.floor(c + 0.5) for c in p))
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


class WeaponFX(Overlay):
    """Full-window overlay for shots, the laser beam, shells and blast rings. Every primitive is in world space and
    is projected with the frame's view-projection matrix, so the effects stay put while the camera moves."""
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, 0, 0)

    def _render(self, vp, prims):
        s = self.surf
        s.fill((0, 0, 0, 0))
        for pr in prims:
            kind = pr[0]
            if kind == 'line':
                pa, pb = _proj(vp, pr[1]), _proj(vp, pr[2])
                if pa and pb:
                    pygame.draw.line(s, pr[3], pa[:2], pb[:2], pr[4])
            elif kind == 'dot':
                q = _proj(vp, pr[1])
                if q:
                    pygame.draw.circle(s, pr[2], (int(q[0]), int(q[1])), pr[3])
            elif kind == 'sphere':                       # a ball of real world radius: its size on screen shrinks with distance
                q = _proj(vp, pr[1])
                if q:
                    px = pr[3] * (HEIGHT / 2) / (q[2] * math.tan(math.radians(pr[4]) / 2))
                    if px >= 1:
                        c = (int(q[0]), int(q[1]))
                        pygame.draw.circle(s, pr[2], c, int(px))
                        pygame.draw.circle(s, (118, 122, 136, 255), (c[0] - int(px * 0.3), c[1] - int(px * 0.3)), max(1, int(px * 0.35)))   # a highlight so it reads as a ball
                        pygame.draw.circle(s, (20, 20, 26, 255), c, int(px), max(1, int(px * 0.08)))
            elif kind == 'ring':
                q = _proj(vp, pr[1])
                if q:
                    px = pr[2] * (HEIGHT / 2) / (q[2] * math.tan(math.radians(pr[4]) / 2))
                    if px >= 1:
                        pygame.draw.circle(s, pr[3], (int(q[0]), int(q[1])), int(px), 3)


class Arsenal:
    """The weapons (just the axe now). Fire is the left mouse button or RT, and the weapon used is the one equipped in the weapons
    hand. Each shot works on the first solid thing along the view: loose pieces and structure cells through STRUCTS,
    and terrain through World.dig. Dug dirt becomes dirt cubes, as it always has."""
    NAMES = ['Axe', 'Melting Laser']     # switched with T
    TEMP_C = 1_000_000                    # the laser's temperature, shown on its meter
    LASER_REACH = 60.0
    LASER_TICK = 0.08                     # seconds between melt ticks (each tick remeshes the terrain it cuts)
    HEAT_MAX = 3.0                        # seconds of dwelling on one spot for the melt pool to reach full size
    AXE_SIZE_RANGE = (0.4, 1.0)   # each chop is scaled by a random amount in this range (1.0 = the original fixed size, now the largest)
    DENT_RADIUS = 0.45     # world radius of the dent an axe swing presses into steel (scaled by the swing's random size)
    DEBRIS_PER_HIT = 120   # most dirt cubes one hit may spawn: a big crater would otherwise add hundreds of bodies at once

    def __init__(self):
        self.fx = WeaponFX()
        self.reset()

    def reset(self):
        self.cool = 0.0
        self.prev_fire = False
        self.last_weapon = None
        self.held = None           # the chunk the staff is holding
        self.hold_dist = 3.0
        self.beam = None           # (muzzle, end) while the laser fires this frame
        self.beam_tick = 0.0
        self.heat = 0.0            # seconds the beam has dwelt on one spot: the longer, the wider the melt pool
        self.last_spot = None
        self.rings = []            # [centre, radius, time left, first time left, colour]
        self.tracers = []          # [start, end, time left, colour]
        self.drawn = False
        self.kick = 0.0           # 1 right after a shot, fades to 0 (recoil of the held model)

    def release(self):
        self.held = None
        self.beam = None
        self.last_spot = None
        self.heat = 0.0
        SFX.loop_set('laser', False)

    def update(self, world, player, dt, fire_now, weapon):
        name = self.NAMES[weapon]
        eye = np.asarray(player.eye, np.float64)
        fwd = np.asarray(player.forward(), np.float64)
        fwd = fwd / np.linalg.norm(fwd)
        view_pt = _held_pose(name, self.kick) @ np.array([*MUZZLE_LOCAL.get(name, (0.0, 0.0, -0.4)), 1.0])
        muzzle = _world_from_view_point(eye, fwd, view_pt[:3])
        edge = bool(fire_now and not self.prev_fire)
        self.prev_fire = bool(fire_now)
        self.cool = max(0.0, self.cool - dt)
        if name != self.last_weapon:
            self.release()
            self.last_weapon = name
        self._advance(world, dt)
        self.beam = None
        self.kick = max(0.0, self.kick - dt * 7.0)
        ready = edge and self.cool <= 0.0
        if ready:
            self.kick = 1.0
        if name == 'Axe':
            if ready:
                self.cool = 0.45
                SFX.play('axe_swing', gap=0.0)
                hit = _first_hit(world, eye, fwd, 3.0)
                if hit is not None:
                    SFX.axe_hit(None if hit[2] is None else _mat_at(hit[2], hit[1], fwd), hit[1])
                    # Every chop is a different size: one random scale drives how wide the cut is on structures and
                    # terrain, with a little extra noise on its force and on how deep the dirt crater goes.
                    k = float(_rng.uniform(*self.AXE_SIZE_RANGE))
                    if isinstance(hit[2], Structure) and _mat_at(hit[2], hit[1], fwd) == STEEL:
                        # steel does not chip: it is pressed aside into a shallow crater with a low rim (never chopped out)
                        hit[2].dent(hit[1], fwd, self.DENT_RADIUS * k)
                        self.rings.append([np.asarray(hit[1], np.float64).copy(), 0.5 * k, 0.3, 0.3, (200, 205, 215)])
                    else:
                        self._impact(world, hit[1], 0.7 * k, 1.6 * k * float(_rng.uniform(0.8, 1.2)),
                                     0.9 * k * float(_rng.uniform(0.85, 1.15)), 1.2 * float(_rng.uniform(0.75, 1.25)))
        elif name == 'Melting Laser':
            self._laser(world, eye, fwd, muzzle, fire_now, dt)
        SFX.loop_set('laser', self.beam is not None)

    def _advance(self, world, dt):
        """Age the impact rings and tracers."""
        for r in self.rings:
            r[2] -= dt
        self.rings = [r for r in self.rings if r[2] > 0.0]
        for tr in self.tracers:
            tr[2] -= dt
        self.tracers = [tr for tr in self.tracers if tr[2] > 0.0]

    def _impact(self, world, p, radius, power, dig_r, dig_s, ring=True, melt=False):
        """A hit at a point: structures and loose pieces are blasted, terrain is carved, dug dirt becomes cubes."""
        p = np.asarray(p, np.float64)
        broken = STRUCTS.blast(world, p, radius, power, shatter=melt, throw=not melt)   # melt: every cell in range is vaporised
        world.dig(p, dig_r, dig_s)
        groups = world.take_dug_cells()
        if groups and not melt:                      # melted ground is vaporised, not turned into cubes
            cells = np.concatenate(groups)
            level = CONFIG.get('dirt_per_block', 1)
            # Keep roughly DEBRIS_PER_HIT cubes per hit: more cubes per block means fewer dug blocks.
            max_blocks = max(1, self.DEBRIS_PER_HIT // (level ** 3))
            if len(cells) > max_blocks:
                cells = cells[_rng.choice(len(cells), max_blocks, replace=False)]
            BODIES.spawn(world, [cells], level)
        if ring:
            self.rings.append([p.copy(), radius, 0.3, 0.3, (255, 200, 120)])

    def _laser(self, world, eye, fwd, muzzle, fire_now, dt):
        """A beam while the trigger is held. It vaporises everything it touches, boring a tunnel; staying on one spot
        builds up heat, and the melt pool widens."""
        if not fire_now:
            self.beam_tick = 0.0
            self.last_spot = None
            self.heat = max(0.0, self.heat - 2.0 * dt)
            return
        hit = _first_hit(world, eye, fwd, self.LASER_REACH)
        end = np.asarray(hit[1] if hit is not None else eye + fwd * self.LASER_REACH, np.float64)
        self.beam = (muzzle.copy(), end.copy())
        if hit is None:
            self.last_spot = None
            return
        if self.last_spot is not None and float(np.linalg.norm(end - self.last_spot)) < 0.6:
            self.heat = min(self.HEAT_MAX, self.heat + dt)
        else:
            self.heat = 0.0
        self.last_spot = end.copy()
        self.beam_tick -= dt
        if self.beam_tick <= 0.0:
            self.beam_tick = self.LASER_TICK
            k = self.heat / self.HEAT_MAX
            r = 0.30 + 0.45 * k
            SFX.play('lava_pop', 0.4, end, gap=0.12)
            self._impact(world, end, r, 2.0, 0.55 + 0.5 * k, 3.0, ring=False, melt=True)
            self.rings.append([end.copy(), r * 1.6, 0.12, 0.12, (170, 205, 255)])

    def draw(self, vp, fov):
        prims = []
        for a, b, ttl, col in self.tracers:
            prims.append(('line', a, b, (col[0], col[1], col[2], int(255 * ttl / 0.12)), 3))
        if self.beam is not None:
            a, b = self.beam                                     # white-hot core, blue glow
            prims.append(('line', a, b, (110, 160, 255, 60), 20))
            prims.append(('line', a, b, (190, 220, 255, 200), 8))
            prims.append(('line', a, b, (255, 255, 255, 255), 3))
            prims.append(('dot', b, (255, 255, 255, 255), 9))
        for p, r, ttl, ttl0, col in self.rings:
            k = ttl / ttl0
            prims.append(('ring', p, r * (1.0 - 0.6 * k), (col[0], col[1], col[2], int(255 * k)), fov))
        if not prims and not self.drawn:
            return
        self.drawn = bool(prims)
        self.fx.draw(vp, prims)


# ---------------------------------------------------------------- held weapon models
# Low-poly models built in code, like the menu icons. Local space: x right, y up, the weapon points down -Z,
# and its grip sits near the origin. Each part is a convex group of triangles with flat (faceted) shading.

_M_STEEL = (0.80, 0.82, 0.88)
_M_STEEL_D = (0.46, 0.49, 0.56)
_M_DARK = (0.20, 0.21, 0.25)
_M_WOOD = (0.56, 0.37, 0.20)
_M_WOOD_D = (0.36, 0.23, 0.13)
_M_PURPLE = (0.66, 0.40, 0.96)
_M_RED = (1.00, 0.28, 0.14)
_M_CYAN = (0.45, 0.92, 1.00)


def _m_box(c, s):
    c, h = np.asarray(c, np.float64), np.asarray(s, np.float64) / 2.0
    P = [c + h * np.array([1 if (i >> 2) & 1 else -1, 1 if (i >> 1) & 1 else -1, 1 if i & 1 else -1], np.float64)
         for i in range(8)]
    quads = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    tris = []
    for q in quads:
        tris.append((P[q[0]], P[q[1]], P[q[2]]))
        tris.append((P[q[0]], P[q[2]], P[q[3]]))
    return tris


def _m_cyl(base, axis, r, n=8):
    base, axis = np.asarray(base, np.float64), np.asarray(axis, np.float64)
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.linspace(0.0, 2 * math.pi, n + 1)
    ring = [r * (math.cos(t) * u + math.sin(t) * v) for t in th]
    b = [base + p for p in ring]
    t = [base + axis + p for p in ring]
    c0, c1 = base, base + axis
    tris = []
    for i in range(n):
        tris.append((b[i], b[i + 1], t[i + 1]))
        tris.append((b[i], t[i + 1], t[i]))
        tris.append((c0, b[i + 1], b[i]))
        tris.append((c1, t[i], t[i + 1]))
    return tris


def _m_octa(c, rad):
    c, rx, ry, rz = np.asarray(c, np.float64), rad[0], rad[1], rad[2]
    top, bot = c + np.array([0.0, ry, 0.0]), c - np.array([0.0, ry, 0.0])
    E = [c + np.array([rx, 0, 0]), c + np.array([0, 0, rz]), c - np.array([rx, 0, 0]), c - np.array([0, 0, rz])]
    tris = []
    for i in range(4):
        tris.append((top, E[i], E[(i + 1) % 4]))
        tris.append((bot, E[(i + 1) % 4], E[i]))
    return tris


def _m_extrude(poly, z0, z1):
    """A convex polygon in the XY plane, pushed from z0 to z1."""
    n = len(poly)
    f = [np.array([x, y, z1], np.float64) for x, y in poly]
    b = [np.array([x, y, z0], np.float64) for x, y in poly]
    tris = []
    for i in range(1, n - 1):
        tris.append((f[0], f[i], f[i + 1]))
        tris.append((b[0], b[i + 1], b[i]))
    for i in range(n):
        j = (i + 1) % n
        tris.append((b[i], b[j], f[j]))
        tris.append((b[i], f[j], f[i]))
    return tris


def _model_axe():
    return [
        (_m_cyl([0, -0.42, 0], [0, 0.92, 0], 0.035), _M_WOOD),
        (_m_box([0, 0.30, 0], [0.09, 0.09, 0.09]), _M_STEEL_D),
        (_m_extrude([(-0.04, 0.28), (0.20, 0.34), (0.30, 0.66), (-0.04, 0.62)], -0.04, 0.04), _M_STEEL),
    ]


def _model_laser():
    return [
        (_m_box([0, 0.0, -0.05], [0.13, 0.13, 0.42]), _M_STEEL_D),
        (_m_box([0, 0.075, -0.08], [0.09, 0.03, 0.30]), _M_STEEL),
        (_m_box([0, -0.15, 0.10], [0.09, 0.20, 0.08]), _M_DARK),
        (_m_cyl([0, 0, -0.26], [0, 0, -0.14], 0.055, 8), _M_STEEL),
        (_m_cyl([0, 0, -0.40], [0, 0, -0.03], 0.045, 8), (0.85, 0.93, 1.0), 1.0),   # white-hot emitter
        (_m_octa([0.07, 0.02, -0.05], (0.02, 0.04, 0.04)), _M_CYAN, 1.0),
    ]


MODEL_BUILDERS = {
    'Axe': _model_axe, 'Melting Laser': _model_laser,
}


def _mesh_array(parts):
    """Flatten parts into rows of 10 floats: position, flat normal, colour, emissive amount."""
    rows = []
    for part in parts:
        tris, col = part[0], part[1]
        emit = part[2] if len(part) > 2 else 0.0
        arr = np.asarray(tris, np.float64)                  # (T, 3, 3)
        centre = arr.reshape(-1, 3).mean(axis=0)            # every part is convex, so its centre is inside
        for a, b, c in arr:
            n = np.cross(b - a, c - a)
            ln = float(np.linalg.norm(n))
            if ln < 1e-12:
                continue
            n = n / ln
            if np.dot(n, (a + b + c) / 3.0 - centre) < 0.0:
                n = -n
            for p in (a, b, c):
                rows.append([p[0], p[1], p[2], n[0], n[1], n[2], col[0], col[1], col[2], emit])
    return np.asarray(rows, np.float32)


def _T(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def _Rz(a):
    m = np.eye(4, dtype=np.float32)
    c, s = math.cos(a), math.sin(a)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


MUZZLE_LOCAL = {'Axe': (0.25, 0.5, 0.0), 'Melting Laser': (0.0, 0.0, -0.43)}


def _world_from_view_point(eye, fwd, pt):
    """A point in camera (view) space -> world space, using the same camera axes as the renderer."""
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(right, fwd)
    return eye + right * pt[0] + up * pt[1] - fwd * pt[2]


def _S(k):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = k
    return m


def _held_pose(name, kick):
    """Where the weapon sits in front of the camera. kick is 1 right after a shot and fades to 0."""
    lean = {'Axe': -0.5}.get(name, 0.0)
    scale = {'Axe': 0.85}.get(name, 0.9)
    return _T(0.34, -0.30 - 0.03 * kick, -0.62 + 0.07 * kick) @ _Rz(lean) @ _S(scale)


_WEAPON_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
layout(location = 2) in vec3 aCol;
layout(location = 3) in float aEmit;
uniform mat4 uProj;
uniform mat4 uModel;
out vec3 vN;
out vec3 vC;
out float vE;
void main() {
    gl_Position = uProj * (uModel * vec4(aPos, 1.0));
    vN = mat3(uModel) * aNrm;
    vC = aCol;
    vE = aEmit;
}
"""

_WEAPON_FRAG = """
#version 330 core
in vec3 vN;
in vec3 vC;
in float vE;
uniform vec3 uLight;
out vec4 fragColor;
void main() {
    float d = max(dot(normalize(vN), normalize(uLight)), 0.0);
    vec3 lit = vC * (0.35 + 0.65 * d);
    fragColor = vec4(mix(lit, vC * 1.4 + 0.1, vE), 1.0);
}
"""


class WeaponModels:
    """Draws the weapon in hand over the world, with its own depth range so it never clips into walls."""
    STRIDE = 40          # 10 floats per vertex

    def __init__(self):
        vs = _compile_shader(_WEAPON_VERT, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_WEAPON_FRAG, gl.GL_FRAGMENT_SHADER)
        self.program = _link_program(vs, fs)
        self.u_proj = gl.glGetUniformLocation(self.program, "uProj")
        self.u_model = gl.glGetUniformLocation(self.program, "uModel")
        self.u_light = gl.glGetUniformLocation(self.program, "uLight")
        self.meshes = {}
        for name, builder in MODEL_BUILDERS.items():
            data = _mesh_array(builder())
            vao = gl.glGenVertexArrays(1)
            vbo = gl.glGenBuffers(1)
            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
            for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 1, 36)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, self.STRIDE, ctypes.c_void_p(off))
            gl.glBindVertexArray(0)
            self.meshes[name] = (vao, len(data))

    def draw(self, name, kick, fov):
        if name not in self.meshes:
            return
        vao, n = self.meshes[name]
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self.u_proj, 1, gl.GL_TRUE, _perspective(fov, WIDTH / HEIGHT, 0.02, 30.0))
        gl.glUniformMatrix4fv(self.u_model, 1, gl.GL_TRUE, _held_pose(name, kick))
        gl.glUniform3f(self.u_light, 0.35, 0.85, 0.45)
        gl.glBindVertexArray(vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, n)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)


def run():
    # Open the window centred on the primary monitor instead of wherever the OS puts it.
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.mixer.pre_init(44100, -16, 2, 512)         # small buffer: sounds start without lag
    pygame.init()
    _fit_window_to_display()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
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
    menu = TabMenu(settings)
    settings_panel = SettingsPanel()
    loader.step("Starting GPU monitor and controller", 0.16)
    poller = GpuPoller()
    pad = Gamepad()
    print(f"Controller: {pad.name}" if pad.connected else "Controller: none found (plug in or pair an Xbox controller)")
    loader.step("Compiling shaders: terrain, and starting mesh workers", 0.22, "%d background processes build chunk meshes" % MESH_WORKERS)
    world = World()
    renderer = Renderer()
    player = PlayerController((0.5, float(terrain_height(0.0, 0.0)) + 2.0, 0.5))
    last_centre = None
    last_caption = None
    last_cpu_ms = None
    paused = False
    show_grid = False            # X: cube grid lines on the cubes
    last_space_tap = -10.0       # time of the previous Space press, for double-tap detection
    auto_walk = False            # on while benchmarking: walk forward without holding W
    bench = BenchmarkSession()
    bench_results = None         # summary shown when a benchmark ends; None = no results screen
    results_panel = BenchmarkResults()
    loader.step("Loading weapons: axe model", 0.30)
    arsenal = Arsenal()
    held_weapon = 0              # index into arsenal.NAMES: the weapon in hand
    weapon_gl = WeaponModels()
    load_ms = draw_ms = 0.0      # per-frame timings for the benchmark breakdown
    loader.step("Compiling shaders: sky, chunk grid, debris", 0.40)
    backdrop = Backdrop()
    grid = ChunkGrid()
    body_gl = DirtBodyRenderer()
    loader.step("Compiling shaders: structures and shadow map", 0.55)
    struct_gl = StructureRenderer()
    shadow = ShadowMap()
    renderer.shadow_tex = shadow.tex
    struct_gl.shadow_tex = shadow.tex
    loader.step("Building HUD: crosshair, monitor", 0.66)
    crosshair = Crosshair()
    laser_meter = LaserMeter()
    monitor = MonitorOverlay()
    monitor.src = dict(world=world, renderer=renderer, player=player, settings=settings,
                       bench=bench, poller=poller, paused=False, auto_walk=False)
    built_res = settings['struct_res']   # cube detail the layout was last built at
    cube_shape = 0               # the only shape is the giant cube
    last_strength = 100
    cube_mat = 1                 # which material the test piece is made of (Q cycles it); starts as stone
    pending_layout = True        # the shapes beside the dirt cube are built once the chunks under them are loaded
    menu_frozen = False          # True once the world has been frozen behind the open menu
    counted = False              # True when the last pass was a real game frame (not the menu)
    clock = pygame.time.Clock()
    # What was on screen last frame, so a menu opening, a pause or a toggle can make its sound once.
    sfx_seen = dict(menu=False, paused=False, grid=False,
                    hud=settings['show_hud'], stats=monitor.show_stats, bench=False, results=False)

    # Generate the terrain around the player now, nearest chunks first, so the game opens on a finished world.
    rd0 = settings['render_distance']
    pcx, pcz = math.floor(player.pos[0] / CHUNK_SIZE), math.floor(player.pos[2] / CHUNK_SIZE)
    todo = sorted(((cx, cz) for cx in range(pcx - rd0, pcx + rd0 + 1) for cz in range(pcz - rd0, pcz + rd0 + 1)),
                  key=lambda c: (c[0] - pcx) ** 2 + (c[1] - pcz) ** 2)
    for i, (cx, cz) in enumerate(todo):
        if i % 12 == 0:
            loader.step("Generating terrain", 0.70 + 0.28 * i / len(todo), "chunk %d of %d" % (i, len(todo)))
        world.ensure_chunk(cx, cz)
    loader.step("Starting the game", 1.0)
    SFX.play('level_up', 0.8, gap=0.0)                              # loading finished
    gl.glClearColor(SKY_RGB[0], SKY_RGB[1], SKY_RGB[2], 1.0)        # the loading screen changed the clear colour: put the sky back

    def _set_menu(on):
        menu.open = on
        menu.dirty = True
        pygame.mouse.set_visible(on)
        pygame.event.set_grab(not on)

    running = True
    while running:
        # With the menu open the world is frozen, so idle at a low rate instead of the FPS limit.
        dt = clock.tick(30 if (menu.open or bench_results is not None) else settings['fps_limit']) / 1000.0    # 0 = uncapped
        work_start = time.perf_counter()                     # everything after the cap's sleep
        if counted:
            PROF.push(dt * 1000.0, last_cpu_ms or 0.0)
        else:
            PROF.cur = {}
        counted = False
        t_in = time.perf_counter()
        rd = settings['render_distance']
        if bench.active:
            bench.sample(dt * 1000.0, last_cpu_ms, GT.total(), poller.stats,
                         (load_ms, renderer.mesh_s * 1000.0, renderer.upload_s * 1000.0, draw_ms,
                          renderer.list_s * 1000.0, renderer.call_s * 1000.0, renderer.collect_s * 1000.0))

        SFX.tick()
        SFX.set_listener(player.eye, player.yaw)
        events = pygame.event.get()
        bench_toggle = bench_stop = False
        pending_reset = False
        shape_step = 0
        cube_step = 0                # +1 / -1 when the cube material should change this frame
        for e in events:
            if e.type in PAD_EVENTS:
                pad.rescan()                 # a controller was plugged in or removed
                continue
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
                SFX.play('ui_click')
                if act == 'exit':
                    running = False
                elif act == 'reset_damage':
                    pending_reset = True
            elif menu.open and e.type == MOUSEBUTTONUP and e.button == 1:
                menu.drag = None
            elif menu.open and e.type == MOUSEMOTION and menu.drag is not None:
                settings_panel.mouse_drag(menu, e.pos)
                SFX.play('ui_click', 0.4, gap=0.07)                 # a slider being dragged ticks
            elif e.type == KEYDOWN and e.key == K_t and not bench.active and not menu.open:
                held_weapon = (held_weapon + 1) % len(arsenal.NAMES)     # T: axe <-> melting laser
                SFX.play('armor_equip', gap=0.0)
            elif e.type == KEYDOWN and e.key == K_x and not bench.active:
                show_grid = not show_grid
            elif e.type == KEYDOWN and e.key == K_g and not bench.active:
                settings['show_hud'] = not settings['show_hud']          # G: the whole monitor overlay on / off
            elif menu.open:
                if e.type == KEYDOWN and e.key in (K_UP, K_DOWN, K_LEFT, K_RIGHT, K_LEFTBRACKET, K_RIGHTBRACKET, K_RETURN, K_KP_ENTER):
                    SFX.play('ui_click')
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
            elif e.type == KEYDOWN and e.key == K_SPACE and not bench.active:
                now = time.perf_counter()
                if now - last_space_tap < 0.3:          # double tap: toggle flying
                    player.toggle_fly()
                    last_space_tap = -10.0             # so a third tap starts a fresh pair
                else:
                    last_space_tap = now
            elif e.type == KEYDOWN and e.key == K_q and not bench.active:
                cube_step += 1                                       # Q: next cube material
            elif e.type == KEYDOWN and e.key == K_r and not bench.active:
                pending_reset = True                                 # R: put the whole map back and clear all damage
                last_centre = None
            elif e.type == KEYDOWN and e.key == K_f and not bench.active:
                paused = not paused
            elif e.type == KEYDOWN and e.key == K_v and not bench.active:
                if settings['show_hud']:
                    monitor.show_stats = not monitor.show_stats      # V with the monitor on: stats columns on / off
                else:
                    player.toggle_fly()                              # V otherwise: fly
            elif e.type == KEYDOWN and e.key == K_F5:
                bench_toggle = True         # F5 starts a benchmark (autowalk on); F5 again ends it
            elif e.type == MOUSEMOTION:                              # look around works while paused too
                player.yaw += e.rel[0] * 0.002
                player.pitch = max(-1.5, min(1.5, player.pitch - e.rel[1] * 0.002))

        pad.poll()
        if menu.open and any(pad.pressed(b_) for b_ in ('UP', 'DOWN', 'LEFT', 'RIGHT', 'A')):
            SFX.play('ui_click')
        if bench_results is not None:
            if pad.pressed('A') or pad.pressed('B') or pad.pressed('START'):
                bench_results = None
        elif menu.open:
            # Controls tab: Back + Start together close it, so every button can be tried on the live view.
            if SETTINGS_TABS[menu.tab][0] == 'CONTROLLER':
                if pad.held('START') and pad.held('BACK') and (pad.pressed('START') or pad.pressed('BACK')):
                    _set_menu(False)
            elif pad.pressed('START'):
                _set_menu(False)
        elif not bench.active:
            if pad.pressed('START'):
                _set_menu(True)
            if pad.pressed('Y'):
                player.toggle_fly()
            if pad.pressed('X'):
                settings['show_hud'] = not settings['show_hud']          # X: same switch as G
            if pad.pressed('R3'):
                paused = not paused
            if pad.pressed('UP'):
                pending_reset = True
                last_centre = None
            if pad.pressed('RIGHT'):
                cube_step += 1
            if pad.pressed('LEFT'):
                cube_step -= 1
            if pad.pressed('BACK'):
                bench_toggle = True
        elif pad.pressed('BACK'):
            bench_toggle = True                                      # Back also ends a running benchmark
        PROF.add('input', (time.perf_counter() - t_in) * 1000.0)

        if cube_step or shape_step:                                  # swap the piece: clear the old one and build the new material / shape
            cube_mat = (cube_mat + cube_step) % len(DEMO_LAYOUT)
            SFX.play('item_pickup')
            cube_shape = (cube_shape + shape_step) % len(SHAPE_NAMES)
            world.reset_damage()                                     # the land comes back whole too, like pressing R
            STRUCTS.clear()
            BODIES.clear()
            world.take_dug_cells()
            pending_layout = True
            last_centre = None
        settings['struct_res'] = settings['dirt_cubes']              # the cube detail sets how fine the structures are cut
        if settings['struct_res'] != built_res:                      # cube detail changed: rebuild the cubes at the new detail
            built_res = settings['struct_res']
            STRUCTS.clear()
            BODIES.clear()
            pending_layout = True
        if pending_reset or menu.take_action() == 'reset_damage':
            SFX.play('totem', 0.9, gap=0.0)
            world.reset_damage()
            arsenal.reset()
            STRUCTS.clear()
            pending_layout = True                                    # the whole map comes back whole
            BODIES.clear()
            world.take_dug_cells()

        if bench_toggle or bench_stop:
            if bench.active:
                bench_results = bench.stop(settings, player)
                auto_walk = False
            else:
                bench.start(settings, player)
                auto_walk = True

        # Sounds for things that just changed: windows opening, pausing, toggles, the benchmark.
        if menu.open != sfx_seen['menu']:
            SFX.play('door_open' if menu.open else 'door_close', gap=0.0)
        if paused != sfx_seen['paused']:
            SFX.play('bucket_fill' if paused else 'door_close', gap=0.0)
        if show_grid != sfx_seen['grid'] or settings['show_hud'] != sfx_seen['hud'] or monitor.show_stats != sfx_seen['stats']:
            SFX.play('ui_click')
        if bench.active != sfx_seen['bench']:
            SFX.play('bell' if bench.active else 'level_up', 0.6 if bench.active else 0.9, gap=0.0)
        if (bench_results is not None) != sfx_seen['results'] and bench_results is not None:
            SFX.play('bell', 0.9, gap=0.0, delay=0.7)                  # the results are up
        sfx_seen.update(menu=menu.open, paused=paused,
                        grid=show_grid, hud=settings['show_hud'], stats=monitor.show_stats, bench=bench.active,
                        results=bench_results is not None)

        modal = menu.open or bench_results is not None   # menu or results screen is up: world frozen

        # Menu open and nothing changed in it: keep the frozen frame on screen and skip the work.
        menu_changed = pad.changed or any(e.type != MOUSEMOTION or menu.drag is not None for e in events)
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
        GFX['physics_device'] = 'gpu' if (settings['gpu_physics'] and GPU_OK) else 'cpu'
        CONFIG['debris_on'] = settings['debris_on']
        CONFIG['debris_amount'] = settings['debris_amount'] / 100.0
        CONFIG['dirt_per_block'] = settings['dirt_cubes']
        if not settings['debris_on']:
            BODIES.clear()
        GT.enabled = bool(settings['show_hud'])                      # GPU timer queries only while the monitor is shown
        GT.poll()
        keys = pygame.key.get_pressed()
        t_load = time.perf_counter()
        if not modal:
            with Timed('chunks', gpu=False):
                load_around(world, player.pos[0], player.pos[2], radius=rd)
                if pending_layout and STRUCTS.layout_ready(world):
                    STRUCTS.build_layout(world, settings['struct_res'], cube_mat, cube_shape)
                    pending_layout = False
        load_ms = (time.perf_counter() - t_load) * 1000.0

        if not modal:
            yaw = player.yaw
            fwd = np.array([math.sin(yaw), 0.0, -math.cos(yaw)])
            right = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
            wish = np.zeros(3)
            if keys[K_w] or auto_walk: wish += fwd
            if keys[K_s]: wish -= fwd
            if keys[K_d]: wish += right
            if keys[K_a]: wish -= right
            mx, my = pad.move()
            pad_move = mx != 0.0 or my != 0.0
            if pad_move:
                wish += fwd * (-my) + right * mx                     # left stick: up is forward
            if pad.connected:
                lx_, ly_ = pad.look()
                player.yaw += lx_ * PAD_LOOK_SPEED * min(dt, 0.05)
                player.pitch = max(-1.5, min(1.5, player.pitch - ly_ * PAD_LOOK_SPEED * min(dt, 0.05)))
            if player.flying:
                if keys[K_SPACE]: wish[1] += 1.0
                if keys[K_LCTRL]: wish[1] -= 1.0
                if pad.held('A'): wish[1] += 1.0
                if pad.held('B'): wish[1] -= 1.0
            if bench.active and player.pos[1] < bench.start_y + BENCH_CLIMB:
                wish = np.array([0.0, 1.0, 0.0])     # climb straight up first
            n = np.linalg.norm(wish)
            if n > 1.0 or (n > 0 and not pad_move):
                wish /= n                                            # keys always walk at full speed; a stick keeps its analog push
            with Timed('player', gpu=False):
                player.update(world, dt, wish, keys[K_SPACE] or pad.held('A'), keys[K_LSHIFT] or pad.held('L3'))
            if not paused:                                           # the simulation freezes; the player keeps moving
                with Timed('structs', gpu=False):
                    STRUCTS.update(world, dt)
                with Timed('debris', gpu=False):
                    BODIES.update(world, dt, settings['cube_sleep_time'])
            with Timed('weapon', gpu=False):
                arsenal.update(world, player, dt, fire_now=bool(pygame.mouse.get_pressed()[0] or pad.held('RT')),
                               weapon=held_weapon)
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
        state = ("BENCHMARK   " if bench.active else "") + ("PAUSED   " if paused else "") + ("AUTO-WALK   " if auto_walk else "")
        mat_txt = DEMO_NAMES[cube_mat] + ' '
        caption = f"VoidCraft   {mat_txt}{SHAPE_NAMES[cube_shape]}   {state}{FPS.update(dt):.0f} fps"
        if caption != last_caption:
            pygame.display.set_caption(caption)
            last_caption = caption

        # Sun shadows: the shadow map is redrawn from the sun each frame, centred on the player.
        lvp = light_matrix(player.eye)
        renderer.light_vp = lvp
        struct_gl.light_vp = lvp
        with Timed('shadow'):
            shadow.render(renderer, struct_gl, lvp)

        if modal:
            arsenal.release()
            if not menu_frozen:
                # First frame with the menu open: draw the world once and keep it as the backdrop.
                renderer.draw(world, player.eye, player.yaw, player.pitch,
                              radius=rd - 1, fov=settings['fov'])
                if settings['show_hud']:
                    grid.draw(renderer.last_vp, centre[0], centre[1], rd - 1)
                    monitor.src.update(paused=paused, auto_walk=auto_walk)
                    monitor.draw(_monitor_tick())
                backdrop.capture()
                menu_frozen = True
            else:
                backdrop.draw()
            if bench_results is not None:
                results_panel.draw(bench_results)
            else:
                settings_panel.draw(menu, pad)
        else:
            menu_frozen = False
            t_draw = time.perf_counter()
            with Timed('terrain'):
                renderer.draw(world, player.eye, player.yaw, player.pitch,
                              radius=rd - 1, fov=settings['fov'])
            draw_ms = (time.perf_counter() - t_draw) * 1000.0
            # renderer.draw times its own parts; they sit inside 'terrain' and are listed under it.
            PROF.add('terrain.mesh', renderer.mesh_s * 1000.0)
            PROF.add('terrain.upload', renderer.upload_s * 1000.0)
            PROF.add('terrain.list', renderer.list_s * 1000.0)
            PROF.add('terrain.collect', renderer.collect_s * 1000.0)
            PROF.add('terrain.call', renderer.call_s * 1000.0)
            with Timed('struct_draw'):
                struct_gl.render(renderer.last_vp, 1.0, overlay=settings['show_hud'], grid=show_grid)
            with Timed('debris_draw'):
                body_gl.render(renderer.last_vp, 1.0, player.yaw, player.pitch)
            with Timed('overlay'):
                if settings['show_hud']:
                    grid.draw(renderer.last_vp, centre[0], centre[1], rd - 1)
                crosshair.draw(0)
                if arsenal.NAMES[held_weapon] == 'Melting Laser':
                    laser_meter.draw(int(round(arsenal.heat / arsenal.HEAT_MAX * LaserMeter.SEGS)))
                arsenal.draw(renderer.last_vp, settings['fov'])
                weapon_gl.draw(arsenal.NAMES[held_weapon], arsenal.kick, settings['fov'])
            if settings['show_hud']:
                with Timed('monitor'):
                    monitor.src.update(paused=paused, auto_walk=auto_walk)
                    monitor.draw(_monitor_tick())
        # Measure before the flip, since flip can wait on vsync and that isn't game work.
        last_cpu_ms = (time.perf_counter() - work_start) * 1000.0
        t_flip = time.perf_counter()
        pygame.display.flip()
        PROF.add(PROF_FLIP, (time.perf_counter() - t_flip) * 1000.0)
        counted = not modal

    poller.stop()
    renderer.mesher.pool.shutdown(wait=False, cancel_futures=True)
    pygame.quit()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
