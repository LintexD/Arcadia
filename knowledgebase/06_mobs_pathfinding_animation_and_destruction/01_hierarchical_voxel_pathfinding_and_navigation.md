# 01. Real-Time Voxel Navigation & Dynamic 3D Pathfinding

## 1. The Challenge: Navigation in a Fully Destructible $6.25\text{ cm}$ World

Traditional 3D game engines rely on **Navigation Meshes (NavMeshes)** generated via polygonization libraries such as Recast/Detour. While well-suited for static environments, NavMeshes break down catastrophically in a real-time destructible micro-voxel world like Arcadia:

1. **Re-Triangulation Latency:** Whenever a rocket or explosion carves a crater, a pillar collapses, or a player digs a trench, standard NavMesh generation requires voxel contour tracing, polygon simplification, and Delaunay triangulation. For a $16\text{ m}^3$ chunk, this takes $15\text{--}80\text{ ms}$ on the CPU—far exceeding our entire $16.6\text{ ms}$ frame budget ($60\text{ FPS}$) and causing intolerable frame stutters.
2. **Resolution Mismatch:** Arcadia’s native voxel resolution is $s = 0.0625\text{ m}$ ($6.25\text{ cm}$, $16\text{ voxels/meter}$). A uniform 3D grid pathfinding graph across a modest $128\text{ m} \times 64\text{ m} \times 128\text{ m}$ battle arena would contain:
   $$N_{\text{voxels}} = (128 \times 16) \times (64 \times 16) \times (128 \times 16) = 2048 \times 1024 \times 2048 = 4.29 \times 10^9 \text{ nodes}$$
   Running raw $A^*$ or Dijkstra on billions of voxel nodes is completely impossible in real time.
3. **Complex 3D Surface Topology:** Unlike standard 2.5D heightfields, Arcadia features full 3D overhangs, gothic cross-vaulted ceilings, hollow ruins, dynamic rubble piles, spiral staircases, and cantilevered bridges. Mob pathfinding must natively support 3D traversal with vertical step-ups, jumping over craters, vaulting ledges, and wall-crawling.

---

## 2. Hierarchical Voxel Navigation Graph (HVG)

To achieve nanosecond path queries with instant $O(1)$ dynamic destruction updates, Arcadia uses a **Three-Tier Hierarchical Voxel Navigation Graph (HVG)** aligned directly with our Chunked Sparse 64-Tree (CS64) memory hierarchy:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Tier 1: Macro-Graph (4.0 m cells, Level 1 CS64 Nodes)                  │
│ Long-range topological pathing (A* across chunks & macro-rooms)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Refines to
┌───────────────────────────────────▼────────────────────────────────────┐
│ Tier 2: Meso-Surface Graph (0.25 m – 1.0 m, Level 2 / Leaf Bricks)     │
│ Surface contour connectivity, drop-offs, 3-voxel step-ups, jumps       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Guides
┌───────────────────────────────────▼────────────────────────────────────┐
│ Tier 3: Micro-Steering & Kinematic Obstacle Avoidance (6.25 cm voxels) │
│ Bitwise AABB collision, separation forces, dynamic rigid body pushes   │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Tier 1: Macro-Graph (Topological Backbone)
- **Cell Size:** $4.0\text{ m} \times 4.0\text{ m} \times 4.0\text{ m}$ ($64^3$ voxels, matching CS64 Level 1 Macro Nodes).
- **Node Representation:** Each macro-cell stores:
  - `passable_mask: u64`: 64-bit mask indicating which of its $4 \times 4 \times 4$ sub-cells ($1.0\text{ m}^3$ meso-nodes) contain traversable floor surfaces.
  - `boundary_portals: [u16; 6]`: Connectivity bitmasks across its $\pm X, \pm Y, \pm Z$ adjacent macro-cells.
- **Pathing Cost:** An $A^*$ search across a $256\text{ m} \times 64\text{ m} \times 256\text{ m}$ active simulation volume queries at most $64 \times 16 \times 64 = 65,536$ macro-nodes. With Manhattan/Euclidean heuristics, Tier 1 pathfinding executes in **$< 15\text{ }\mu\text{s}$**.

### 2.2 Tier 2: Meso-Surface Graph (Walkable & Jumpable Manifold)
Mobs do not fly through empty space; terrestrial mobs traverse the **discrete solid-air boundary surface**:
- **Cell Size:** $1.0\text{ m}^3$ (CS64 Level 2 Meso-Node) or $0.25\text{ m}^3$ (CS64 Level 3 Leaf Brick).
- **Surface Node Extraction:** A leaf brick is defined as *traversable* if it contains at least one solid voxel with an empty air voxel directly above it ($+Y$ clearance $\ge \text{mob height}$).
- **Annotated Transitions:** Directed edges between adjacent meso-cells are tagged with physical locomotion constraints:
  - `WALK`: Continuous surface with height delta $|\Delta y| \le 3\text{ voxels}$ ($18.75\text{ cm}$, matching Arcadia’s character auto-step).
  - `VAULT_UP`: Ledge rise of $4\text{--}16\text{ voxels}$ ($0.25\text{ m}\text{--}1.0\text{ m}$) requiring a jump/mantle animation.
  - `DROP_DOWN`: Fall of $4\text{--}48\text{ voxels}$ ($0.25\text{ m}\text{--}3.0\text{ m}$) safely traversable without fatal fall damage.
  - `GAP_JUMP`: Horizontal gap of $4\text{--}32\text{ voxels}$ ($0.25\text{ m}\text{--}2.0\text{ m}$) requiring forward ballistic trajectory leap.
  - `CLIMB_WALL`: Vertical wall surface for agile mobs (demonic imps, arachnids) capable of climbing mossy stone.

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NavEdge {
    pub target_cell: u32,       // Flat index into local chunk meso-grid
    pub edge_type: NavEdgeType, // Walk, Step, Vault, Drop, Jump, Climb
    pub cost_units: u8,         // Traversal cost (1 unit = 1/16 meter equivalent)
    pub headroom_voxels: u8,    // Vertical clearance directly above edge (0..255 voxels = up to 16.0m)
    pub width_voxels: u8,       // Horizontal lateral clearance from walls (0..255 voxels)
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavEdgeType {
    Walk = 0,
    StepUp = 1,
    Vault = 2,
    DropDown = 3,
    GapJump = 4,
    ClimbWall = 5,
}
```

### 2.3 Multi-Size Agent Clearance Fields (Preventing Head & Shoulder Collisions)
A common pitfall in voxel pathfinding is assuming a single universal agent size, causing tall demons ($3.5\text{m}$ Cyberdemons) to hit their heads on low gothic arches or wide beasts ($2.5\text{m}$ Arachnids) to wedge in narrow doorways.

Arcadia prevents head and shoulder clipping without duplicating graph memory through **Annotated Clearance Fields**:

```text
Low Gothic Archway Cross-Section:
 Ceiling ════════════════════════════════════════════════════
                             ┌────────────────┐
                             │ Low Stone Arch │ ◄── Headroom: 24 voxels (1.5 m)
 ────────────────────────────┴────────────────┴──────────────
   Traversable by Class S (Imp: 16 voxels) ──► FITS!
   Traversable by Class M (Zombie: 30 voxels) ──► BLOCKED! (Routes around)
   Traversable by Class L (Baron: 56 voxels) ──► BLOCKED! (Routes around)
 Floor   ════════════════════════════════════════════════════
```

1. **Nanosecond Vertical Headroom Probing:**
   - To compute `headroom_voxels` above floor position $(x, y, z)$, the engine scans upward through the CS64 tree.
   - Within the $4^3$ Leaf Brick, testing the vertical bit-column is a single bitwise shift and count:
     $$\text{headroom} = (!\text{col\_solid\_bits}).\text{trailing\_zeros}()$$
   - Scanning across Level 2 and Level 1 nodes uses child bitmask testing, determining clearance up to $16.0\text{ m}$ in **$< 4\text{ ns}$**.
2. **Horizontal Lateral Clearance:**
   - Evaluated by measuring distance to the nearest solid voxel along the lateral axis $\pm X, \pm Z$.
3. **Edge Gating Filter:**
   - When an agent executes an $A^*$ or JPS path expansion, any candidate edge $e$ is strictly filtered against the agent's bounding dimensions:
     ```rust
     if edge.headroom_voxels < mob.height_voxels || edge.width_voxels < mob.radius_voxels {
         continue; // Prune edge: mob will hit head or wedge shoulders
     }
     ```
   - **Result:** Large demons automatically path through grand porticos and wide vaulted halls, while small hounds and imps exploit low cracks, collapsed walls, and ventilation tunnels.
4. **Bucketed Horde Flow Fields (Multi-Size Swarms):**
   - For swarms, agents are grouped into **Three Clearance Classes**:
     - **Class S (Small / Agile):** $h \le 20\text{ voxels}$ ($1.25\text{ m}$), $r \le 8\text{ voxels}$ ($0.5\text{ m}$) — imps, hounds, crawlers.
     - **Class M (Medium / Humanoid):** $h \le 36\text{ voxels}$ ($2.25\text{ m}$), $r \le 12\text{ voxels}$ ($0.75\text{ m}$) — humanoids, soldiers, Hellknights.
     - **Class L (Large / Behemoth):** $h \le 72\text{ voxels}$ ($4.50\text{ m}$), $r \le 32\text{ voxels}$ ($2.00\text{ m}$) — cyberdemons, siege beasts, bosses.
   - Background worker threads calculate at most 2–3 clearance-masked flow fields ($< 1.2\text{ ms}$ total), guaranteeing that each size class follows a clearance-verified navigation stream.
5. **Tier 3 Kinematic Swept AABB Safety:**
   - Even if unexpected rubble falls directly in front of an advancing demon during run-time, the mob's kinematic controller tests its exact 3D AABB against the CS64 tree every tick before stepping forward. If the head or shoulder contacts geometry, forward momentum halts immediately, preventing clipping and triggering a dynamic re-path or wall breach attack.

---

## 3. Real-Time Dynamic Invalidation via CS64 `ChunkEditReport`

When a crater is carved by [`carve.rs`](../../crates/arcadia-core/src/chunk_tree/edit/carve.rs) or an explosion triggers structural collapse via [`islands.rs`](../../crates/arcadia-world/src/islands.rs), Arcadia’s core mutation engine emits a fine-grained [`ChunkEditReport`](../../crates/arcadia-core/src/chunk_tree/edit.rs):
- `min_dirty_node` to `max_dirty_node`
- `min_dirty_brick` to `max_dirty_brick`
- Bounded spatial AABB $[\mathbf{p}_{\text{min}}, \mathbf{p}_{\text{max}}]$

### 3.1 Localized Edge Invalidation Protocol ($O(1)$ per Brick)
Rather than regenerating the nav graph for the entire scene or even an entire chunk:
1. **Identify Intersected Meso-Cells:** Expand the dirty AABB by $1.0\text{ m}$ (to account for neighbor step-up transitions).
2. **Bitwise Surface Re-Evaluation:** For each dirty $4^3$ Leaf Brick:
   - Extract the 64-bit occupancy mask `M_solid = node.mask_u64()`.
   - Compute the air mask directly above: `M_air = !M_solid_above`.
   - The walkable surface mask is calculated via bitwise shifts:
     $$\mathbf{M}_{\text{walkable}} = \mathbf{M}_{\text{solid}} \ \& \ (\mathbf{M}_{\text{air}} \ll 4)$$
   - *(A 4-bit shift corresponds exactly to a $+Y$ offset in our `(z<<4) | (y<<2) | x` layout).*
3. **Graph Patching:** If a previously traversable bridge now has zero walkable bits, its directed edges are immediately pruned from the Tier 2 graph. If an explosion carves a new passage through a wall, new `WALK` edges are instantiated.
4. **Time Complexity:** Updates take **$< 0.2\text{ }\mu\text{s}$ per dirty brick**, allowing full dynamic graph synchronization within **$< 0.05\text{ ms}$** following a massive explosion.

```text
Carve Sphere Intersection:
   [Air]          [Air]          [Air]
     ▲              ▲              ▲
     │              │              │
[Solid Floor] ──▶ [Crater Void] ──▶ [Solid Floor]
(Valid Edge)      (BROKEN EDGE)     (Valid Edge)
                        │
                        ▼
            Edge Pruned in 0.15 µs
```

---

## 4. Path Search Algorithms for Voxel Terrains

### 4.1 3D Jump Point Search (JPS-3D) on Voxel Grids
While standard $A^*$ evaluates every adjacent voxel node, **Jump Point Search (JPS)** exploits the symmetry of grid spaces to prune up to $95\%$ of open-space expansions:
- When searching along a horizontal or diagonal direction across flat voxel terrain or open air, the algorithm "jumps" ahead across contiguous identical cells until:
  1. A terrain obstruction is reached.
  2. A "forced neighbor" is detected (e.g., a hole in the floor or a corner wall allowing a turn).
  3. The goal node is intercepted.
- **Arcadia Optimization:** JPS-3D jumps can test 16 voxels in a single CPU instruction using bitwise masking (`_pext_u64` or `count_zeros`) on `LeafBrick64` rows.

### 4.2 Dynamic Dijkstra Flow Fields (Vector Fields) for Swarms & Hordes
When 20 to 100 demonic mobs are all pursuing the player simultaneously, running 100 individual $A^*$ searches causes redundant path queries and high CPU overhead.

**Solution: 3D Surface Dijkstra Flow Field:**
1. **Goal Seeding:** Seed the player's current meso-cell with distance $D = 0$.
2. **Breadth-First Dijkstra Wave:** Propagate distance costs outwards across Tier 2 surface graph edges up to a maximum radius $R = 48\text{ m}$ (768 voxels):
   $$D(v) = \min_{u \in \text{Neighbors}(v)} (D(u) + \text{cost}(u, v))$$
3. **Gradient Vector Field:** Each cell computes a normalized 3D flow vector pointing towards the steepest downhill potential:
   $$\mathbf{v}_{\text{flow}}(\mathbf{x}) = -\nabla D(\mathbf{x}) \approx -\left( \frac{D(x+1) - D(x-1)}{2}, \frac{D(y+1) - D(y-1)}{2}, \frac{D(z+1) - D(z-1)}{2} \right)$$
4. **Mob Traversal:** Every mob within the $48\text{ m}$ radius simply samples the local flow vector $\mathbf{v}_{\text{flow}}$ at its feet and drives its kinematic controller in that direction!
5. **Cost:** One single flow field calculation ($< 0.8\text{ ms}$ on a worker thread) guides **unlimited** mobs with zero per-agent pathfinding cost.

```text
Dijkstra Flow Field Convergence:
  [Mob 1] ───► ──┐
                 ├──► [Corner] ──► ──┐
  [Mob 2] ───► ──┘                   ├──► [PLAYER]
                                     │
  [Mob 3] ───► ───────► ─────────────┘
```

---

## 5. Locomotion Mechanics: Kinematic Voxel Controller for Mobs

Mobs in Arcadia do not float or slide through geometry. They utilize a specialized version of Arcadia’s **Bitwise AABB Kinematic Controller** ([`arcadia-physics/src/character.rs`](../../crates/arcadia-physics/src/character.rs)):

### 5.1 Auto-Stepping & Slope Ascending
- When moving horizontally, if forward progress is blocked by a voxel ledge of height $h \in [1, 3]$ voxels ($6.25\text{ cm}\text{--}18.75\text{ cm}$):
  1. Elevate mob bounding box by $h \cdot s$.
  2. Test if the elevated AABB has clearance using bitwise `test_voxel_aabb`.
  3. If clear, snap mob up to the new elevation and continue horizontal momentum without velocity loss.
- Enables demons and beasts to run smoothly up fluted colonnade bases, jagged rock scree, and spiral staircases.

### 5.2 Dynamic Jump Trajectory Solver
When a mob reaches a `GAP_JUMP` or `VAULT_UP` edge:
- Compute ballistic trajectory impulse vector $\mathbf{v}_0$:
  $$\mathbf{x}(t) = \mathbf{x}_0 + \mathbf{v}_0 t + \frac{1}{2} \mathbf{g} t^2$$
  Given gravity $g = -18.0\text{ m/s}^2$, target apex height $h_{\text{apex}}$, and target landing position $\mathbf{x}_{\text{target}}$:
  $$v_{0, y} = \sqrt{2 |g| h_{\text{apex}}}$$
  $$t_{\text{flight}} = \frac{v_{0, y} + \sqrt{v_{0, y}^2 - 2 |g| (y_{\text{target}} - y_0)}}{|g|}$$
  $$\mathbf{v}_{0, xz} = \frac{\mathbf{x}_{\text{target}, xz} - \mathbf{x}_{0, xz}}{t_{\text{flight}}}$$
- The mob executes a leaping animation, enters airborne ballistic kinematic state, and lands cleanly on the opposing voxel ledge.

### 5.3 Wall-Crawling & Surface Adhesion (Arachnid / Imp Locomotion)
For agile subterranean mobs:
- Instead of orienting gravity strictly down ($-Y$), wall-crawling mobs define local gravity directed opposite to the underlying voxel surface normal $\mathbf{n}_{\text{surface}}$:
  $$\mathbf{g}_{\text{local}} = -g \cdot \mathbf{n}_{\text{surface}}$$
- Surface normals are sampled instantaneously using the 64-bit gradient of the adjacent leaf brick.
- Mobs can crawl along vertical cliff walls, hang from cross-vaulted ceilings, and drop down on the player from above.

---

## 6. Performance Budget & Threading Model

| Navigation Phase | Execution Target | Update Frequency | Cost Budget |
|---|---|---|---|
| **Tier 1 Macro-Graph A\*** | Rayon Background Thread | Event-driven / On spawn | $< 0.02\text{ ms}$ |
| **Tier 2 Dijkstra Flow Field** | Rayon Worker (`rayon::spawn`) | 5 Hz – 10 Hz (Staggered) | $< 0.80\text{ ms}$ |
| **Dynamic Invalidation** | Main Simulation Thread | Per carve event ($< 200\text{ bricks}$) | $< 0.05\text{ ms}$ |
| **Tier 3 Kinematic Steer (50 Mobs)** | Parallel For (`par_iter_mut`) | 60 Hz | $< 0.45\text{ ms}$ |
| **Total Navigation CPU Footprint** | — | — | **$< 1.32\text{ ms}$ / frame** |

With this architecture, pathfinding consumes less than **$8\%$ of a 60 FPS frame**, ensuring completely fluid real-time combat in fully destructible voxel worlds.

