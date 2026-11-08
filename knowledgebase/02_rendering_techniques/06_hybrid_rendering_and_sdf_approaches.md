# 06. Hybrid Rendering & SDF Approximations

## 1. Overview & Motivation

Real-time rendering architectures are rarely dogmatic. Leading production engines (e.g. Unreal Engine 5 Lumen, Frostbite, Cyberpunk 2077) combine the strengths of multiple rendering representations:
- **Rasterization / Hardware Meshes** for primary camera visibility and skinned character animation.
- **Signed Distance Fields (SDFs) and Voxel Grids** for coarse global illumination, soft distance shadows, and screen-space ray tracing fallbacks.

In a dedicated voxel engine, hybrid rendering operates in reverse:
- **Primary Compute Ray Marching** for pixel-exact, destructible micro-voxel worlds.
- **Voxel-Approximated SDFs (Signed Distance Fields)** for long-distance ray stepping and ultra-soft contact shadows.
- **Visibility Buffers** for decoupling geometric intersection from expensive PBR material shading.

---

## 2. Generating SDFs from Discrete Voxel Grids

### 2.1 The Distance Field Transformation
A Signed Distance Field $S(\mathbf{p})$ returns the shortest Euclidean distance from any point $\mathbf{p}$ to the nearest solid voxel surface:
$$S(\mathbf{p}) = \begin{cases} -\min_{\mathbf{q} \in \partial V} \|\mathbf{p} - \mathbf{q}\| & \text{if } \mathbf{p} \in \text{Solid} \\ +\min_{\mathbf{q} \in \partial V} \|\mathbf{p} - \mathbf{q}\| & \text{if } \mathbf{p} \in \text{Air} \end{cases}$$

### 2.2 Fast Jump Flooding Algorithm (JFA) on the GPU
Computing Euclidean distance fields naively takes $O(N^6)$ for an $N^3$ volume.
The **Jump Flooding Algorithm (JFA)** (Rong & Tan, 2006) computes an exact 3D distance field in $O(\log N)$ passes:
1. Initialize seed voxels (surface voxels know their own coordinates; empty voxels are set to $\infty$).
2. In pass $k$ (with step length $s = N/2, N/4, \dots, 1$):
   - Each cell queries its 26 neighbors at distance $s$ ($x \pm s, y \pm s, z \pm s$).
   - Propagate the closest known seed point.
3. For a $256^3$ chunk, JFA completes in **8 fast compute dispatches ($< 1.0\text{ ms}$)**.

---

## 3. Sphere Tracing with SDFs vs DDA

Once a coarse 3D distance field texture is generated:
- Instead of stepping cell-by-cell ($0.0625\text{ m}$ per step), the ray can safely take a giant leap equal to the distance field value:
  $$t_{\text{next}} = t + S(\mathbf{o} + t \cdot \mathbf{d})$$
- When far from surfaces, a ray can traverse hundreds of meters in **3 to 5 steps**!
- As the ray nears a surface ($S(\mathbf{p}) < \epsilon$), execution switches to exact CS64 HDDA for micro-voxel precision.

```text
Camera (t=0)
  ═════════════════► Large Sphere Step (20m) ═══► Medium Step (4m) ─► CS64 HDDA (6.25cm)
```

---

## 4. Soft Contact Shadows via SDF Cone Tracing

Using a distance field representation, directional sun shadows with realistic penumbra (soft shadows that sharpen near contact points) can be evaluated without multi-sample jittering:
$$k = \min_{t \in [t_{\text{min}}, t_{\text{max}}]} \left( \frac{w \cdot S(\mathbf{o} + t \cdot \mathbf{d}_{\text{sun}})}{t} \right)$$
$$\text{ShadowIntensity} = \text{clamp}(k, 0.0, 1.0)$$

This yields stunning photographic contact shadows at **fractional GPU compute cost** ($< 0.3\text{ ms}$ on 1080p).

---

## 5. Visibility Buffer Architecture for Voxels

### 5.1 Decoupling Ray Marching from Material Shading
In a naive ray marcher, when a ray hits a voxel, the shader executes full PBR lighting, reads texture maps, evaluates normal maps, and calculates specular highlights. Because different threads in a warp hit different materials (stone, water, mirror, glowing crystal), warps suffer massive **instruction divergence**.

### 5.2 The 2-Pass Visibility Architecture
1. **Pass 1 (Primary Ray March):**
   - Output only a 64-bit Visibility Buffer: 32-bit linear depth + 16-bit Material ID + 16-bit Instance/Normal ID.
   - Zero material shading logic is executed! All threads remain 100% lockstep.
2. **Pass 2 (Deferred Material Shading):**
   - Read the visibility buffer.
   - Material shading is grouped and evaluated coherently, eliminating instruction cache eviction.
