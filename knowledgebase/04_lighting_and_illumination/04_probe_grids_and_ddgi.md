# 04. Probe Grids & Dynamic Diffuse Global Illumination (DDGI)

## 1. Overview & Core Architecture

Introduced by Majercik et al. (NVIDIA, 2019–2021), **Dynamic Diffuse Global Illumination (DDGI)** provides real-time multi-bounce diffuse lighting by maintaining a regular 3D grid of world-space light probes ($N_x \times N_y \times N_z$):
- Probes are updated asynchronously using GPU ray tracing.
- Surface shading samples the 8 enclosing grid probes via trilinear interpolation.
- To eliminate light leaking through thin walls, DDGI uses **Chebyshev depth variance tests** (adapted from Variance Shadow Maps).

---

## 2. Octahedral Mapping & Probe Textures

Directional probe data is packed onto 2D texture atlases using **octahedral projection**:

```text
Octahedral Projection Mapping
          ▲ +Y
        / | \
       /  |  \
  ◄───┼───┼───┼───► +X
       \  |  /
        \ | /
          ▼
```

Each probe stores two small textures in a global atlas:
1. **Irradiance Texture:** $8 \times 8$ texels (+ 1-texel border padding = $10 \times 10$) storing filtered cosine-weighted irradiance $E(\mathbf{n})$ in `R11G11B10F` ($400\text{ bytes}$ / probe).
2. **Depth Texture:** $16 \times 16$ texels (+ 1-texel border = $18 \times 18$) storing mean distance $\mu$ and mean squared distance $\mu^{(2)}$ in `RG16F` ($1,296\text{ bytes}$ / probe).

---

## 3. Chebyshev Visibility Tests to Prevent Light Leaks

When shading surface point $\mathbf{x}$ with normal $\mathbf{n}$:
1. Identify the 8 enclosing probe grid points $\mathbf{p}_i$.
2. For probe $\mathbf{p}_i$, vector to surface: $\mathbf{v} = \mathbf{x} - \mathbf{p}_i$, distance $r = \|\mathbf{v}\|$, direction $\hat{\mathbf{v}} = \mathbf{v} / r$.
3. Sample probe depth texture in direction $\hat{\mathbf{v}}$ to read $(\mu, \mu^{(2)})$.
4. Compute depth variance:
   $$\sigma^2 = \max(\mu^{(2)} - \mu^2, 10^{-4})$$
5. If $r \le \mu$: Probe has direct line-of-sight ($w_{\text{vis}} = 1.0$).
6. If $r > \mu$: Evaluate the Chebyshev upper bound:
   $$P(\text{dist} \ge r) \le \frac{\sigma^2}{\sigma^2 + (r - \mu)^2}$$
   $$w_{\text{vis}, i} = \text{clamp}\left( \frac{P - v_{\text{bias}}}{1.0 - v_{\text{bias}}}, 0.0, 1.0 \right)^3$$
7. Modulate trilinear weight: $w_i = w_{\text{trilinear}, i} \cdot \max(0, \mathbf{n} \cdot \hat{\mathbf{v}}) \cdot w_{\text{vis}, i}$.
8. Final Irradiance:
   $$E(\mathbf{x}) = \frac{\sum_{i=1}^8 w_i E_i(\mathbf{n})}{\sum_{i=1}^8 w_i + \epsilon}$$

---

## 4. Voxel-Specific Adaptations: Relocation & Sleep States

In a micro-voxel world, naive probe grids suffer from probes being spawned **inside solid stone voxels**, resulting in black blotches or leaked darkness.

### 4.1 Automatic Voxel Probe Relocation
1. Test probe coordinate $\mathbf{p}$ against the CS64 tree.
2. If inside a solid voxel, trace 6 exploratory axial rays ($\pm X, \pm Y, \pm Z$).
3. Snap the probe out of the solid block into the nearest empty air voxel:
   $$\mathbf{p}_{\text{relocated}} = \mathbf{p} + \mathbf{n}_{\text{exit}} \cdot \text{VoxelSize}$$
4. If a probe is deeply buried in bedrock with no empty neighbors within $2\text{ meters}$, set `PROBE_STATE_CULLED` and disable ray tracing.

### 4.2 Probe Sleep States
- In static subterranean caves or rooms where lighting does not change, probes reach convergence.
- If texel delta $\Delta E < \epsilon$ for 30 consecutive frames, the probe enters **Sleep Mode**, tracing only 4 rays/frame instead of 128 rays.
- When an explosion or voxel edit occurs in the chunk, an AABB trigger instantly awakens all probes within radius $R$.
- **Bandwidth Reduction:** Slashes probe ray tracing load by **$60\%\text{--}80\%$**.

---

## 5. Performance & Memory Profile

| Metric | DDGI ($32 \times 16 \times 32$ Probes = 16,384 Probes) |
|---|---|
| **Probe Ray Tracing (128 rays/probe, 20% active)** | $1.2\text{ ms}$ (RTX 3080) |
| **Octahedral Accumulation & Filter** | $0.4\text{ ms}$ |
| **Screen Shading Surface Gather** | $0.6\text{ ms}$ |
| **Total DDGI Frame Time** | **$2.2\text{ ms}$** |
| **VRAM Footprint** | **$\approx 27.5\text{ MB}$** (Irradiance: 6.5 MB, Depth: 21 MB) |
| **Texture Bandwidth** | **$12\text{--}22\text{ GB/s}$** (Low) |
