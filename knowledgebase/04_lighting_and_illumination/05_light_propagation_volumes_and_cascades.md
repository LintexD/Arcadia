# 05. Light Propagation Volumes (LPV) & Cascades

## 1. Overview & Core Concept

Developed by Anton Kaplanyan (Crytek, 2009–2010) for *CryEngine 3*, **Light Propagation Volumes (LPV)** was one of the earliest fully dynamic, real-time indirect illumination techniques that avoided ray tracing entirely.

LPV simulates the physical propagation of radiant energy through a discrete 3D spatial grid using **Spherical Harmonics (SH)** to encode directional radiance.

---

## 2. Spherical Harmonics Radiance Representation

To prevent light from propagating isotropically (which would destroy directional shadowing and contrast), each cell in the 3D grid stores radiant flux as a low-order **Spherical Harmonics (SH)** expansion:
- Typically **2-band Spherical Harmonics ($L_1$, 4 coefficients)** per color channel (R, G, B), requiring $4 \times 3 = 12$ scalar floats per cell:
  $$L(\omega) = \sum_{l=0}^1 \sum_{m=-l}^l c_l^m Y_l^m(\omega) = c_0 Y_0^0 + c_1^{-1} Y_1^{-1} + c_1^0 Y_1^0 + c_1^1 Y_1^1$$
- $Y_0^0$ encodes ambient omnidirectional flux.
- $(Y_1^{-1}, Y_1^0, Y_1^1)$ encode the principal direction and strength of light flow.

---

## 3. The 3-Step LPV Simulation Pipeline

```text
[ Shadow Maps ] ───► [ 1. Light Injection (Virtual Point Lights) ]
                               │
                               ▼
                     [ 2. Iterative Propagation (8–12 steps) ]
                               │
                               ▼
                     [ 3. Screen-Space Sampling & Shading ]
```

### Step 1: Light Injection
- Reflective Shadow Maps (RSMs) render the scene from the sun's perspective, capturing depth, flux, and normals.
- Surface texels in the RSM are treated as **Virtual Point Lights (VPLs)**.
- In a compute pass, VPLs are projected into the 3D LPV grid and splatted as cosine-distributed SH lobes pointing along the surface normal.

### Step 2: Iterative Wave Propagation
Over $N$ simulation iterations (typically 8 to 12 iterations):
- Each cell transfers radiant energy to its 6 face-adjacent and 12 edge-adjacent neighbor cells.
- Directional flux is projected through the solid angle subtended by the neighbor face.
- **Blocking Geometry (Geometry Volume):** A secondary 3D grid stores voxel opacity. If a neighbor cell contains a solid voxel, light transfer is attenuated, casting coarse indirect shadows.

### Step 3: Surface Shading
During deferred shading, surface pixels query the LPV grid via trilinear interpolation, evaluating the SH expansion along the surface normal $\mathbf{n}$:
$$E(\mathbf{n}) = \int_{\Omega} L(\omega) \max(0, \mathbf{n} \cdot \omega) \, d\omega$$

---

## 4. Cascaded LPV for Open Worlds

A single uniform $32^3$ grid covers only $32\text{ meters}$ at $1\text{ m}$ resolution.
**Cascaded Light Propagation Volumes (cLPV)** maintains 3 to 4 nested concentric grids:
- Grid 0: $32^3$ cells covering $16\text{ m}$ ($0.5\text{ m}$ cells).
- Grid 1: $32^3$ cells covering $64\text{ m}$ ($2.0\text{ m}$ cells).
- Grid 2: $32^3$ cells covering $256\text{ m}$ ($8.0\text{ m}$ cells).

---

## 5. Failure Modes & Modern Assessment

While revolutionary in 2010, LPV exhibits severe limitations for micro-voxel engines:
1. **Coarse Spatial Resolution:** At $0.5\text{--}1.0\text{ m}$ cell size, light bleeds through thin walls ($6.25\text{ cm}$), illuminating closed interior rooms.
2. **Low Angular Fidelity:** 2-band SH cannot represent sharp specular reflections or high-frequency directional shadows.
3. **High Iteration Overhead:** Running 10 propagation iterations across 3 cascades consumes **$3\text{--}6\text{ ms}$** of compute time without producing specular lighting.

> **Modern Verdict:** LPV has been largely superseded by **ReSTIR GI**, **Radiance Cascades**, and **DDGI**, which deliver higher spatial resolution, zero wall bleeding, and support for specular reflections.
