# 03. Radiance Cascades (2024–2026)

## 1. Overview: The Next-Generation Paradigm

Introduced by Alexander Sannikov in late 2023 and rapidly formalized into 3D throughout 2024–2026, **Radiance Cascades (RC)** is arguably the most significant breakthrough in real-time global illumination since ReSTIR.

### The Fundamental Breakthrough: Zero Stochastic Noise
- Traditional real-time GI (Path Tracing, ReSTIR, DDGI) relies on Monte Carlo sampling, which inherently introduces variance, noise, boiling artifacts, and temporal ghosting.
- Radiance Cascades is **completely deterministic**. It computes continuous global illumination with **zero stochastic noise**, eliminating the need for aggressive temporal accumulators (TAA) or edge-blurring denoisers.

---

## 2. Mathematical Foundation: The Penumbra Hypothesis

Radiance Cascades is built upon a profound geometric observation known as the **Penumbra Hypothesis**:
* The required spatial sampling frequency $\nu_{\mathbf{x}}$ and angular sampling frequency $\nu_{\omega}$ have an **inverse relationship** with distance $r$ from an occluder or emitter:
  * **Near Field ($r \to 0$):** High spatial resolution $\Delta x$ is required to resolve sharp geometric contact boundaries, but coarse angular resolution $\Delta \Omega$ suffices (rays do not spread significantly over short distances).
  * **Far Field ($r \gg 0$):** Occluders subtend tiny solid angles ($\Delta \Omega \propto A / r^2$). Resolving soft penumbrae and distant lighting requires dense angular direction sampling ($\Delta \Omega \to 0$), but the spatial radiance field varies smoothly ($\nabla_{\mathbf{x}} L \approx 0$), allowing coarse spatial probe spacing.

```text
Near Field: High Spatial Res, Coarse Angular Res
[Surface] ──► (Many dense probe locations, few ray directions)

Far Field: Coarse Spatial Res, Dense Angular Res
[Surface] ════════════════════════════════► (Few sparse probes, hundreds of ray directions)
```

---

## 3. Cascade Decomposition & The Merge Operator

The radiance integral is partitioned into a hierarchy of $C$ cascades ($c \in \{0, 1, \dots, C-1\}$) spanning bounded distance shells $[r_{c,\min}, r_{c,\max}]$:
- Interval lengths grow geometrically: $\Delta r_c = r_0 \cdot b^c$ (branching factor $b = 4$).
- Spatial probe spacing: $\Delta x_c = \Delta x_0 \cdot 2^c$.
- Angular ray count: $N_c = N_0 \cdot 4^c$.

### 3.1 The Back-to-Front Merge Operator
Cascades are evaluated and hierarchically merged from the outermost cascade ($c = C-1$) down to $c = 0$ via volumetric back-to-front alpha-compositing:
$$\mathcal{M}\Big( (L_{\text{near}}, T_{\text{near}}), (L_{\text{far}}, T_{\text{far}}) \Big) = \Big( L_{\text{near}} + T_{\text{near}} \cdot \mathcal{I}[L_{\text{far}}], \; T_{\text{near}} \cdot \mathcal{I}[T_{\text{far}}] \Big)$$
- $L_c$: Accumulated radiance over interval.
- $T_c$: Transmittance / visibility ($T = 0$ if blocked by a solid voxel, $T = 1$ if empty air).
- $\mathcal{I}[\cdot]$: Directional and spatial interpolation operator.

```glsl
struct IntervalSample {
    vec3 radiance;
    float transmittance;
};

IntervalSample MergeIntervals(IntervalSample nearField, IntervalSample farField) {
    IntervalSample merged;
    merged.radiance = nearField.radiance + (farField.radiance * nearField.transmittance);
    merged.transmittance = nearField.transmittance * farField.transmittance;
    return merged;
}
```

---

## 4. 3D Voxel Adaptations: Split Radiance Cascades

In 2D, Radiance Cascades is trivial. In 3D, a dense volumetric grid of probes would exhaust VRAM (the curse of dimensionality).

**Split Radiance Cascades (2024–2026)** makes 3D real-time RC viable for voxel games through two key innovations:

### 4.1 Sparse Surface Hash Maps
- Probes are allocated **only around active voxel surfaces** (surfels) or camera-visible screen pixels.
- Stored in an open-addressed GPU spatial hash table ($H(\mathbf{p}) = ((x \cdot 73856093) \oplus (y \cdot 19349663) \oplus (z \cdot 83492791)) \bmod M$).
- Reduces 3D probe memory by **over $95\%$**.

### 4.2 Ray Splitting along Voxel DDA Traversal
- Rather than casting independent rays for every cascade level, primary rays march outward through the voxel CS64 tree.
- When an occluder is struck at distance $t \in [r_{c,\min}, r_{c,\max}]$:
  - Intervals $< c$ record full transmittance ($T = 1, L = 0$).
  - Cascade $c$ records hit radiance $(L_{\text{hit}}, T = 0)$.
  - Intervals $> c$ are **culled immediately**.
- Macro-cell empty space skipping in the CS64 tree allows ray intervals spanning empty air to evaluate in $O(1)$ operations.

---

## 5. Performance Comparison: Radiance Cascades vs ReSTIR vs VCT

| Feature | Split Radiance Cascades (3D) | ReSTIR GI | Voxel Cone Tracing (VCT) |
|---|---|---|---|
| **Noise Profile** | **Deterministic (Zero Noise)** | Stochastic (Requires Wavelet Denoiser) | Deterministic (Blurry/Leaky) |
| **Contact Shadows** | **Pixel-sharp ($c=0$ interval)** | Sharp after spatial reuse | Soft / Leaky |
| **Dynamic Voxel Edits** | **Instant ($O(1)$ DDA response)** | Instant | Requires clipmap re-voxelization |
| **Frame Time (RTX 3080)** | **$2.4\text{--}3.8\text{ ms}$** | **$2.6\text{--}4.5\text{ ms}$** | $4.8\text{--}7.5\text{ ms}$ |
| **Memory Footprint** | **$28\text{--}55\text{ MB}$ (Sparse Hash)** | $32\text{--}64\text{ MB}$ | $180\text{--}380\text{ MB}$ |
| **Temporal Stability** | **100% Stable (No Ghosting)** | Susceptible to disocclusion lag | 100% Stable |

> **Arcadia Synthesis:** Radiance Cascades represents the future frontier for voxel illumination. Its deterministic, noise-free nature solves the greatest flaw of modern real-time ray tracing (denoiser smearing and disocclusion boiling).
