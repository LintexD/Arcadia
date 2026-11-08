# 09. Spatiotemporal Denoising & Reconstruction

## 1. Overview & The Real-Time Dilemma

At 60+ FPS, real-time ray-traced lighting (such as ReSTIR GI or stochastic diffuse bounces) can only trace **1 or 2 rays per pixel**. Raw Monte Carlo output is heavily undersampled and looks like static snow or grain.

The role of a **Spatiotemporal Denoiser** (such as SVGF - Spatiotemporal Variance-Guided Filtering, Schied et al., 2017) is to reconstruct smooth, physically accurate illumination from sparse samples without blurring geometric edges or creating temporal ghosting.

---

## 2. The Edge-Stopping À-Trous Wavelet Filter

The cornerstone of modern denoisers is the **À-Trous Wavelet Transform** (meaning "with holes"):
- Instead of using a giant $33 \times 33$ Gaussian blur kernel (which would require over 1,000 memory reads per pixel!), the À-Trous filter runs **4 to 5 successive $5 \times 5$ filter passes**.
- In iteration $i$, the filter kernel samples at stride $s = 2^i$ ($s \in \{1, 2, 4, 8, 16\}$).
- A 5-pass À-Trous filter covers a receptive field of **$65 \times 65$ pixels** using only $5 \times 25 = 125$ total texture samples!

```text
Iteration 0 (Stride 1):  •  •  •  •  •  (Local crevice filtering)
Iteration 1 (Stride 2):  •  .  •  .  •  .  •  .  •  (Small blur)
Iteration 2 (Stride 4):  •  .  .  .  •  .  .  .  •  (Medium blur)
Iteration 3 (Stride 8):  •  .  .  .  .  .  .  .  •  (Broad global bounce)
```

---

## 3. Voxel Plane-Distance Edge-Stopping Weights

A standard Gaussian blur blends across all adjacent pixels, causing light to bleed through walls and blurring away geometry.
An **Edge-Stopping Wavelet Filter** weights each neighbor sample by geometric affinity:
$$w(\mathbf{p}, \mathbf{q}) = w_{\text{spatial}} \cdot w_{\text{normal}}(\mathbf{p}, \mathbf{q}) \cdot w_{\text{depth}}(\mathbf{p}, \mathbf{q}) \cdot w_{\text{luminance}}(\mathbf{p}, \mathbf{q})$$

### 3.1 The Voxel Corner Bleeding Problem
Standard depth weighting ($|z_p - z_q| < \sigma_z$) fails on micro-voxels:
- When looking diagonally across a flight of voxel stairs or along a terraced hill, adjacent pixels have different depths even though they belong to the same physical step.
- Standard depth weights think the step is an edge and refuse to blur, leaving noisy speckles on flat terraces. Conversely, loosening the depth threshold causes light to bleed around sharp $90^\circ$ corners!

### 3.2 Exact Plane-Distance Weighting
In Arcadia's denoiser, edge-stopping uses the **Voxel Plane Distance**:
$$d_{\text{plane}} = \left| \mathbf{n}_p \cdot (\mathbf{x}_q - \mathbf{x}_p) \right|$$
$$w_{\text{plane}} = \exp\left( -\frac{d_{\text{plane}}^2}{2 \sigma_{\text{plane}}^2} \right)$$
- If point $\mathbf{q}$ lies on the same mathematical plane as point $\mathbf{p}$ (e.g. along the flat top of a stone terrace), $d_{\text{plane}} \equiv 0$ regardless of camera viewing angle! The filter blurs with full strength.
- If point $\mathbf{q}$ steps around a $90^\circ$ voxel edge, $d_{\text{plane}} \ge 6.25\text{ cm}$. The weight instantly drops to **zero**, eliminating edge bleeding across voxel corners.

---

## 4. Temporal Accumulation & YCoCg Variance Bounding Boxes

### 4.1 Temporal Reprojection
Using camera motion vectors:
$$\mathbf{x}_{\text{prev}} = \mathbf{x}_{\text{curr}} - \mathbf{v}_{\text{motion}}$$
Read historical radiance $L_{\text{hist}}$ from the previous frame.

### 4.2 Ghosting Prevention via YCoCg Variance Clipping
When an object moves, or the player carves a voxel, stale historical lighting creates blurry trails (**ghosting**).
To eliminate ghosting:
1. Transform local $3 \times 3$ neighbor colors from RGB to **YCoCg color space** (which decouples luminance $Y$ from chrominance $Co, Cg$).
2. Calculate the local mean $\boldsymbol{\mu}$ and standard deviation $\boldsymbol{\sigma}$.
3. Form a bounding box $[\boldsymbol{\mu} - \gamma \boldsymbol{\sigma}, \boldsymbol{\mu} + \gamma \boldsymbol{\sigma}]$ (typically $\gamma = 1.0\text{--}1.5$).
4. **Clip the history sample** to this bounding box:
   $$L_{\text{hist}}^* = \text{clamp}(L_{\text{hist}}, \, \boldsymbol{\mu} - \gamma \boldsymbol{\sigma}, \, \boldsymbol{\mu} + \gamma \boldsymbol{\sigma})$$
5. Blend current sample with clipped history:
   $$L_{\text{accum}} = (1 - \alpha) L_{\text{curr}} + \alpha L_{\text{hist}}^* \quad (\alpha \approx 0.90\text{--}0.95)$$

---

## 5. Denoiser Frame Budget (RTX 3080 @ 1080p)

| Stage | Frame Time |
|---|---|
| **Temporal Reprojection & YCoCg Variance Clipping** | $0.25\text{ ms}$ |
| **Pass 1: À-Trous Wavelet (Stride 1)** | $0.28\text{ ms}$ |
| **Pass 2: À-Trous Wavelet (Stride 2)** | $0.26\text{ ms}$ |
| **Pass 3: À-Trous Wavelet (Stride 4)** | $0.24\text{ ms}$ |
| **Pass 4: À-Trous Wavelet (Stride 8)** | $0.22\text{ ms}$ |
| **Total Denoiser Execution Time** | **$1.25\text{ ms}$** |
