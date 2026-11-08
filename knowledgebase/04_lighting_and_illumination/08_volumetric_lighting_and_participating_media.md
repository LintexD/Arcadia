# 08. Volumetric Lighting & Participating Media

## 1. Overview & Physics of Participating Media

Light traveling through an atmosphere (fog, dust, smoke, underwater murky depths) interacts with suspended microscopic particles through three physical phenomena:
1. **Absorption:** Light energy converted into heat.
2. **Out-Scattering:** Light deflected away from the view ray.
3. **In-Scattering:** Directional sunlight or emissive light scattered into the view ray toward the camera (creating **God Rays / Light Shafts**).

The governing equation is the **Radiative Transfer Equation (RTE)**:
$$\frac{dL(\mathbf{x}, \omega)}{ds} = -\sigma_t(\mathbf{x}) L(\mathbf{x}, \omega) + \sigma_s(\mathbf{x}) \int_{4\pi} p(\omega, \omega') L(\mathbf{x}, \omega') \, d\omega' + L_e(\mathbf{x}, \omega)$$
Where $\sigma_a$ is the absorption coefficient, $\sigma_s$ is the scattering coefficient, and $\sigma_t = \sigma_a + \sigma_s$ is the **extinction coefficient**.

---

## 2. Beer-Lambert Law & Phase Functions

### 2.1 Beer-Lambert Transmittance
Over a distance $d$ through medium with uniform extinction $\sigma_t$, transmittance $T(d) \in [0, 1]$ decays exponentially:
$$T(d) = \exp(-\sigma_t \cdot d)$$

In water volumes, $\sigma_t$ is wavelength-dependent (e.g. red light is absorbed $10\times$ faster than blue light), creating natural turquoise and deep-blue underwater coloration.

### 2.2 Henyey-Greenstein Phase Function
The angular probability distribution of scattered light is modeled by the **Henyey-Greenstein Phase Function**:
$$p(\theta) = \frac{1}{4\pi} \frac{1 - g^2}{(1 + g^2 - 2g \cos\theta)^{3/2}}$$
- $g = 0$: Isotropic scattering (light scatters uniformly in all directions).
- $g > 0$ (e.g. $g = 0.6\text{--}0.8$): Strong forward scattering (produces brilliant, glowing light halos around the sun and bright light shafts through forest canopies).

---

## 3. Real-Time Froxel Grids (Frustum Voxel Grids)

Marching physical rays through volumetric fog for every 1080p screen pixel would require hundreds of shadow ray evaluations per pixel, which is computationally prohibitive ($>15\text{ ms}$).

The modern solution (pioneered by Bart Wronski, *Assassin's Creed IV*, SIGGRAPH 2014) is the **Froxel Grid** (Frustum-aligned Voxel Grid):
- Subdivide the camera view frustum into a 3D grid: $160 \times 90 \times 64$ cells.
- $X$ and $Y$ map to screen tiles.
- $Z$ uses **exponential depth slicing** to allocate fine slices near the camera and coarse slices far away:
  $$z_k = z_{\text{near}} \left( \frac{z_{\text{far}}}{z_{\text{near}}} \right)^{\frac{k}{N_z}}$$

```text
Camera Frustum Slices (Exponential Depth)
   🎥 ───► |  |   |    |     |      |        |          | (Far)
         Fine slices near camera   Coarse slices at distance
```

### 3.1 The 3-Pass Froxel Compute Pipeline
1. **Pass 1 (Scatter & Shadow Injection):**
   - For each froxel cell, test its world-space center against the directional shadow map.
   - Evaluate in-scattering: $L_{\text{in}} = L_{\text{sun}} \cdot V(\mathbf{x}) \cdot p(\theta) \cdot \sigma_s$.
   - Output to a 3D texture: `RGBA16F`.
2. **Pass 2 (Temporal Reprojection):**
   - Reproject the previous frame's 3D froxel volume using camera motion vectors.
   - Blend with exponential moving average (alpha $\approx 0.95$) to eliminate spatial stepping noise.
3. **Pass 3 (Ray-Marching Integration):**
   - For each screen pixel, step through the 64 depth slices from front to back, accumulating transmittance and in-scattered radiance:
     $$\mathbf{c}_{\text{accum}} += T_{\text{accum}} \cdot L_{\text{slice}} \cdot (1 - \exp(-\sigma_t \Delta z))$$
     $$T_{\text{accum}} *= \exp(-\sigma_t \Delta z)$$

---

## 4. Performance & Resource Budget

| Parameter | Value |
|---|---|
| **Froxel Dimensions** | $160 \times 90 \times 64$ ($921,600$ cells) |
| **VRAM Consumption** | $160 \times 90 \times 64 \times 8\text{ bytes} \approx \mathbf{7.37\text{ MB}}$ |
| **Pass 1: Light & Shadow Injection** | $0.65\text{ ms}$ (RTX 3080) |
| **Pass 2: Temporal Reprojection** | $0.12\text{ ms}$ |
| **Pass 3: Screen Integration** | $0.28\text{ ms}$ |
| **Total Volumetric Fog Frame Cost** | **$\approx 1.05\text{ ms}$** |

This achieves cinematic volumetric god rays, mist, and atmospheric depth well within a 60 FPS frame budget.
