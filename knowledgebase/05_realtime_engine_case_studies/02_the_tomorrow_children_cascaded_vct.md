# 02. The Tomorrow Children: Cascaded VCT

## 1. Overview & Historic Context

Developed by Q-Games and Sony Computer Entertainment for the PlayStation 4 (released 2016), *The Tomorrow Children* was a landmark technical achievement:
- It was the first commercial console game to render an entire open world with **fully real-time cascaded Voxel Cone Tracing (VCT)**.
- Achieved a distinct toy-like, claymation aesthetic characterized by smooth ambient occlusion, soft color bleeding, and glistening reflections on voxel terrain.

---

## 2. Geometry Architecture: Sparse Voxel DAGs (SVDAG)

To fit massive stylized Soviet monuments, mountains, and floating islands into the PS4's unified $8\text{ GB}$ memory pool:
- The game utilized **Sparse Voxel Directed Acyclic Graphs (SVDAGs)** (James McLaren, GDC 2015).
- Shared subtrees were deduplicated across the entire world volume.
- Massive geometric islands covering hundreds of meters occupied only **a few dozen megabytes of RAM**.

---

## 3. Lighting Architecture: 3D Cascaded Clipmap VCT

### 3.1 The Cascaded 3D Clipmap Structure
The lighting pipeline maintained **3 cascaded concentric 3D textures** centered on the player:
- **Clipmap 0 (Near):** $64\text{ m}$ diameter at fine voxel resolution.
- **Clipmap 1 (Mid):** $128\text{ m}$ diameter at half resolution.
- **Clipmap 2 (Far):** $256\text{ m}$ diameter at coarse resolution.

### 3.2 Dynamic Voxel Injection
1. Direct sunlight was injected into Clipmap 0 and Clipmap 1 using shadow map unprojection.
2. Emissive objects (mining tools, glowing crystal beacons, player lanterns) splatted radiance directly into the 3D texture using compute shaders.
3. Downsampling passes generated anisotropic directional mipmaps across 6 principal directions ($\pm X, \pm Y, \pm Z$).

### 3.3 Screen-Space Cone Tracing
Deferred shading passes evaluated:
- **Diffuse GI:** 5 wide cones per pixel sampled through the cascaded clipmaps, delivering multi-bounce color bleeding (e.g. red Soviet banners casting rich red bounces onto pale stone floors).
- **Directional Occlusion:** Wide cones evaluated ambient sky darkening in narrow ravines.
- **Glossy Reflections:** Narrow specular cones along reflection vectors for polished marble, ice, and metal.

---

## 4. Key Strengths & Production Pitfalls

### Strengths
- **Cinematic Global Illumination:** Perfectly smooth, noiseless indirect bounces with zero stochastic grain.
- **Dynamic Lighting:** Moving flashlights and glowing crystal mining tools dynamically illuminated cave walls in real time.

### Pitfalls & Why VCT Was Later Abandoned by Industry
1. **Severe Memory Footprint:** Allocating multi-cascade 3D textures consumed hundreds of megabytes of memory.
2. **Light Leaking on Thin Geometry:** 1-voxel thick walls frequently leaked light or shadow due to trilinear mipmap interpolation.
3. **Heavy Texture Filtering Bandwidth:** Marching 5 to 9 cones per pixel across 3D textures consumed over $50\%$ of the PS4's total memory bandwidth, forcing the engine to run at 1080p 30 FPS.

> **Lesson for Arcadia:** While VCT proved real-time indirect illumination was possible, modern **ReSTIR GI** and **Radiance Cascades** provide superior spatial accuracy, zero light leaking through thin walls, and lower memory bandwidth.
