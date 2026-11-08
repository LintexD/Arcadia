# 03. Voxel Skeletal Rigging, Forward Kinematics & GPU Ray Marching

## 1. The Voxel Character Paradox

In traditional rasterization engines, character meshes are animated using **Linear Blend Skinning (LBS)** or **Dual Quaternion Skinning (DQS)**, where vertices interpolate smoothly between bone transform matrices:
$$\mathbf{v}' = \sum_{j=1}^B w_j \mathbf{M}_j \mathbf{v}$$

Applying this directly to volumetric voxel grids in real time fails:
1. **Grid Tearing & Non-Orthogonal Distortion:** Voxels are axis-aligned cubic elements ($6.25\text{ cm}^3$). Applying continuous shear and rotation to individual voxel coordinates warps cubes into arbitrary parallelepipeds, breaking uniform CS64 Morton layouts and hardware `POPCNT` indexing.
2. **Re-Voxelization Overhead:** Re-rasterizing or voxelizing a deforming polygonal mesh every frame on the CPU or GPU consumes $5\text{--}15\text{ ms}$ per character, entirely exhausting the frame budget for a single enemy.

### The Real-Time Paradigm: Articulated Kinematic Multi-Body
Arcadia resolves this paradox by treating animated voxel mobs as **Articulated Kinematic Multi-Bodies**. 

Rather than deforming a single continuous volume, each limb, torso, and head is an **independent, rigid local CS64 voxel volume** bound to a bone in a kinematic skeleton. Joint articulation is achieved via geometric overlap and forward kinematics:

```text
                        [Head Volume]
                              │
                        [Neck Pivot]
                              │
     [Left Arm] ─── [Upper Torso Volume] ─── [Right Arm]
     (Clavicle)               │               (Clavicle)
     (Upper Arm)        [Spine Pivot]         (Upper Arm)
     (Forearm)                │               (Forearm)
     (Hand)             [Pelvis Volume]       (Hand)
                         ┌────┴────┐
                   [Left Leg]   [Right Leg]
                   (Thigh)      (Thigh)
                   (Calf)       (Calf)
                   (Foot)       (Foot)
```

---

## 2. Skeletal Hierarchy & Forward Kinematics

### 2.1 Bone Data Structure
A demon or humanoid skeleton consists of $B \in [12, 24]$ bones organized in a strict parent-child tree:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneTransform {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: f32, // Uniform scale (typically 1.0)
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct MobBone {
    pub name: &'static str,
    pub parent_idx: Option<u16>,
    pub local_bind_pose: BoneTransform,
    pub current_local_pose: BoneTransform,
    pub model_matrix: glam::Mat4,
    pub prev_model_matrix: glam::Mat4, // For analytic DLAA motion vectors
    pub obb_half_extents: glam::Vec3,
    pub limb_tree_idx: u32,             // Index into global LimbPool
}
```

### 2.2 Forward Kinematics Propagation
Each frame, animation keyframes or procedural physics controllers update `current_local_pose`. Global world matrices are evaluated in topological depth order:
$$\mathbf{M}_i = \begin{cases} 
\mathbf{M}_{\text{mob\_root}} \times \mathbf{T}(\text{pose}_i), & \text{if } \text{parent}(i) = \emptyset \\
\mathbf{M}_{\text{parent}(i)} \times \mathbf{T}(\text{pose}_i), & \text{otherwise}
\end{cases}$$
For 20 bones, computing forward kinematics takes **$< 1.2\text{ }\mu\text{s}$** per mob on the CPU.

---

## 3. Seamless Joint Mechanics: Ball-and-Socket Overlap

In polygonal modeling, unskinned multi-part meshes reveal ugly seams or open gaps when bent. In a compute ray marcher, we solve this with **Convex Spherical / Cylindrical Joint Caps**:

```text
  Unbent Joint (0°):                     Bent Joint (90°):
     ┌───────────┐                          ┌───────────┐
     │ Upper Arm │                          │ Upper Arm │
     │  Voxel    │                          │  Voxel    │
     │  Volume   │                          │  Volume   │
     └───┬───┬───┘                          └─────┬─────┘
         │ S │  ◄── Overlapping Spherical Cap     │ S │ ──┐
     ┌───┴───┴───┐                                └───┬───┘  │ Smooth Voxel
     │  Forearm  │                                    │ F │  │ Shoulder Contour!
     │  Voxel    │                                    │ O │  │ (No Cracks)
     │  Volume   │                                    │ R │  │
     └───────────┘                                    └───┘ ◄┘
```

1. **Pivot Sphere Inclusion:** The proximal end of the forearm volume and the distal end of the upper arm volume both include a concentric hemisphere of matching radius $R_{\text{joint}}$ centered exactly at the joint pivot.
2. **Ray Marching Invariance:** When the joint bends by an angle $\theta \in [-120^\circ, +120^\circ]$, the two volumes overlap in 3D world space. 
3. **Zero Ray-Tracing Artifacts:** Because Arcadia’s compute shader evaluates ray intersections along the ray direction, the ray simply halts at whichever voxel surface it strikes first. There are **zero visible seams, zero polygon clipping artifacts, and zero gaps**.

---

## 4. GPU Compute Ray Marching of Articulated Mobs

Rendering dozens of articulated voxel mobs with zero triangle mesh extraction is achieved via **Inverse-Space Hierarchical DDA** integrated directly into [`primary_gbuffer.wgsl`](../../shaders/primary_gbuffer.wgsl).

### 4.1 Two-Level Bounding Volume Traversal
Passing 20 individual OBBs per mob directly to all camera rays would cause excessive intersection testing. Instead, we use a two-tiered test:

```text
Primary Camera Ray
        │
        ▼
[Mob Root Bounding Sphere / OBB]  ──► (Miss: Skip all 20 limbs)
        │
        ▼ (Hit: t_in, t_out)
[Limb OBB Broadphase: 15-20 limbs]
        │
        ▼ (Hit 1 or 2 overlapping limbs)
[Ray Transformed to Bone Local Space]
        │
        ▼
[Hierarchical CS64 DDA March inside Limb]
```

### 4.2 Local Ray Transformation & Intersection Math
When ray $\mathbf{r}_{\text{world}}(t) = \mathbf{o}_{\text{world}} + t \mathbf{d}_{\text{world}}$ strikes a limb's world OBB:
1. Transform origin and direction into the bone's **local coordinate space** using the inverse model matrix $\mathbf{M}_{\text{bone}}^{-1}$:
   $$\mathbf{o}_{\text{local}} = \mathbf{M}_{\text{bone}}^{-1} \begin{bmatrix} \mathbf{o}_{\text{world}} \\ 1 \end{bmatrix}, \quad \mathbf{d}_{\text{local}} = \mathbf{M}_{\text{bone}}^{-1} \begin{bmatrix} \mathbf{d}_{\text{world}} \\ 0 \end{bmatrix}$$
2. Re-normalize local direction and rescale ray interval:
   $$s_d = \|\mathbf{d}_{\text{local}}\|, \quad \hat{\mathbf{d}}_{\text{local}} = \frac{\mathbf{d}_{\text{local}}}{s_d}, \quad t_{\text{local}} = t \cdot s_d$$
3. Traverse the limb’s local 64-Tree using standard hierarchical DDA ([`traversal64.wgsl`](../../shaders/common/traversal64.wgsl)).
4. Transform the resulting hit normal back to world space:
   $$\mathbf{n}_{\text{world}} = \text{normalize}\left( (\mathbf{M}_{\text{bone}}^{-1})^T \mathbf{n}_{\text{local}} \right)$$

---

## 5. Analytic 3D Motion Vectors for Sub-Pixel DLAA

Arcadia relies on Halton sub-pixel jitter with DLAA/TSR temporal accumulation ([`composite_dlaa.wgsl`](../../shaders/composite_dlaa.wgsl)). For moving and rotating demon limbs to avoid ghosting or temporal smearing, every hit pixel must output an **exact 2D screen motion vector**:

```wgsl
// WGSL snippet executed in primary_gbuffer.wgsl on mob limb hit:
fn compute_mob_motion_vector(
    p_local: vec3<f32>, 
    curr_mat: mat4x4<f32>, 
    prev_mat: mat4x4<f32>, 
    curr_vp: mat4x4<f32>, 
    prev_vp: mat4x4<f32>
) -> vec2<f32> {
    // Current world and clip position
    let p_curr_world = (curr_mat * vec4<f32>(p_local, 1.0)).xyz;
    let p_curr_clip  = curr_vp * vec4<f32>(p_curr_world, 1.0);
    let uv_curr      = (p_curr_clip.xy / p_curr_clip.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);

    // Previous world and clip position using last frame's bone transform
    let p_prev_world = (prev_mat * vec4<f32>(p_local, 1.0)).xyz;
    let p_prev_clip  = prev_vp * vec4<f32>(p_prev_world, 1.0);
    let uv_prev      = (p_prev_clip.xy / p_prev_clip.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);

    return uv_curr - uv_prev;
}
```

This ensures fast-slashing demon claws and leaping beasts remain **razor sharp at 1080p**, fully integrated with our YCoCg variance clipping denoiser.

---

## 6. Procedural Inverse Kinematics (IK) for Voxel Terrain

Because Arcadia’s voxel terrain is dynamic and stepped ($6.25\text{ cm}$ increments), pre-baked walk animations will float or penetrate rubble. We employ **Two-Bone Analytical IK** on legs and arms:

### 6.1 Two-Bone Analytical Leg IK
Given hip origin $\mathbf{H}$, desired foot ground target $\mathbf{F}$ (found via a quick downward DDA probe against the static CS64 world), upper leg length $L_1$, and lower leg length $L_2$:

```text
               (Hip: H)
                 o
                / \
  (Thigh: L1)  /   \  Target Distance: D = ||F - H||
              /     \
    (Knee: K) o      o (Foot: F)
               \    /
   (Calf: L2)   \  /
                 o (Foot Placed on Voxel Step)
```

1. Distance constraint: $D = \text{clamp}(\|\mathbf{F} - \mathbf{H}\|, \epsilon, L_1 + L_2 - \epsilon)$.
2. By the Law of Cosines, the interior knee bend angle $\alpha$ is:
   $$\cos \alpha = \frac{L_1^2 + L_2^2 - D^2}{2 L_1 L_2}$$
   $$\alpha = \arccos(\cos \alpha)$$
3. The hip pitch offset angle $\beta$ is:
   $$\cos \beta = \frac{L_1^2 + D^2 - L_2^2}{2 L_1 D}$$
4. Foot orientation snaps to match the underlying voxel step surface normal, ensuring demon feet plant firmly on rock ledges, stairs, and crater edges.

