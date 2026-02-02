---
name: python-rendering
description: Use when writing Python 3D rendering code. Automatically applies when using Python rendering libraries like nvdiffrast, PyTorch3D, pyrender, Blender Python API, trimesh, Open3D. Essential for 3D mesh visualization, texture rendering, camera setup, and coordinate system conversion.
---

# Python 3D Rendering Skill

## Core Principle: Always Verify Visually

After writing rendering code, you **must** visually verify the result image:

1. Save the image to a file (`output.png`)
2. Open the image file with the Read tool to visually verify
3. If there are issues, fix and re-render
4. Repeat until the result is correct

**Never just write code and call it done!**

---

## Coordinate Systems by Framework (Axis Conventions)

Always check which coordinate system your project uses before rendering.

### Important: Two Meanings of "Forward"

- **Object Forward**: The direction an object/character faces as "front"
- **Camera Look**: The direction the camera looks at (usually -Z)

These are different! Don't confuse them.

### Coordinate System Table

| Framework | Up | Object Forward | Camera Look | View Space | Notes |
|-----------|-----|----------------|-------------|------------|-------|
| **OpenGL** | +Y | - | -Z | Right-handed | NDC is left-handed! |
| **Blender** | +Z | +Y | -Z (local) | Right-handed | Camera local -Z |
| **PyTorch3D** | +Y | - | -Z (world) | Right-handed | +Z=viewer side in NDC |
| **nvdiffrast** | +Y | - | -Z | Right-handed | OpenGL clip space |
| **pyrender** | +Y | - | -Z | Right-handed | Follows OpenGL |
| **Open3D** | +Y | - | -Z | Right-handed | |
| **trimesh** | (none) | - | -Z (scene) | - | No enforced coord system! |
| **Unity** | +Y | +Z | +Z | Left-handed | |
| **DirectX** | +Y | +Z | +Z | Left-handed | |

### Cautions

1. **trimesh does not enforce any coordinate system** - Uses the file's original coordinate system
2. **OpenGL NDC is left-handed** - Z flips when projecting from View Space (right-handed)
3. **Blender camera** - Camera object looks at its local -Z (different from object forward +Y)

### Coordinate Conversion Examples

```python
import numpy as np

def convert_blender_to_opengl(vertices):
    """Blender (Z-up) -> OpenGL (Y-up)"""
    # Rotate -90 degrees around X axis
    rotation = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=np.float32)
    return vertices @ rotation.T

def convert_opengl_to_blender(vertices):
    """OpenGL (Y-up) -> Blender (Z-up)"""
    rotation = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=np.float32)
    return vertices @ rotation.T
```

### Verification Method

After coordinate conversion, always:
1. Render and save to an image
2. Check the image with the Read tool
3. Visually verify the model is oriented correctly

---

## Checklist: Pre-Rendering Checks

### 1. Check Mesh Data

```python
# After loading mesh, always check
print(f"Vertices: {vertices.shape}")  # (N, 3)
print(f"Faces: {faces.shape}")        # (M, 3)
print(f"Vertex range: {vertices.min(axis=0)} ~ {vertices.max(axis=0)}")

# Does it have texture coordinates?
if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
    print(f"UV coords: {mesh.visual.uv.shape}")
    HAS_TEXTURE = True

# Does it have vertex colors?
if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
    print(f"Vertex colors: {mesh.visual.vertex_colors.shape}")
    HAS_VERTEX_COLORS = True
```

### 2. Texture/Vertex Color Handling

**If there's a texture, you MUST render with the texture!**
**If there are vertex colors, you MUST render with vertex colors!**

```python
# Checking in trimesh
import trimesh

mesh = trimesh.load('model.obj')

if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
    # Has texture -> texture rendering required
    texture_image = mesh.visual.material.image
    uv_coords = mesh.visual.uv
    print("Texture rendering required!")

elif isinstance(mesh.visual, trimesh.visual.ColorVisuals):
    # Has vertex colors -> vertex color rendering required
    vertex_colors = mesh.visual.vertex_colors
    print("Vertex color rendering required!")
```

### 3. Lighting Check

**Rendering without lighting results in a black screen!**

If there's no lighting, inform the user and:
- Add default lighting
- Or use flat shading / unlit rendering

```python
# pyrender lighting example
import pyrender

# Add basic light
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=np.eye(4))

# Or add ambient light
scene.ambient_light = [0.3, 0.3, 0.3]
```

### pyrender DirectionalLight Direction Setup (Important!)

**DirectionalLight emits light in the -Z axis direction of its local coordinate system.**

```
Light object's local coordinate system:

        Y (up)
        |
        |
        +-----> X (right)
       /
      /
     Z (forward)

Light is emitted in -Z direction
```

**pose matrix = defines how to place the local coordinate system in world space**

```python
# DirectionalLight has no direction parameter!
light = pyrender.DirectionalLight(color=[1,1,1], intensity=10.0)

# Direction is set via pose (pose's Z axis = local Z axis direction in world)
scene.add(light, pose=light_pose)
```

**To emit light in a specific direction:**

```python
def make_light_pose(direction):
    """Create pose to emit light in the given direction"""
    d = np.array(direction) / np.linalg.norm(direction)

    # Light emits in local -Z
    # For local -Z to become world d -> set local Z to -d
    pose = np.eye(4)
    pose[:3, 2] = -d  # Z axis -> -d (then -Z axis -> d)

    # Calculate X, Y axes too (complete orthogonal coordinate system)
    up = np.array([0, 1, 0])
    right = np.cross(up, -d)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right /= np.linalg.norm(right)
    pose[:3, 0] = right
    pose[:3, 1] = np.cross(-d, right)

    return pose

# Usage: emit light in [1, 1.5, 1] direction
light = pyrender.DirectionalLight(color=[1,1,1], intensity=20.0)
scene.add(light, pose=make_light_pose([1, 1.5, 1]))
```

---

## Basic Rendering Templates by Framework

### pyrender (Recommended - 3-point Lighting + RGBA Output)

```python
import numpy as np
import trimesh
import pyrender
from PIL import Image
import os
import random

os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Headless environment

def render_mesh(glb_path, output_path, azimuth=45, elevation=30, image_size=512):
    """3-point lighting + transparent background RGBA rendering"""

    # Load mesh (apply scene graph transforms!)
    scene = trimesh.load(glb_path, force='scene')
    mesh = scene.to_geometry()
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes)

    # Normalize: center + fit in unit sphere
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh.vertices /= np.abs(mesh.vertices).max()

    # Create scene - low ambient (for shading contrast)
    pr_scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],  # Transparent background!
        ambient_light=[0.02, 0.02, 0.02]  # Low ambient
    )
    pr_scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    # Camera - phone camera FOV (can randomize), adjust framing with distance
    fov_deg = random.uniform(60, 75)  # Phone camera range
    fov = np.radians(fov_deg)
    distance = 2.5   # Adjust framing with distance (for normalized mesh)

    azim_rad, elev_rad = np.radians(azimuth), np.radians(elevation)
    cam_pos = np.array([
        distance * np.cos(elev_rad) * np.sin(azim_rad),
        distance * np.sin(elev_rad),
        distance * np.cos(elev_rad) * np.cos(azim_rad)
    ])

    # Look-at matrix
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(forward, [0, 1, 0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    cam_pose = np.eye(4)
    cam_pose[:3, 0], cam_pose[:3, 1], cam_pose[:3, 2], cam_pose[:3, 3] = right, up, -forward, cam_pos
    pr_scene.add(pyrender.PerspectiveCamera(yfov=fov), pose=cam_pose)

    # === 3-Point Lighting (shading with lights only, no ambient) ===
    def make_light_pose(direction):
        d = np.array(direction) / np.linalg.norm(direction)
        pose = np.eye(4)
        pose[:3, 2] = -d
        r = np.cross([0, 1, 0], -d)
        if np.linalg.norm(r) < 1e-6:
            r = np.array([1, 0, 0])
        pose[:3, 0] = r / np.linalg.norm(r)
        pose[:3, 1] = np.cross(-d, pose[:3, 0])
        return pose

    # 5-point lighting (final)
    pr_scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=40.0),
                 pose=make_light_pose([1, 1.5, 1]))       # Key (upper right front)
    pr_scene.add(pyrender.DirectionalLight(color=[0.9, 0.9, 1], intensity=35.0),
                 pose=make_light_pose([-1, 0.5, 0.5]))    # Fill (left)
    pr_scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=30.0),
                 pose=make_light_pose([0, 0.3, -1]))      # Rim (back)
    pr_scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=30.0),
                 pose=make_light_pose([0, 1, 0]))         # Top
    pr_scene.add(pyrender.DirectionalLight(color=[0.8, 0.8, 0.9], intensity=15.0),
                 pose=make_light_pose([0, -1, 0.2]))      # Bottom (weak)

    # Render (2x supersampling for AA)
    renderer = pyrender.OffscreenRenderer(image_size * 2, image_size * 2)
    color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # LANCZOS downsampling + RGBA PNG save
    img = Image.fromarray(color).resize((image_size, image_size), Image.LANCZOS)
    img.save(output_path)  # Save RGBA as-is (preserve transparent background)
```

**Key Points:**
- `ambient_light=[0.02, 0.02, 0.02]`: Low ambient -> maximum shading contrast
- `bg_color=[0, 0, 0, 0]`: Transparent background
- `fov = 60~75 degrees`: Phone camera level, can be randomized
- 5-point lighting: Key(40) + Fill(35) + Rim(30) + Top(30) + Bottom(15)
- `distance = 2.5`: For normalized mesh, adjust framing with distance (not FOV)
- RGBA PNG save: Preserve transparent background

### PyTorch3D

```python
import torch
import numpy as np
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV,
)
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mesh (with texture)
mesh = load_objs_as_meshes(["model.obj"], device=device)

# Camera setup (PyTorch3D: Y-up, +Z forward)
R, T = look_at_view_transform(dist=2.5, elev=30, azim=45)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Lighting (required!)
lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

# Rasterization settings
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# Render
images = renderer(mesh)
image = images[0, ..., :3].cpu().numpy()
image = (image * 255).astype(np.uint8)

Image.fromarray(image).save('output.png')
print("Rendering complete: output.png")
```

### nvdiffrast

```python
import torch
import nvdiffrast.torch as dr
import numpy as np
from PIL import Image

# CUDA context
glctx = dr.RasterizeCudaContext()

# Prepare mesh data (OpenGL coordinate system: Y-up, -Z forward)
vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')  # (N, 3)
faces = torch.tensor(faces, dtype=torch.int32, device='cuda')          # (M, 3)

# Transform vertices to clip space
# MVP matrix needed
def make_mvp(eye, target, up, fov, aspect, near, far):
    # View matrix
    z = eye - target
    z = z / torch.norm(z)
    x = torch.cross(up, z)
    x = x / torch.norm(x)
    y = torch.cross(z, x)

    view = torch.eye(4, device='cuda')
    view[:3, 0] = x
    view[:3, 1] = y
    view[:3, 2] = z
    view[:3, 3] = -torch.tensor([torch.dot(x, eye), torch.dot(y, eye), torch.dot(z, eye)], device='cuda')

    # Projection matrix
    f = 1.0 / np.tan(fov / 2)
    proj = torch.zeros(4, 4, device='cuda')
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0

    return proj @ view

mvp = make_mvp(
    eye=torch.tensor([0, 0, 3], dtype=torch.float32, device='cuda'),
    target=torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda'),
    up=torch.tensor([0, 1, 0], dtype=torch.float32, device='cuda'),
    fov=np.pi/3, aspect=1.0, near=0.1, far=100.0
)

# Clip space coordinates
pos_clip = vertices @ mvp[:3, :3].T + mvp[:3, 3]
pos_clip = torch.cat([pos_clip, torch.ones_like(pos_clip[:, :1])], dim=-1)

# Rasterize
rast, _ = dr.rasterize(glctx, pos_clip[None], faces, resolution=[512, 512])

# Interpolate (vertex colors or texture coordinates)
if HAS_VERTEX_COLORS:
    colors = torch.tensor(vertex_colors[:, :3] / 255.0, dtype=torch.float32, device='cuda')
    color, _ = dr.interpolate(colors[None], rast, faces)
else:
    # Default gray
    color = torch.ones_like(rast[..., :3]) * 0.8

# Anti-aliasing
color = dr.antialias(color, rast, pos_clip[None], faces)

# Save
image = (color[0].cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(image).save('output.png')
```

### Open3D

```python
import open3d as o3d
import numpy as np

# Load mesh
mesh = o3d.io.read_triangle_mesh("model.obj")

# Check for texture
if mesh.has_triangle_uvs() and len(mesh.textures) > 0:
    print("Has texture - texture rendering")
elif mesh.has_vertex_colors():
    print("Has vertex colors")
else:
    # Apply default color
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

# Compute normals (needed for lighting)
mesh.compute_vertex_normals()

# Offscreen rendering
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=800, height=600)
vis.add_geometry(mesh)

# Camera setup
ctr = vis.get_view_control()
ctr.set_zoom(0.8)
ctr.set_front([0, 0, -1])
ctr.set_up([0, 1, 0])

vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output.png")
vis.destroy_window()

print("Rendering complete: output.png")
```

---

## Headless Rendering (EGL Setup)

EGL is required when rendering on a server or headless environment.

### EGL Installation Check and Setup

```bash
# Check EGL installation
python -c "import OpenGL.EGL"

# If not installed
pip install PyOpenGL PyOpenGL_accelerate

# Install EGL libraries on Ubuntu
sudo apt-get install libegl1-mesa-dev libgl1-mesa-dev
```

### pyrender Headless Setup

```python
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Must be before import!

import pyrender
# ... rest of code
```

### Open3D Headless Setup

```python
import os
os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # Use CPU rendering

import open3d as o3d
# ... rest of code
```

---

## Debugging Checklist

When rendering results are wrong:

### Black Screen
- [ ] Is there lighting?
- [ ] Is the camera looking at the mesh?
- [ ] Is the mesh inside the camera frustum?
- [ ] Are near/far planes appropriate?

### Mesh Appears Flipped
- [ ] Is coordinate system conversion needed?
- [ ] Is face winding order correct? (CCW vs CW)
- [ ] Are normals pointing correctly?

### Texture Not Visible
- [ ] Are there UV coordinates?
- [ ] Is the texture image loaded?
- [ ] Is UV range [0, 1]?

### Strange Colors
- [ ] Is the color value range correct? ([0,1] vs [0,255])
- [ ] Is RGB vs BGR order correct?

---

## Pre-Completion Checklist

1. **Did you save the rendered image?**
2. **Did you open the image with the Read tool to visually verify?**
3. **Does the result match user expectations?**
4. **Does the coordinate system match other parts of the project?**

Rendering code is **not complete until visually verified!**

---

## Environment Setup and Troubleshooting

### Library Compatibility Matrix

| Library | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | Notes |
|---------|-------------|-------------|-------------|-------------|-------|
| trimesh | OK | OK | OK | OK | Requires pyglet<2 |
| pyrender | OK | OK | Warning | No | OpenGL context issues |
| PyTorch3D | OK | OK | OK | Warning | Requires CUDA |
| nvdiffrast | OK | OK | OK | Warning | Requires CUDA |
| Open3D | OK | OK | OK | Warning | |

### Quick Environment Check

```bash
# Check Python version
python --version

# Check installed rendering libraries
python -c "import trimesh; print(f'trimesh: {trimesh.__version__}')"
python -c "import pyrender; print('pyrender: OK')"
python -c "import open3d as o3d; print(f'Open3D: {o3d.__version__}')"
python -c "import torch; import pytorch3d; print('PyTorch3D: OK')"
python -c "import nvdiffrast; print('nvdiffrast: OK')"

# Check pyglet version (important for trimesh rendering!)
python -c "import pyglet; print(f'pyglet: {pyglet.version}')"
```

### trimesh Rendering Setup

trimesh's `save_image()` requires **pyglet<2**:

```bash
# Downgrade if pyglet 2.x is installed
pip install "pyglet<2"

# Verify
python -c "import pyglet; print(pyglet.version)"  # Should be 1.5.x
```

**Error when rendering with trimesh with pyglet 2.x installed:**
```
ImportError: `trimesh.viewer.windowed` requires `pip install "pyglet<2"`
```

### pyrender Windows Troubleshooting

**When OpenGL error occurs on Python 3.12+:**
```
ctypes.ArgumentError: argument 2: TypeError: No array-type handler...
```

**Solutions:**
1. Use Python 3.11 or lower (recommended)
2. Or use trimesh instead:

```python
# Use trimesh instead of pyrender
import trimesh

scene = trimesh.load('model.glb')
png_data = scene.save_image(resolution=[1024, 768], visible=False)
with open('output.png', 'wb') as f:
    f.write(png_data)
```

### Open3D Installation

```bash
# Basic installation
pip install open3d

# GPU support (optional)
pip install open3d-gpu
```

### PyTorch3D Installation

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch first (matching CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch3D
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
```

### nvdiffrast Installation

```bash
# Requires CUDA toolkit
pip install nvdiffrast
```

### Recommended Installation Order

When installing rendering libraries in a new environment:

```bash
# 1. Basic dependencies
pip install numpy pillow

# 2. trimesh (most versatile)
pip install trimesh
pip install "pyglet<2"  # For rendering

# 3. Open3D (optional)
pip install open3d

# 4. pyrender (only on Python 3.11 or lower)
pip install pyrender

# 5. PyTorch3D / nvdiffrast (when GPU rendering needed)
# Install according to CUDA environment
```

### Renderer Selection Guide

| Situation | Recommended Renderer | Notes |
|-----------|---------------------|-------|
| Quick preview | trimesh | Requires pyglet<2 |
| Texture + lighting rendering | pyrender | Basic PBR support |
| Differentiable rendering (training) | PyTorch3D, nvdiffrast | Requires CUDA |
| Headless server (Linux) | pyrender + EGL | |
| Headless server (Windows) | trimesh | EGL not supported |
| Windows + Python 3.13 | pyrender or trimesh | Both work |

### trimesh Basic Rendering Template (Most Stable)

```python
import numpy as np
import trimesh

# Load mesh
scene = trimesh.load('model.glb')

# Bounding info
bounds = scene.bounds
center = scene.centroid
extent = np.linalg.norm(bounds[1] - bounds[0])

# Camera setup
scene.camera.resolution = [1024, 768]
scene.camera.fov = [45, 35]

# Camera position (from slightly above)
distance = extent * 1.0
azimuth, elevation = np.radians(30), np.radians(25)

cam_pos = center + np.array([
    distance * np.sin(azimuth) * np.cos(elevation),
    distance * np.sin(elevation),
    distance * np.cos(azimuth) * np.cos(elevation)
])

# Look-at matrix
def look_at(eye, target, up=[0,1,0]):
    f = (target - eye); f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = s, u, -f, eye
    return m

scene.camera_transform = look_at(cam_pos, center)

# Render and save
png_data = scene.save_image(resolution=[1024, 768], visible=False)
with open('output.png', 'wb') as f:
    f.write(png_data)

print("Rendering complete!")
# Always open output.png with Read tool to visually verify!
```

### pyrender Texture Rendering Template (Verified)

**Important: pyrender supports texture rendering. No need to convert to vertex colors!**

```python
import numpy as np
import trimesh
import pyrender
from PIL import Image

# Load mesh
tm_scene = trimesh.load('model.glb')

# Extract geometry if Scene
if isinstance(tm_scene, trimesh.Scene):
    geometries = list(tm_scene.geometry.values())
else:
    geometries = [tm_scene]

# Create pyrender Scene
scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.95, 0.95, 0.95, 1.0])

# Add all geometry (texture auto-applied)
for geom in geometries:
    pr_mesh = pyrender.Mesh.from_trimesh(geom)  # Includes texture!
    scene.add(pr_mesh)

# Calculate bounding box
all_bounds = np.array([g.bounds for g in geometries])
bounds_min = all_bounds[:, 0, :].min(axis=0)
bounds_max = all_bounds[:, 1, :].max(axis=0)
center = (bounds_min + bounds_max) / 2
extent = np.linalg.norm(bounds_max - bounds_min)

# Camera setup
camera_distance = extent * 1.5
azimuth, elevation = np.radians(35), np.radians(25)

eye = center + np.array([
    camera_distance * np.sin(azimuth) * np.cos(elevation),
    camera_distance * np.sin(elevation),
    camera_distance * np.cos(azimuth) * np.cos(elevation)
])

# Look-at matrix
f = center - eye; f = f / np.linalg.norm(f)
s = np.cross(f, [0,1,0]); s = s / np.linalg.norm(s)
u = np.cross(s, f)

camera_pose = np.eye(4)
camera_pose[:3, 0] = s
camera_pose[:3, 1] = u
camera_pose[:3, 2] = -f
camera_pose[:3, 3] = eye

camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
scene.add(camera, pose=camera_pose)

# Lighting (required!)
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=camera_pose)

# Render
renderer = pyrender.OffscreenRenderer(1024, 768)
color, depth = renderer.render(scene)
renderer.delete()

Image.fromarray(color).save('output.png')
# Always open output.png with Read tool to visually verify!
```

---

## Windows Environment Notes

### Unavailable Settings

```python
# Not available on Windows
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # No OSMesa
os.environ['PYOPENGL_PLATFORM'] = 'egl'     # EGL not supported by default
```

### Recommended Methods on Windows

1. **trimesh + pyglet<2**: Most stable
2. **pyrender default settings**: Works in most cases
3. **When headless needed**: Use virtual display or GPU server

---

## Texture Rendering Cautions

### Avoid: Texture to Vertex Color Conversion

```python
# This method loses 90%+ of texture information
vertex_colors = geom.visual.to_color().vertex_colors  # Only as last resort!
```

**Information loss example:**
- Original texture: 1024 x 1024 = 1,048,576 pixels
- Vertex colors: 63,892 points
- Loss rate: **93.9%**

pyrender directly supports textures, so no conversion needed!

---

## trimesh Scene Loading Transform Application (Important!)

### Problem: GLB/GLTF Scene Graph Transform Not Applied

GLB/GLTF files store transformation matrices (rotation, scale, translation) in the scene graph.
`scene.geometry.values()` returns **only original vertices** without applying transforms!

```python
# Wrong method (transforms not applied - model orientation incorrect)
scene = trimesh.load(path, force='scene')
meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
mesh = trimesh.util.concatenate(meshes)

# Correct method (apply scene graph transforms)
scene = trimesh.load(path, force='scene')
mesh = scene.to_geometry()  # Apply all transforms!
if isinstance(mesh, trimesh.Scene):
    # Still a Scene, concatenate
    meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
    mesh = trimesh.util.concatenate(meshes)
```

### Why Is This Important?

GLB files exported from Sketchfab include a Z-up to Y-up transformation matrix in the root node:
```
Matrix: [1, 0, 0, 0,  0, 0, -1, 0,  0, 1, 0, 0,  0, 0, 0, 1]
# = -90 degree rotation around X axis (from Z-up to Y-up)
```

Loading without `to_geometry()` will make the model appear **lying sideways or flipped**.

---

## Objaverse / Sketchfab Model Characteristics

### Coordinate System Conversion

- Original models: Mostly **Z-up** (created in Blender, 3ds Max, etc.)
- GLTF export: Sketchfab converts to **Y-up** (applies matrix to root node)
- trimesh load: Must use `to_geometry()` for transforms to apply

### Cautions

1. **Some models use non-standard transforms** - Creators model in different orientations
2. **100% consistent orientation not guaranteed** - Most are fine but some may still be wrong
3. **How to check transformation matrix**:
```python
import pygltflib
gltf = pygltflib.GLTF2().load(path)
node = gltf.nodes[0]
print(f"Root transform: {node.matrix}")
```

---

## ObjaversePlusPlus Recommended Filtering Conditions

To select high-quality textured models from Objaverse, use the [ObjaversePlusPlus](https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus) dataset:

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("cindyxl/ObjaversePlusPlus")
df = ds['train'].to_pandas()

# Recommended filtering conditions (single objects)
filtered = df[
    (df['is_scene'] == 'false') &        # Exclude scenes
    (df['is_multi_object'] == 'false') & # Exclude multi-objects (single objects only!)
    (df['is_single_color'] == 'false') & # Exclude single color
    (df['is_transparent'] == 'false') &  # Exclude transparent
    (df['style'] != 'scanned') &         # Exclude scanned (often no texture)
    (df['score'] == 3)                   # Highest quality only
]

# Result: Approximately 200K high-quality single objects
print(f"Filtered: {len(filtered)}")
```

### Quality Score Meanings

| Score | Meaning | Texture |
|-------|---------|---------|
| 0 | Meaningless, damaged | No |
| 1 | Identifiable but incomplete | Maybe |
| 2 | Clear shape + adequate texture | Yes |
| 3 | Excellent quality + professional texture | Yes (high quality) |

### Style Characteristics

| Style | Texture | Notes |
|-------|---------|-------|
| realistic | Mostly present | Recommended |
| anime | Present | Character-focused |
| cartoon | Present | |
| scanned | Often missing | Often only vertex colors |

---

## Anti-Aliasing (Smooth Edge Processing)

### pyrender: 2x Supersampling + LANCZOS (Recommended)

pyrender is OpenGL-based, so supersampling produces clean anti-aliasing.

```python
import pyrender
from PIL import Image

# Render at 2x resolution
final_size = 512
render_size = final_size * 2  # 1024

renderer = pyrender.OffscreenRenderer(render_size, render_size)
color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
renderer.delete()

# LANCZOS downsampling for anti-aliasing
image = Image.fromarray(color)
image = image.resize((final_size, final_size), Image.LANCZOS)
image.save('output.png')
```

**Result**: Smooth edges without triangle artifacts

### PyTorch3D: SoftPhongShader Is Not Suitable for Visualization!

**Warning**: PyTorch3D's `SoftPhongShader` is for **differentiable rendering**,
and produces **triangle mesh artifacts**. Not suitable for visualization!

```python
# Not suitable for visualization - triangle artifacts
from pytorch3d.renderer import SoftPhongShader
shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

# HardPhongShader is also not perfect
from pytorch3d.renderer import HardPhongShader
shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
```

**If you must use PyTorch3D for visualization**:
- Apply 2x+ supersampling
- Or use pyrender/Blender instead

### Other Rendering APIs

Each rendering API has different appropriate anti-aliasing methods:

| API | Anti-Aliasing Method | Notes |
|-----|---------------------|-------|
| **pyrender** | 2x Supersampling + LANCZOS | Verified |
| **PyTorch3D** | Supersampling only (shader artifacts) | SoftPhongShader not suitable for visualization |
| **nvdiffrast** | `dr.antialias()` function provided | Needs verification |
| **Blender EEVEE** | `scene.eevee.taa_render_samples` | Temporal AA |
| **Open3D** | Separate settings required | Needs verification |
| **trimesh** | pyglet dependent, limited | Needs verification |

---

## trimesh Texture Check Method (Accurate Version)

```python
import trimesh

def has_texture(mesh: trimesh.Trimesh) -> bool:
    """Accurately check if trimesh mesh has texture"""
    return (
        isinstance(mesh.visual, trimesh.visual.TextureVisuals) and
        hasattr(mesh.visual, 'uv') and
        mesh.visual.uv is not None and
        hasattr(mesh.visual.material, 'baseColorTexture') and
        mesh.visual.material.baseColorTexture is not None
    )

def has_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    """Check if trimesh mesh has vertex colors"""
    return (
        isinstance(mesh.visual, trimesh.visual.ColorVisuals) and
        mesh.visual.vertex_colors is not None
    )

# Usage example
mesh = trimesh.load('model.glb').to_geometry()
if has_texture(mesh):
    print("Texture rendering required")
    tex_img = mesh.visual.material.baseColorTexture
    uv = mesh.visual.uv
elif has_vertex_colors(mesh):
    print("Vertex color rendering required")
    colors = mesh.visual.vertex_colors
else:
    print("No texture/color - use default material")
```
