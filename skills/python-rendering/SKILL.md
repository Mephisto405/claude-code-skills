---
name: python-rendering
description: Python 3D 렌더링 코드 작성 시 사용. nvdiffrast, PyTorch3D, pyrender, Blender Python API, trimesh, Open3D 등 Python 렌더링 라이브러리 사용 시 자동 적용. 3D 메시 시각화, 텍스처 렌더링, 카메라 설정, 좌표계 변환 작업 시 필수.
---

# Python 3D Rendering 스킬

## 핵심 원칙: 반드시 시각적으로 검증하라

렌더링 코드를 작성한 후 **반드시** 결과 이미지를 직접 확인해야 한다:

1. 이미지 파일로 저장 (`output.png`)
2. Read 도구로 이미지 파일을 열어 시각적으로 확인
3. 문제가 있으면 수정하고 다시 렌더링
4. 올바른 결과가 나올 때까지 반복

**절대 코드만 작성하고 끝내지 말 것!**

---

## 프레임워크별 좌표계 (축 컨벤션)

렌더링 전 반드시 현재 프로젝트에서 사용하는 좌표계를 확인하라.

### 중요: "Forward"의 두 가지 의미

- **Object Forward**: 오브젝트/캐릭터가 "앞"을 향하는 방향
- **Camera Look**: 카메라가 바라보는 방향 (대부분 -Z)

이 둘은 다르다! 혼동하지 말 것.

### 좌표계 표

| 프레임워크 | Up | Object Forward | Camera Look | View Space | 비고 |
|-----------|-----|----------------|-------------|------------|------|
| **OpenGL** | +Y | - | -Z | 우손 | NDC는 왼손! |
| **Blender** | +Z | +Y | -Z (local) | 우손 | 카메라 로컬 -Z |
| **PyTorch3D** | +Y | - | -Z (world) | 우손 | NDC에서 +Z=viewer쪽 |
| **nvdiffrast** | +Y | - | -Z | 우손 | OpenGL 클립공간 |
| **pyrender** | +Y | - | -Z | 우손 | OpenGL 따름 |
| **Open3D** | +Y | - | -Z | 우손 | |
| **trimesh** | (없음) | - | -Z (scene) | - | 좌표계 강제 없음! |
| **Unity** | +Y | +Z | +Z | 왼손 | |
| **DirectX** | +Y | +Z | +Z | 왼손 | |

### 주의사항

1. **trimesh는 좌표계를 강제하지 않는다** - 로드한 파일의 좌표계를 그대로 사용
2. **OpenGL NDC는 왼손 좌표계** - View Space(우손)에서 투영 시 Z가 뒤집힘
3. **Blender 카메라** - 카메라 오브젝트는 로컬 -Z를 바라봄 (오브젝트 forward +Y와 다름)

### 좌표계 변환 예시

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

### 검증 방법

좌표계 변환 후 반드시:
1. 렌더링해서 이미지로 저장
2. Read 도구로 이미지 확인
3. 모델이 올바른 방향으로 보이는지 시각적으로 확인

---

## 체크리스트: 렌더링 전 확인사항

### 1. 메시 데이터 확인

```python
# 메시 로드 후 반드시 확인
print(f"Vertices: {vertices.shape}")  # (N, 3)
print(f"Faces: {faces.shape}")        # (M, 3)
print(f"Vertex range: {vertices.min(axis=0)} ~ {vertices.max(axis=0)}")

# 텍스처 좌표가 있는가?
if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv'):
    print(f"UV coords: {mesh.visual.uv.shape}")
    HAS_TEXTURE = True

# 버텍스 컬러가 있는가?
if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
    print(f"Vertex colors: {mesh.visual.vertex_colors.shape}")
    HAS_VERTEX_COLORS = True
```

### 2. 텍스처/버텍스 컬러 처리

**텍스처가 있으면 반드시 텍스처를 렌더링하라!**
**버텍스 컬러가 있으면 반드시 버텍스 컬러를 렌더링하라!**

```python
# trimesh에서 확인
import trimesh

mesh = trimesh.load('model.obj')

if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
    # 텍스처 있음 -> 텍스처 렌더링 필요
    texture_image = mesh.visual.material.image
    uv_coords = mesh.visual.uv
    print("텍스처 렌더링 필요!")

elif isinstance(mesh.visual, trimesh.visual.ColorVisuals):
    # 버텍스 컬러 있음 -> 버텍스 컬러 렌더링 필요
    vertex_colors = mesh.visual.vertex_colors
    print("버텍스 컬러 렌더링 필요!")
```

### 3. 라이팅 확인

**라이팅 없이 렌더링하면 검은 화면이 나온다!**

라이팅이 없는 경우 사용자에게 알리고:
- 기본 조명 추가
- 또는 flat shading / unlit 렌더링 사용

```python
# pyrender 라이팅 예시
import pyrender

# 기본 조명 추가
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=np.eye(4))

# 또는 ambient light 추가
scene.ambient_light = [0.3, 0.3, 0.3]
```

---

## 프레임워크별 기본 렌더링 템플릿

### pyrender (가장 간단)

```python
import numpy as np
import trimesh
import pyrender
from PIL import Image

# 메시 로드
mesh = trimesh.load('model.obj')

# pyrender 메시로 변환
if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
    # 텍스처가 있는 경우
    pr_mesh = pyrender.Mesh.from_trimesh(mesh)
else:
    # 텍스처 없는 경우 - 기본 머티리얼 적용
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.8, 0.8, 0.8, 1.0]
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

# 씬 생성
scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
scene.add(pr_mesh)

# 조명 추가 (필수!)
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
light_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 3],
    [0, 0, 0, 1]
])
scene.add(light, pose=light_pose)

# 카메라 설정
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# 메시 바운딩 박스 기준으로 카메라 위치 계산
bounds = mesh.bounds
center = (bounds[0] + bounds[1]) / 2
extent = np.linalg.norm(bounds[1] - bounds[0])
camera_distance = extent * 1.5

camera_pose = np.array([
    [1, 0, 0, center[0]],
    [0, 1, 0, center[1]],
    [0, 0, 1, center[2] + camera_distance],
    [0, 0, 0, 1]
])
scene.add(camera, pose=camera_pose)

# 렌더링
renderer = pyrender.OffscreenRenderer(800, 600)
color, depth = renderer.render(scene)

# 저장
Image.fromarray(color).save('output.png')
print("렌더링 완료: output.png")
```

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

# 메시 로드 (텍스처 포함)
mesh = load_objs_as_meshes(["model.obj"], device=device)

# 카메라 설정 (PyTorch3D: Y-up, +Z forward)
R, T = look_at_view_transform(dist=2.5, elev=30, azim=45)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# 조명 (필수!)
lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

# 래스터화 설정
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# 렌더러
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# 렌더링
images = renderer(mesh)
image = images[0, ..., :3].cpu().numpy()
image = (image * 255).astype(np.uint8)

Image.fromarray(image).save('output.png')
print("렌더링 완료: output.png")
```

### nvdiffrast

```python
import torch
import nvdiffrast.torch as dr
import numpy as np
from PIL import Image

# CUDA 컨텍스트
glctx = dr.RasterizeCudaContext()

# 메시 데이터 준비 (OpenGL 좌표계: Y-up, -Z forward)
vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')  # (N, 3)
faces = torch.tensor(faces, dtype=torch.int32, device='cuda')          # (M, 3)

# 버텍스를 클립 공간으로 변환
# MVP 행렬 적용 필요
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

# 클립 공간 좌표
pos_clip = vertices @ mvp[:3, :3].T + mvp[:3, 3]
pos_clip = torch.cat([pos_clip, torch.ones_like(pos_clip[:, :1])], dim=-1)

# 래스터화
rast, _ = dr.rasterize(glctx, pos_clip[None], faces, resolution=[512, 512])

# 인터폴레이션 (버텍스 컬러 또는 텍스처 좌표)
if HAS_VERTEX_COLORS:
    colors = torch.tensor(vertex_colors[:, :3] / 255.0, dtype=torch.float32, device='cuda')
    color, _ = dr.interpolate(colors[None], rast, faces)
else:
    # 기본 회색
    color = torch.ones_like(rast[..., :3]) * 0.8

# 안티앨리어싱
color = dr.antialias(color, rast, pos_clip[None], faces)

# 저장
image = (color[0].cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(image).save('output.png')
```

### Open3D

```python
import open3d as o3d
import numpy as np

# 메시 로드
mesh = o3d.io.read_triangle_mesh("model.obj")

# 텍스처 확인
if mesh.has_triangle_uvs() and len(mesh.textures) > 0:
    print("텍스처 있음 - 텍스처 렌더링")
elif mesh.has_vertex_colors():
    print("버텍스 컬러 있음")
else:
    # 기본 색상 적용
    mesh.paint_uniform_color([0.8, 0.8, 0.8])

# 노멀 계산 (조명에 필요)
mesh.compute_vertex_normals()

# 오프스크린 렌더링
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=800, height=600)
vis.add_geometry(mesh)

# 카메라 설정
ctr = vis.get_view_control()
ctr.set_zoom(0.8)
ctr.set_front([0, 0, -1])
ctr.set_up([0, 1, 0])

vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("output.png")
vis.destroy_window()

print("렌더링 완료: output.png")
```

---

## Headless 렌더링 (EGL 설정)

서버나 headless 환경에서 렌더링할 때 EGL이 필요하다.

### EGL 설치 확인 및 설치

```bash
# EGL 설치 확인
python -c "import OpenGL.EGL"

# 설치되지 않은 경우
pip install PyOpenGL PyOpenGL_accelerate

# Ubuntu에서 EGL 라이브러리 설치
sudo apt-get install libegl1-mesa-dev libgl1-mesa-dev
```

### pyrender headless 설정

```python
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # 반드시 import 전에!

import pyrender
# ... 나머지 코드
```

### Open3D headless 설정

```python
import os
os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # CPU 렌더링 사용

import open3d as o3d
# ... 나머지 코드
```

---

## 디버깅 체크리스트

렌더링 결과가 이상할 때:

### 검은 화면
- [ ] 조명이 있는가?
- [ ] 카메라가 메시를 바라보고 있는가?
- [ ] 메시가 카메라 frustum 안에 있는가?
- [ ] near/far plane이 적절한가?

### 메시가 뒤집혀 보임
- [ ] 좌표계 변환이 필요한가?
- [ ] face winding order가 맞는가? (CCW vs CW)
- [ ] 노멀 방향이 맞는가?

### 텍스처가 안 보임
- [ ] UV 좌표가 있는가?
- [ ] 텍스처 이미지가 로드되었는가?
- [ ] UV 범위가 [0, 1]인가?

### 이상한 색상
- [ ] 색상 값 범위가 맞는가? ([0,1] vs [0,255])
- [ ] RGB vs BGR 순서가 맞는가?

---

## 작업 완료 전 필수 확인

1. **렌더링 결과 이미지를 저장했는가?**
2. **Read 도구로 이미지를 열어 시각적으로 확인했는가?**
3. **결과가 사용자의 기대와 일치하는가?**
4. **좌표계가 프로젝트의 다른 부분과 일치하는가?**

렌더링 코드는 **눈으로 확인하기 전까지는 완료가 아니다!**

---

## 환경 설정 및 문제 해결

### 라이브러리 호환성 매트릭스

| 라이브러리 | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 | 비고 |
|-----------|-------------|-------------|-------------|-------------|------|
| trimesh | ✅ | ✅ | ✅ | ✅ | pyglet<2 필요 |
| pyrender | ✅ | ✅ | ⚠️ | ❌ | OpenGL 컨텍스트 이슈 |
| PyTorch3D | ✅ | ✅ | ✅ | ⚠️ | CUDA 필요 |
| nvdiffrast | ✅ | ✅ | ✅ | ⚠️ | CUDA 필요 |
| Open3D | ✅ | ✅ | ✅ | ⚠️ | |

### 빠른 환경 점검

```bash
# Python 버전 확인
python --version

# 설치된 렌더링 라이브러리 확인
python -c "import trimesh; print(f'trimesh: {trimesh.__version__}')"
python -c "import pyrender; print('pyrender: OK')"
python -c "import open3d as o3d; print(f'Open3D: {o3d.__version__}')"
python -c "import torch; import pytorch3d; print('PyTorch3D: OK')"
python -c "import nvdiffrast; print('nvdiffrast: OK')"

# pyglet 버전 확인 (trimesh 렌더링에 중요!)
python -c "import pyglet; print(f'pyglet: {pyglet.version}')"
```

### trimesh 렌더링 설정

trimesh의 `save_image()`는 **pyglet<2**가 필요하다:

```bash
# pyglet 2.x가 설치되어 있으면 다운그레이드
pip install "pyglet<2"

# 확인
python -c "import pyglet; print(pyglet.version)"  # 1.5.x 여야 함
```

**pyglet 2.x가 설치된 상태에서 trimesh 렌더링 시 오류:**
```
ImportError: `trimesh.viewer.windowed` requires `pip install "pyglet<2"`
```

### pyrender Windows 문제 해결

**Python 3.12+ 에서 OpenGL 오류 발생 시:**
```
ctypes.ArgumentError: argument 2: TypeError: No array-type handler...
```

**해결책:**
1. Python 3.11 이하 사용 (권장)
2. 또는 trimesh로 대체:

```python
# pyrender 대신 trimesh 사용
import trimesh

scene = trimesh.load('model.glb')
png_data = scene.save_image(resolution=[1024, 768], visible=False)
with open('output.png', 'wb') as f:
    f.write(png_data)
```

### Open3D 설치

```bash
# 기본 설치
pip install open3d

# GPU 지원 (선택)
pip install open3d-gpu
```

### PyTorch3D 설치

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch 먼저 설치 (CUDA 버전에 맞게)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# PyTorch3D 설치
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
```

### nvdiffrast 설치

```bash
# CUDA toolkit 필요
pip install nvdiffrast
```

### 권장 설치 순서

새 환경에서 렌더링 라이브러리 설치 시:

```bash
# 1. 기본 의존성
pip install numpy pillow

# 2. trimesh (가장 범용적)
pip install trimesh
pip install "pyglet<2"  # 렌더링용

# 3. Open3D (선택)
pip install open3d

# 4. pyrender (Python 3.11 이하에서만)
pip install pyrender

# 5. PyTorch3D / nvdiffrast (GPU 렌더링 필요시)
# CUDA 환경에 맞게 설치
```

### 렌더러 선택 가이드

| 상황 | 권장 렌더러 | 비고 |
|------|-----------|------|
| 빠른 미리보기 | trimesh | pyglet<2 필요 |
| 텍스처 + 조명 렌더링 | pyrender | 기본적인 PBR 지원 |
| 미분 가능 렌더링 (학습) | PyTorch3D, nvdiffrast | CUDA 필요 |
| Headless 서버 (Linux) | pyrender + EGL | |
| Headless 서버 (Windows) | trimesh | EGL 미지원 |
| Windows + Python 3.13 | pyrender 또는 trimesh | 둘 다 작동 |

### trimesh 기본 렌더링 템플릿 (가장 안정적)

```python
import numpy as np
import trimesh

# 메시 로드
scene = trimesh.load('model.glb')

# 바운딩 정보
bounds = scene.bounds
center = scene.centroid
extent = np.linalg.norm(bounds[1] - bounds[0])

# 카메라 설정
scene.camera.resolution = [1024, 768]
scene.camera.fov = [45, 35]

# 카메라 위치 (비스듬히 위에서)
distance = extent * 1.0
azimuth, elevation = np.radians(30), np.radians(25)

cam_pos = center + np.array([
    distance * np.sin(azimuth) * np.cos(elevation),
    distance * np.sin(elevation),
    distance * np.cos(azimuth) * np.cos(elevation)
])

# Look-at 행렬
def look_at(eye, target, up=[0,1,0]):
    f = (target - eye); f = f / np.linalg.norm(f)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = s, u, -f, eye
    return m

scene.camera_transform = look_at(cam_pos, center)

# 렌더링 및 저장
png_data = scene.save_image(resolution=[1024, 768], visible=False)
with open('output.png', 'wb') as f:
    f.write(png_data)

print("렌더링 완료!")
# 반드시 Read 도구로 output.png를 열어 시각적으로 확인할 것!
```

### pyrender 텍스처 렌더링 템플릿 (검증됨)

**중요: pyrender는 텍스처 렌더링을 지원한다. 버텍스 컬러로 변환할 필요 없음!**

```python
import numpy as np
import trimesh
import pyrender
from PIL import Image

# 메시 로드
tm_scene = trimesh.load('model.glb')

# Scene인 경우 geometry 추출
if isinstance(tm_scene, trimesh.Scene):
    geometries = list(tm_scene.geometry.values())
else:
    geometries = [tm_scene]

# pyrender Scene 생성
scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0.95, 0.95, 0.95, 1.0])

# 모든 geometry 추가 (텍스처 자동 적용)
for geom in geometries:
    pr_mesh = pyrender.Mesh.from_trimesh(geom)  # 텍스처 포함!
    scene.add(pr_mesh)

# 바운딩 박스 계산
all_bounds = np.array([g.bounds for g in geometries])
bounds_min = all_bounds[:, 0, :].min(axis=0)
bounds_max = all_bounds[:, 1, :].max(axis=0)
center = (bounds_min + bounds_max) / 2
extent = np.linalg.norm(bounds_max - bounds_min)

# 카메라 설정
camera_distance = extent * 1.5
azimuth, elevation = np.radians(35), np.radians(25)

eye = center + np.array([
    camera_distance * np.sin(azimuth) * np.cos(elevation),
    camera_distance * np.sin(elevation),
    camera_distance * np.cos(azimuth) * np.cos(elevation)
])

# Look-at 행렬
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

# 조명 (필수!)
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
scene.add(light, pose=camera_pose)

# 렌더링
renderer = pyrender.OffscreenRenderer(1024, 768)
color, depth = renderer.render(scene)
renderer.delete()

Image.fromarray(color).save('output.png')
# 반드시 Read 도구로 output.png를 열어 시각적으로 확인할 것!
```

---

## Windows 환경 주의사항

### 사용 불가능한 설정

```python
# ❌ Windows에서 사용 불가
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # OSMesa 없음
os.environ['PYOPENGL_PLATFORM'] = 'egl'     # EGL 기본 미지원
```

### Windows에서 권장하는 방법

1. **trimesh + pyglet<2**: 가장 안정적
2. **pyrender 기본 설정**: 대부분 작동함
3. **headless 필요시**: 가상 디스플레이 또는 GPU 서버 사용

---

## 텍스처 렌더링 주의사항

### ❌ 피해야 할 방법: 텍스처 → 버텍스 컬러 변환

```python
# ❌ 이 방법은 텍스처 정보의 90% 이상을 손실함
vertex_colors = geom.visual.to_color().vertex_colors  # 최후의 수단으로만!
```

**정보 손실 예시:**
- 원본 텍스처: 1024 x 1024 = 1,048,576 픽셀
- 버텍스 컬러: 63,892 포인트
- 손실률: **93.9%**

pyrender가 텍스처를 직접 지원하므로 변환할 필요 없음!

---

## trimesh Scene 로딩 시 변환 적용 (중요!)

### 문제: GLB/GLTF 씬 그래프 변환 미적용

GLB/GLTF 파일은 씬 그래프에 변환 행렬(회전, 스케일, 이동)을 저장한다.
`scene.geometry.values()`는 **원본 버텍스만 반환**하고 변환을 적용하지 않는다!

```python
# ❌ 잘못된 방법 (변환 미적용 - 모델 방향이 틀어짐)
scene = trimesh.load(path, force='scene')
meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
mesh = trimesh.util.concatenate(meshes)

# ✅ 올바른 방법 (씬 그래프 변환 적용)
scene = trimesh.load(path, force='scene')
mesh = scene.to_geometry()  # 모든 변환 적용!
if isinstance(mesh, trimesh.Scene):
    # 여전히 Scene이면 concatenate
    meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
    mesh = trimesh.util.concatenate(meshes)
```

### 왜 중요한가?

Sketchfab에서 내보낸 GLB 파일은 루트 노드에 Z-up → Y-up 변환 행렬이 포함되어 있다:
```
Matrix: [1, 0, 0, 0,  0, 0, -1, 0,  0, 1, 0, 0,  0, 0, 0, 1]
# = X축 기준 -90도 회전 (Z-up에서 Y-up으로)
```

`to_geometry()` 없이 로드하면 모델이 **옆으로 누워있거나 뒤집혀** 보인다.

---

## Objaverse / Sketchfab 모델 특성

### 좌표계 변환

- 원본 모델: 대부분 **Z-up** (Blender, 3ds Max 등에서 제작)
- GLTF 내보내기: Sketchfab이 **Y-up**으로 변환 (루트 노드에 행렬 적용)
- trimesh 로드: `to_geometry()` 사용해야 변환 적용됨

### 주의사항

1. **일부 모델은 비표준 변환 사용** - 제작자마다 다른 방향으로 모델링
2. **100% 일관된 방향 보장 불가** - 대부분은 정상이지만 일부는 여전히 틀어질 수 있음
3. **변환 행렬 확인 방법**:
```python
import pygltflib
gltf = pygltflib.GLTF2().load(path)
node = gltf.nodes[0]
print(f"Root transform: {node.matrix}")
```

---

## ObjaversePlusPlus 필터링 권장 조건

Objaverse에서 고품질 텍스처 모델을 선별하려면 [ObjaversePlusPlus](https://huggingface.co/datasets/cindyxl/ObjaversePlusPlus) 데이터셋 사용:

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("cindyxl/ObjaversePlusPlus")
df = ds['train'].to_pandas()

# 권장 필터링 조건
filtered = df[
    (df['is_scene'] == 'false') &        # 씬 제외 (단일 오브젝트만)
    (df['is_single_color'] == 'false') & # 단색 제외
    (df['is_transparent'] == 'false') &  # 투명 제외
    (df['style'] != 'scanned') &         # 스캔 모델 제외 (텍스처 없는 경우 많음)
    (df['score'] == 3)                   # 최고 품질만
]

# 결과: 약 15만개의 고품질 텍스처 모델
print(f"Filtered: {len(filtered)}")
```

### 품질 점수 의미

| 점수 | 의미 | 텍스처 |
|------|------|--------|
| 0 | 의미 없음, 손상 | ❌ |
| 1 | 식별 가능하나 불완전 | △ |
| 2 | 명확한 형태 + 적절한 텍스처 | ✅ |
| 3 | 우수한 품질 + 전문적 텍스처 | ✅✅ |

### 스타일별 특성

| 스타일 | 텍스처 | 비고 |
|--------|--------|------|
| realistic | ✅ 대부분 있음 | 권장 |
| anime | ✅ 있음 | 캐릭터 위주 |
| cartoon | ✅ 있음 | |
| scanned | △ 종종 없음 | 버텍스 컬러만 있는 경우 많음 |

---

## trimesh 텍스처 확인 방법 (정확한 버전)

```python
import trimesh

def has_texture(mesh: trimesh.Trimesh) -> bool:
    """trimesh 메시에 텍스처가 있는지 정확히 확인"""
    return (
        isinstance(mesh.visual, trimesh.visual.TextureVisuals) and
        hasattr(mesh.visual, 'uv') and
        mesh.visual.uv is not None and
        hasattr(mesh.visual.material, 'baseColorTexture') and
        mesh.visual.material.baseColorTexture is not None
    )

def has_vertex_colors(mesh: trimesh.Trimesh) -> bool:
    """trimesh 메시에 버텍스 컬러가 있는지 확인"""
    return (
        isinstance(mesh.visual, trimesh.visual.ColorVisuals) and
        mesh.visual.vertex_colors is not None
    )

# 사용 예시
mesh = trimesh.load('model.glb').to_geometry()
if has_texture(mesh):
    print("텍스처 렌더링 필요")
    tex_img = mesh.visual.material.baseColorTexture
    uv = mesh.visual.uv
elif has_vertex_colors(mesh):
    print("버텍스 컬러 렌더링 필요")
    colors = mesh.visual.vertex_colors
else:
    print("텍스처/컬러 없음 - 기본 머티리얼 사용")
```
