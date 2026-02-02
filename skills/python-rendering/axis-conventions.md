# 축 컨벤션 상세 가이드

## 왜 모두가 다르게 말하는가?

"Forward"라는 단어가 3가지 의미로 쓰이기 때문:

| 용어 | 의미 | 예시 |
|-----|------|------|
| **Object Forward** | 캐릭터가 앞을 향하는 방향 | Blender: +Y, Unity: +Z |
| **Camera Look** | 카메라가 바라보는 방향 | 대부분 -Z (OpenGL 계열) |
| **NDC Z** | 깊이 증가 방향 | OpenGL: +Z가 멀어지는 방향 |

## 정확한 좌표계 정보

### OpenGL (pyrender, nvdiffrast, Open3D 포함)

```
        +Y (up)
         |
         |
         +------ +X (right)
        /
       /
      +Z (toward viewer)

카메라는 -Z 방향을 바라봄
View Space: 우손 좌표계
NDC (투영 후): 왼손 좌표계! (Z가 뒤집힘)
```

### Blender

```
        +Z (up)
         |
         |
         +------ +X (right)
        /
       /
      +Y (object forward)

오브젝트 Forward: +Y
카메라 Look: 로컬 -Z (카메라 오브젝트 자체의 -Z 축)
```

### PyTorch3D

```
        +Y (up)
         |
         |
         +------ +X (right)
        /
       /
      +Z (toward viewer, NDC 기준)

Camera: 월드에서 -Z 방향을 바라봄
NDC: +Z가 뷰어 쪽 (화면 밖으로 나오는 방향)
look_at_view_transform: dist, elev, azim으로 카메라 위치 지정
```

### Unity / DirectX (왼손 좌표계)

```
        +Y (up)
         |
         |
         +------ +X (right)
         |
         |
        +Z (forward, into screen)

오브젝트 Forward: +Z
카메라 Look: +Z
왼손 좌표계: 엄지(X), 검지(Y), 중지(Z)가 각각 +방향
```

### trimesh

**좌표계를 강제하지 않음!**

- 로드한 파일의 좌표계를 그대로 유지
- OBJ 파일이 Y-up이면 Y-up
- STL 파일이 Z-up이면 Z-up
- Scene 카메라만 OpenGL 스타일 (-Z look)

## 변환 행렬

### Z-up to Y-up (Blender → OpenGL 계열)

```python
import numpy as np

# X축 기준 -90도 회전
Z_UP_TO_Y_UP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
], dtype=np.float32)

vertices_yup = vertices @ Z_UP_TO_Y_UP.T
```

### Y-up to Z-up (OpenGL → Blender)

```python
# X축 기준 +90도 회전
Y_UP_TO_Z_UP = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
], dtype=np.float32)

vertices_zup = vertices @ Y_UP_TO_Z_UP.T
```

### 왼손 ↔ 오른손 좌표계

```python
# Z축 뒤집기 (가장 일반적)
FLIP_Z = np.array([
    [1, 0,  0],
    [0, 1,  0],
    [0, 0, -1]
], dtype=np.float32)

vertices_flipped = vertices @ FLIP_Z.T
# 주의: face winding order도 뒤집어야 할 수 있음!
```

## 일반적인 파일 포맷

| 포맷 | 기본 Up | 참고 |
|------|---------|------|
| OBJ | 관례상 Y-up | 명시적 정의 없음 |
| GLTF/GLB | Y-up | 스펙에 명시 |
| FBX | 설정에 따라 다름 | 내보내기 시 선택 |
| STL | 관례상 Z-up | 명시적 정의 없음 |
| PLY | 관례상 Z-up | 명시적 정의 없음 |
| USD | Y-up | 스펙에 명시 |

## 검증 방법: 축 화살표 렌더링

가장 확실한 방법은 RGB 축 화살표를 렌더링하는 것:

```python
import trimesh
import numpy as np

def create_axis_arrows(length=1.0, radius=0.02):
    """RGB 축 화살표 생성 (R=X, G=Y, B=Z)"""
    arrows = []

    # X축 (빨강)
    x_cyl = trimesh.creation.cylinder(radius=radius, height=length)
    x_cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    x_cyl.apply_translation([length/2, 0, 0])
    x_cyl.visual.vertex_colors = [255, 0, 0, 255]
    arrows.append(x_cyl)

    # Y축 (초록)
    y_cyl = trimesh.creation.cylinder(radius=radius, height=length)
    y_cyl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    y_cyl.apply_translation([0, length/2, 0])
    y_cyl.visual.vertex_colors = [0, 255, 0, 255]
    arrows.append(y_cyl)

    # Z축 (파랑)
    z_cyl = trimesh.creation.cylinder(radius=radius, height=length)
    z_cyl.apply_translation([0, 0, length/2])
    z_cyl.visual.vertex_colors = [0, 0, 255, 255]
    arrows.append(z_cyl)

    return trimesh.util.concatenate(arrows)

# 사용
axes = create_axis_arrows()
axes.export('axes.glb')
```

렌더링 후 확인:
- **빨강(X)**: 오른쪽으로 가는가?
- **초록(Y)**: Y-up 시스템이면 위로, Z-up이면 앞으로
- **파랑(Z)**: Z-up 시스템이면 위로, Y-up이면 카메라 쪽으로

## 실전 팁

1. **새 프로젝트 시작 시**: 먼저 축 화살표를 렌더링해서 좌표계 확인
2. **다른 소스에서 메시 가져올 때**: 항상 좌표계 변환 필요 여부 확인
3. **Blender에서 내보낼 때**: 내보내기 설정에서 "Forward"와 "Up" 축 명시적으로 설정
4. **혼란스러우면**: 무조건 렌더링해서 눈으로 확인!
