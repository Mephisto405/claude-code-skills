# Python Rendering Plugin for Claude Code

Python 3D 렌더링 라이브러리(nvdiffrast, PyTorch3D, pyrender, trimesh, Open3D) 사용을 위한 Claude Code 플러그인.

## Installation

```bash
/plugin install python-rendering@Mephisto405/python-rendering-cc-skill
```

또는 로컬 설치:

```bash
git clone https://github.com/Mephisto405/python-rendering-cc-skill.git
claude --plugin-dir ./python-rendering-guide
```

## Usage

설치 후 Claude Code가 렌더링 관련 작업 시 자동으로 스킬을 참조합니다.

직접 호출:
```
/python-rendering
```

## Features

- **좌표계 컨벤션**: OpenGL, Blender, PyTorch3D, nvdiffrast 등 프레임워크별 축 컨벤션
- **렌더링 템플릿**: pyrender, trimesh 검증된 코드 템플릿
- **텍스처 렌더링**: GLB/GLTF 텍스처 처리 방법
- **환경 설정**: Python 버전별 호환성, Windows/Linux 설정
- **문제 해결**: 일반적인 렌더링 오류 해결 방법

## Contents

```
python-rendering-guide/
├── .claude-plugin/
│   └── plugin.json
├── skills/
│   └── python-rendering/
│       ├── SKILL.md              # 메인 스킬
│       └── axis-conventions.md   # 축 컨벤션 상세
└── README.md
```

## License

MIT
