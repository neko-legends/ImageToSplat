# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


project_dir = Path(SPECPATH)

datas = [
    (str(project_dir / "static"), "static"),
    (str(project_dir / "README.md"), "."),
    (str(project_dir / "LICENSE"), "."),
]
binaries = []
hiddenimports = []

for package in [
    "gradio",
    "gradio_client",
    "hf_gradio",
    "safehttpx",
    "groovy",
    "fastapi",
    "starlette",
    "uvicorn",
]:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

hiddenimports += collect_submodules("uvicorn.protocols")
hiddenimports += collect_submodules("uvicorn.lifespan")
hiddenimports += collect_submodules("uvicorn.loops")
hiddenimports += [
    "model",
    "triposplat",
    "torch",
    "torchvision",
    "safetensors",
    "safetensors.torch",
    "PIL",
    "PIL.Image",
    "huggingface_hub",
]


a = Analysis(
    ["run_gradio.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "notebook",
        "matplotlib",
        "pytest",
        "boto3",
        "botocore",
        "cv2",
        "deepspeed",
        "imageio_ffmpeg",
        "llvmlite",
        "moviepy",
        "numba",
        "pyarrow",
        "tensorboard",
        "tensorflow",
        "torchaudio",
        "triton",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NekoSplatForge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_dir / "static" / "app-icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NekoSplatForge",
)
