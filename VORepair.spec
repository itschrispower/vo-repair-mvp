# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/gui_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
    ],
    hiddenimports=[
        'librosa',
        'numpy',
        'scipy',
        'soundfile',
        'numba',
        'joblib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VORepair',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VORepair',
)

app = BUNDLE(
    coll,
    name='VORepair.app',
    icon=None,
)
