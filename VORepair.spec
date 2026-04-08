# -*- mode: python ; coding: utf-8 -*-
#
# VORepair.spec — PyInstaller build spec for VO Repair
#
# Build command (run from repo root with PyInstaller installed):
#   pyinstaller VORepair.spec
#
# Output: dist/VORepair.app  (macOS .app bundle)
#

block_cipher = None

a = Analysis(
    ['run_app.py'],          # entry point — do NOT change to src/gui_app.py
    pathex=['src'],          # so engine/matcher/run_job imports resolve
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
        'aaf2',
        'run_job',
        'engine',
        'matcher',
        'aaf_io',
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
    bundle_identifier='com.vorepair.app',
)
