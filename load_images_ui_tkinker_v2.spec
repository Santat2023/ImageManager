# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['load_images_ui_tkinker_v2.py'],
    pathex=['F:/MyProjects/ImageManager/venv/Lib/site-packages'],
    binaries=[],
    datas=[
        ('F:/MyProjects/ImageManager/venv/Lib/site-packages/clip/bpe_simple_vocab_16e6.txt.gz', 'clip')
    ],
    hiddenimports=['chromadb.telemetry.product.posthog', 'chromadb.api.fastapi', 'posthog'],
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
    a.binaries,
    a.datas,
    [],
    name='load_images_ui_tkinker_v2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
