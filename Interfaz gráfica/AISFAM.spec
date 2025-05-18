# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['AISFAM.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\mario.adame\\.spyder-py3\\SECMF\\sistema_experto\\dnn_bm_gkv.keras', '.'), ('C:\\Users\\mario.adame\\.spyder-py3\\SECMF\\sistema_experto\\lstm_bm_gkv.keras', '.'), ('C:\\Users\\mario.adame\\.spyder-py3\\SECMF\\sistema_experto\\mlp_bm_gkv.keras', '.')],
    hiddenimports=[],
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
    name='AISFAM',
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
