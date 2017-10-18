# -*- mode: python -*-

block_cipher = None


a = Analysis(['Gui.py'],
             pathex=['C:\\Users\\gilad\\ownCloud\\Shared\\gui\\Python GUi\\v1.09'],
             binaries=[],
             datas=[],
             hiddenimports=['sklearn.neighbors.typedefs'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='Gui',
          debug=False,
          strip=False,
          upx=True,
          console=True , icon='favicon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='Gui')
