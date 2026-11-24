import sys

# 1. 指向 weapon2glb 的 src 绝对路径
scripts_dir = r"E:\3dmodels\blenders\weapon2glb\src"
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# 2. 清理模块缓存
for m in list(sys.modules.keys()):
    if m == "main" or m.startswith("step") or m.startswith("utils"):
        del sys.modules[m]

# 3. 零配置直接运行（自动按当前 .blend 文件名导出同名 .glb）
import main

main.run()
