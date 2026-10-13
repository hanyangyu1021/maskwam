#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from importlib.util import find_spec

# 设置Qt环境变量
pyqt_spec = find_spec("PyQt5")
if pyqt_spec is None or not pyqt_spec.submodule_search_locations:
    raise RuntimeError("PyQt5 is not installed in the current Python environment")

pyqt_dir = Path(next(iter(pyqt_spec.submodule_search_locations)))
qt_plugin_path = pyqt_dir / "Qt5" / "plugins"
if qt_plugin_path.is_dir():
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(qt_plugin_path)

# 尝试不同的Qt平台插件
print("Available Qt platform plugins:")
print("1. xcb (X11)")
print("2. wayland (Wayland)")
print("3. linuxfb (Linux Framebuffer)")
print("4. offscreen (Offscreen rendering)")

# 尝试使用offscreen模式
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# 设置Qt库路径
qt_lib_path = pyqt_dir / "Qt5" / "lib"
conda_lib_path = Path(sys.prefix) / "lib"

# 清理LD_LIBRARY_PATH，移除所有CoppeliaSim路径
if 'LD_LIBRARY_PATH' in os.environ:
    current_ld_path = os.environ['LD_LIBRARY_PATH']
    paths = current_ld_path.split(':')
    filtered_paths = [p for p in paths if 'CoppeliaSim' not in p]
    os.environ['LD_LIBRARY_PATH'] = ':'.join(filtered_paths)

# 添加conda路径到LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = str(conda_lib_path) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '')}")
print(f"QT_QPA_PLATFORM_PLUGIN_PATH: {os.environ.get('QT_QPA_PLATFORM_PLUGIN_PATH', '')}")
print(f"QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM', '')}")

try:
    # 导入PyQt5
    from PyQt5.QtWidgets import QApplication, QLabel, QWidget
    from PyQt5.QtCore import Qt

    print("PyQt5 imported successfully!")

    # 创建简单的Qt应用
    app = QApplication(sys.argv)
    print("QApplication created successfully!")

    # 创建窗口
    window = QWidget()
    window.setWindowTitle('Qt Test')
    window.setGeometry(100, 100, 300, 200)

    # 添加标签
    label = QLabel('Qt is working!', window)
    label.setAlignment(Qt.AlignCenter)
    label.setGeometry(50, 50, 200, 100)

    # 显示窗口
    window.show()

    print("Qt application started successfully!")
    print("If you can see a window with 'Qt is working!' text, then Qt is working correctly.")

    # 运行应用
    sys.exit(app.exec_())

except Exception as e:
    print(f"Error: {e}")
    print("Qt application failed to start.")
