import sys
import os
import warnings

# Suppress FutureWarning from pynvml (PyTorch imports it internally)
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)

# Initialize CUDA before PySide6 to avoid OpenGL/CUDA conflicts
import torch
if torch.cuda.is_available():
    torch.cuda.init()
    print(f"CUDA initialized: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application metadata for taskbar/dock integration
    app.setApplicationName("ASPIR")
    app.setApplicationDisplayName("ASPIR")
    app.setDesktopFileName("aspir")

    # Set application icon (appears in taskbar)
    assets_dir = os.path.join(os.path.dirname(__file__), '..', 'assets')
    icon_path = os.path.join(assets_dir, 'icon_app.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    ventana = MainWindow()
    ventana.show()
    sys.exit(app.exec())
