"""
Resource Monitor Widget - Shows CPU, RAM, and GPU usage bars.
"""
import logging
import os
from typing import Optional

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QProgressBar, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

# Try to import psutil for CPU/RAM monitoring
try:
    import psutil
    HAS_PSUTIL = True
    # Seed the CPU percent measurement (first call always returns 0)
    psutil.cpu_percent(interval=None)
except ImportError:
    HAS_PSUTIL = False

# Try to import torch for GPU monitoring
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False


def _is_jetson() -> bool:
    """Cheap Jetson detector: the L4T release file only exists on Tegra."""
    return os.path.isfile("/etc/nv_tegra_release")


# On Jetson NVML is not available. ``jtop`` (from the ``jetson-stats``
# package, shipped by the ``[jetson]`` extra) talks to the tegrastats-style
# daemon and exposes GPU load + RAM, which on Tegra is shared between CPU
# and GPU — effectively the VRAM figure we care about.
try:
    from jtop import jtop  # noqa: F401
    HAS_JTOP = True
except ImportError:
    HAS_JTOP = False


def _jtop_gpu_load(jt) -> Optional[float]:
    """Best-effort GPU load reader across jtop API revisions.

    jtop changed its accessor layout several times; try the two stable
    ones and fall back to ``None`` if nothing matches.
    """
    # jtop >= 4.x  — ``jt.gpu`` is a dict keyed by chip (ga10b, orin, ...)
    gpu = getattr(jt, "gpu", None)
    if isinstance(gpu, dict):
        for entry in gpu.values():
            status = entry.get("status") if isinstance(entry, dict) else None
            if isinstance(status, dict) and "load" in status:
                return float(status["load"])
            if isinstance(entry, dict) and "load" in entry:
                return float(entry["load"])
    # jtop 3.x fallback — ``jt.stats['GPU']`` (percent as float/int)
    stats = getattr(jt, "stats", None)
    if isinstance(stats, dict) and "GPU" in stats:
        return float(stats["GPU"])
    return None


def _jtop_mem_percent(jt) -> Optional[float]:
    """Percent of RAM in use. On Tegra the GPU shares system RAM, so this
    is the closest thing to "VRAM used"."""
    mem = getattr(jt, "memory", None)
    if isinstance(mem, dict):
        ram = mem.get("RAM", {})
        tot = ram.get("tot") or ram.get("total")
        used = ram.get("used")
        if tot and used is not None:
            return 100.0 * used / tot
    return None


class ResourceMonitorWidget(QWidget):
    """
    Compact widget showing CPU, RAM, and GPU usage with progress bars.
    Updates automatically at a configurable interval.
    """

    def __init__(self, update_interval_ms: int = 1000, parent=None, logger=None):
        super().__init__(parent)

        if logger:
            self.logger = logger.getChild("ResourceMonitor")
        else:
            self.logger = logging.getLogger("ResourceMonitor")

        self._update_interval = update_interval_ms
        self._is_monitoring = False

        # Jetson: open a persistent jtop handle so each timer tick is just a
        # dict lookup instead of relaunching the daemon client. Falls back
        # silently if the jtop.service daemon isn't reachable.
        self._jtop = None
        if HAS_JTOP and _is_jetson():
            try:
                from jtop import jtop
                self._jtop = jtop()
                self._jtop.start()
                self.logger.debug("jtop started for Jetson GPU monitoring")
            except Exception as e:
                self.logger.warning(
                    "Could not start jtop (%s); falling back to torch memory only. "
                    "On Jetson install and start the jtop service:\n"
                    "  sudo pip install -U jetson-stats\n"
                    "  sudo systemctl enable --now jtop", e,
                )
                self._jtop = None

        self._setup_ui()
        self._setup_timer()

        # Initial update and start monitoring automatically
        self._update_resources()
        self.start_monitoring()

    def _setup_ui(self):
        """Setup the compact resource monitor UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 2, 5, 2)
        main_layout.setSpacing(10)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #ccc;")
        main_layout.addWidget(sep)

        # Common style for progress bars
        bar_style = """
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #f0f0f0;
                text-align: center;
                font-size: 9px;
            }
            QProgressBar::chunk {
                border-radius: 2px;
            }
        """

        # CPU usage
        cpu_layout = QVBoxLayout()
        cpu_layout.setSpacing(1)
        cpu_layout.setContentsMargins(0, 0, 0, 0)

        cpu_label = QLabel("CPU")
        cpu_label.setStyleSheet("font-size: 9px; color: #666;")
        cpu_label.setAlignment(Qt.AlignCenter)
        cpu_layout.addWidget(cpu_label)

        self.cpu_bar = QProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_bar.setFixedSize(60, 14)
        self.cpu_bar.setFormat("%p%")
        self.cpu_bar.setStyleSheet(bar_style + """
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        cpu_layout.addWidget(self.cpu_bar)
        main_layout.addLayout(cpu_layout)

        # RAM usage
        ram_layout = QVBoxLayout()
        ram_layout.setSpacing(1)
        ram_layout.setContentsMargins(0, 0, 0, 0)

        ram_label = QLabel("RAM")
        ram_label.setStyleSheet("font-size: 9px; color: #666;")
        ram_label.setAlignment(Qt.AlignCenter)
        ram_layout.addWidget(ram_label)

        self.ram_bar = QProgressBar()
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setFixedSize(60, 14)
        self.ram_bar.setFormat("%p%")
        self.ram_bar.setStyleSheet(bar_style + """
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        ram_layout.addWidget(self.ram_bar)
        main_layout.addLayout(ram_layout)

        # GPU usage (only if available)
        if HAS_TORCH:
            gpu_layout = QVBoxLayout()
            gpu_layout.setSpacing(1)
            gpu_layout.setContentsMargins(0, 0, 0, 0)

            gpu_label = QLabel("GPU")
            gpu_label.setStyleSheet("font-size: 9px; color: #666;")
            gpu_label.setAlignment(Qt.AlignCenter)
            gpu_layout.addWidget(gpu_label)

            self.gpu_bar = QProgressBar()
            self.gpu_bar.setRange(0, 100)
            self.gpu_bar.setFixedSize(60, 14)
            self.gpu_bar.setFormat("%p%")
            self.gpu_bar.setStyleSheet(bar_style + """
                QProgressBar::chunk {
                    background-color: #FF9800;
                }
            """)
            gpu_layout.addWidget(self.gpu_bar)
            main_layout.addLayout(gpu_layout)

            # GPU Memory
            gpu_mem_layout = QVBoxLayout()
            gpu_mem_layout.setSpacing(1)
            gpu_mem_layout.setContentsMargins(0, 0, 0, 0)

            gpu_mem_label = QLabel("VRAM")
            gpu_mem_label.setStyleSheet("font-size: 9px; color: #666;")
            gpu_mem_label.setAlignment(Qt.AlignCenter)
            gpu_mem_layout.addWidget(gpu_mem_label)

            self.gpu_mem_bar = QProgressBar()
            self.gpu_mem_bar.setRange(0, 100)
            self.gpu_mem_bar.setFixedSize(60, 14)
            self.gpu_mem_bar.setFormat("%p%")
            self.gpu_mem_bar.setStyleSheet(bar_style + """
                QProgressBar::chunk {
                    background-color: #9C27B0;
                }
            """)
            gpu_mem_layout.addWidget(self.gpu_mem_bar)
            main_layout.addLayout(gpu_mem_layout)
        else:
            self.gpu_bar = None
            self.gpu_mem_bar = None

    def _setup_timer(self):
        """Setup the update timer."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_resources)

    def start_monitoring(self):
        """Start automatic resource monitoring."""
        if not self._is_monitoring:
            self._is_monitoring = True
            self._timer.start(self._update_interval)
            self.logger.debug("Resource monitoring started")

    def stop_monitoring(self):
        """Stop automatic resource monitoring."""
        if self._is_monitoring:
            self._is_monitoring = False
            self._timer.stop()
            self.logger.debug("Resource monitoring stopped")

    def closeEvent(self, event):
        """Tear down the jtop background thread cleanly on shutdown."""
        try:
            if self._jtop is not None:
                self._jtop.close()
        except Exception:
            pass
        super().closeEvent(event)

    def _update_resources(self):
        """Update all resource bars."""
        try:
            # CPU usage
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_bar.setValue(int(cpu_percent))
                self._update_bar_color(self.cpu_bar, cpu_percent, "#2196F3")
            else:
                self.cpu_bar.setValue(0)
                self.cpu_bar.setFormat("N/A")

            # RAM usage
            if HAS_PSUTIL:
                ram = psutil.virtual_memory()
                ram_percent = ram.percent
                self.ram_bar.setValue(int(ram_percent))
                self._update_bar_color(self.ram_bar, ram_percent, "#4CAF50")
            else:
                self.ram_bar.setValue(0)
                self.ram_bar.setFormat("N/A")

            # GPU usage and memory
            if HAS_TORCH and self.gpu_bar is not None:
                updated = False

                # 1) Jetson path — use the jtop daemon.
                if self._jtop is not None and self._jtop.ok():
                    try:
                        gpu_percent = _jtop_gpu_load(self._jtop)
                        mem_percent = _jtop_mem_percent(self._jtop)
                        if gpu_percent is not None:
                            self.gpu_bar.setValue(int(gpu_percent))
                            self.gpu_bar.setFormat("%p%")
                            self._update_bar_color(self.gpu_bar, gpu_percent, "#FF9800")
                        if mem_percent is not None:
                            self.gpu_mem_bar.setValue(int(mem_percent))
                            self.gpu_mem_bar.setFormat("%p%")
                            self._update_bar_color(self.gpu_mem_bar, mem_percent, "#9C27B0")
                        updated = (gpu_percent is not None) or (mem_percent is not None)
                    except Exception as e:
                        self.logger.debug("jtop read failed: %s", e)

                # 2) Desktop/server path — NVML.
                if not updated:
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_percent = util.gpu
                        self.gpu_bar.setValue(int(gpu_percent))
                        self.gpu_bar.setFormat("%p%")
                        self._update_bar_color(self.gpu_bar, gpu_percent, "#FF9800")

                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_mem_percent = (mem_info.used / mem_info.total) * 100
                        self.gpu_mem_bar.setValue(int(gpu_mem_percent))
                        self.gpu_mem_bar.setFormat("%p%")
                        self._update_bar_color(self.gpu_mem_bar, gpu_mem_percent, "#9C27B0")
                        pynvml.nvmlShutdown()
                        updated = True
                    except Exception:
                        pass

                # 3) Last-resort fallback: torch-only VRAM (no utilisation).
                if not updated:
                    try:
                        allocated = torch.cuda.memory_allocated(0)
                        total = torch.cuda.get_device_properties(0).total_memory
                        gpu_mem_percent = (allocated / total) * 100
                        self.gpu_mem_bar.setValue(int(gpu_mem_percent))
                        self._update_bar_color(self.gpu_mem_bar, gpu_mem_percent, "#9C27B0")
                        self.gpu_bar.setValue(0)
                        self.gpu_bar.setFormat("--")
                    except Exception:
                        self.gpu_bar.setValue(0)
                        self.gpu_mem_bar.setValue(0)

        except Exception as e:
            self.logger.warning("Failed to update resources: %s", e)

    def _update_bar_color(self, bar: QProgressBar, value: float, base_color: str):
        """Update bar color based on usage level (green -> yellow -> red)."""
        if value >= 90:
            color = "#F44336"  # Red - critical
        elif value >= 75:
            color = "#FF9800"  # Orange - warning
        elif value >= 50:
            color = "#FFC107"  # Yellow - moderate
        else:
            color = base_color  # Normal

        bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: #f0f0f0;
                text-align: center;
                font-size: 9px;
            }}
            QProgressBar::chunk {{
                border-radius: 2px;
                background-color: {color};
            }}
        """)

    def set_update_interval(self, interval_ms: int):
        """Set the update interval in milliseconds."""
        self._update_interval = interval_ms
        if self._is_monitoring:
            self._timer.setInterval(interval_ms)
