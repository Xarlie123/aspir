"""
Log Manager - Configures application logging with file rotation.
"""
import logging
import os
import glob
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List
import yaml

# Get the project root directory (where the log folder is)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "log"
LOG_CONFIG_FILE = PROJECT_ROOT / "settings" / "log_config.yaml"

# Default configuration
DEFAULT_CONFIG = {
    "log_level": "INFO",
    "max_log_files": 10,
    "max_file_size_mb": 5,
    "log_to_console": True,
    "log_to_file": True,
}

# Log level mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LogManager:
    """
    Manages application logging configuration.

    Features:
    - Rotating file handler (keeps last N log files)
    - Configurable log levels
    - Console and file output
    - Configuration persistence
    """

    _instance: Optional['LogManager'] = None
    _initialized: bool = False

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if LogManager._initialized:
            return

        self._config = DEFAULT_CONFIG.copy()
        self._file_handler: Optional[logging.Handler] = None
        self._console_handler: Optional[logging.Handler] = None
        self._current_log_file: Optional[str] = None
        self._root_logger = logging.getLogger()

        # Load configuration
        self._load_config()

        LogManager._initialized = True

    def _load_config(self):
        """Load configuration from file."""
        if LOG_CONFIG_FILE.exists():
            try:
                with open(LOG_CONFIG_FILE, 'r') as f:
                    loaded = yaml.safe_load(f) or {}
                    self._config.update(loaded)
            except Exception as e:
                print(f"Warning: Could not load log config: {e}")

    def save_config(self):
        """Save configuration to file."""
        try:
            LOG_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_CONFIG_FILE, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
        except Exception as e:
            print(f"Warning: Could not save log config: {e}")

    def setup_logging(self, app_name: str = "ASPIR") -> logging.Logger:
        """
        Setup the logging system.

        Args:
            app_name: Application name for the logger

        Returns:
            The configured root logger
        """
        # Ensure log directory exists
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Clean up old log files
        self._cleanup_old_logs()

        # Create new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_log_file = str(LOG_DIR / f"{app_name}_{timestamp}.log")

        # Get log level
        level = LOG_LEVELS.get(self._config["log_level"], logging.INFO)

        # Configure root logger
        self._root_logger.setLevel(logging.DEBUG)  # Capture all, filter at handlers

        # Remove existing handlers
        for handler in self._root_logger.handlers[:]:
            self._root_logger.removeHandler(handler)

        # File handler
        if self._config["log_to_file"]:
            max_bytes = self._config["max_file_size_mb"] * 1024 * 1024
            self._file_handler = RotatingFileHandler(
                self._current_log_file,
                maxBytes=max_bytes,
                backupCount=3,
                encoding='utf-8'
            )
            self._file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self._file_handler.setFormatter(file_formatter)
            self._root_logger.addHandler(self._file_handler)

        # Console handler (with colors if available)
        if self._config["log_to_console"]:
            self._console_handler = logging.StreamHandler()
            self._console_handler.setLevel(level)

            # Use colored formatter if available
            try:
                from ui.utils.color_log import ColorFormatter
                console_formatter = ColorFormatter()
            except ImportError:
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )

            self._console_handler.setFormatter(console_formatter)
            self._root_logger.addHandler(self._console_handler)

        # Return logger for the app
        logger = logging.getLogger(app_name)
        logger.info(f"Logging initialized - Level: {self._config['log_level']}, File: {self._current_log_file}")

        return logger

    def _cleanup_old_logs(self):
        """Remove old log files, keeping only the last N."""
        max_files = self._config["max_log_files"]

        # Get all log files sorted by modification time
        log_pattern = str(LOG_DIR / "*.log")
        log_files = sorted(glob.glob(log_pattern), key=os.path.getmtime, reverse=True)

        # Remove excess files (keep max_files - 1 to make room for new one)
        for old_file in log_files[max_files - 1:]:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"Warning: Could not remove old log file {old_file}: {e}")

    def get_log_files(self) -> List[str]:
        """Get list of log files sorted by date (newest first)."""
        log_pattern = str(LOG_DIR / "*.log")
        return sorted(glob.glob(log_pattern), key=os.path.getmtime, reverse=True)

    def get_current_log_file(self) -> Optional[str]:
        """Get the current log file path."""
        return self._current_log_file

    def set_log_level(self, level: str):
        """Change the log level dynamically."""
        if level in LOG_LEVELS:
            self._config["log_level"] = level
            log_level = LOG_LEVELS[level]

            if self._file_handler:
                self._file_handler.setLevel(log_level)
            if self._console_handler:
                self._console_handler.setLevel(log_level)

            logging.getLogger().info(f"Log level changed to {level}")

    def get_log_level(self) -> str:
        """Get current log level."""
        return self._config["log_level"]

    def set_max_log_files(self, count: int):
        """Set maximum number of log files to keep."""
        self._config["max_log_files"] = max(1, count)

    def get_max_log_files(self) -> int:
        """Get maximum number of log files."""
        return self._config["max_log_files"]

    def set_log_to_file(self, enabled: bool):
        """Enable/disable file logging."""
        self._config["log_to_file"] = enabled

    def get_log_to_file(self) -> bool:
        """Check if file logging is enabled."""
        return self._config["log_to_file"]

    def set_log_to_console(self, enabled: bool):
        """Enable/disable console logging."""
        self._config["log_to_console"] = enabled

    def get_log_to_console(self) -> bool:
        """Check if console logging is enabled."""
        return self._config["log_to_console"]

    def get_config(self) -> dict:
        """Get current configuration."""
        return self._config.copy()

    def read_current_log(self, max_lines: int = 1000) -> str:
        """Read the current log file content."""
        if not self._current_log_file or not os.path.exists(self._current_log_file):
            return "No log file available."

        try:
            with open(self._current_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    lines = lines[-max_lines:]
                return ''.join(lines)
        except Exception as e:
            return f"Error reading log file: {e}"

    def read_log_file(self, filepath: str, max_lines: int = 1000) -> str:
        """Read a specific log file."""
        if not os.path.exists(filepath):
            return f"Log file not found: {filepath}"

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > max_lines:
                    lines = lines[-max_lines:]
                return ''.join(lines)
        except Exception as e:
            return f"Error reading log file: {e}"


# Global instance
_log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get the global LogManager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager
