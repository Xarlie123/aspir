# File: ui/custom_widgets/common/button_styles.py
"""
Common button styles for consistent UI across the application.
"""

# Green - Primary action buttons (Generate, Run, Create)
BUTTON_STYLE_GREEN = """
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        font-size: 13px;
        padding: 8px 16px;
        min-height: 32px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QPushButton:pressed {
        background-color: #3d8b40;
    }
    QPushButton:disabled {
        background-color: #ccc;
        color: #666;
    }
"""

# Blue - Secondary action buttons (Select, Browse, Report)
BUTTON_STYLE_BLUE = """
    QPushButton {
        background-color: #0078d7;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        font-size: 13px;
        padding: 8px 16px;
        min-height: 32px;
    }
    QPushButton:hover {
        background-color: #005a9e;
    }
    QPushButton:pressed {
        background-color: #004275;
    }
    QPushButton:disabled {
        background-color: #ccc;
        color: #666;
    }
"""

# Orange - Tertiary actions (Add, Profile)
BUTTON_STYLE_ORANGE = """
    QPushButton {
        background-color: #FF9800;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        font-size: 13px;
        padding: 8px 16px;
        min-height: 32px;
    }
    QPushButton:hover {
        background-color: #F57C00;
    }
    QPushButton:pressed {
        background-color: #E65100;
    }
    QPushButton:disabled {
        background-color: #ccc;
        color: #666;
    }
"""

# Red - Destructive actions (Remove, Delete)
BUTTON_STYLE_RED = """
    QPushButton {
        background-color: #f44336;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        font-size: 13px;
        padding: 8px 16px;
        min-height: 32px;
    }
    QPushButton:hover {
        background-color: #d32f2f;
    }
    QPushButton:pressed {
        background-color: #b71c1c;
    }
    QPushButton:disabled {
        background-color: #ccc;
        color: #666;
    }
"""

# Gray - Utility buttons (Detect, Export)
BUTTON_STYLE_GRAY = """
    QPushButton {
        background-color: #607D8B;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        font-size: 13px;
        padding: 8px 16px;
        min-height: 32px;
    }
    QPushButton:hover {
        background-color: #546E7A;
    }
    QPushButton:pressed {
        background-color: #455A64;
    }
    QPushButton:disabled {
        background-color: #ccc;
        color: #666;
    }
"""


def apply_button_style(button, style: str):
    """
    Apply a style to a QPushButton.

    Args:
        button: QPushButton instance
        style: One of BUTTON_STYLE_GREEN, BUTTON_STYLE_BLUE, etc.
    """
    button.setStyleSheet(style)
    button.setMinimumHeight(36)
