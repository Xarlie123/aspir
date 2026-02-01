"""
Container for Pipeline mode using existing UIPipelineHandler content.
"""
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QSizePolicy, QScrollArea, QFrame
)


class PipelineContainer(QWidget):
    """
    Container for pipeline mode.
    Wraps existing UIPipelineHandler content with minimal changes.
    """

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self._handler = None
        self._setup_ui()

        self.logger.debug("PipelineContainer initialized")

    def _setup_ui(self):
        """Setup the container UI."""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)

        # Content will be added when handler is set

    def set_handler(self, handler):
        """
        Set the pipeline handler and embed its content.

        Args:
            handler: UIPipelineHandler instance
        """
        self._handler = handler

        # Get content widget from handler if available
        if hasattr(handler, 'get_content_widget'):
            content = handler.get_content_widget()
            if content:
                self.layout.addWidget(content)
                self.logger.debug("Added pipeline handler content widget")
                return

        # Fallback: wrap existing UI elements
        # The pipeline handler typically uses UI elements directly from ui_main_window
        # We'll handle this in main_window.py by reparenting widgets

        self.logger.debug("Pipeline handler set (content managed externally)")

    def add_content(self, widget: QWidget):
        """Add a content widget directly."""
        self.layout.addWidget(widget)

    def on_visible(self):
        """Called when this container becomes visible."""
        if self._handler and hasattr(self._handler, 'on_tab_visible'):
            self._handler.on_tab_visible()
