"""Zoomable image view widget with mouse-wheel zoom and pan support."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)


class ZoomableImageView(QGraphicsView):
    """
    A QGraphicsView that supports mouse wheel zoom and pan.

    Features:
    - Initial fit to window
    - Zoom in/out with mouse wheel
    - Pan by dragging with mouse
    - Emits zoom level changes
    """

    zoom_changed = Signal(float)  # Emits zoom percentage (100 = 100%)

    MIN_ZOOM = 0.1   # 10%
    MAX_ZOOM = 10.0  # 1000%
    ZOOM_STEP = 1.15  # 15% per wheel step

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Setup scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Setup view properties
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setBackgroundBrush(Qt.white)
        self.setFrameShape(QFrame.NoFrame)

        # State
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_level = 1.0
        self._fit_zoom = 1.0  # Zoom level that fits the image to view

    def set_pixmap(self, pixmap: QPixmap):
        """Set the image to display and fit it to the view."""
        # Clear previous content
        self._scene.clear()

        # Add new pixmap
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)

        # Set scene rect to image size
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        # Fit to view
        self.fit_to_view()

    def fit_to_view(self):
        """Fit the image to the current view size."""
        if self._pixmap_item is None:
            return

        # Reset any existing transform
        self.resetTransform()

        # Calculate scale to fit
        view_rect = self.viewport().rect()
        scene_rect = self._scene.sceneRect()

        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            return

        x_ratio = view_rect.width() / scene_rect.width()
        y_ratio = view_rect.height() / scene_rect.height()

        # Use the smaller ratio to ensure entire image fits
        self._fit_zoom = min(x_ratio, y_ratio) * 0.95  # 95% to leave small margin
        self._zoom_level = self._fit_zoom

        self.scale(self._fit_zoom, self._fit_zoom)
        self.centerOn(self._pixmap_item)

        self.zoom_changed.emit(self._zoom_level * 100)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self._pixmap_item is None:
            return

        # Get zoom direction
        if event.angleDelta().y() > 0:
            # Zoom in
            factor = self.ZOOM_STEP
        else:
            # Zoom out
            factor = 1.0 / self.ZOOM_STEP

        # Calculate new zoom level
        new_zoom = self._zoom_level * factor

        # Clamp to min/max
        if new_zoom < self.MIN_ZOOM:
            factor = self.MIN_ZOOM / self._zoom_level
            new_zoom = self.MIN_ZOOM
        elif new_zoom > self.MAX_ZOOM:
            factor = self.MAX_ZOOM / self._zoom_level
            new_zoom = self.MAX_ZOOM

        # Apply zoom
        self._zoom_level = new_zoom
        self.scale(factor, factor)

        self.zoom_changed.emit(self._zoom_level * 100)

    def reset_zoom(self):
        """Reset zoom to fit the image in view."""
        self.fit_to_view()

    def zoom_in(self):
        """Zoom in by one step."""
        if self._pixmap_item is None:
            return

        factor = self.ZOOM_STEP
        new_zoom = self._zoom_level * factor

        if new_zoom <= self.MAX_ZOOM:
            self._zoom_level = new_zoom
            self.scale(factor, factor)
            self.zoom_changed.emit(self._zoom_level * 100)

    def zoom_out(self):
        """Zoom out by one step."""
        if self._pixmap_item is None:
            return

        factor = 1.0 / self.ZOOM_STEP
        new_zoom = self._zoom_level * factor

        if new_zoom >= self.MIN_ZOOM:
            self._zoom_level = new_zoom
            self.scale(factor, factor)
            self.zoom_changed.emit(self._zoom_level * 100)

    def resizeEvent(self, event):
        """Handle resize to maintain fit if at fit zoom level."""
        super().resizeEvent(event)

        # If we're close to the fit zoom, re-fit on resize
        if self._pixmap_item is not None and abs(self._zoom_level - self._fit_zoom) < 0.01:
            self.fit_to_view()
