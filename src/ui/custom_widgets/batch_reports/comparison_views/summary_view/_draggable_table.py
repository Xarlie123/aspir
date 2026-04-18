"""QTableWidget subclass with row drag-and-drop reordering support."""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QAbstractItemView, QTableWidget


class DraggableRowTableWidget(QTableWidget):
    """
    QTableWidget subclass that supports dragging entire rows.

    Uses custom drag-drop handling to avoid Qt's InternalMove item manipulation:
    the widget emits ``rows_reordered(from_row, to_row)`` and lets the caller
    refresh the table from the updated data model.
    """

    rows_reordered = pyqtSignal(int, int)  # from_row, to_row

    def __init__(self, parent=None):
        super().__init__(parent)

        # Enable row selection mode
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

        # Use DragDrop mode but ignore the default drop action
        # This gives us the visual feedback without Qt moving items
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(Qt.IgnoreAction)
        self.setDropIndicatorShown(True)

        # Track the row being dragged
        self._drag_start_row = -1

    def startDrag(self, supportedActions):
        """Remember which row we're dragging before the drag starts."""
        self._drag_start_row = self.currentRow()
        super().startDrag(supportedActions)

    def dropEvent(self, event):
        """Handle drop event to reorder rows."""
        # Only accept drops from this widget
        if event.source() != self:
            event.ignore()
            return

        # Get the row being dropped on
        drop_row = self.rowAt(event.pos().y())
        if drop_row == -1:
            # Dropped below last row
            drop_row = self.rowCount() - 1

        # Use the saved drag start row
        drag_row = self._drag_start_row
        if drag_row == -1 or drag_row == drop_row:
            event.ignore()
            return

        # Emit signal for external handling (data reordering)
        # The handler will refresh the table completely
        self.rows_reordered.emit(drag_row, drop_row)

        # Ignore the event to prevent Qt from manipulating items
        event.ignore()

        # Reset drag state
        self._drag_start_row = -1

    def dragEnterEvent(self, event):
        """Accept drag events from this widget."""
        if event.source() == self:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Show drop indicator during drag."""
        if event.source() == self:
            event.accept()
        else:
            event.ignore()
