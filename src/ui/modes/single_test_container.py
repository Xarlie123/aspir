"""
Container for Single Test mode with stepper/wizard interface.
The stepper widget is managed externally (in main_window) for top bar placement.
Navigation is done by clicking on step indicators.
"""
import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QStackedWidget, QSizePolicy
)
from PyQt5.QtCore import pyqtSignal

from ui.custom_widgets.stepper import StepperWidget, StepState
from ui.modes.stepper_state_manager import StepperStateManager


class SingleTestContainer(QWidget):
    """
    Main container for single test mode.
    Contains: QStackedWidget for step content.
    The StepperWidget is exposed for external placement in a top bar.
    Navigation is done by clicking step indicators.
    """
    # Signal when step changes (for external tracking)
    step_changed = pyqtSignal(int)

    def __init__(self, simulation, logger=None, status_manager=None, parent=None):
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self.simulation = simulation
        self.status_manager = status_manager

        # Handlers will be set by main_window after creation
        self._handlers = {}

        # State manager
        self.state_manager = StepperStateManager(logger=self.logger)

        # Create stepper widget (will be reparented to top bar by main_window)
        self.stepper_widget = StepperWidget(logger=self.logger)

        self._setup_ui()
        self._connect_signals()

        self.logger.debug("SingleTestContainer initialized")

    def _setup_ui(self):
        """Setup the container UI with content stack."""
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Center: Stacked widget for step content
        self.content_stack = QStackedWidget()
        self.content_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.content_stack, 1)  # stretch=1 to fill space

    def _connect_signals(self):
        """Connect internal signals."""
        # Stepper widget clicks
        self.stepper_widget.step_clicked.connect(self._on_step_clicked)

        # State manager
        self.state_manager.state_changed.connect(self._on_state_changed)
        self.state_manager.current_step_changed.connect(self._on_current_step_changed)

    def get_stepper_widget(self):
        """Return the stepper widget for external placement in top bar."""
        return self.stepper_widget

    def set_handlers(self, handlers: dict):
        """
        Set the handlers and add their content widgets to the stack.

        handlers should be a dict with keys:
        - 'dataset': UIDatasetHandler
        - 'masks': UIMaskHandler
        - 'test': UITestMaskHandler
        - 'postprocessor': UIPostprocessorHandler
        - 'reports': UIReportsHandler
        """
        self._handlers = handlers

        # Add content widgets to stack (order must match step indices)
        step_keys = ['dataset', 'masks', 'test', 'postprocessor', 'reports']

        for key in step_keys:
            handler = handlers.get(key)
            if handler and hasattr(handler, 'get_content_widget'):
                widget = handler.get_content_widget()
                if widget:
                    self.content_stack.addWidget(widget)
                    self.logger.debug("Added content widget for step: %s", key)
                else:
                    # Add placeholder
                    placeholder = QWidget()
                    self.content_stack.addWidget(placeholder)
                    self.logger.warning("No content widget for step: %s", key)
            else:
                # Add placeholder
                placeholder = QWidget()
                self.content_stack.addWidget(placeholder)
                self.logger.warning("Handler missing or no get_content_widget for: %s", key)

        # Connect handler completion signals
        self._connect_handler_signals()

        # Set initial state
        self.content_stack.setCurrentIndex(0)

    def _connect_handler_signals(self):
        """Connect handler step_completed and data_changed signals."""
        # Dataset completion
        dataset_handler = self._handlers.get('dataset')
        if dataset_handler:
            if hasattr(dataset_handler, 'step_completed'):
                dataset_handler.step_completed.connect(
                    lambda: self._on_step_completed(0)
                )
            elif hasattr(dataset_handler, 'dataset_updated'):
                # Use existing signal as completion indicator
                dataset_handler.dataset_updated.connect(
                    lambda size: self._on_step_completed(0)
                )

        # Masks completion (also completes Test step automatically)
        masks_handler = self._handlers.get('masks')
        if masks_handler:
            if hasattr(masks_handler, 'step_completed'):
                masks_handler.step_completed.connect(
                    lambda: self._complete_masks_and_test()
                )
            elif hasattr(masks_handler, 'mask_created'):
                # Use existing signal
                masks_handler.mask_created.connect(
                    lambda d, m, a: self._complete_masks_and_test()
                )

        # Postprocessor completion
        pp_handler = self._handlers.get('postprocessor')
        if pp_handler:
            if hasattr(pp_handler, 'step_completed'):
                pp_handler.step_completed.connect(
                    lambda: self._on_step_completed(3)
                )
            elif hasattr(pp_handler, 'training_finished'):
                pp_handler.training_finished.connect(
                    lambda: self._on_step_completed(3)
                )

        # Connect invalidation: when dataset changes, invalidate steps 1-4
        if dataset_handler and hasattr(dataset_handler, 'dataset_updated'):
            dataset_handler.dataset_updated.connect(
                lambda size: self._invalidate_from_step(1)
            )

    def _complete_masks_and_test(self):
        """Complete both Masks (step 2) and Test (step 3) together."""
        self._on_step_completed(1)  # Masks
        self._on_step_completed(2)  # Test (auto-complete)

    def _on_step_completed(self, step_index: int):
        """Handle step completion."""
        self.logger.debug("Step %d completed", step_index)
        self.state_manager.complete_step(step_index)
        self._update_stepper_visual()

    def _invalidate_from_step(self, start_index: int):
        """Invalidate steps from start_index onwards, but only if there are completed steps."""
        current_step = self.state_manager.get_current_step()

        # Only invalidate if we're before the invalidated steps
        if current_step < start_index:
            # Check if there are any COMPLETED steps beyond start_index
            # Don't invalidate steps that are just ACTIVE (user hasn't worked on them yet)
            has_completed_beyond = any(
                self.state_manager.get_state(i) == StepState.COMPLETED
                for i in range(start_index, 5)  # 5 = total steps
            )
            if has_completed_beyond:
                self.state_manager.invalidate_from(start_index)
                self._update_stepper_visual()
                self.logger.debug("Invalidated steps from %d", start_index)

    def _on_step_clicked(self, index: int):
        """Handle click on stepper step indicator."""
        if self.state_manager.can_navigate_to(index):
            self.state_manager.set_current_step(index)

    def _on_state_changed(self, index: int, state: str):
        """Handle state change from state manager."""
        state_enum = StepState(state)
        self.stepper_widget.set_step_state(index, state_enum)

    def _on_current_step_changed(self, index: int):
        """Handle current step change from state manager."""
        self._navigate_to_step(index)

    def _navigate_to_step(self, index: int):
        """Navigate to a specific step."""
        self.content_stack.setCurrentIndex(index)
        self.stepper_widget.set_current_step(index)

        # Notify handlers about visibility
        step_keys = ['dataset', 'masks', 'test', 'postprocessor', 'reports']
        if index < len(step_keys):
            handler = self._handlers.get(step_keys[index])
            if handler and hasattr(handler, 'on_tab_visible'):
                handler.on_tab_visible()

        self.step_changed.emit(index)
        self.logger.debug("Navigated to step %d", index)

    def _update_stepper_visual(self):
        """Update stepper widget visual states from state manager."""
        for i, state in enumerate(self.state_manager.get_all_states()):
            self.stepper_widget.set_step_state(i, state)

    def get_current_step(self) -> int:
        """Get current step index."""
        return self.state_manager.get_current_step()

    def reset(self):
        """Reset to initial state."""
        self.state_manager.reset()
        self._update_stepper_visual()
        self._navigate_to_step(0)
        self.logger.debug("SingleTestContainer reset")
