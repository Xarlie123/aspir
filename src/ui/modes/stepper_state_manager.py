"""
State manager for the stepper workflow.
Tracks step states, validates transitions, and handles invalidation cascade.
"""
import logging
from PySide6.QtCore import QObject, Signal
from ui.custom_widgets.stepper.stepper_widget import StepState


class StepperStateManager(QObject):
    """
    Central state management for the stepper workflow.

    Tracks step states:
    - pending: Not yet reachable
    - active: Current step
    - completed: Done, can revisit
    - invalidated: Was completed but needs redo

    Handles invalidation cascade: changing step N invalidates steps N+1 onwards.
    """
    state_changed = Signal(int, str)  # (step_index, new_state)
    current_step_changed = Signal(int)  # new current step index

    # Step definitions
    STEPS = [
        {"name": "Dataset", "index": 0},
        {"name": "Masks", "index": 1},
        {"name": "Test", "index": 2},
        {"name": "DNN", "index": 3},
        {"name": "Reports", "index": 4},
    ]

    def __init__(self, logger=None, parent=None):
        super().__init__(parent)

        if logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger.getChild(self.__class__.__name__)

        self._num_steps = len(self.STEPS)
        self._states = [StepState.PENDING] * self._num_steps
        self._states[0] = StepState.ACTIVE  # First step is active by default
        self._current_step = 0

        self.logger.debug("StepperStateManager initialized with %d steps", self._num_steps)

    def get_state(self, index: int) -> StepState:
        """Get the state of a specific step."""
        if 0 <= index < self._num_steps:
            return self._states[index]
        return StepState.PENDING

    def get_current_step(self) -> int:
        """Get the current active step index."""
        return self._current_step

    def set_current_step(self, index: int):
        """
        Set the current step. Only allowed if step is completed, active, or invalidated.
        """
        if not 0 <= index < self._num_steps:
            self.logger.warning("Invalid step index: %d", index)
            return

        state = self._states[index]
        if state == StepState.PENDING:
            self.logger.warning("Cannot navigate to pending step %d", index)
            return

        old_step = self._current_step
        self._current_step = index

        # Mark the new current step as active if it was invalidated
        if self._states[index] == StepState.INVALIDATED:
            self._states[index] = StepState.ACTIVE
            self.state_changed.emit(index, StepState.ACTIVE.value)

        self.logger.debug("Current step changed from %d to %d", old_step, index)
        self.current_step_changed.emit(index)

    def complete_step(self, index: int):
        """
        Mark a step as completed.
        This enables navigation to the next step.
        """
        if not 0 <= index < self._num_steps:
            return

        old_state = self._states[index]
        self._states[index] = StepState.COMPLETED

        # If next step is pending, make it active
        if index + 1 < self._num_steps and self._states[index + 1] == StepState.PENDING:
            self._states[index + 1] = StepState.ACTIVE
            self.state_changed.emit(index + 1, StepState.ACTIVE.value)

        self.logger.debug("Step %d completed (was %s)", index, old_state.value)
        self.state_changed.emit(index, StepState.COMPLETED.value)

    def invalidate_from(self, start_index: int):
        """
        Invalidate all steps from start_index onwards.
        Called when a previous step's data changes.
        """
        for i in range(start_index, self._num_steps):
            if self._states[i] in (StepState.COMPLETED, StepState.ACTIVE):
                self._states[i] = StepState.INVALIDATED
                self.state_changed.emit(i, StepState.INVALIDATED.value)
                self.logger.debug("Step %d invalidated", i)

    def can_navigate_to(self, index: int) -> bool:
        """Check if navigation to a step is allowed."""
        if not 0 <= index < self._num_steps:
            return False
        return self._states[index] != StepState.PENDING

    def can_proceed_next(self) -> bool:
        """Check if user can proceed to next step (current step must be complete)."""
        if self._current_step >= self._num_steps - 1:
            return False  # Already on last step
        return self._states[self._current_step] == StepState.COMPLETED

    def can_go_back(self) -> bool:
        """Check if user can go back (not on first step)."""
        return self._current_step > 0

    def go_next(self):
        """Navigate to next step if allowed."""
        if self.can_proceed_next():
            self.set_current_step(self._current_step + 1)

    def go_back(self):
        """Navigate to previous step if allowed."""
        if self.can_go_back():
            self.set_current_step(self._current_step - 1)

    def reset(self):
        """Reset all steps to initial state."""
        self._states = [StepState.PENDING] * self._num_steps
        self._states[0] = StepState.ACTIVE
        self._current_step = 0

        for i, state in enumerate(self._states):
            self.state_changed.emit(i, state.value)

        self.current_step_changed.emit(0)
        self.logger.debug("State manager reset")

    def get_all_states(self) -> list:
        """Get list of all step states."""
        return self._states.copy()
