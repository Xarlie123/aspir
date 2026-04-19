# pipeline_control_widget.py

import logging
from PyQt5.QtWidgets import QWidget
from ui.custom_widgets.pipeline_control.pipeline_control import Ui_pipeline_control

class PipelineControlWidget(QWidget):
    def __init__(self, simulation, logger=None, parent=None):
        super().__init__(parent)
        self.simulation = simulation
        self.logger = logger.getChild("PipelineControlWidget") if logger else logging.getLogger("PipelineControlWidget")
        self.logger.debug("Initializing PipelineControlWidget")

        # Set up UI from .ui file
        self.ui = Ui_pipeline_control()
        self.ui.setupUi(self)

        # Connect UI elements to logic
        self.ui.add_test_button.clicked.connect(self.add_test)
        self.ui.remove_test_button.clicked.connect(self.remove_test)

    def add_test(self):
        """
        Add a test name from line edit to the list widget.
        """
        test_name = self.ui.test_name_lineedit.text()
        if test_name:
            self.ui.tests_listwidget.addItem(test_name)
            self.logger.info(f"Added test: {test_name}")
        else:
            self.logger.warning("Test name is empty. No test was added.")

    def remove_test(self):
        """
        Remove selected test(s) from the list widget.
        """
        selected_items = self.ui.tests_listwidget.selectedItems()
        for item in selected_items:
            self.logger.info(f"Removing test: {item.text()}")
            self.ui.tests_listwidget.takeItem(self.ui.tests_listwidget.row(item))
