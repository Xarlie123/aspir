# ui_pipeline_handler.py

import logging
from ui.utils.widget_helpers import embed_widget
from ui.custom_widgets.pipeline_control.pipeline_control_widget import PipelineControlWidget

class UIPipelineHandler:
    def __init__(self, ui, simulation, logger=None):
        self.ui = ui
        self.simulation = simulation
        self.logger = logger.getChild("UIPipelineHandler") if logger else logging.getLogger("UIPipelineHandler")
        self.logger.debug("Initializing UIPipelineHandler")

        # Instantiate and embed the PipelineControlWidget in the placeholder
        self.pipeline_widget = PipelineControlWidget(simulation=self.simulation, logger=self.logger)
        embed_widget(self.pipeline_widget, self.ui.pipeline_control_placeholder, self.ui.pipeline_control_layout)

    def get_content_widget(self):
        """Return the main container widget for stepper integration."""
        return self.ui.test_pipeline_tab
