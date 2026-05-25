"""
Worker thread for LaTeX compilation.

Prevents UI freeze during pdflatex execution by running compilation
in a background QThread.
"""
import logging
from typing import List, Tuple, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from .plotneuralnet_generator import PlotNeuralNetGenerator
from .network_visualizer import SemanticBlock


class LaTeXCompilationWorker(QThread):
    """
    Background thread for generating PlotNeuralNet diagrams.

    Signals:
        finished(str): Emitted with PDF path on success
        error(str): Emitted with error message on failure
        progress(str): Status updates during compilation
        tex_ready(str): Emitted with .tex source code
    """

    finished = Signal(str)    # PDF path
    error = Signal(str)       # Error message
    progress = Signal(str)    # Status message
    tex_ready = Signal(str)   # .tex source code

    def __init__(self,
                 blocks: List[SemanticBlock],
                 skip_connections: List[Tuple[int, int]],
                 model_name: str,
                 input_image: Optional[np.ndarray] = None,
                 output_image: Optional[np.ndarray] = None,
                 colormap: Optional[str] = None,
                 output_path: Optional[str] = None,
                 keep_tex: bool = False,
                 logger: Optional[logging.Logger] = None,
                 parent=None):
        """
        Initialize the compilation worker.

        Args:
            blocks: List of SemanticBlocks from NetworkVisualizer
            skip_connections: List of (encoder_idx, decoder_idx) tuples
            model_name: Display name for the network
            input_image: Optional sample input image (numpy array)
            output_image: Optional sample output image (numpy array)
            colormap: Optional colormap name for grayscale images
            output_path: Optional path for output PDF
            keep_tex: Keep .tex file alongside PDF
            logger: Logger instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.blocks = blocks
        self.skip_connections = skip_connections
        self.model_name = model_name
        self.input_image = input_image
        self.output_image = output_image
        self.colormap = colormap
        self.output_path = output_path
        self.keep_tex = keep_tex

        if logger:
            self.logger = logger.getChild("LaTeXCompilationWorker")
        else:
            self.logger = logging.getLogger(__name__)

        self._generator: Optional[PlotNeuralNetGenerator] = None

    def run(self):
        """Execute compilation in background."""
        try:
            self._generator = PlotNeuralNetGenerator(logger=self.logger)

            # Check availability
            if not self._generator.is_available():
                self.error.emit(
                    "pdflatex not found.\n\n"
                    "Install with:\n"
                    "  apt install texlive-latex-base texlive-latex-extra"
                )
                return

            self.progress.emit("Generating TikZ code...")

            # Emit .tex source for potential saving
            tex_source = self._generator.get_tex_source(
                self.blocks,
                self.skip_connections,
                self.model_name
            )
            self.tex_ready.emit(tex_source)

            self.progress.emit("Compiling LaTeX (this may take a few seconds)...")

            # Generate PDF
            pdf_path = self._generator.generate(
                self.blocks,
                self.skip_connections,
                self.model_name,
                self.input_image,
                self.output_image,
                self.output_path,
                self.keep_tex,
                self.colormap
            )

            if pdf_path:
                self.progress.emit("Done!")
                self.finished.emit(pdf_path)
            else:
                self._cleanup_generator()
                self.error.emit(
                    "LaTeX compilation failed.\n\n"
                    "Check that all required LaTeX packages are installed:\n"
                    "  apt install texlive-latex-extra"
                )

        except Exception as e:
            self.logger.error(f"Compilation worker error: {e}", exc_info=True)
            self._cleanup_generator()
            self.error.emit(f"Compilation error:\n{str(e)}")

    def _cleanup_generator(self):
        """Clean up generator temp files."""
        if self._generator:
            self._generator.cleanup()

    def cleanup(self):
        """Public method to clean up resources. Call when done with PDF."""
        self._cleanup_generator()

    def get_generator(self) -> Optional[PlotNeuralNetGenerator]:
        """Get the generator instance (available after run starts)."""
        return self._generator
