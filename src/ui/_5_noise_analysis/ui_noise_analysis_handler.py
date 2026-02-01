"""Handler for the noise analysis tab."""
import logging
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore    import pyqtSignal, QObject
from ui.utils.widget_helpers import embed_widget

from ui.custom_widgets.noise_analysis_metric.noise_analysis_metrics_widget import NoiseAnalysisMetricsWidget
from ui.custom_widgets.visualizers.visual_analysis.visual_analysis_noise_widget import VisualNoiseAnalysisWidget

from simulation_engine._5_analyzer.analyzer import Analyzer


class UINoiseAnalysisHandler(QObject):
    """
    Manage noise analysis tab: embed widgets, run analysis, and display results.
    Updated to handle LPIPS metrics.
    """
    def __init__(self, ui, simulation, logger):
        super().__init__()
        self.ui = ui
        self.simulation = simulation

        self.logger = logger.getChild("UINoiseAnalysisHandler")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Initializing UINoiseAnalysisHandler")

        # Numeric metrics widget
        ph1     = self.ui.noise_analysis_placeholder
        layout1 = self.ui.noise_analysis_layout
        self.metrics_widget = NoiseAnalysisMetricsWidget(
            simulation=simulation,
            parent=ph1.parentWidget(),
            logger=self.logger  # Pass logger if supported/needed
        )
        embed_widget(self.metrics_widget, ph1, layout1)
        self.logger.debug("Embedded NoiseAnalysisMetricsWidget")

        # Connect run request signal
        self.metrics_widget.runAnalysisRequested.connect(self._do_analysis)

        # Visual widget for noise analysis
        ph2     = self.ui.preview_noise_placeholder
        layout2 = self.ui.preview_noise_layout
        self.visual_widget = VisualNoiseAnalysisWidget(parent=ph2.parentWidget())
        embed_widget(self.visual_widget, ph2, layout2)
        self.logger.debug("Embedded VisualNoiseAnalysisWidget")

        # Button trigger (optional)
        if hasattr(self.ui, 'analizar_button'):
            self.ui.analyze_button.clicked.connect(self._do_analysis)

    def _do_analysis(self):
        """
        Perform noise analysis using stored validation results,
        or synthesize them from the loaded model if possible.
        """
        self.logger.info("Starting noise analysis")

        vr = getattr(self.simulation, 'validation_results', None)

        # Auto-synthesize preview if missing but a trained/loaded model exists
        if not vr:
            post = getattr(self.simulation, 'postprocessor', None)
            if post is not None and getattr(post, 'trained', False):
                self.logger.debug("No validation_results; generating from trained model...")
                try:
                    orig, recons, denoised = post.test_dataset()
                    vr = self.simulation.validation_results = {
                        'original': orig,
                        'recons':   recons,
                        'denoised': denoised
                    }
                except Exception as e:
                    self.logger.error("Could not generate validation preview: %s", e, exc_info=True)
                    QMessageBox.warning(None, "Attention", "Run postprocessing first.")
                    return
            else:
                self.logger.warning("No validation_results found, aborting analysis")
                QMessageBox.warning(None, "Attention", "Run postprocessing first.")
                return

        # Use validation_results
        orig, recons, denoised = vr['original'], vr['recons'], vr['denoised']
        self.logger.debug(
            "Validation results loaded: orig=%d, recons=%d, denoised=%d",
            len(orig), len(recons), len(denoised)
        )

        # Analyze noise metrics
        analyzer = Analyzer(orig, recons, denoised)
        try:
            # FIX: Unpack 6 values (PSNR, SSIM, LPIPS for noise & recon)
            pn, sn, ln, pr, sr, lr = analyzer.analyze_noise()
            self.logger.info("Noise analysis completed")
        except Exception as e:
            self.logger.error("Error during noise analysis: %s", e, exc_info=True)
            QMessageBox.critical(None, "Analysis Error", str(e))
            return

        # Update numeric metrics (Now passing 6 arguments + LPIPS means)
        # Note: analyzer.noise stores the means computed during analyze()
        try:
            self.metrics_widget.setOverallMetrics(
                analyzer.noise.psnr_noise_mean,
                analyzer.noise.ssim_noise_mean,
                analyzer.noise.lpips_noise_mean, # LPIPS Noise
                analyzer.noise.psnr_rec_mean,
                analyzer.noise.ssim_rec_mean,
                analyzer.noise.lpips_rec_mean    # LPIPS Recon
            )
            self.metrics_widget.setPerImageMetrics(pn, sn, ln, pr, sr, lr)
            self.logger.debug("Numeric metrics updated in metrics_widget")
        except Exception as e:
            self.logger.error("Error updating metrics widget: %s", e, exc_info=True)

        # Prepare metadata for visual widget
        num_images    = len(orig)
        img_size      = orig[0].shape[0] if num_images else 0
        dataset_type  = getattr(self.simulation.dataset, 'dataset_type', 'Unknown')
        mask_type     = type(getattr(self.simulation, 'mask', None)).__name__ if self.simulation.mask else 'Unknown'
        postproc_type = getattr(self.simulation.postprocessor, 'postproc_type', 'Unknown') if self.simulation.postprocessor else 'Unknown'

        self.visual_widget.update_info(
            num_images=num_images,
            img_size=img_size,
            dataset_type=dataset_type,
            mask_type=mask_type,
            postprocessor_type=postproc_type
        )
        self.logger.debug("Visual widget info updated: images=%d, size=%d", num_images, img_size)

        # Update images and bar chart (VisualWidget still accepts standard metrics)
        # If VisualNoiseAnalysisWidget hasn't been updated to accept LPIPS,
        # we only pass the original 4 metrics to avoid TypeError there.
        self.visual_widget.set_images_and_metrics(
            orig_images  = orig,
            noise_images = recons,
            recon_images = denoised,
            psnr_noise   = pn,
            ssim_noise   = sn,
            psnr_recon   = pr,
            ssim_recon   = sr
        )
        self.logger.info("Visual results rendered in visual_widget")