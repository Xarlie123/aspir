from time import perf_counter

# Import mask generators
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal

# Import applicators
from simulation_engine._3_applicator.applicator_scatter import ApplicatorScatter
from simulation_engine._3_applicator.applicator_scatter_pseudoinverse import ApplicatorScatterPseudoinverse
from simulation_engine._3_applicator.applicator_scatter_fista import ApplicatorScatterFISTA
from simulation_engine._3_applicator.applicator_scatter_tv_norm import ApplicatorScatterTV
from simulation_engine._3_applicator.applicator_sweep import ApplicatorSweep
from simulation_engine._3_applicator.applicator_hadamard import ApplicatorHadamard

from simulation_engine._4_postprocessor.postprocessor_nn import PostprocessorNN

# Import the Analyzer facade
from simulation_engine._5_analyzer.analyzer import Analyzer

class Simulation:
    """
    Main ASPIR simulation engine.
    Manages dataset, mask, applicator, postprocessor and analyzer.
    """
    def __init__(self, logger):
        self.logger = logger.getChild("Simulation")
        self.logger.debug("Initializing Simulation")
        self.dataset = None
        self.mask = None
        self.applicator = None
        self.postprocessor = None
        self.analyzer = None

    def set_dataset(self, dataset):
        """Associates a dataset to the simulation."""
        self.dataset = dataset
        self.logger.info("Dataset assigned: %r", getattr(dataset, 'name', dataset))

    def set_mask(self, mask, applicator_type_scatter=None):
        """
        Associates the mask and configures the applicator.
        If the mask defines applicator_type_scatter, uses it.
        """
        self.mask = mask
        self.logger.info("Mask assigned: %s", type(mask).__name__)
        if self.dataset is None:
            raise RuntimeError("Must establish the dataset before the mask.")
        option = getattr(mask, 'applicator_type_scatter', None) or applicator_type_scatter
        self.set_applicator(applicator_type_scatter=option)

    def set_applicator(self, applicator_type_scatter=None):
        """
        Instantiates and associates the correct applicator according to the current mask.
        """
        if self.mask is None or self.dataset is None:
            raise RuntimeError("Dataset and mask must be defined.")
        cls_name = type(self.mask).__name__
        self.logger.debug("Configuring applicator for %s", cls_name)

        if isinstance(self.mask, MaskScatter):
            if applicator_type_scatter == 'Pseudoinverse':
                self.applicator = ApplicatorScatterPseudoinverse(self.dataset, self.mask)
            elif applicator_type_scatter == 'FISTA':
                self.applicator = ApplicatorScatterFISTA(self.dataset, self.mask)
            elif applicator_type_scatter == 'TV-norm':
                self.applicator = ApplicatorScatterTV(self.dataset, self.mask)
            else:
                self.applicator = ApplicatorScatter(self.dataset, self.mask)

        elif isinstance(self.mask, MaskSweep):
            self.applicator = ApplicatorSweep(self.dataset, self.mask)

        elif isinstance(self.mask, (
            MaskHadamard,
            MaskHadamardCakeCutting,
            MaskHadamardWalshPaley,
            MaskCalSal
        )):
            self.applicator =  ApplicatorHadamard(self.dataset, self.mask)

        else:
            raise ValueError(f"Unsupported mask: {cls_name}")

        self.logger.info("Applicator configured: %s", type(self.applicator).__name__)

    def set_postprocessor(
        self,
        dataset,
        mask,
        applicator,
        *,
        postprocessor_cls: type = PostprocessorNN,
        model_name: str = "autoencoder",
        model_overrides: dict = None,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.0,
        loss_function: str = 'mse',
        optimizer_name: str = 'adam',
        use_gpu: bool = True,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ):
        """
        Associates a postprocessor to the simulation.
        """
        self.logger.debug("Configuring postprocessor: %s (use_gpu=%s, loss=%s, optimizer=%s, dropout=%.2f)",
                         postprocessor_cls.__name__, use_gpu, loss_function, optimizer_name, dropout)
        if dataset is None or mask is None or applicator is None:
            raise ValueError("Dataset, mask and applicator are required")
        if postprocessor_cls is PostprocessorNN:
            overrides = model_overrides or {}
            # Add dropout to model overrides if supported
            overrides['dropout'] = dropout
            self.postprocessor = postprocessor_cls(
                model_name=model_name,
                model_overrides=overrides,
                dataset=dataset,
                applicator=applicator,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                loss_function=loss_function,
                optimizer_name=optimizer_name,
                use_gpu=use_gpu,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
        else:
            self.postprocessor = postprocessor_cls(dataset=dataset, applicator=applicator)
        self.logger.info("Postprocessor ready: %s", type(self.postprocessor).__name__)

    def set_analyzer(self):
        """
        Creates and associates the Analyzer facade from test_dataset().
        """
        if self.postprocessor is None:
            raise RuntimeError("Configure the postprocessor first.")
        self.logger.debug("Getting data from test_dataset()")
        orig, recons, denoised = self.postprocessor.test_dataset()
        start = perf_counter()
        self.analyzer = Analyzer(orig, recons, denoised)
        elapsed = perf_counter() - start
        self.logger.info("Analyzer created in %.3f s", elapsed)
