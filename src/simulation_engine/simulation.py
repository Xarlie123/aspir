from time import perf_counter

# Import mask generators
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._2_mask_gen.mask_fourier import MaskFourier
from simulation_engine._2_mask_gen.mask_measured import MaskMeasured

# Import applicators
from simulation_engine._3_applicator.applicator_scatter import ApplicatorScatter
from simulation_engine._3_applicator.applicator_pseudoinverse import ApplicatorPseudoinverse
from simulation_engine._3_applicator.applicator_fista import ApplicatorFISTA
from simulation_engine._3_applicator.applicator_tv_norm import ApplicatorTV
from simulation_engine._3_applicator.applicator_sweep import ApplicatorSweep
from simulation_engine._3_applicator.applicator_hadamard import ApplicatorHadamard
from simulation_engine._3_applicator.applicator_fourier import ApplicatorFourier

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

    # Canonical reconstruction-method keys accepted by set_applicator.
    # "native" delegates to the mask's natural reconstruction (Ghost Imaging,
    # Sweep Linear, Hadamard Linear, Fourier Inverse). The rest are
    # mask-agnostic iterative solvers that reuse the sensing matrix
    # ``mask.masks``.
    _RECON_METHOD_ALIASES = {
        # Canonical
        "native": "native",
        "pseudoinverse": "pseudoinverse",
        "fista": "fista",
        "tv_norm": "tv_norm",
        # Display / legacy names accepted for GUI + old batch configs
        None: "native",
        "": "native",
        "conventional": "native",
        "ghost imaging": "native",
        "ghost_imaging": "native",
        "sweep linear": "native",
        "hadamard linear": "native",
        "fourier": "native",
        "fourier inverse": "native",
        "fourier_inverse": "native",
        "pinv": "pseudoinverse",
        "tv-norm": "tv_norm",
        "tv norm": "tv_norm",
    }

    @classmethod
    def _canonical_recon_method(cls, name):
        if isinstance(name, str):
            key = name.strip().lower()
        else:
            key = name
        return cls._RECON_METHOD_ALIASES.get(key, key)

    def set_mask(self, mask, reconstruction_method=None,
                 applicator_type_scatter=None):
        """
        Associate the mask and configure the applicator.

        ``reconstruction_method`` accepts ``"native"``, ``"pseudoinverse"``,
        ``"fista"``, ``"tv_norm"`` (and a handful of display aliases). The
        old ``applicator_type_scatter`` parameter is still honored for
        backward compatibility. If the mask instance carries an
        ``applicator_type_scatter`` attribute it wins, mirroring how the
        scatter mask control stores the user's dropdown selection.
        """
        self.mask = mask
        self.logger.info("Mask assigned: %s", type(mask).__name__)
        if self.dataset is None:
            raise RuntimeError("Must establish the dataset before the mask.")
        option = (getattr(mask, 'applicator_type_scatter', None)
                  or reconstruction_method
                  or applicator_type_scatter)
        # The mask widgets stash the device choice on the mask, like the method.
        device = getattr(mask, 'reconstruction_device', None) or "auto"
        # ...and the FISTA / TV parameters, when the panel exposes them.
        solver_params = getattr(mask, 'solver_params', None)
        self.set_applicator(reconstruction_method=option, device=device,
                            solver_params=solver_params)

    @staticmethod
    def _solver_kwargs(solver_params, iterations_name):
        """Map ``{"lam", "iterations"}`` onto an iterative applicator's arguments."""
        kwargs = {}
        if solver_params:
            if solver_params.get("lam") is not None:
                kwargs["lam"] = float(solver_params["lam"])
            if solver_params.get("iterations") is not None:
                kwargs[iterations_name] = int(solver_params["iterations"])
        return kwargs

    def set_applicator(self, reconstruction_method=None,
                       applicator_type_scatter=None, device="auto",
                       solver_params=None):
        """
        Instantiate the applicator implementing ``reconstruction_method`` over
        the current mask. ``"native"`` picks the mask-specific algorithm
        (Ghost Imaging, Sweep Linear, Hadamard Linear or Fourier Inverse);
        the iterative solvers work with any mask whose sensing matrix is
        exposed via ``mask.masks``.

        ``device`` selects where the reconstruction runs: ``"auto"`` (CUDA if
        available, else CPU), ``"cpu"`` or ``"cuda"`` — the same convention as
        the NN stage, which uses CUDA only when it is both requested and
        available.

        ``solver_params`` sets the iterative solvers' regularisation and
        iteration count, ``{"lam": float, "iterations": int}``; omitted keys
        keep the defaults (FISTA lambda 1e-3, 500 iterations; TV lambda 0.1,
        100 iterations), which are the values the solvers have always used.
        """
        if self.mask is None or self.dataset is None:
            raise RuntimeError("Dataset and mask must be defined.")
        cls_name = type(self.mask).__name__
        method = self._canonical_recon_method(
            reconstruction_method if reconstruction_method is not None
            else applicator_type_scatter
        )
        self.logger.debug("Configuring applicator for %s (method=%s)",
                          cls_name, method)

        if method == "pseudoinverse":
            self.applicator = ApplicatorPseudoinverse(self.dataset, self.mask, device=device)
        elif method == "fista":
            self.applicator = ApplicatorFISTA(self.dataset, self.mask, device=device,
                                              **self._solver_kwargs(solver_params, "maxit"))
        elif method == "tv_norm":
            self.applicator = ApplicatorTV(self.dataset, self.mask, device=device,
                                           **self._solver_kwargs(solver_params, "niter"))
        else:
            # Native: mask-specific default.
            if isinstance(self.mask, MaskScatter):
                self.applicator = ApplicatorScatter(self.dataset, self.mask, device=device)
            elif isinstance(self.mask, MaskSweep):
                self.applicator = ApplicatorSweep(self.dataset, self.mask, device=device)
            elif isinstance(self.mask, MaskFourier):
                self.applicator = ApplicatorFourier(self.dataset, self.mask, device=device)
            elif isinstance(self.mask, (
                MaskHadamard,
                MaskHadamardCakeCutting,
                MaskHadamardWalshPaley,
                MaskCalSal,
            )):
                self.applicator = ApplicatorHadamard(self.dataset, self.mask, device=device)
            elif isinstance(self.mask, MaskMeasured) and self.mask.is_bipolar:
                # Measured +-1 patterns (e.g. SPIHIM): Hadamard Linear, S^T y / M.
                self.applicator = ApplicatorHadamard(self.dataset, self.mask, device=device)
            else:
                raise ValueError(f"Unsupported mask: {cls_name}")

        self.logger.info("Applicator configured: %s (device request=%s)",
                         type(self.applicator).__name__, device)

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
        orig, _, denoised = self.postprocessor.test_dataset()
        # The linear stage is scored on its own output, on the image's scale
        linear = self.postprocessor.test_linear()
        start = perf_counter()
        self.analyzer = Analyzer(orig, linear, denoised)
        elapsed = perf_counter() - start
        self.logger.info("Analyzer created in %.3f s", elapsed)
