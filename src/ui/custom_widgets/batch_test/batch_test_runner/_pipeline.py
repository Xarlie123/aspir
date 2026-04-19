"""Pipeline builders for batch tests: masks and applicators."""
from __future__ import annotations

from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep
from simulation_engine._3_applicator.applicator_hadamard import ApplicatorHadamard
from simulation_engine._3_applicator.applicator_scatter import ApplicatorScatter
from simulation_engine._3_applicator.applicator_fista import ApplicatorFISTA
from simulation_engine._3_applicator.applicator_pseudoinverse import ApplicatorPseudoinverse
from simulation_engine._3_applicator.applicator_tv_norm import ApplicatorTV
from simulation_engine._3_applicator.applicator_sweep import ApplicatorSweep
from ui.custom_widgets.batch_test.test_config_model import TestConfiguration


def create_mask(config: TestConfiguration, dataset, logger):
    """Create mask based on configuration."""
    img_size = dataset.img_size

    if config.mask_type == "scatter":
        return MaskScatter(
            img_size=img_size,
            point_density=config.scatter_point_density,
            num_patterns=config.scatter_num_patterns,
            seed=config.mask_seed,
            logger=logger
        )
    elif config.mask_type == "hadamard_natural":
        max_idx = min(config.hadamard_max_idx, img_size * img_size)
        return MaskHadamard(
            img_size=img_size,
            min_idx=config.hadamard_min_idx,
            max_idx=max_idx,
            logger=logger
        )
    elif config.mask_type == "hadamard_cake_cutting":
        max_idx = min(config.hadamard_max_idx, img_size * img_size)
        return MaskHadamardCakeCutting(
            img_size=img_size,
            min_idx=config.hadamard_min_idx,
            max_idx=max_idx,
            logger=logger
        )
    elif config.mask_type == "hadamard_walsh_paley":
        max_idx = min(config.hadamard_max_idx, img_size * img_size)
        return MaskHadamardWalshPaley(
            img_size=img_size,
            min_idx=config.hadamard_min_idx,
            max_idx=max_idx,
            logger=logger
        )
    elif config.mask_type == "sweep":
        # Convert sweep angles to parameters list format
        parametros = []
        for i, angle in enumerate(config.sweep_angles):
            parametros.append({
                "angle": angle,
                "bar_width": config.sweep_bar_widths[i] if i < len(config.sweep_bar_widths) else 2,
                "stride": config.sweep_strides[i] if i < len(config.sweep_strides) else 4,
            })
        return MaskSweep(
            img_size=img_size,
            parametros=parametros,
            logger=logger
        )
    elif config.mask_type == "cal_sal":
        return MaskCalSal(
            img_size=img_size,
            logger=logger
        )
    else:
        raise ValueError(f"Unknown mask type: {config.mask_type}")


def _native_applicator_for_mask(mask, dataset):
    """Return the mask-specific default ("native") applicator."""
    if isinstance(mask, MaskScatter):
        return ApplicatorScatter(dataset, mask)
    if isinstance(mask, MaskSweep):
        return ApplicatorSweep(dataset, mask)
    if isinstance(mask, (MaskHadamard, MaskHadamardCakeCutting,
                         MaskHadamardWalshPaley, MaskCalSal)):
        return ApplicatorHadamard(dataset, mask)
    raise ValueError(f"Unsupported mask type for applicator: {type(mask)}")


def create_applicator(config: TestConfiguration, mask, dataset):
    """Create the applicator for ``mask`` using ``config.reconstruction_method``.

    ``method`` is mask-agnostic: ``pseudoinverse``/``fista``/``tv_norm`` use
    the generic iterative solvers for any mask, and any other value
    (including ``conventional`` / ``native`` / legacy empty strings) falls
    back to the mask's native algorithm (Ghost Imaging, Sweep Linear,
    Hadamard Linear).
    """
    mask.generate_masks()

    method = (config.reconstruction_method or "").lower()

    if method == "pseudoinverse":
        return ApplicatorPseudoinverse(dataset, mask)
    if method == "fista":
        applicator = ApplicatorFISTA(dataset, mask)
        applicator.lambda_val = config.fista_lambda
        applicator.max_iter = config.fista_iterations
        return applicator
    if method == "tv_norm":
        applicator = ApplicatorTV(dataset, mask)
        applicator.lambda_val = config.tv_lambda
        applicator.max_iter = config.tv_iterations
        return applicator

    return _native_applicator_for_mask(mask, dataset)
