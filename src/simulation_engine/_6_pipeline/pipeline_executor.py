"""
Module to execute the SPIm simulation pipeline based on a YAML configuration.
Handles loading datasets, generating masks (with nested applicator), instantiating applicators,
performing progress callbacks, measuring per-image and full-dataset performance,
and returning results.
"""
import yaml
from time import perf_counter

from simulation_engine.simulation import Simulation
from simulation_engine._1_dataset_gen.DatasetFromIRBeam import DatasetFromIRBeam
from simulation_engine._1_dataset_gen.DatasetFromImage import DatasetFromImage
from simulation_engine._1_dataset_gen.DatasetFromFolder import DatasetFromFolder
from simulation_engine._2_mask_gen.mask_hadamard import MaskHadamard
from simulation_engine._2_mask_gen.mask_hadamard_cake_cutting import MaskHadamardCakeCutting
from simulation_engine._2_mask_gen.mask_hadamard_walsh_paley import MaskHadamardWalshPaley
from simulation_engine._2_mask_gen.mask_cal_sal import MaskCalSal
from simulation_engine._2_mask_gen.mask_scatter import MaskScatter
from simulation_engine._2_mask_gen.mask_sweep import MaskSweep


def load_tests(cfg_path):
    """
    Load YAML configuration and return the list of test dicts.
    """
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or 'tests' not in cfg or not isinstance(cfg['tests'], list):
        raise ValueError("YAML must contain a 'tests' list")
    return cfg['tests']


def execute_pipeline(tests, progress_per_task=None, progress_overall=None):
    """
    Run the SPIm pipeline given test configurations.
    Reads 'size_px' from YAML (ir_beam) or ignores for single/folder types.
    """
    sim = Simulation()
    results = []
    total_tasks = len(tests)

    mask_map = {
        'sweep': MaskSweep,
        'scatter': MaskScatter,
        'hadamard': MaskHadamard,
        'hadamard_cake_cutting': MaskHadamardCakeCutting,
        'hadamard_walsh_paley': MaskHadamardWalshPaley,
        'cal_sal': MaskCalSal
    }

    for i, cfg in enumerate(tests, start=1):
        if progress_overall:
            progress_overall(i, total_tasks)

        # 1) Dataset selection
        ds_cfg = cfg.get('dataset', {})
        dt = ds_cfg.get('type')
        if dt == 'single_image':
            ds = DatasetFromImage(ds_cfg['image_path'])
        elif dt == 'folder_image':
            ds = DatasetFromFolder(ds_cfg['folder_path'])
        elif dt == 'ir_beam':
            size = ds_cfg.get('size_px')
            if size is None or size <= 0:
                raise ValueError(f"Invalid 'size_px' ({size}) for test '{cfg.get('name','')}'")
            ds = DatasetFromIRBeam(
                ds_cfg.get('name', 'ir_beam'),
                size,
                ds_cfg.get('number_images', 0),
                ds_cfg.get('random_seed', 0)
            )
        else:
            raise ValueError(f"Unknown dataset type: {dt}")

        ds.load_data()
        sim.set_dataset(ds)

        # 2) Mask generation
        m_cfg = cfg.get('mask', {})
        mtype = m_cfg.get('type', 'sweep').lower()
        MaskClass = mask_map.get(mtype)
        if MaskClass is None:
            raise ValueError(f"Unknown mask type: {m_cfg.get('type')}")
        # Exclude 'type' and 'applicator' from kwargs
        kwargs = {k: v for k, v in m_cfg.items() if k not in ('type', 'applicator')}
        mask = MaskClass(sim.dataset.img_size, **kwargs)

        sim.set_mask(mask)
        sim.mask.generate_masks(progress_callback=progress_per_task)

        # 3) Applicator (YAML key "applicator" carries the reconstruction
        # method: "native" / "pseudoinverse" / "fista" / "tv_norm").
        sim.set_applicator(reconstruction_method=m_cfg.get('applicator'))

        # 4) Measure per-image performance
        per_image_times = []
        for idx in range(len(sim.dataset.data)):
            t0 = perf_counter()
            sim.applicator.process_image(idx)
            t1 = perf_counter()
            per_image_times.append(t1 - t0)

        # 5) Measure full-dataset performance
        t0_all = perf_counter()
        sim.applicator.process_dataset()
        t1_all = perf_counter()
        total_time = t1_all - t0_all

        results.append({
            'name': cfg.get('name', ''),
            'params': cfg,
            'per_image_times': per_image_times,
            'total_time': total_time
        })

    return results
