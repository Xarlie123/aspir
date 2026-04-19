import yaml

class TestConfigManager:
    """
    Manages loading, saving and transforming test configurations.
    Moves top-level 'applicator' into each test['mask'].
    Prunes dataset fields based on dataset type when saving.
    """
    def __init__(self):
        self.tests = []

    def load(self, path: str) -> list:
        """
        Load tests from a YAML file, normalize structure and return list of tests.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        tests = data['tests'] if isinstance(data, dict) and 'tests' in data else data
        # Move top-level applicator inside mask
        for t in tests:
            if 'applicator' in t:
                t.setdefault('mask', {})['applicator'] = t.pop('applicator')
        self.tests = tests
        return tests

    def save(self, path: str):
        """
        Prune extraneous fields and save current self.tests into a YAML file at path.
        For each test:
          - 'single_image': dataset only keeps 'type' and 'image_path'
          - 'folder_image': dataset only keeps 'type' and 'folder_path'
          - 'ir_beam': dataset keeps 'type', 'size_px', 'number_images', 'random_seed'
          - mask keeps 'type', 'applicator', and any parameters specific to its type
        """
        pruned_tests = []
        for t in self.tests:
            new_t = {}
            # Preserve test name
            if 'name' in t:
                new_t['name'] = t['name']

            # Dataset pruning
            ds = t.get('dataset', {})
            dtype = ds.get('type')
            new_ds = {'type': dtype}
            if dtype == 'single_image':
                if 'image_path' in ds:
                    new_ds['image_path'] = ds['image_path']
            elif dtype == 'folder_image':
                if 'folder_path' in ds:
                    new_ds['folder_path'] = ds['folder_path']
            elif dtype == 'ir_beam':
                for key in ('size_px', 'number_images', 'random_seed'):
                    if key in ds:
                        new_ds[key] = ds[key]
            else:
                # Unknown type: include all fields
                for k, v in ds.items():
                    new_ds[k] = v
            new_t['dataset'] = new_ds

            # Mask pruning
            m = t.get('mask', {})
            new_m = {'type': m.get('type')}
            # Include applicator if present
            if 'applicator' in m:
                new_m['applicator'] = m['applicator']
            # Include mask-specific parameters
            mtype = m.get('type')
            if mtype == 'scatter':
                for key in ('num_patterns', 'point_density', 'seed'):
                    if key in m:
                        new_m[key] = m[key]
            elif mtype in ('hadamard', 'hadamard_cake_cutting', 'hadamard_walsh_paley'):
                # No extra parameters for Hadamard variants
                pass
            elif mtype == 'sweep':
                if 'parametros' in m:
                    new_m['parametros'] = m['parametros']
            else:
                # For other mask types, include all except nested objects
                for k, v in m.items():
                    if k not in ('type', 'applicator'):
                        new_m[k] = v
            new_t['mask'] = new_m

            pruned_tests.append(new_t)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump({'tests': pruned_tests}, f)

    def get_tests(self) -> list:
        """Return the current list of tests."""
        return self.tests

    def set_tests(self, tests: list):
        """Replace the current tests list."""
        self.tests = tests
