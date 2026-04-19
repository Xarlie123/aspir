import os
import tempfile
import webbrowser
import numpy as np
import matplotlib.pyplot as plt

class PipelineReporter:
    """
    Generatestes an HTML report from precomputed SPIm pipeline results.
    Each result dict must contain:
      - 'name': str
      - 'params': dict
      - 'per_image_times': list of floats
      - 'total_time': float
    """
    def __init__(self, report_title="Informe de Tests"):
        """Initialize reporter with a custom report title."""
        self.report_title = report_title

    def generate_report(self, results):
        """
        Generateste HTML report using provided pipeline results, without re-executing simulation.
        :param results: list of result dicts from execute_pipeline
        :return: path to the generated HTML report
        """
        # Prepare temporary directory
        report_dir = tempfile.mkdtemp(prefix="pipeline_report_")
        img_dir = os.path.join(report_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        html_path = os.path.join(report_dir, "report.html")

        # Build HTML blocks for each test
        html_blocks = []
        for idx, res in enumerate(results):
            name = res['name']
            params = res.get('params', {})
            total = res.get('total_time', 0.0)
            per = res.get('per_image_times', [])
            avg = float(np.mean(per)) if per else 0.0

            # Create bar chart for total time
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.barh([0], [total], height=0.5)
            ax.set_xlim(0, max(total, avg) * 1.2)
            ax.set_yticks([])
            ax.set_xlabel("s")
            ax.set_title(name)
            fig.tight_layout()

            img_file = f"chart_{idx}.png"
            img_path = os.path.join(img_dir, img_file)
            fig.savefig(img_path)
            plt.close(fig)

            # HTML block
            block = [f"<h2>{name}</h2>", "<h3>Parámetros</h3><ul>"]
            for section in ("dataset", "mask"):  # show dataset/mask parameters
                sec = params.get(section, {}) or {}
                if isinstance(sec, dict):
                    block.append(f"<li><strong>{section}:</strong><ul>")
                    for k, v in sec.items():
                        block.append(f"<li>{k}: {v}</li>")
                    block.append("</ul></li>")
            if 'applicator' in params:
                block.append(f"<li><strong>applicator:</strong> {params['applicator']}</li>")
            block.append("</ul>")
            block.append("<h3>Tiempos</h3>")
            block.append(f"<p>Total: {total:.3f} s, Promedio: {avg:.3f} s</p>")
            block.append(f"<img src='images/{img_file}' alt='Chart {name}'/>")

            html_blocks.append("\n".join(block))

        # Assemble full HTML
        full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>{self.report_title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    h2 {{ margin-top: 40px; }}
    img {{ max-width: 100%; height: auto; }}
    ul {{ margin: 5px 0 15px 20px; }}
  </style>
</head>
<body>
  <h1>{self.report_title}</h1>
  {''.join(html_blocks)}
</body>
</html>"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        return html_path