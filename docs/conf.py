# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = 'ASPIR'
copyright = '2026, Universitat Jaume I, Carlos Chabert Ull'
author = 'Carlos Chabert Ull'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',              # Markdown support
    'sphinx.ext.autodoc',       # API documentation from docstrings
    'sphinx.ext.napoleon',      # Google/NumPy style docstrings
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx.ext.intersphinx',   # Link to other projects' docs
    'sphinx_copybutton',        # Copy button for code blocks
    'sphinx_design',            # Cards, tabs, grids
    'sphinxcontrib.mermaid',    # Diagrams
    'sphinx.ext.mathjax',       # Math rendering
]

# Markdown configuration
myst_enable_extensions = [
    'colon_fence',      # ::: directive syntax
    'deflist',          # Definition lists
    'fieldlist',        # Field lists
    'tasklist',         # Task lists with checkboxes
    'attrs_inline',     # Inline attributes
    'dollarmath',       # $inline$ and $$block$$ math syntax
]
myst_heading_anchors = 3

# Source file settings
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '../assets/logo_banner.png'
html_theme_options = {
    'logo_only': True,
    'navigation_depth': 3,
}
html_css_files = ['custom.css']
html_extra_path = ['animations', 'dataset_samples']

# -- Mermaid configuration ---------------------------------------------------
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    maxTextSize: 90000,
    flowchart: {
        useMaxWidth: false,
        padding: 2
    }
});
"""
mermaid_fullscreen = False

# -- Options for PDF output --------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
}

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
