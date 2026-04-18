"""Small stateless / light-state helpers for the energy view."""
from __future__ import annotations


def shorten_backend_name(name: str, max_length: int = 20) -> str:
    """Shorten a backend name for display."""
    if " with " in name:
        name = name.split(" with ")[0]
    if len(name) > max_length:
        return name[:max_length - 3] + "..."
    return name


def get_nested_value(data: dict, key: str):
    """Get a value from a nested dictionary using dot notation."""
    keys = key.split(".")
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value


def apply_axes_config(view, ax, default_title: str = "", default_xlabel: str = "",
                      default_ylabel: str = ""):
    """Apply axes configuration from ``view._chart_config`` to a matplotlib axis."""
    axes_cfg = view._chart_config.get('axes', {})

    title = axes_cfg.get('title', '') or default_title
    title_fontsize = axes_cfg.get('title_fontsize', 13)
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold')

    xlabel = axes_cfg.get('xlabel', '') or default_xlabel
    xlabel_fontsize = axes_cfg.get('xlabel_fontsize', 11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)

    ylabel = axes_cfg.get('ylabel', '') or default_ylabel
    ylabel_fontsize = axes_cfg.get('ylabel_fontsize', 11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    xtick_fontsize = axes_cfg.get('xtick_fontsize', 9)
    ytick_fontsize = axes_cfg.get('ytick_fontsize', 9)
    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    ax.tick_params(axis='y', labelsize=ytick_fontsize)

    if not axes_cfg.get('auto_scale', True):
        ymin = axes_cfg.get('ymin', 0)
        ymax = axes_cfg.get('ymax', 100)
        ax.set_ylim(ymin, ymax)


def apply_legend(view, ax):
    """Apply legend configuration from ``view._chart_config`` to a matplotlib axis."""
    legend_cfg = view._chart_config['legend']
    legend_pos = legend_cfg['position']
    legend_kwargs = {
        'fontsize': legend_cfg['fontsize'],
        'frameon': legend_cfg['frameon'],
        'shadow': legend_cfg['shadow'],
        'fancybox': legend_cfg['fancybox'],
        'framealpha': legend_cfg['framealpha'],
    }

    loc_map = {
        0: 'upper right',
        1: 'upper left',
        2: 'lower right',
        3: 'lower left',
    }

    if legend_pos in loc_map:
        ax.legend(loc=loc_map[legend_pos], ncol=legend_cfg['ncol'], **legend_kwargs)
    elif legend_pos == 4:
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  ncol=legend_cfg['ncol'], **legend_kwargs)
    else:
        legend_kwargs['frameon'] = False
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=4, **legend_kwargs)


def draw_no_data_message(view, message: str):
    """Draw a message on the view's figure when no data is available."""
    ax = view.figure.add_subplot(111)
    ax.text(0.5, 0.5, message,
            ha='center', va='center', fontsize=12, color='#999',
            multialignment='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    view.canvas.draw()
    view.report_btn.setEnabled(False)
