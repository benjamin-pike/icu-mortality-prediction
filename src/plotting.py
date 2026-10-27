import matplotlib.pyplot as plt

from .io import ROOT

INK, NAVY, TEAL = '#1C2730', '#245C85', '#008779'
ORANGE, TERRACOTTA, PLUM = '#C85A22', '#C45545', '#8050A0'
LIGHT_GREY, MID_GREY, DARK_GREY, GRID, WHITE = (
    '#DCE4E7',
    '#8A969C',
    '#56636A',
    '#D8E0E3',
    '#FFFFFF'
)


def style():
    plt.rcParams.update(
        {
            'font.family': 'Arial',
            'font.size': 9,
            'axes.titlesize': 10.5,
            'axes.labelsize': 9,
            'axes.titleweight': 'semibold',
            'axes.labelcolor': INK,
            'axes.edgecolor': DARK_GREY,
            'axes.linewidth': 0.75,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.prop_cycle': plt.cycler(color = [NAVY, TEAL, ORANGE, TERRACOTTA, PLUM]),
            'xtick.color': DARK_GREY,
            'ytick.color': DARK_GREY,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'text.color': INK,
            'legend.fontsize': 9,
            'legend.frameon': False,
            'lines.linewidth': 1.1,
            'lines.markersize': 4.5,
            'figure.facecolor': WHITE,
            'axes.facecolor': WHITE,
            'savefig.facecolor': WHITE,
            'savefig.transparent': False,
            'savefig.bbox': 'tight',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none'
        }
    )


def clean_axis(axis, grid_axis = None):
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['left', 'bottom']].set_color(LIGHT_GREY)
    axis.tick_params(length = 2.8, width = 0.65, color = MID_GREY)
    
    axis.grid(False)
    if grid_axis:
        axis.grid(axis = grid_axis, color = GRID, linewidth = 0.65, alpha = 0.7)
    
    axis.set_axisbelow(True)
    axis.xaxis.labelpad = 10
    axis.yaxis.labelpad = 7.5


def panel_title(axis, title, legend = False):
    axis.set_title(title, loc = 'left', pad = 28 if legend else 8)


def save_figure(figure, name):
    directory = ROOT / 'results' / 'figures'
    directory.mkdir(parents = True, exist_ok = True)
    
    figure.savefig(directory / f'{name}.pdf')
    figure.savefig(directory / f'{name}.svg')
    figure.savefig(directory / f'{name}.png', dpi = 600)


def save_table(frame, name, directory = None):
    directory = ROOT / 'results' / 'tables' if directory is None else directory
    directory.mkdir(parents = True, exist_ok = True)
    
    frame.to_csv(directory / f'{name}.csv', index = False)
