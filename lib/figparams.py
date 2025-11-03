import numpy as np
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple, HandlerBase
import matplotlib.pyplot as plt

fig_width_pt = 2*246.0  
inches_per_pt = 1.0 / 72.27
golden_mean = (np.sqrt(5) - 1.0) / 2.0
fig_width = fig_width_pt * inches_per_pt
fig_height = fig_width * golden_mean
square_size = [fig_width, fig_width]
rect_size = [fig_width, fig_height]

rc_params = {'axes.labelsize': 18,
          'axes.titlesize': 24,
          'font.size': 18,
          'legend.fontsize': 18,
          'font.family': 'serif',
          'font.sans-serif': ["Bitstream Vera Sans"],
          'mathtext.fontset': "cm",
          "font.serif": ["Computer Modern Roman"],
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'text.usetex': True,
          'text.latex.preamble': r"""\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}""",
          'figure.figsize': rect_size,
         }

class HandlerVerticalLines(HandlerTuple):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line1, line2,line3 = orig_handle
        y_offset = height/1.5
        line1_artist = super().create_artists(legend, (line1,), xdescent, ydescent + y_offset, width, height, fontsize, trans)[0]
        line2_artist = super().create_artists(legend, (line2,), xdescent, ydescent, width, height, fontsize, trans)[0]
        line3_artist = super().create_artists(legend, (line3,), xdescent, ydescent - y_offset, width, height, fontsize, trans)[0]        
        return [line1_artist, line2_artist, line3_artist]

class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        c = orig_handle.get_color()
        l1 = plt.Line2D([x0,y0+width], [1.0*height,1.0*height],
                         color=c, ls="-")
        l2 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height], 
                           color=c, ls="--")
        l3 = plt.Line2D([x0,y0+width], [0.0*height,0.0*height],
                         color=c, ls=":")
        return [l1, l2, l3]

class HandlerThreeLines(HandlerBase):
    def __init__(self, colors, spacing, **kwargs):
        super().__init__(**kwargs)
        self.colors = colors  # List or tuple of three colors
        self.spacing = spacing
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y_offsets = [-height * self.spacing, 0, height * self.spacing]
        lines = []
        for y_off, color in zip(y_offsets, self.colors):
            line = mlines.Line2D(
                [xdescent, xdescent + width],
                [ydescent + height / 2 + y_off, ydescent + height / 2 + y_off],
                color=color,
                linestyle="solid",
                transform=trans,
            )
            lines.append(line)
        return lines
        
class HandlerThreeMarkers(HandlerBase):
    def __init__(self, colors, spacing=0.25, marker="o", markersize=8, markeredgecolor=None,  **kwargs):
        super().__init__(**kwargs)
        self.colors = colors  # List or tuple of three colors
        self.marker = marker
        self.markersize = markersize
        self.spacing = spacing
        self.markeredgecolor = markeredgecolor

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Define horizontal offsets for the three markers
        x_offsets = [-width * self.spacing, 0, width * self.spacing]
        artists = []

        for x_off, color in zip(x_offsets, self.colors):
            marker = mlines.Line2D(
                [xdescent + width / 2 + x_off],
                [ydescent + height / 2],
                color=color,
                marker=self.marker,
                linestyle="None",
                markersize=self.markersize,
                transform=trans,
                markeredgecolor = self.markeredgecolor if self.markeredgecolor else None
            )
            artists.append(marker)

        return artists


