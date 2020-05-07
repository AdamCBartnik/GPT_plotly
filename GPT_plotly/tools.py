import numpy as np
import matplotlib as mpl
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from .nicer_units import *

def get_dist_plot_type(dist_y):
    dist_y = dist_y.lower()
    
    if (dist_y == 'charge density'):
        var='charge'
    if (dist_y == 'emittance x'):
        var='norm_emit_x'
    if (dist_y == 'emittance y'):
        var='norm_emit_y'
    if (dist_y == 'emittance 4d'):
        var='sqrt_norm_emit_4d'
    if (dist_y == 'sigma x'):
        var='sigma_x'
    if (dist_y == 'sigma y'):
        var='sigma_y'
    
    return var

def get_trend_vars(trend_y):
    trend_y = trend_y.lower()
    
    if (trend_y == 'beam size'):
        var=['sigma_x', 'sigma_y']
    if (trend_y == 'bunch length'):
        var='sigma_t'
    if (trend_y == 'emittance (x,y)'):
        var=['norm_emit_x', 'norm_emit_y']
    if (trend_y == 'emittance (4d)'):
        var=['sqrt_norm_emit_4d']
    if (trend_y == 'slice emit. (x,y)'):
        var=['slice_emit_x', 'slice_emit_y']
    if (trend_y == 'slice emit. (4d)'):
        var=['slice_emit_4d']
    if (trend_y == 'energy'):
        var='mean_energy'
    if (trend_y == 'trajectory'):
        var=['mean_x', 'mean_y']
    if (trend_y == 'charge'):
        var='mean_charge'
    
    return var
        
    
    
    
def format_label(s, latex=True, use_base=False, add_underscore=True):
    if (use_base):
        s = s.replace("mean_", "").replace("sigma_", "").replace("norm_", "").replace("sqrt_", "").replace("slice_", "")
    if (add_underscore):
        s = s.replace('px', 'p_x')
        s = s.replace('py', 'p_y')
        s = s.replace('pz', 'p_z')
    if (latex):
        s = s.replace('sigma','\sigma')
        s = s.replace('theta', '\theta')
        s = s.replace('slice_emit', '\mbox{Slice }\epsilon')
        s = s.replace('sqrt_norm_emit', '\epsilon')
        s = s.replace('norm_emit', '\epsilon')
        s = s.replace('emit', '\epsilon')
        s = s.replace('4d','{4d}')
    else:
        s = s.replace('\sigma','sigma')
        s = s.replace('\theta', 'theta')
        s = s.replace('\epsilon', 'emit')
        s = s.replace('\mu ', 'μ')
        s = s.replace('\mu', 'μ')
        s = s.replace('sigma','σ')
        s = s.replace('theta', 'θ')
        s = s.replace('slice_emit', 'slice ε')
        s = s.replace('sqrt_norm_emit', 'ε')
        s = s.replace('norm_emit', 'ε')
        s = s.replace('emit', 'ε')
    s = s.replace('kinetic_energy', 'K')
    s = s.replace('energy', 'E')
    return s

def get_y_label(var):
    ylabel_str = '\mbox{Value}'
    if all('norm_' in var_str for var_str in var):
        ylabel_str = '\mbox{Emittance}'
    if all('slice_emit' in var_str for var_str in var):
        ylabel_str = '\mbox{Slice Emittance}'
    if all('sigma_' in var_str for var_str in var):
        ylabel_str = '\mbox{Beam Size}'
    if all('sigma_t' in var_str for var_str in var):
        ylabel_str = '\mbox{Bunch Length}'
    if all('charge' in var_str for var_str in var):
        ylabel_str = '\mbox{Charge}'
    if all('energy' in var_str for var_str in var):
        ylabel_str = '\mbox{Energy}'
    
    return ylabel_str


def mean_weights(x,w):
    return np.sum(x*w)/np.sum(w)


def std_weights(x,w):
    w_norm = np.sum(w)
    x_mean = np.sum(x*w)/w_norm
    x_m = x - x_mean
    return np.sqrt(np.sum(x_m*x_m*w)/w_norm)
    
    
def corr_weights(x,y,w):
    w_norm = np.sum(w)
    x_mean = np.sum(x*w)/w_norm
    y_mean = np.sum(y*w)/w_norm
    x_m = x - x_mean
    y_m = y - y_mean
    return np.sum(x_m*y_m*w)/w_norm



def duplicate_points_for_hist_plot(edges, hist):
    hist_plt = np.empty((hist.size*2,), dtype=hist.dtype)
    edges_plt = np.empty((hist.size*2,), dtype=hist.dtype)
    hist_plt[0::2] = hist
    hist_plt[1::2] = hist
    edges_plt[0::2] = edges[:-1]
    edges_plt[1::2] = edges[1:]
    
    return (edges_plt, hist_plt)


def get_screen_data(gpt_data, **params):
    if (len(gpt_data.screen) == 0):
         raise ValueError('No screen data found.')
    
    screen_key = None
    screen_value = None
    
    if ('screen_key' in params and 'screen_value' in params):
        screen_key = params['screen_key']
        screen_value = params['screen_value']
        
    if ('screen_z' in params):
        screen_key = 'z'
        screen_value = params['screen_z']
        
    if ('screen_t' in params):
        screen_key = 't'
        screen_value = params['screen_t']
        
    if (screen_key is not None and screen_value is not None):

        values = np.zeros(len(gpt_data.screen)) * np.nan
        
        for ii, screen in enumerate(gpt_data.screen, start=0):
            values[ii] = np.mean(screen[screen_key])
        
        screen_index = np.argmin(np.abs(values-screen_value))
        found_screen_value = values[screen_index]
        #print(f'Found screen at {screen_key} = {values[screen_index]}')
    else:
        print('Defaulting to screen[0]')
        screen_index = 0
        screen_key = 'index'
        found_screen_value = 0
    
    screen = gpt_data.screen[screen_index]
    
    return (screen, screen_key, found_screen_value)
        

def make_default_plot(fig, plot_width=600, plot_height=300, column_widths = None, specs = None, is_table=False, **params):
    
    if (fig is None):
        fig = go.FigureWidget()

    fig.update_layout(overwrite=True,
                      autosize=True, 
                      width=plot_width, 
                      height=plot_height,
                      margin=dict(
                        l=30,
                        r=30,
                        b=30,
                        t=30,
                        pad=0),
                      paper_bgcolor="White",
                      plot_bgcolor="White")
    
    #fig.layout.xaxis.pop('scaleanchor')
    #fig.layout.xaxis.pop('scaleratio')
    #fig.layout.yaxis.pop('scaleanchor')
    #fig.layout.yaxis.pop('scaleratio')
        
    #fig.update_yaxes(automargin=True)
    
    if (is_table==False):
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    else:
        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, 
                                xaxis=dict(showgrid=False, zeroline=False), 
                                yaxis=dict(showgrid=False, zeroline=False), 
                                paper_bgcolor="White", plot_bgcolor="White")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
    
    fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)

    return fig



def scale_and_get_units(x, x_base_units):
    x, x_scale, x_prefix = nicer_array(x)
    x_unit_str = check_mu(x_prefix)+x_base_units
    
    return (x, x_unit_str, x_scale)
    

    
def scale_mean_and_get_units(x, x_base_units, subtract_mean=True, weights=None):
    if (weights is None):
        mean_x = np.mean(x)
    else:
        mean_x = mean_weights(x,weights)
    if (subtract_mean):
        x = x - mean_x
    if (np.abs(mean_x) < 1.0e-24):
        mean_x_scale = 1
        mean_x_prefix = ''
    else:
        mean_x, mean_x_scale, mean_x_prefix = nicer_array(mean_x)
    mean_x_unit_str = check_mu(mean_x_prefix)+x_base_units
    x, x_unit_str, x_scale = scale_and_get_units(x, x_base_units)
    
    return (x, x_unit_str, x_scale, mean_x, mean_x_unit_str, mean_x_scale)
    

def check_mu(str):
    if (str == 'u'):
        return '\mu '
    return str
    
    

def pad_data_with_zeros(edges_plt, hist_plt, sides=[True,True]):
    dx = edges_plt[1] - edges_plt[0]
    if (sides[0]):
        edges_plt = np.concatenate((np.array([edges_plt[0]-2*dx,edges_plt[0]-dx]), edges_plt))
        hist_plt = np.concatenate((np.array([0,0]), hist_plt))
    if (sides[1]):
        edges_plt = np.concatenate((edges_plt, np.array([edges_plt[-1]+dx,edges_plt[-1]+2*dx])))
        hist_plt = np.concatenate((hist_plt, np.array([0,0])))
    return (edges_plt, hist_plt)



def hist2d(fig,x,y,weights=None,bins=[100,100],colormap=None, density=False, is_radial_var=[False,False], colorbar_x=None):
    if (is_radial_var[0]):
        x = x*x
    if (is_radial_var[1]):
        y = y*y
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights, density=density)
    if (is_radial_var[0]):
        xedges = np.sqrt(xedges)
    if (is_radial_var[1]):
        yedges = np.sqrt(yedges)
    H = H.T
    
    colorscale = 'Viridis'
    if (colormap is not None):
        colorscale = [[i/(colormap.N-1), "#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2]))] for i, c in enumerate(255*colormap(range(colormap.N)))]
    
    trace = go.Heatmap(
        hoverinfo="none",  #hovertemplate = '%{x}, %{y}, %{z}<extra></extra>',
        x = xedges,
        y = yedges,
        z = H,
        type = 'heatmap',
        colorscale = colorscale)     
        
    return trace


def scatter_hist2d(fig, x, y, bins=10, range=None, density=False, weights=None,
                   colormap = mpl.cm.get_cmap('jet'), dens_func=None, is_radial_var=[False, False], **kwargs):
    if (is_radial_var[0]):
        x = x*x
    if (is_radial_var[1]):
        y = y*y
    h, xe, ye = np.histogram2d(x, y, bins=bins, range=range, density=density, weights=weights)
    dens = map_hist(x, y, h, bins=(xe, ye))
    if (is_radial_var[0]):
        x = np.sqrt(x)
        xe = np.sqrt(xe)
    if (is_radial_var[1]):
        y = np.sqrt(y)
        ye = np.sqrt(ye)
    if dens_func is not None:
        dens = dens_func(dens)
    iorder = slice(None)  # No ordering by default
    iorder = np.argsort(dens)
    x = x[iorder]
    y = y[iorder]
    dens = dens[iorder]

    if (colormap is None):
        colormap = mpl.cm.get_cmap('jet') 
        
    #colorscale = [[i/(colormap.N-1), "#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2]))] for i, c in enumerate(255*colormap(range(colormap.N)))]    
    colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*colormap(mpl.colors.Normalize(vmin=0.0)(dens))]
    
    scatter = go.Scattergl(x=x, y=y, mode='markers',
                     hoverinfo="none",
                     marker=dict(
                         size=4,
                         color=colors,
                         line_width=0
                         )
                     )

    #palette = [RGB(*tuple(rgb)).to_hex() for rgb in (255*colormap(np.arange(256))).astype('int')]
    #color_mapper = LinearColorMapper(palette=palette, low=0.0, high=np.max(dens))
    #color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0,0))   
    #p.add_layout(color_bar, 'right')
    
    return scatter
    
    
def scatter_color(fig, pmd, x, y, color_var='density', bins=10, weights=None, colormap = None, is_radial_var=[False, False], **kwargs):
    
    force_zero = False
       
    if (color_var=='density'):
        x2 = x
        y2 = y
        if (is_radial_var[0]):
            x2 = x*x
        if (is_radial_var[1]):
            y2 = y*y
        h, xe, ye = np.histogram2d(x2, y2, bins=bins, weights=weights)
        c = map_hist(x2, y2, h, bins=(xe, ye))
        force_zero = True
        title_str = 'Density'
    else:
        q = pmd.weight
        (c, c_units, c_scale, avgc, avgc_units, avgc_scale) = scale_mean_and_get_units(getattr(pmd, color_var), pmd.units(color_var).unitSymbol, subtract_mean=True, weights=q)
        title_str = f'{color_var} ({c_units})'
    
    if (colormap is None):
        colormap = mpl.cm.get_cmap('jet') 
    colorscale = [[i/(colormap.N-1), "#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2]))] for i, c in enumerate(255*colormap(range(colormap.N)))]
   
    cmin = np.min(c)
    cmax = np.max(c)
    if (force_zero):
        cmin = 0.0

    scatter = go.Scattergl(x=x, y=y, mode='markers',
                     hoverinfo="none",
                     marker=dict(
                         size=4,
                         color=c,
                         cmin=cmin,
                         cmax=cmax,
                         line_width=0,
                         colorscale=colorscale,
                         colorbar=dict(title=title_str)
                         )
                     )

    #palette = [RGB(*tuple(rgb)).to_hex() for rgb in (255*colormap(np.arange(256))).astype('int')]
    #color_mapper = LinearColorMapper(palette=palette, low=0.0, high=np.max(dens))
    #color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0,0))   
    #p.add_layout(color_bar, 'right')
    
    return scatter
    
#    color='#e41a1c'
    
    
def map_hist(x, y, h, bins):
    xi = np.digitize(x, bins[0]) - 1
    yi = np.digitize(y, bins[1]) - 1
    inds = np.ravel_multi_index((xi, yi),
                                (len(bins[0]) - 1, len(bins[1]) - 1),
                                mode='clip')
    vals = h.flatten()[inds]
    bads = ((x < bins[0][0]) | (x > bins[0][-1]) |
            (y < bins[1][0]) | (y > bins[1][-1]))
    vals[bads] = np.NaN
    return vals


def radial_histogram_no_units(r, weights=None, nbins=1000):

    """ Performs histogramming of the varibale r using non-equally space bins """
    r2 = r*r
    dr2 = (max(r2)-min(r2))/(nbins-2);
    r2_edges = np.linspace(min(r2), max(r2) + dr2, nbins);
    dr2 = r2_edges[1]-r2_edges[0]
    edges = np.sqrt(r2_edges)
    
    which_bins = np.digitize(r2, r2_edges)-1
    minlength = r2_edges.size-1
    hist = np.bincount(which_bins, weights=weights, minlength=minlength)/(np.pi*dr2)

    return (hist, edges)
    
    

def make_parameter_table(fig, data, headers):
    
    for key in data:
        if (key not in headers):
            raise ValueError(f'Header dictionary does not contain: {key}')  
    
    header_values = [headers[key] for key in headers]
    data_values = [data[key] for key in data]
    
    table = go.Table(
                header=dict(
                    values=header_values,
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=data_values,
                    align = "left")
            )

    return table


def add_row(data, **params):
    for p in params:
        if (p not in data):
            raise ValueError('Column not found')
        data[p] = data[p] + [params[p]]
        
    return data

