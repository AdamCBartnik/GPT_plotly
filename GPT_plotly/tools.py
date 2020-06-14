import copy
import numpy as np
import matplotlib as mpl
import plotly.graph_objects as go
from .nicer_units import *
from .ParticleGroupExtension import ParticleGroupExtension
from scipy.stats import binned_statistic_2d

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
        s = s.replace('charge','\mbox{charge }')
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
    s = s.replace('mean_', '')
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


def special_screens(z_input, decimals=6, min_length=10):
    if (len(z_input) < min_length):
        return list(range(0, len(z_input)))
    
    z = np.sort(z_input)
    z_copy = copy.copy(z)
    
    if (z[0] == 0.0):
        z = np.delete(z, 0)
        special_z = [0.0]
    else:
        special_z = []
    
    dz = np.diff(z)
    (values,indices,counts) = np.unique(dz.round(decimals=decimals), return_counts=True, return_inverse=True)
    
    while (len(values) > 1):
        z_index = np.argmax(indices != np.argmax(counts))
        if (z_index > 0):
            z_index = z_index+1  # Assume the special screen is the former when the index is zero, otherwise it is the latter
        special_z += [z[z_index]]
        z = np.delete(z, z_index)
        dz = np.diff(z)
        (values,indices,counts) = np.unique(dz.round(decimals=6), return_counts=True, return_inverse=True)
        
    special_indices = [i for i, zz in enumerate(z_copy) if zz in special_z ]
    return special_indices


def get_screen_data(gpt_data, verbose=False, use_extension=True, **params):   
    use_touts = False
    screen_key = None
    screen_value = None
    
    if ('screen_key' in params and 'screen_value' in params):
        screen_key = params['screen_key']
        screen_value = params['screen_value']
        
    if ('tout_z' in params):
        use_touts = True
        screen_key = 'z'
        screen_value = params['tout_z']
        
    if ('tout_t' in params):
        use_touts = True
        screen_key = 't'
        screen_value = params['tout_t']
        
    if ('screen_z' in params):
        screen_key = 'z'
        screen_value = params['screen_z']
        
    if ('screen_t' in params):
        screen_key = 't'
        screen_value = params['screen_t']
        
    if (use_touts == False):
        screen_list = gpt_data.screen
        if (len(gpt_data.screen) == 0):
             raise ValueError('No screen data found.')
    else:
        screen_list = gpt_data.tout
        if (len(gpt_data.tout) == 0):
             raise ValueError('No tout data found.')
        
    if (screen_key is not None and screen_value is not None):

        values = np.zeros(len(screen_list)) * np.nan
        
        for ii, screen in enumerate(screen_list, start=0):
            values[ii] = screen['mean_'+screen_key]
        
        screen_index = np.argmin(np.abs(values-screen_value))
        found_screen_value = values[screen_index]
        if (verbose):
            print(f'Found screen at {screen_key} = {values[screen_index]}')
    else:
        if (verbose):
            print('Defaulting to screen[0]')
        screen_index = 0
        screen_key = 'index'
        found_screen_value = 0
    
    if (use_extension):
        screen = ParticleGroupExtension(input_particle_group=screen_list[screen_index])
    else:
        screen = copy.deepcopy(screen_list[screen_index])
    
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

    
def check_subtract_mean(var):
    subtract_mean = False
    if (var in ['x', 'y', 'z', 't']):
        subtract_mean = True

    #subtract_mean = True
    #if ('action' in var):
    #    subtract_mean = False
    #if (var in ['r', 'pr']):
    #    subtract_mean = False
        
    return subtract_mean

    

def hist2d(fig, pmd, x, y, weights, color_var='density', bins=[100,100], colormap=mpl.cm.get_cmap('jet'), is_radial_var=[False,False]):
    force_zero = False
    use_separate_data = False
        
    if (isinstance(color_var, tuple)):
        color_var_data = color_var[1]
        color_var = color_var[0]
        use_separate_data = True
        x_c = color_var_data.x
        y_c = color_var_data.y
    else:
        color_var_data = pmd
        x_c = x
        y_c = y
    
    if (is_radial_var[0]):
        x = x*x
    if (is_radial_var[1]):
        y = y*y
                
    if (color_var == 'density'):
        # Here I use mean*count instead of sum, in order to have empty bins = NaN
        H_mean, xedges, yedges, _ = binned_statistic_2d(x, y, weights, statistic='mean', bins=bins)
        H_count, xedges, yedges, _ = binned_statistic_2d(x, y, weights, statistic='count', bins=bins)
        H = H_mean*H_count / (xedges[1]-xedges[0]) / (yedges[1]-yedges[0])
        title_str = 'Density'
    else:
        q = color_var_data.weight
        
        (c, c_units, c_scale, avgc, avgc_units, avgc_scale) = scale_mean_and_get_units(getattr(color_var_data, color_var), color_var_data.units(color_var).unitSymbol, 
                                                                                       subtract_mean=check_subtract_mean(color_var), weights=q)
        
        title_str = f'{color_var} ({format_label(c_units, latex=False)})'

        if (use_separate_data):
            # Reorder to order from pmd
            c_id = color_var_data.id
            c_dict = {id : i for i,id in enumerate(c_id)}
            c = np.array([c[c_dict[id]] if id in c_dict else np.nan for id in pmd.id])

        has_color = np.logical_not(np.isnan(c))
        x = x[has_color]
        y = y[has_color]
        c = c[has_color]
        H, xedges, yedges, _ = binned_statistic_2d(x, y, c, statistic='mean', bins=bins)
    
    if (is_radial_var[0]):
        xedges = np.sqrt(xedges)
    if (is_radial_var[1]):
        yedges = np.sqrt(yedges)
    H = H.T
        
    colorscale = [[i/(colormap.N-1), "#%02x%02x%02x" % (int(cc[0]), int(cc[1]), int(cc[2]))] for i, cc in enumerate(255*colormap(range(colormap.N)))]
        
    force_zero_vars = ['r', 'pr','density']
    force_zero = color_var in force_zero_vars
        
    zmin = np.nanmin(H)
    zmax = np.nanmax(H)
    if (force_zero):
        zmin = 0.0
        
    trace = go.Heatmap(
        hoverinfo="none",  #hovertemplate = '%{x}, %{y}, %{z}<extra></extra>',
        x = xedges,
        y = yedges,
        z = H,
        type = 'heatmap',
        zmin = zmin,
        zmax = zmax,
        colorscale = colorscale,
        colorbar=dict(title=title_str))     
        
    return trace

    
def scatter_color(fig, pmd, x, y, weights=None, color_var='density', bins=100, colormap=mpl.cm.get_cmap('jet'), is_radial_var=[False, False], **kwargs):
    
    force_zero = False
    use_separate_data = False
        
    if (isinstance(color_var, tuple)):
        color_var_data = color_var[1]
        color_var = color_var[0]
        use_separate_data = True
        x_c = color_var_data.x
        y_c = color_var_data.y
    else:
        color_var_data = pmd
        x_c = x
        y_c = y
        
    if (color_var=='density'):
        x2 = x_c
        y2 = y_c
        if (is_radial_var[0]):
            x2 = x_c*x_c
        if (is_radial_var[1]):
            y2 = y_c*y_c
        h, xe, ye = np.histogram2d(x2, y2, bins=bins, weights=weights)
        c = map_hist(x2, y2, h, bins=(xe, ye))
        force_zero = True
        title_str = 'Density'
    else:
        q = color_var_data.weight
        
        (c, c_units, c_scale, avgc, avgc_units, avgc_scale) = scale_mean_and_get_units(getattr(color_var_data, color_var), color_var_data.units(color_var).unitSymbol, 
                                                                                       subtract_mean=check_subtract_mean(color_var), weights=q)
        
        title_str = f'{color_var} ({format_label(c_units, latex=False)})'
    
    if (use_separate_data):
        # Reorder to order from pmd
        c_id = color_var_data.id
        c_dict = {id : i for i,id in enumerate(c_id)}
        c = [c[c_dict[id]] if id in c_dict else np.nan for id in pmd.id]
    
    colorscale = [[i/(colormap.N-1), "#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2]))] for i, c in enumerate(255*colormap(range(colormap.N)))]
    
    cmin = np.nanmin(c)
    cmax = np.nanmax(c)
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

    
    return scatter
        
    
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


