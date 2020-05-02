
import numpy as np
import matplotlib as mpl
import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .tools import *
from .nicer_units import *


def gpt_plot(gpt_data, var1, var2, units=None, fig=None, show_plot=True, **params):
    
    fig = make_default_plot(fig, plot_width=600, plot_height=400, **params)
    
    # Use MatPlotLib default line colors
    mpl_cmap = mpl.pyplot.get_cmap('Set1') # 'Set1', 'tab10'
    cmap = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl_cmap(range(mpl_cmap.N))]
    
    # Find good units for x data
    (x, x_units, x_scale) = scale_and_get_units(gpt_data.stat(var1, 'tout'), gpt_data.stat_units(var1).unitSymbol)
    screen_x = gpt_data.stat(var1, 'screen') / x_scale
    
    if (not isinstance(var2, list)):
        var2 = [var2]

    # Combine all y data into single array to find good units
    all_y = np.array([])
    all_y_base_units = gpt_data.stat_units(var2[0]).unitSymbol
    for var in var2:
        if (gpt_data.stat_units(var).unitSymbol != all_y_base_units):
            raise ValueError('Plotting data with different units not allowed.')
        if (var == 'sigma_t'):
            # sigma_t is a special case. Make a copy of the GPT data and use drift_to_z to get fake screen outputs
            gpt_data_copy = copy.deepcopy(gpt_data)
            for tout in gpt_data_copy.tout:
                tout.drift_to_z()
            all_y = np.concatenate((all_y, gpt_data_copy.stat(var)))  # touts and screens for unit choices
        else:
            all_y = np.concatenate((all_y, gpt_data.stat(var)))  # touts and screens for unit choices

    # In the case of emittance, use 2*median(y) as a the default scale, to avoid solenoid growth dominating the choice of scale
    use_median_y_strs = ['norm_']
    if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
        (_, y_units, y_scale) = scale_and_get_units(2.0*np.median(all_y), all_y_base_units)
        all_y = all_y / y_scale
    else:
        (all_y, y_units, y_scale) = scale_and_get_units(all_y, all_y_base_units)
    
    # Finally, actually plot the data
    for i, var in enumerate(var2):
        if (var == 'sigma_t'):
            y = gpt_data_copy.stat(var, 'tout') / y_scale
        else:
            y = gpt_data.stat(var, 'tout') / y_scale
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'${format_label(var)}$',
                        hovertemplate = '%{x}, %{y}<extra></extra>',
                        line=dict(
                             color=cmap[i % len(cmap)]
                         )
                     ))

        screen_y = gpt_data.stat(var, 'screen') / y_scale
        fig.add_trace(go.Scatter(x=screen_x, y=screen_y, mode='markers', name=f'$\mbox{{Screen: }}{format_label(var)}$',
                         hovertemplate = '%{x}, %{y}<extra></extra>',
                         marker=dict(
                             size=8,
                             color=cmap[i % len(cmap)],
                             line_width=1
                         )
                     ))
        
    # Axes labels
    ylabel_str = get_y_label(var2)
    fig.update_xaxes(title_text=f"${format_label(var1, use_base=True)} \, ({x_units})$")
    fig.update_yaxes(title_text=f"${ylabel_str} \, ({y_units})$")
    
    # Turn off legend if not desired
    if('legend' in params):
        if (isinstance(params['legend'], bool)):
            fig.update_layout(showlegend=params['legend'])
    
    # Cases where the y-axis should be forced to start at 0
    # plotly seems to require specifying both sides of the range, so I also set the max range here
    zero_y_strs = ['sigma_', 'norm_', 'charge', 'energy']
    if (any(any(substr in varstr for substr in zero_y_strs) for varstr in var2)):
        fig.update_yaxes(range=[0, 1.1*np.max(all_y)])      
    
    # Cases where the y-axis range should use the median, instead of the max (e.g. emittance plots)
    use_median_y_strs = ['norm_']
    if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
        fig.update_yaxes(range=[0, 2.0*np.median(all_y)])
        
    if show_plot:
        fig.show(config = {'displaylogo': False})
    


def gpt_plot_dist1d(pmd, var, units=None, fig=None, show_plot=True, table_on=True, **params):
    
    if (table_on):
        fig = make_default_plot(fig, cols=2, plot_width=900, plot_height=400, column_widths=[500, 400],  
                                specs=[[{"type": "scatter"}, {"type": "table"}]], **params)
    else:
        fig = make_default_plot(fig, plot_width=500, plot_height=400, **params)
    
    # Use MatPlotLib default line colors
    mpl_cmap = mpl.pyplot.get_cmap('Set1') # 'Set1', 'tab10'
    cmap = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl_cmap(range(mpl_cmap.N))]
    
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    else:
        pmd = postprocess_screen(pmd, **params)
        screen_key = None
        screen_value = None
            
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50

    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
        
    subtract_mean = False
    if (var in ['x','y','z','t']):
        subtract_mean = True
    (x, x_units, x_scale, mean_x, mean_x_units, mean_x_scale) = scale_mean_and_get_units(getattr(pmd, var), pmd.units(var).unitSymbol,
                                                                                         subtract_mean=subtract_mean, weights=q)
    
    if (var == 'r'):
        hist, edges = radial_histogram_no_units(x, weights=q, nbins=nbins)
    else:
        hist, edges = np.histogram(x,bins=nbins,weights=q,density=True)
        hist *= q_total;

    edges, hist = duplicate_points_for_hist_plot(edges, hist)
    
    if (var != 'r'):
        edges, hist = pad_data_with_zeros(edges, hist)

    fig.add_trace(go.Scatter(x=edges, y=hist, mode='lines', name=f'${format_label(var)}$',
                    hovertemplate = '%{x}, %{y}<extra></extra>',
                    line=dict(
                         color=cmap[0]
                     )
                 ), row=1, col=1)
    fig.update_xaxes(title_text=f"${format_label(var)} \, ({x_units})$")
    
    if (var == 'r'):
        fig.update_yaxes(title_text=f"$\mbox{{Charge density}} \, ({q_units}/{x_units}^2)$")
    else:
        fig.update_yaxes(title_text=f"$\mbox{{Charge density}} \, ({q_units}/{x_units})$")
    
    stdx = std_weights(x,q)
    
    if(table_on):
        var_label = format_label(var, add_underscore=False)
        data = dict(col1=[], col2=[], col3=[])
        if (screen_key is not None):
            data = add_row(data, col1=f'Screen {screen_key}', col2=f'{screen_value:G}', col3='')
        data = add_row(data, col1=f'Total charge', col2=f'{q_total:G}', col3=f'{q_units}')
        data = add_row(data, col1=f'Mean {var_label}', col2=f'{mean_x:G}', col3=f'{mean_x_units}')
        data = add_row(data, col1=f'σ_{var_label}', col2=f'{stdx:G}', col3=f'{x_units}')
        headers = dict(col1='Name', col2='Value', col3='Units')
        fig.add_trace(
            make_parameter_table(fig, data, headers),
            row=1, col=2)
    
    if show_plot:
        fig.show(config = {'displaylogo': False})
            

def gpt_plot_dist2d(pmd, var1, var2, ptype='hist2d', units=None, fig=None, show_plot=True, table_on=True, **params):

    if (table_on):
        column_widths=[500, 75, 400]
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "table"}]]
        fig = make_default_plot(fig, cols=len(column_widths), plot_width=sum(column_widths), plot_height=400, column_widths=column_widths,  
                                specs=specs, **params)
        colorbar_x = column_widths[0] / sum(column_widths)
    else:
        fig = make_default_plot(fig, plot_width=500, plot_height=400, **params)
         
    screen_key = None
    screen_value = None
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    else:
        pmd = postprocess_screen(pmd, **params)
        screen_key = None
        screen_value = None
            
    if('axis' in params and params['axis']=='equal'):
        fig.update_layout(
            yaxis = dict(
              scaleanchor = "x",
              scaleratio = 1,
            )
        )
    
    is_radial_var = [False, False]
    if (var1 == 'r'):
        is_radial_var[0] = True
    if (var2 == 'r'):
        is_radial_var[1] = True
        
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50
        
    if (not isinstance(nbins, list)):
        nbins = [nbins, nbins]
        
    if ('colormap' in params):
        colormap = mpl.cm.get_cmap(params[colormap])
    else:
        colormap = mpl.cm.get_cmap('jet') 

    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
    
    (x, x_units, x_scale, avgx, avgx_units, avgx_scale) = scale_mean_and_get_units(getattr(pmd, var1), pmd.units(var1).unitSymbol, subtract_mean= not is_radial_var[0], weights=q)
    (y, y_units, y_scale, avgy, avgy_units, avgy_scale) = scale_mean_and_get_units(getattr(pmd, var2), pmd.units(var2).unitSymbol, subtract_mean= not is_radial_var[1], weights=q)
                
    if(ptype=="scatter"):
        fig.add_trace(go.Scattergl(x=x, y=y, mode='markers',
                     hoverinfo="none",
                     marker=dict(
                         size=4,
                         color='#e41a1c',  # first color in matplotlib cmap 'Set1'
                         line_width=0
                         )
                     ), row=1, col=1)
    if(ptype=="hist2d"):
        fig.add_trace(
            hist2d(fig, x, y, bins=nbins, weights=q, colormap=colormap, is_radial_var=is_radial_var, colorbar_x=colorbar_x),
            row=1, col=1
        )
    if(ptype=="scatter_hist2d"):
        fig.add_trace(
            scatter_hist2d(fig, x, y, bins=nbins, weights=q, colormap=colormap, is_radial_var=is_radial_var),
            row=1, col=1
        )
        
    fig.update_xaxes(title_text=f"${format_label(var1)} \, ({x_units})$")
    fig.update_yaxes(title_text=f"${format_label(var2)} \, ({y_units})$")
             
    stdx = std_weights(x,q)
    stdy = std_weights(y,q)
    corxy = corr_weights(x,y,q)
    if (x_units == y_units):
        corxy_units = f'{x_units}²'
    else:
        corxy_units = f'{x_units}·{y_units}'
    
    show_emit = False
    if ((var1 == 'x' and var2 == 'px') or (var1 == 'y' and var2 == 'py')):
        show_emit = True
        factor = c_light**2 /e_charge # kg -> eV
        particle_mass = 9.10938356e-31  # kg
        emitxy = (x_scale*y_scale/factor/particle_mass)*np.sqrt(stdx**2 * stdy**2 - corxy**2)
        (emitxy, emitxy_units, emitxy_scale) = scale_and_get_units(emitxy, pmd.units(var1).unitSymbol)
    
    if(table_on):
        x_units = format_label(x_units, latex=False)
        y_units = format_label(y_units, latex=False)
        avgx_units = format_label(avgx_units, latex=False)
        avgy_units = format_label(avgy_units, latex=False)
        emitxy_units = format_label(emitxy_units, latex=False)
        var1_label = format_label(var1, add_underscore=False, latex=False)
        var2_label = format_label(var2, add_underscore=False, latex=False)
        data = dict(col1=[], col2=[], col3=[])
        if (screen_key is not None):
            data = add_row(data, col1=f'Screen {screen_key}', col2=f'{screen_value:G}', col3='')
        data = add_row(data, col1=f'Mean {var1_label}', col2=f'{avgx:G}', col3=f'{avgx_units}')
        data = add_row(data, col1=f'Mean {var2_label}', col2=f'{avgy:G}', col3=f'{avgy_units}')
        data = add_row(data, col1=f'σ_{var1_label}', col2=f'{stdx:G}', col3=f'{x_units}')
        data = add_row(data, col1=f'σ_{var2_label}', col2=f'{stdy:G}', col3=f'{y_units}')
        data = add_row(data, col1=f'Corr({var1_label}, {var2_label})', col2=f'{corxy:G}', col3=f'{corxy_units}')
        if (show_emit):
            data = add_row(data, col1=f'ε_{var1_label}', col2=f'{emitxy:G}', col3=f'{emitxy_units}')
        headers = dict(col1='Name', col2='Value', col3='Units')
        fig.add_trace(
            make_parameter_table(fig, data, headers),
            row=1,col=3)
    
    
    if show_plot:
        fig.show(config = {'displaylogo': False})
        
    
    
    
