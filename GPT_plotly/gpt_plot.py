import numpy as np
import matplotlib as mpl
from gpt.gpt import GPT as GPT
import plotly.graph_objects as go
from .tools import *
from .nicer_units import *
from .postprocess import postprocess_screen
from pmd_beamphysics.units import c_light, e_charge
from .ParticleGroupExtension import ParticleGroupExtension, convert_gpt_data, divide_particles
from ipywidgets import HBox

def gpt_plot(gpt_data_input, var1, var2, units=None, fig=None, format_input_data=True, show_survivors_at_z=None, **params):
    if (format_input_data):
        gpt_data = convert_gpt_data(gpt_data_input)
    else:
        gpt_data = copy.deepcopy(gpt_data_input)
        
    if (show_survivors_at_z is not None):
        survivor_params = {}
        survivor_params['screen_z'] = show_survivors_at_z
        pmd, _, _ = get_screen_data(gpt_data, **survivor_params)
        pmd = postprocess_screen(pmd, **params)
        survivor_ids = pmd.id[pmd.weight > 0]

        survivor_params = {}
        survivor_params['include_ids'] = survivor_ids
        survivor_params['need_copy'] = False
        for i, s in enumerate(gpt_data.particles):
            gpt_data.particles[i] = postprocess_screen(s, **survivor_params)
    else:
        if ('include_ids' in params):
            survivor_params = {}
            survivor_params['include_ids'] = params['include_ids']
            survivor_params['need_copy'] = False
            for i, s in enumerate(gpt_data.particles):
                gpt_data.particles[i] = postprocess_screen(s, **survivor_params)
        
    
    fig = make_default_plot(fig, plot_width=600, plot_height=400, **params)
    
    # Use MatPlotLib default line colors
    mpl_cmap = mpl.pyplot.get_cmap('Set1') # 'Set1', 'tab10'
    cmap = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl_cmap(range(mpl_cmap.N))]
    
    # Find good units for x data
    special_i = special_screens(gpt_data.stat('mean_z', 'screen'))
    (all_x, x_units, x_scale) = scale_and_get_units(gpt_data.stat(var1, 'screen'), gpt_data.units(var1).unitSymbol)
    
    #x = gpt_data.stat(var1, 'tout') / x_scale
    #screen_x = gpt_data.stat(var1, 'screen') / x_scale
    x = all_x
    screen_x = [xx for i, xx in enumerate(all_x) if i in special_i]
        
    if (not isinstance(var2, list)):
        var2 = [var2]

    if ('n_slices' in params):
        for p in gpt_data.particles:
            p.n_slices = params['n_slices']
    if ('slice_key' in params):
        for p in gpt_data.particles:
            p.slice_key = params['slice_key']
        
    # Combine all y data into single array to find good units
    all_y = np.array([])
    all_y_base_units = gpt_data.units(var2[0]).unitSymbol
    for var in var2:
        if (gpt_data.units(var).unitSymbol != all_y_base_units):
            raise ValueError('Plotting data with different units not allowed.')
        #all_y = np.concatenate((all_y, gpt_data.stat(var)))  # touts and screens for unit choices
        all_y = np.concatenate((all_y, gpt_data.stat(var, 'screen')))  # touts and screens for unit choices

    # In the case of emittance, use 2*median(y) as a the default scale, to avoid solenoid growth dominating the choice of scale
    use_median_y_strs = ['norm', 'slice']
    if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
        (_, y_units, y_scale) = scale_and_get_units(2.0*np.median(all_y), all_y_base_units)
        all_y = all_y / y_scale
    else:
        (all_y, y_units, y_scale) = scale_and_get_units(all_y, all_y_base_units)
                
    # Finally, actually plot the data
    for i, var in enumerate(var2):
        #y = gpt_data.stat(var, 'tout') / y_scale
        y = all_y
        legend_name = f'${format_label(var)}$'
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=legend_name,
                        hovertemplate = '%{x}, %{y}<extra></extra>',
                        line=dict(
                             color=cmap[i % len(cmap)]
                         )
                     ))

        #screen_y = gpt_data.stat(var, 'screen') / y_scale
        screen_y = [yy for i, yy in enumerate(all_y) if i in special_i]
        legend_name = f'$\mbox{{Screen: }}{format_label(var)}$'
        fig.add_trace(go.Scatter(x=screen_x, y=screen_y, mode='markers', name=legend_name,
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
        
    #fig.update_xaxes(title_font=dict(size=18))
        
    # Turn off legend if not desired
    if('legend' in params):
        if (isinstance(params['legend'], bool)):
            fig.update_layout(showlegend=params['legend'])
    
    fig.update_xaxes(range=[np.min(all_x), np.max(all_x)])
    
    # Cases where the y-axis should be forced to start at 0
    # plotly seems to require specifying both sides of the range, so I also set the max range here
    zero_y_strs = ['sigma_', 'charge', 'energy', 'slice']
    if (any(any(substr in varstr for substr in zero_y_strs) for varstr in var2)):
        fig.update_yaxes(range=[0, 1.1*np.max(all_y)])      
    
    # Cases where the y-axis range should use the median, instead of the max (e.g. emittance plots)
    use_median_y_strs = ['norm_emit_x','norm_emit_y']
    if (any(any(substr in varstr for substr in use_median_y_strs) for varstr in var2)):
        fig.update_yaxes(range=[0, 2.0*np.median(all_y)])
        
    if ('xlim' in params):
        fig.update_xaxes(range=params['xlim'])
    if ('ylim' in params):
        fig.update_yaxes(range=params['ylim'])
    
    return fig
    


def gpt_plot_trajectory(gpt_data_input, var1, var2, fig=None, format_input_data=True, **params):
    if (format_input_data):
        gpt_data = convert_gpt_data(gpt_data_input)
    else:
        gpt_data = copy.deepcopy(gpt_data_input)
        
    fig = make_default_plot(fig, plot_width=500, plot_height=400, **params)
        
    # Get set of particle IDs to plot trajectories of
    pmd, _, _ = get_screen_data(gpt_data_input, **params)
    pmd = postprocess_screen(pmd, **params)
    plot_ids = pmd.id[pmd.weight > 0]
    
    if(len(plot_ids)==0):
        return None

    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
    
    x = np.empty( (len(gpt_data.screen),len(plot_ids)) )
    y = np.empty( (len(gpt_data.screen),len(plot_ids)) )
    
    for i, s in enumerate(gpt_data.screen):
        index = np.argsort(s.id)
        sorted_s_id = s.id[index]
        sorted_index = np.searchsorted(sorted_s_id, plot_ids)

        found_index = np.take(index, sorted_index, mode="clip")
        mask = s.id[found_index] != plot_ids

        index_of_found_ids = np.ma.array(found_index, mask=mask)
        x[i,:] = getattr(s, var1)[index_of_found_ids]
        x[i,index_of_found_ids.mask] = np.nan
        y[i,:] = getattr(s, var2)[index_of_found_ids]
        y[i,index_of_found_ids.mask] = np.nan
            
    all_x = x.flatten()
    all_x = all_x[np.logical_not(np.isnan(all_x))]
    (_, x_units, x_scale, _, _, _) = scale_mean_and_get_units(all_x, pmd.units(var1).unitSymbol, subtract_mean=False)
    x = x/x_scale
    
    all_y = y.flatten()
    all_y = all_y[np.logical_not(np.isnan(all_y))]
    (_, y_units, y_scale, _, _, _) = scale_mean_and_get_units(all_y, pmd.units(var2).unitSymbol, subtract_mean=False)
    y = y/y_scale
    
    for j, id in enumerate(plot_ids):
        fig.add_trace(go.Scattergl(x=x[:,j], y=y[:,j], mode='lines',
                    hoverinfo="none"   # hovertemplate = '%{x}, %{y}<extra></extra>',
                 ))
        
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text=f"${format_label(var1)} \, ({x_units})$")
    fig.update_yaxes(title_text=f"${format_label(var2)} \, ({y_units})$")    
    
    if ('xlim' in params):
        fig.update_xaxes(range=params['xlim'])
    if ('ylim' in params):
        fig.update_yaxes(range=params['ylim'])
    
    return fig
    

def gpt_plot_dist1d(pmd, var, plot_type='charge', units=None, fig=None, table_fig=None, table_on=True, **params):
    screen_key = None
    screen_value = None
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    if (not isinstance(pmd, ParticleGroupExtension)):
        pmd = ParticleGroupExtension(input_particle_group=pmd)
    pmd = postprocess_screen(pmd, **params)
                
    plot_type = plot_type.lower()
    density_types = {'charge'}
    is_density = False
    if (plot_type in density_types):
        is_density = True
        
    positive_types = {'charge', 'norm_emit', 'sigma', 'slice'}
    is_positive = False
    if any([d in plot_type for d in positive_types]):
        is_positive = True
        
    min_particles = 1
    needs_many_particles_types = {'norm_emit', 'sigma'}
    if any([d in plot_type for d in positive_types]):
        min_particles = 3
        
    if (table_on):
        fig = make_default_plot(fig, plot_width=500, plot_height=400, **params)
        table_fig = make_default_plot(table_fig, plot_width=400, plot_height=400, is_table=True, **params)
    else:
        fig = make_default_plot(fig, plot_width=500, plot_height=400, **params)
    
    # Use MatPlotLib default line colors
    mpl_cmap = mpl.pyplot.get_cmap('Set1') # 'Set1', 'tab10'
    cmap = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl_cmap(range(mpl_cmap.N))]
    
    if('nbins' in params):
        nbins = params['nbins']
    else:
        nbins = 50
                
    charge_base_units = pmd.units('charge').unitSymbol
    q_total, charge_scale, charge_prefix = nicer_array(pmd.charge)
    q = pmd.weight / charge_scale
    q_units = check_mu(charge_prefix)+charge_base_units
    
    subtract_mean=check_subtract_mean(var)
    (x, x_units, x_scale, mean_x, mean_x_units, mean_x_scale) = scale_mean_and_get_units(getattr(pmd, var), pmd.units(var).unitSymbol,
                                                                                         subtract_mean=subtract_mean, weights=q)
    p_list, edges, density_norm = divide_particles(pmd, nbins=nbins, key=var)
    
    if (var == 'r'):
        density_norm = density_norm*(x_scale*x_scale)
    else:
        density_norm = density_norm*x_scale
    
    if (subtract_mean==True):
        edges = edges - mean_x*mean_x_scale
    edges = edges/x_scale
    
    plot_type_base_units = pmd.units(plot_type).unitSymbol
    _, plot_type_scale, plot_type_prefix = nicer_array(pmd[plot_type])
    plot_type_units = check_mu(plot_type_prefix)+plot_type_base_units
    norm = 1.0/plot_type_scale
    if (is_density):
        norm = norm*density_norm
    
    weights = np.array([0.0 for p in p_list])
    hist = np.array([0.0 for p in p_list])
    for p_i, p in enumerate(p_list):
        if (p.n_particle >= min_particles):
            hist[p_i] = p[plot_type]*norm
            weights[p_i] = p['charge']
    weights = weights/np.sum(weights)
    avg_hist = np.sum(hist*weights)
                
    edges, hist = duplicate_points_for_hist_plot(edges, hist)
    
    if (var != 'r'):
        edges, hist = pad_data_with_zeros(edges, hist)

    fig.add_trace(go.Scatter(x=edges, y=hist, mode='lines', name=f'${format_label(var)}$',
                    hovertemplate = '%{x}, %{y}<extra></extra>',
                    line=dict(
                         color=cmap[0]
                     )
                 ))
    
    fig.update_xaxes(title_text=f"${format_label(var)} \, ({x_units})$")
    
    plot_type_label = get_y_label([plot_type])
    if (is_positive):
        fig.update_yaxes(range=[0, 1.1*np.max(hist)])  
    if (is_density):
        if (var == 'r'):
            y_axis_label=f"${plot_type_label} \, \mbox{{density}} \, ({plot_type_units}/{x_units}^2)$"
        else:
            y_axis_label=f"${plot_type_label} \, \mbox{{density}} \, ({plot_type_units}/{x_units})$"
    else:
        y_axis_label=f"${plot_type_label} \, ({plot_type_units})$"
    
    fig.update_yaxes(title_text=y_axis_label)
    
    if ('xlim' in params):
        fig.update_xaxes(range=params['xlim'])
    if ('ylim' in params):
        fig.update_yaxes(range=params['ylim'])
    
    stdx = std_weights(x,q)
    
    if(table_on):
        x_units = format_label(x_units, latex=False)
        mean_x_units = format_label(mean_x_units, latex=False)
        plot_type_units = format_label(plot_type_units, latex=False)
        q_units = format_label(q_units, latex=False)
        var_label = format_label(var, add_underscore=False)
        plot_type_label = format_label(plot_type, add_underscore=False, latex=False)
        data = dict(col1=[], col2=[], col3=[])
        if (screen_key is not None):
            data = add_row(data, col1=f'Screen {screen_key}', col2=f'{screen_value:G}', col3='')
        data = add_row(data, col1=f'Total charge', col2=f'{q_total:G}', col3=f'{q_units}')
        if (not is_density):
            data = add_row(data, col1=f'Mean {plot_type_label}', col2=f'{avg_hist:G}', col3=f'{plot_type_units}')
        data = add_row(data, col1=f'Mean {var_label}', col2=f'{mean_x:G}', col3=f'{mean_x_units}')
        data = add_row(data, col1=f'σ_{var_label}', col2=f'{stdx:G}', col3=f'{x_units}')
        headers = dict(col1='Name', col2='Value', col3='Units')
        table_fig.add_trace(make_parameter_table(fig, data, headers))
        
    if (table_on):
        return HBox([fig, table_fig])
    else:
        return fig

    
    
    
    
def gpt_plot_dist2d(pmd, var1, var2, plot_type='histogram', units=None, fig=None, table_fig=None, table_on=True, plot_width=500, plot_height=400, **params):

    if (table_on):
        fig = make_default_plot(fig, plot_width=plot_width, plot_height=plot_height, **params)
        table_fig = make_default_plot(table_fig, plot_width=400, plot_height=plot_height, is_table=True, **params)
    else:
        fig = make_default_plot(fig, plot_width=plot_width, plot_height=plot_height, **params)
    
    screen_key = None
    screen_value = None
    if (isinstance(pmd, GPT)):
        pmd, screen_key, screen_value = get_screen_data(pmd, **params)
    if (not isinstance(pmd, ParticleGroupExtension)):
        pmd = ParticleGroupExtension(input_particle_group=pmd)
    pmd = postprocess_screen(pmd, **params)
            
    if (isinstance(var2, tuple)):
        use_separate_data = True
        pmd2 = var2[1]
        var2 = var2[0]
        if (not isinstance(pmd2, ParticleGroupExtension)):
            pmd2 = ParticleGroupExtension(input_particle_group=pmd2)
    else:
        use_separate_data = False
        pmd2 = pmd
        
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
    
    
    
    (x, x_units, x_scale, avgx, avgx_units, avgx_scale) = scale_mean_and_get_units(getattr(pmd, var1), pmd.units(var1).unitSymbol, subtract_mean=check_subtract_mean(var1), weights=q)
    
    y = getattr(pmd2, var2)
    q_y = pmd2.weight / charge_scale
    if (use_separate_data):
        # Reorder to order from pmd
        y_id = pmd2.id
        y_dict = {id : i for i,id in enumerate(y_id)}
        y = np.array([y[y_dict[id]] if id in y_dict else 0.0 for id in pmd.id])  # The value on failure here doesn't matter since it will have weight = 0
        q_y = np.array([q_y[y_dict[id]] if id in y_dict else 0.0 for id in pmd.id])
    
    (y, y_units, y_scale, avgy, avgy_units, avgy_scale) = scale_mean_and_get_units(y, pmd2.units(var2).unitSymbol, subtract_mean=check_subtract_mean(var2), weights=q_y)
    
    color_var = 'density'
    if ('color_var' in params):
        color_var = params['color_var']
    if(plot_type=="scatter"):
        fig.add_trace(scatter_color(fig, pmd, x, y, weights=q, color_var=color_var, bins=nbins, colormap=colormap, is_radial_var=is_radial_var))
    if(plot_type=="histogram"):
        fig.add_trace(hist2d(fig, pmd, x, y, weights=q, color_var=color_var, bins=nbins, colormap=colormap, is_radial_var=is_radial_var))
                
    fig.update_xaxes(title_text=f"${format_label(var1)} \, ({x_units})$")
    fig.update_yaxes(title_text=f"${format_label(var2)} \, ({y_units})$")
             
    if ('xlim' in params):
        fig.update_xaxes(range=params['xlim'])
    if ('ylim' in params):
        fig.update_yaxes(range=params['ylim'])
        
    stdx = std_weights(x,q)
    stdy = std_weights(y,q)
    corxy = corr_weights(x,y,q)
    if (x_units == y_units):
        corxy_units = f'{x_units}²'
    else:
        corxy_units = f'{x_units}·{y_units}'
    
    show_emit = False
    if ((var1 == 'x' and 'px' in var2 ) or (var1 == 'y' and 'py' in var2)):
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
        q_units = format_label(q_units, latex=False)
        if (show_emit):
            emitxy_units = format_label(emitxy_units, latex=False)
        var1_label = format_label(var1, add_underscore=False, latex=False)
        var2_label = format_label(var2, add_underscore=False, latex=False)
        data = dict(col1=[], col2=[], col3=[])
        if (screen_key is not None):
            data = add_row(data, col1=f'Screen {screen_key}', col2=f'{screen_value:G}', col3='')
        data = add_row(data, col1=f'Total charge', col2=f'{q_total:G}', col3=f'{q_units}')
        data = add_row(data, col1=f'Mean {var1_label}', col2=f'{avgx:G}', col3=f'{avgx_units}')
        data = add_row(data, col1=f'Mean {var2_label}', col2=f'{avgy:G}', col3=f'{avgy_units}')
        data = add_row(data, col1=f'σ_{var1_label}', col2=f'{stdx:G}', col3=f'{x_units}')
        data = add_row(data, col1=f'σ_{var2_label}', col2=f'{stdy:G}', col3=f'{y_units}')
        data = add_row(data, col1=f'Corr({var1_label}, {var2_label})', col2=f'{corxy:G}', col3=f'{corxy_units}')
        if (show_emit):
            data = add_row(data, col1=f'ε_{var1_label}', col2=f'{emitxy:G}', col3=f'{emitxy_units}')
        headers = dict(col1='Name', col2='Value', col3='Units')
        table_fig.add_trace(make_parameter_table(fig, data, headers))
    
    if (table_on):
        return HBox([fig, table_fig])
    else:
        return fig
        
    
    
    
