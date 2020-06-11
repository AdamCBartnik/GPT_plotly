import numpy as np
import plotly.graph_objects as go
from .gpt_plot import *
from .tools import special_screens
from .ParticleGroupExtension import convert_gpt_data
import ipywidgets as widgets


def gpt_plot_gui(gpt_data_input):
    gpt_data = convert_gpt_data(gpt_data_input)
    z = gpt_data.stat('mean_z', 'screen')
    screen_z_list = z.tolist()
    special_z_list = z[special_screens(z)].tolist()
    
    gui = widgets.HBox(layout={'border': '1px solid grey'})
    #figure_hbox = widgets.HBox()
    tab_panel = widgets.Tab()
    
    # Layouts
    layout_200px = widgets.Layout(width='200px',height='30px')
    layout_150px = widgets.Layout(width='150px',height='30px')
    layout_125px = widgets.Layout(width='125px',height='30px')
    layout_100px = widgets.Layout(width='100px',height='30px')
    layout_20px = widgets.Layout(width='20px',height='30px')
    label_layout = layout_150px
    
    # Make widgets
    dist_list = ['t','x','y','r','px','py','pz','pr','action_x','action_y','action_4d']
    
    plottype_list = ['Trends', '1D Distribution', '2D Distribution']
    plottype_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(plottype_list)], value=0)
    
    screen_type_list = ['Special', 'All']
    screen_type_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(screen_type_list)], value=0, layout=layout_150px)
    screen_z_dropdown = widgets.Dropdown(options=[(f'{z:.3f}', i) for (i,z) in enumerate(special_z_list)], layout=layout_150px)
    
    trend_x_list = ['z', 't']
    trend_y_list = ['Beam Size', 'Bunch Length', 'Emittance (x,y)', 'Emittance (4D)', 'Slice emit. (x,y)', 'Slice emit. (4D)', 'Charge', 'Energy', 'Trajectory']
    trend_x_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(trend_x_list)], value=0, layout=layout_150px)
    trend_y_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(trend_y_list)], value=0, layout=layout_150px)
    trend_slice_var_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    trend_slice_nslices_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    
    dist_x_1d_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    dist_type_1d_list = ['Charge Density', 'Emittance X', 'Emittance Y', 'Emittance 4D', 'Sigma X', 'Sigma Y']
    dist_type_1d_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_type_1d_list)], value=0, layout=layout_150px)
    nbin_1d_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    
    dist2d_type_dropdown = widgets.Dropdown(options=[('Scatter', 'scatter'), ('Histogram', 'histogram')], value='histogram', layout=layout_150px)
    scatter_color = ['density'] + dist_list
    dist2d_color_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(scatter_color)], value=0, layout=layout_150px)
    dist2d_color_source_dropdown = widgets.Dropdown(options=[('Same screen', 'same'), ('Alternate screen', 'alt')], value='same', layout=layout_150px)
    dist2d_color_screen_type_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(screen_type_list)], value=0, layout=layout_150px)
    dist2d_color_screen_z_dropdown = widgets.Dropdown(options=[(f'{z:.3f}', i) for (i,z) in enumerate(special_z_list)], layout=layout_100px)
    
    dist_x_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=1, layout=layout_150px)
    dist_y_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=2, layout=layout_150px)
    dist_y_source_dropdown = widgets.Dropdown(options=[('Same screen', 'same'), ('Alternate screen', 'alt')], value='same', layout=layout_150px)
    dist_y_screen_type_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(screen_type_list)], value=0, layout=layout_150px)
    dist_y_screen_z_dropdown = widgets.Dropdown(options=[(f'{z:.3f}', i) for (i,z) in enumerate(special_z_list)], layout=layout_100px)
    
    axis_equal_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    nbin_x_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    nbin_y_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    
    remove_zero_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    
    radial_aperture_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    radial_aperture_r_text = widgets.BoundedFloatText(value=1, min=0, layout=layout_150px)
    
    cyl_copies_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    cyl_copies_text = widgets.BoundedIntText(value=50, min=10, max=500, step=1, layout=layout_150px)
    
    remove_spinning_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    
    remove_correlation_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    remove_correlation_n_text = widgets.BoundedIntText(value=1, min=0, max=10, step=1, layout=layout_150px)
    remove_correlation_var1_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    remove_correlation_var2_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=6, layout=layout_150px)
    
    take_slice_checkbox = widgets.Checkbox(value=False,description='Enabled',disabled=False,indent=False, layout=layout_100px)
    take_slice_var_dropdown = widgets.Dropdown(options=[(a, i) for (i,a) in enumerate(dist_list)], value=0, layout=layout_150px)
    take_slice_nslices_text = widgets.BoundedIntText(value=50, min=5, max=500, step=1, layout=layout_150px)
    take_slice_index_text = widgets.BoundedIntText(value=0, min=0, max=take_slice_nslices_text.value-1, step=1, layout=layout_150px)
    
        
    def make_plot():   
        # Clear plot window
        for old_plot in gui.children[2:]:
            old_plot.close()
        gui.children = gui.children[0:2]

        # Get local copy of widget settings
        plottype = plottype_dropdown.label.lower()
        
        trend_x = trend_x_dropdown.label
        trend_y = trend_y_dropdown.label
        trend_slice_var = trend_slice_var_dropdown.label
        trend_slice_nslices = trend_slice_nslices_text.value
        
        dist_x_1d = dist_x_1d_dropdown.label
        dist_y_1d = dist_type_1d_dropdown.label
        nbins_1d = nbin_1d_text.value
        
        dist_x = dist_x_dropdown.label
        dist_y = dist_y_dropdown.label
        dist_y_alt_source = dist_y_source_dropdown.value.lower() != 'same'
        ptype = dist2d_type_dropdown.value.lower()
        dist2d_color_var = dist2d_color_dropdown.label.lower()
        dist2d_color_alt_source = dist2d_color_source_dropdown.value.lower() != 'same'
        axis_equal = axis_equal_checkbox.value
        nbins = [nbin_x_text.value, nbin_y_text.value]
        
        remove_zero_weight = remove_zero_checkbox.value and (plottype!='trends')
        radial_aperture_on = radial_aperture_checkbox.value and (plottype!='trends')
        radial_aperture_r = radial_aperture_r_text.value * 1.0e-3 # (mm)
        
        cyl_copies = cyl_copies_text.value
        cyl_copies_on = cyl_copies_checkbox.value and (plottype!='trends')
        cyl_copies_text.disabled = not cyl_copies_on
        
        remove_spinning = remove_spinning_checkbox.value and (plottype!='trends')
        
        remove_correlation = remove_correlation_checkbox.value and (plottype!='trends')
        remove_correlation_n = remove_correlation_n_text.value
        remove_correlation_var1 = remove_correlation_var1_dropdown.label
        remove_correlation_var2 = remove_correlation_var2_dropdown.label
        
        take_slice = take_slice_checkbox.value and (plottype!='trends')
        take_slice_var = take_slice_var_dropdown.label
        take_slice_index = take_slice_index_text.value
        take_slice_nslices = take_slice_nslices_text.value
        take_slice_index_text.max = take_slice_nslices-1
        
        is_trend = (plottype=='trends')
        is_dist1d = (plottype=='1d distribution')
        is_dist2d = (plottype=='2d distribution')
        is_slice_trend = ('slice' in trend_y.lower())
                
        # Enable / disable widgets depending on their settings     
        screen_type_dropdown.disabled = not (is_dist1d or is_dist2d)
        screen_z_dropdown.disabled = not (is_dist1d or is_dist2d)
            
        trend_x_dropdown.disabled = not is_trend
        trend_y_dropdown.disabled = not is_trend
        trend_slice_var_dropdown.disabled = not is_slice_trend
        trend_slice_nslices_text.disabled = not is_slice_trend
        
        dist_type_1d_dropdown.disabled = not is_dist1d
        dist_x_1d_dropdown.disabled = not is_dist1d
        nbin_1d_text.disabled = not is_dist1d
        
        dist2d_type_dropdown.disabled = not is_dist2d
        dist2d_color_dropdown.disabled = not is_dist2d
        dist2d_color_screen_type_dropdown.disabled = not dist2d_color_alt_source
        dist2d_color_screen_z_dropdown.disabled = not dist2d_color_alt_source
        dist_x_dropdown.disabled = not is_dist2d
        dist_y_dropdown.disabled = not is_dist2d
        dist_y_dropdown.disabled = not is_dist2d
        dist_y_screen_type_dropdown.disabled = not dist_y_alt_source
        dist_y_screen_z_dropdown.disabled = not dist_y_alt_source
        axis_equal_checkbox.disabled = not is_dist2d
        nbin_x_text.disabled = not is_dist2d
        nbin_y_text.disabled = not is_dist2d
        
        # Add extra parameters to pass into plotting functions
        params = {}
        if (not is_trend):
            if (screen_type_dropdown.label.lower() == 'all'):
                params['screen_z'] = screen_z_list[screen_z_dropdown.value]
            if (screen_type_dropdown.label.lower() == 'special'):
                params['screen_z'] = special_z_list[screen_z_dropdown.value]
            if (remove_zero_weight):
                params['kill_zero_weight'] = remove_zero_weight
            if (radial_aperture_on):
                params['radial_aperture'] = radial_aperture_r
            if (cyl_copies_on):
                params['cylindrical_copies'] = cyl_copies
            if (remove_spinning):
                params['remove_spinning'] = remove_spinning
            if (remove_correlation):
                params['remove_correlation'] = (remove_correlation_var1, remove_correlation_var2, remove_correlation_n)
            if (take_slice):
                params['take_slice'] = (take_slice_var, take_slice_index, take_slice_nslices)
        else:
            if (is_slice_trend):
                params['slice_key'] = trend_slice_var
                params['n_slices'] = trend_slice_nslices
        
        # Make the plot, assign to (only) child of figure_hbox
        if (is_trend):
            if (tab_panel.selected_index < 3):
                tab_panel.selected_index = 0
            var1 = 'mean_'+trend_x
            var2 = get_trend_vars(trend_y)
            gui.children += (gpt_plot(gpt_data, var1, var2, **params), )
        if (is_dist1d):
            if (tab_panel.selected_index < 3):
                tab_panel.selected_index = 1
            ptype_1d = get_dist_plot_type(dist_y_1d)
            gui.children += (gpt_plot_dist1d(gpt_data, dist_x_1d, plot_type=ptype_1d, nbins=nbins_1d, **params), )
        if (is_dist2d):
            if (tab_panel.selected_index < 3):
                tab_panel.selected_index = 2
            if (dist_y_alt_source):
                if (dist_y_screen_type_dropdown.label.lower() == 'all'):
                    dist_y = (dist_y, get_screen_data(gpt_data, screen_z=screen_z_list[dist_y_screen_z_dropdown.value])[0])
                if (dist_y_screen_type_dropdown.label.lower() == 'special'):
                    dist_y = (dist_y, get_screen_data(gpt_data, screen_z=special_z_list[dist_y_screen_z_dropdown.value])[0])
            if (dist2d_color_alt_source):
                if (dist2d_color_screen_type_dropdown.label.lower() == 'all'):
                    params['color_var'] = (dist2d_color_var, get_screen_data(gpt_data, screen_z=screen_z_list[dist2d_color_screen_z_dropdown.value])[0])
                if (dist2d_color_screen_type_dropdown.label.lower() == 'special'):
                    params['color_var'] = (dist2d_color_var, get_screen_data(gpt_data, screen_z=special_z_list[dist2d_color_screen_z_dropdown.value])[0])
            else:
                params['color_var'] = dist2d_color_var
            if (axis_equal):
                params['axis'] = 'equal'
            gui.children += (gpt_plot_dist2d(gpt_data, dist_x, dist_y, plot_type=ptype, nbins=nbins, **params), )
            
            
    # Callback functions
    def remake_on_value_change(change):
        make_plot()

    def fill_screen_list(change):
        if (screen_type_dropdown.label.lower() == 'all'):
            screen_z_dropdown.options = [(f'{z:.3f}', i) for (i,z) in enumerate(screen_z_list)]
        if (screen_type_dropdown.label.lower() == 'special'):
            screen_z_dropdown.options = [(f'{z:.3f}', i) for (i,z) in enumerate(special_z_list)]
        make_plot()
        
    def dist2d_color_fill_screen_list(change):
        if (dist2d_color_screen_type_dropdown.label.lower() == 'all'):
            dist2d_color_screen_z_dropdown.options = [(f'{z:.3f}', i) for (i,z) in enumerate(screen_z_list)]
        if (dist2d_color_screen_type_dropdown.label.lower() == 'special'):
            dist2d_color_screen_z_dropdown.options = [(f'{z:.3f}', i) for (i,z) in enumerate(special_z_list)]
        make_plot()
        
    def dist_y_fill_screen_list(change):
        if (dist_y_screen_type_dropdown.label.lower() == 'all'):
            dist_y_screen_z_dropdown.options = [(f'{z:.3f}', i) for (i,z) in enumerate(screen_z_list)]
        if (dist_y_screen_type_dropdown.label.lower() == 'special'):
            dist_y_screen_z_dropdown.options = [(f'{z:.3f}', i) for (i,z) in enumerate(special_z_list)]
        make_plot()
        
    # Widget layout within GUI
    trends_tab = widgets.VBox([
        widgets.HBox([widgets.Label('X axis', layout=label_layout), trend_x_dropdown]),
        widgets.HBox([widgets.Label('Y axis', layout=label_layout), trend_y_dropdown]),
        widgets.HBox([widgets.Label('Slice variable', layout=label_layout), trend_slice_var_dropdown]),
        widgets.HBox([widgets.Label('Number of slices', layout=label_layout), trend_slice_nslices_text])
    ])
    
    dist_1d_tab = widgets.VBox([
        widgets.HBox([widgets.Label('X axis', layout=label_layout), dist_x_1d_dropdown]), 
        widgets.HBox([widgets.Label('Y axis', layout=label_layout), dist_type_1d_dropdown]),
        widgets.HBox([widgets.Label('Histogram bins', layout=label_layout), nbin_1d_text])
    ])

    dist_2d_tab = widgets.VBox([
        widgets.HBox([widgets.Label('Plot method', layout=label_layout), dist2d_type_dropdown]),
        widgets.HBox([widgets.Label('Color variable', layout=label_layout), dist2d_color_dropdown]),
        widgets.HBox([widgets.Label('Color source', layout=label_layout), dist2d_color_source_dropdown]),
        widgets.HBox([widgets.Label('Color screen type', layout=label_layout), dist2d_color_screen_type_dropdown, dist2d_color_screen_z_dropdown]),
        widgets.HBox([widgets.Label('X axis', layout=label_layout), dist_x_dropdown]), 
        widgets.HBox([widgets.Label('Y axis', layout=label_layout), dist_y_dropdown]),
        widgets.HBox([widgets.Label('Y source', layout=label_layout), dist_y_source_dropdown]),
        widgets.HBox([widgets.Label('Y screen type', layout=label_layout), dist_y_screen_type_dropdown, dist_y_screen_z_dropdown]),
        widgets.HBox([widgets.Label('Equal scale axes', layout=label_layout), axis_equal_checkbox]),
        widgets.HBox([widgets.Label('Histogram bins, X', layout=label_layout), nbin_x_text]),
        widgets.HBox([widgets.Label('Histogram bins, Y', layout=label_layout), nbin_y_text])
    ])
    
    postprocess_tab = widgets.VBox([
        widgets.HBox([widgets.Label('Remove zero weight', layout=label_layout), remove_zero_checkbox]),
        widgets.HBox([widgets.Label('Radial aperture', layout=label_layout), radial_aperture_checkbox]),
        widgets.HBox([widgets.Label('Radius (mm)', layout=label_layout), radial_aperture_r_text]),
        widgets.HBox([widgets.Label('Cylindrical copies', layout=label_layout), cyl_copies_checkbox]),
        widgets.HBox([widgets.Label('Number of copies', layout=label_layout), cyl_copies_text]),
        widgets.HBox([widgets.Label('Remove Spinning', layout=label_layout), remove_spinning_checkbox]),
        widgets.HBox([widgets.Label('Remove Correlation', layout=label_layout), remove_correlation_checkbox]),
        widgets.HBox([widgets.Label('Max polynomial power', layout=label_layout), remove_correlation_n_text]),
        widgets.HBox([widgets.Label('Independent var (x)', layout=label_layout), remove_correlation_var1_dropdown]), 
        widgets.HBox([widgets.Label('Dependent var (y)', layout=label_layout), remove_correlation_var2_dropdown]),
        widgets.HBox([widgets.Label('Take slice of data', layout=label_layout), take_slice_checkbox]), 
        widgets.HBox([widgets.Label('Slice variable', layout=label_layout), take_slice_var_dropdown]), 
        widgets.HBox([widgets.Label('Slice index', layout=label_layout), take_slice_index_text]), 
        widgets.HBox([widgets.Label('Number of slices', layout=label_layout), take_slice_nslices_text])
    ], description='Postprocessing')
    
    # Make tab panel
    tab_list = [trends_tab, dist_1d_tab, dist_2d_tab, postprocess_tab]
    tab_label_list = ['Trends', '1D Dist.', '2D Dist.', 'Postprocess']
    tab_panel.children = tab_list
    for i,t in enumerate(tab_label_list):
        tab_panel.set_title(i, t)
    
    # Main main controls panel
    tools_panel = widgets.VBox([
        widgets.HBox([widgets.Label('Plot Type', layout=label_layout), plottype_dropdown]),
        widgets.HBox([widgets.Label('Screen type', layout=label_layout), screen_type_dropdown, screen_z_dropdown]),
        tab_panel
    ])
    
    # Place controls panel to the left of the plot
    gui.children += (tools_panel, )
    gui.children += (widgets.HBox([], layout=layout_20px), )
        
    # Force the plot redraw function to be called once at start
    make_plot()
                    
    # Register callbacks
    plottype_dropdown.observe(remake_on_value_change, names='value')
    trend_x_dropdown.observe(remake_on_value_change, names='value')
    trend_y_dropdown.observe(remake_on_value_change, names='value')
    dist_x_1d_dropdown.observe(remake_on_value_change, names='value')
    dist_x_dropdown.observe(remake_on_value_change, names='value')
    dist_y_dropdown.observe(remake_on_value_change, names='value')
    dist_y_source_dropdown.observe(remake_on_value_change, names='value')
    dist_y_screen_type_dropdown.observe(dist_y_fill_screen_list, names='value')
    dist_y_screen_z_dropdown.observe(remake_on_value_change, names='value')
    screen_z_dropdown.observe(remake_on_value_change, names='value')
    screen_type_dropdown.observe(fill_screen_list, names='value')
    nbin_1d_text.observe(remake_on_value_change, names='value')
    nbin_x_text.observe(remake_on_value_change, names='value')
    nbin_y_text.observe(remake_on_value_change, names='value')
    cyl_copies_checkbox.observe(remake_on_value_change, names='value')
    cyl_copies_text.observe(remake_on_value_change, names='value')
    dist2d_type_dropdown.observe(remake_on_value_change, names='value')
    dist2d_color_dropdown.observe(remake_on_value_change, names='value')
    dist2d_color_source_dropdown.observe(remake_on_value_change, names='value')
    dist2d_color_screen_type_dropdown.observe(dist2d_color_fill_screen_list, names='value')
    dist2d_color_screen_z_dropdown.observe(remake_on_value_change, names='value')
    axis_equal_checkbox.observe(remake_on_value_change, names='value')
    dist_type_1d_dropdown.observe(remake_on_value_change, names='value')
    remove_spinning_checkbox.observe(remake_on_value_change, names='value')
    remove_correlation_checkbox.observe(remake_on_value_change, names='value')
    remove_correlation_n_text.observe(remake_on_value_change, names='value')
    remove_correlation_var1_dropdown.observe(remake_on_value_change, names='value')
    remove_correlation_var2_dropdown.observe(remake_on_value_change, names='value')
    take_slice_checkbox.observe(remake_on_value_change, names='value')
    take_slice_var_dropdown.observe(remake_on_value_change, names='value')
    take_slice_index_text.observe(remake_on_value_change, names='value')
    take_slice_nslices_text.observe(remake_on_value_change, names='value')
    trend_slice_var_dropdown.observe(remake_on_value_change, names='value')
    trend_slice_nslices_text.observe(remake_on_value_change, names='value')
    remove_zero_checkbox.observe(remake_on_value_change, names='value')
    radial_aperture_checkbox.observe(remake_on_value_change, names='value')
    radial_aperture_r_text.observe(remake_on_value_change, names='value')
        
    return gui

