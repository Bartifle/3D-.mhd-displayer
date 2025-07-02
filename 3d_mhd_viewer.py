import numpy as np
import plotly.graph_objects as go
import argparse
import os
import sys
import dash
from dash import dcc, html, Patch
from dash.dependencies import Input, Output, State
import multiprocessing
import signal

class Geant4Viewer:
    def __init__(self, mhd_file, raw_file=None, downsample_factor=1, unit='MeV'):
        self.mhd_file = mhd_file
        if raw_file is None:
            if mhd_file.endswith('.mhd'):
                self.raw_file = mhd_file.replace('.mhd', '.raw')
            else:
                raise ValueError("A .mhd file is required, or specify the .raw file")
        else:
            self.raw_file = raw_file

        if not os.path.exists(self.mhd_file):
            raise FileNotFoundError(f"MHD file not found: {self.mhd_file}")
        if not os.path.exists(self.raw_file):
            raise FileNotFoundError(f"RAW file not found: {self.raw_file}")

        self.data = None
        self.metadata = {}
        self.unit = unit
        self.downsample_factor = downsample_factor
        self.load_data()

    def parse_mhd_file(self):
        with open(self.mhd_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split(' = ', 1)
                    self.metadata[key] = value

        if self.metadata.get('NDims') != '3':
            raise ValueError(f"Unsupported number of dimensions: {self.metadata.get('NDims')}. This application currently only supports 3D data.")
        
        self.dim_size = [int(x) for x in self.metadata['DimSize'].split()]
        self.element_spacing = [float(x) for x in self.metadata['ElementSpacing'].split()]
        self.offset = [float(x) for x in self.metadata['Offset'].split()]

    def load_raw_data(self):
        element_type = self.metadata.get('ElementType', 'MET_DOUBLE')
        dtype = {'MET_DOUBLE': np.float64, 'MET_FLOAT': np.float32}.get(element_type, np.float64)
        
        with open(self.raw_file, 'rb') as f:
            data_bytes = f.read()
        
        data_array = np.frombuffer(data_bytes, dtype=dtype)
        dims = self.dim_size
        self.data = data_array.reshape((dims[2], dims[1], dims[0]))

        # If the Z-axis is 'Inferior', flip it to match the viewer's 'Superior' Z-axis.
        # Because .mhd files I work with are all RAI
        orientation = self.metadata.get('AnatomicalOrientation')
        if orientation and len(orientation) == 3 and orientation[2] == 'I':
            self.data = np.flip(self.data, axis=0)

    def calculate_statistics(self):
        stats = {}
        # From MHD metadata
        stats['dim_size'] = self.dim_size
        stats['element_spacing'] = self.element_spacing
        stats['total_physical_volume'] = np.prod(np.array(self.dim_size) * np.array(self.element_spacing))

        # From RAW data
        stats['min_value'] = float(self.data.min())
        stats['max_value'] = float(self.data.max())
        stats['total_value'] = float(self.data.sum())
        stats['unit'] = self.unit
        
        # The definition of an 'active' voxel should depend on the data type.
        # For energy deposition (MeV), any voxel with energy is active.
        # For CT scans (HU), we are generally interested in tissue and bone, not air.
        # A good threshold to exclude most of the air is around -500 HU.
        active_threshold = 0 if self.unit == 'MeV' else -500
        active_voxels_mask = self.data > active_threshold

        stats['active_voxels_count'] = int(np.sum(active_voxels_mask))
        stats['total_voxels'] = int(np.prod(self.dim_size))
        stats['active_voxels_percentage'] = (stats['active_voxels_count'] / stats['total_voxels']) * 100 if stats['total_voxels'] > 0 else 0.0
        
        return stats

    def load_data(self):
        self.parse_mhd_file()
        self.load_raw_data()

        if self.downsample_factor > 1:
            self.data = self.data[::self.downsample_factor, ::self.downsample_factor, ::self.downsample_factor]
            # Update metadata to reflect the new reality
            self.dim_size = [self.data.shape[2], self.data.shape[1], self.data.shape[0]]
            self.element_spacing = [s * self.downsample_factor for s in self.element_spacing]

        self.statistics = self.calculate_statistics()

    def run_slice_app(self):
        _run_slice_app_process(self.data, self.dim_size, self.element_spacing, self.statistics, 8050, False)

    def run_3d_app(self, show_all, threshold_percentile):
        run_3d_app_process(self.data, self.dim_size, self.element_spacing, self.offset, self.statistics, show_all, threshold_percentile, 8051, False)

def _run_slice_app_process(data, dim_size, element_spacing, statistics, port, debug_mode):
    app = dash.Dash(__name__, external_stylesheets=['/assets/style.css'])
    x_max, y_max, z_max = dim_size[0]-1, dim_size[1]-1, dim_size[2]-1
    x_center, y_center, z_center = x_max // 2, y_max // 2, z_max // 2
    unit = statistics.get('unit', 'MeV')

    app.layout = html.Div(className='container slice-viewer-layout', children=[
        html.H1("Interactive Slice Viewer", style={'textAlign': 'center'}),
        html.Div(className='graph-container', children=[
            html.Div(className='statistics-box', children=[
                html.H3("Data Statistics", style={'textAlign': 'center'}),
                html.P(f"Dimensions: {statistics['dim_size'][0]}x{statistics['dim_size'][1]}x{statistics['dim_size'][2]} voxels"),
                html.P(f"Voxel Spacing: {statistics['element_spacing'][0]:.2f}x{statistics['element_spacing'][1]:.2f}x{statistics['element_spacing'][2]:.2f} mm/voxel"),
                html.P(f"Total Physical Volume: {statistics['total_physical_volume']:.2f} mm³"),
                html.P(f"Value Range: {statistics['min_value']:.2e} - {statistics['max_value']:.2e} {unit}"),
                html.P(f"Total Value: {statistics['total_value']:.2e} {unit}"),
                html.P(f"Active Voxels: {statistics['active_voxels_count']} ({statistics['active_voxels_percentage']:.2f}%)"),
            ]),
            html.Div(className='slice-cards-container', children=[
                html.Div(className='graph-item', children=[
                    dcc.Graph(id='slice-xy', className='graph'),
                    html.P("Z-Slice"),
                    html.Div(className='slider-input-group', children=[
                        dcc.Slider(id='slider-z', min=0, max=z_max, value=z_center, step=1, marks=None),
                        dcc.Input(id='input-z', type='number', min=0, max=z_max, value=z_center, step=1, style={'width': '100%'}),
                    ]),
                ]),
                html.Div(className='graph-item', children=[
                    dcc.Graph(id='slice-xz', className='graph'),
                    html.P("Y-Slice"),
                    html.Div(className='slider-input-group', children=[
                        dcc.Slider(id='slider-y', min=0, max=y_max, value=y_center, step=1, marks=None),
                        dcc.Input(id='input-y', type='number', min=0, max=y_max, value=y_center, step=1, style={'width': '100%'}),
                    ]),
                ]),
                html.Div(className='graph-item', children=[
                    dcc.Graph(id='slice-yz', className='graph'),
                    html.P("X-Slice"),
                    html.Div(className='slider-input-group', children=[
                        dcc.Slider(id='slider-x', min=0, max=x_max, value=x_center, step=1, marks=None),
                        dcc.Input(id='input-x', type='number', min=0, max=x_max, value=x_center, step=1, style={'width': '100%'}),
                    ]),
                ]),
            ])
        ])
    ])

    vmin = data.min()
    vmax = data.max()

    def create_slice_figure(data_slice, title, x_label, y_label, aspect_ratio):
        return {
            'data': [go.Heatmap(
                z=data_slice, colorscale='Viridis', zmin=vmin, zmax=vmax,
                colorbar=dict(title=f"Value ({unit})", len=0.75)
            )],
            'layout': go.Layout(
                title=title, xaxis_title=x_label, yaxis_title=y_label,
                yaxis=dict(scaleanchor='x', scaleratio=aspect_ratio),
                margin=dict(l=40, r=40, b=40, t=40),
                autosize=True
            )
        }

    @app.callback(
        Output('slice-xy', 'figure'),
        Output('slider-z', 'value'),
        Output('input-z', 'value'),
        Input('slider-z', 'value'),
        Input('input-z', 'value'),
        State('slider-z', 'value'),
        State('input-z', 'value')
    )
    def update_slice_xy(slider_z_val, input_z_val, current_slider_z_val, current_input_z_val):
        ctx = dash.callback_context

        if not ctx.triggered:
            z_idx = current_slider_z_val # Or current_input_z_val, they should be the same initially
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'slider-z':
                z_idx = slider_z_val
            else: # button_id == 'input-z'
                z_idx = input_z_val

        # Ensure z_idx is within bounds
        z_idx = int(np.clip(z_idx, 0, dim_size[2]-1))

        data_slice = data[z_idx, :, :]
        aspect_ratio = element_spacing[1] / element_spacing[0]
        return create_slice_figure(data_slice, f'XY Slice (z={z_idx})', 'X (voxels)', 'Y (voxels)', aspect_ratio), z_idx, z_idx

    @app.callback(
        Output('slice-xz', 'figure'),
        Output('slider-y', 'value'),
        Output('input-y', 'value'),
        Input('slider-y', 'value'),
        Input('input-y', 'value'),
        State('slider-y', 'value'),
        State('input-y', 'value')
    )
    def update_slice_xz(slider_y_val, input_y_val, current_slider_y_val, current_input_y_val):
        ctx = dash.callback_context

        if not ctx.triggered:
            y_idx = current_slider_y_val
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'slider-y':
                y_idx = slider_y_val
            else: # button_id == 'input-y'
                y_idx = input_y_val

        y_idx = int(np.clip(y_idx, 0, dim_size[1]-1))

        data_slice = data[:, y_idx, :]
        aspect_ratio = element_spacing[2] / element_spacing[0]
        return create_slice_figure(data_slice, f'XZ Slice (y={y_idx})', 'X (voxels)', 'Z (voxels)', aspect_ratio), y_idx, y_idx

    @app.callback(
        Output('slice-yz', 'figure'),
        Output('slider-x', 'value'),
        Output('input-x', 'value'),
        Input('slider-x', 'value'),
        Input('input-x', 'value'),
        State('slider-x', 'value'),
        State('input-x', 'value')
    )
    def update_slice_yz(slider_x_val, input_x_val, current_slider_x_val, current_input_x_val):
        ctx = dash.callback_context

        if not ctx.triggered:
            x_idx = current_slider_x_val
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'slider-x':
                x_idx = slider_x_val
            else: # button_id == 'input-x'
                x_idx = input_x_val

        x_idx = int(np.clip(x_idx, 0, dim_size[0]-1))

        data_slice = data[:, :, x_idx]
        aspect_ratio = element_spacing[2] / element_spacing[1]
        return create_slice_figure(data_slice, f'YZ Slice (x={x_idx})', 'Y (voxels)', 'Z (voxels)', aspect_ratio), x_idx, x_idx

    app.run(debug=debug_mode, port=port)

def run_3d_app_process(data, dim_size, element_spacing, offset, statistics, show_all, threshold_percentile, port, debug_mode):
    app = dash.Dash(__name__, external_stylesheets=['/assets/style.css'])
    unit = statistics.get('unit', 'MeV')
    min_val, max_val = statistics['min_value'], statistics['max_value']

    # Pre-filter data once based on the initial threshold
    if np.any(data > 0):
        mask = data > 0 if show_all else data > np.percentile(data[data > 0], threshold_percentile)
    else:
        mask = np.zeros_like(data, dtype=bool)

    z_coords = np.arange(dim_size[2]) * element_spacing[2] + offset[2]
    y_coords = np.arange(dim_size[1]) * element_spacing[1] + offset[1]
    x_coords = np.arange(dim_size[0]) * element_spacing[0] + offset[0]

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    x_filtered, y_filtered, z_filtered = xx[mask], yy[mask], zz[mask]
    values_filtered = data[mask]

    # Create the initial figure
    initial_fig = go.Figure(data=go.Scatter3d(
        x=x_filtered, y=y_filtered, z=z_filtered, mode='markers',
        marker=dict(size=3, color=values_filtered, colorscale='Plasma', colorbar=dict(title=f"Value ({unit})"), opacity=0.2),
        text=[f'Value: {v:.2e} {unit}' for v in values_filtered],
        hovertemplate='X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm<br>%{text}<extra></extra>'
    ))
    initial_fig.update_layout(
        title='3D Interactive Visualization',
        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    app.layout = html.Div(className='container three-d-viewer-layout', children=[
        html.H1("3D Viewer", style={'textAlign': 'center'}),
        html.Div(className='statistics-box', children=[
            html.H3("Data Statistics", style={'textAlign': 'center'}),
            html.P(f"Dimensions: {statistics['dim_size'][0]}x{statistics['dim_size'][1]}x{statistics['dim_size'][2]} voxels"),
            html.P(f"Voxel Spacing: {statistics['element_spacing'][0]:.2f}x{statistics['element_spacing'][1]:.2f}x{statistics['element_spacing'][2]:.2f} mm/voxel"),
            html.P(f"Total Physical Volume: {statistics['total_physical_volume']:.2f} mm³"),
            html.P(f"Value Range: {statistics['min_value']:.2e} - {statistics['max_value']:.2e} {unit}"),
            html.P(f"Total Value: {statistics['total_value']:.2e} {unit}"),
            html.P(f"Active Voxels: {statistics['active_voxels_count']} ({statistics['active_voxels_percentage']:.2f}%)"),
        ]),
        html.Div(className='sliders-wrapper', children=[
            html.Div(className='slider-container', children=[
                html.P("Opacity:"),
                dcc.Slider(
                    id='opacity-slider',
                    min=0, max=1, value=0.2, step=0.05,
                    marks={i/10: str(i/10) for i in range(0, 11, 2)}
                ),
            ]),
            html.Div(className='slider-container', children=[
                html.P(f"Value Range ({unit}):"),
                dcc.RangeSlider(
                    id='value-range-slider',
                    min=min_val, max=max_val, value=[min_val, max_val],
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ]),
        ]),
        dcc.Graph(id='3d-plot', className='graph-3d', figure=initial_fig),
        dcc.Store(id='filtered-data-store', data={
            'x': x_filtered.tolist(),
            'y': y_filtered.tolist(),
            'z': z_filtered.tolist(),
            'values': values_filtered.tolist()
        })
    ])

    @app.callback(
        Output('3d-plot', 'figure'),
        Input('opacity-slider', 'value'),
        Input('value-range-slider', 'value'),
        State('filtered-data-store', 'data'),
        prevent_initial_call=True
    )
    def update_3d_plot_properties(opacity_value, value_range, filtered_data):
        patched_figure = Patch()

        x_all = np.array(filtered_data['x'])
        y_all = np.array(filtered_data['y'])
        z_all = np.array(filtered_data['z'])
        values_all = np.array(filtered_data['values'])

        value_mask = (values_all >= value_range[0]) & (values_all <= value_range[1])
        
        x_display = x_all[value_mask]
        y_display = y_all[value_mask]
        z_display = z_all[value_mask]
        values_display = values_all[value_mask]

        patched_figure['data'][0]['x'] = x_display
        patched_figure['data'][0]['y'] = y_display
        patched_figure['data'][0]['z'] = z_display
        patched_figure['data'][0]['marker']['color'] = values_display
        patched_figure['data'][0]['marker']['opacity'] = opacity_value
        patched_figure['data'][0]['text'] = [f'Value: {v:.2e} {unit}' for v in values_display]

        return patched_figure

    app.run(debug=debug_mode, port=port)

def parse_arguments():
    parser = argparse.ArgumentParser(description='3D viewer for Geant4 energy deposition data.')
    parser.add_argument('mhd_file', help='Path to the .mhd metadata file.')
    parser.add_argument('raw_file', nargs='?', help='Path to the .raw binary data file (optional).')
    parser.add_argument('--mode', choices=['slices', '3d', 'all'], default='all', help='Visualization mode.')
    parser.add_argument('--unit', choices=['MeV', 'HU'], default='MeV', help='The unit of the data being loaded. Default: MeV.')
    parser.add_argument('--show-all', action='store_true', help='Show all voxels (no threshold).')
    parser.add_argument('--threshold', type=float, default=40, help='Percentile threshold for filtering voxels (default: 85).')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample by factor N. Takes every Nth voxel. Default is 1 (no downsampling).')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        viewer = Geant4Viewer(args.mhd_file, args.raw_file, args.downsample, args.unit)

        if args.mode == 'slices':
            viewer.run_slice_app()
        elif args.mode == '3d':
            viewer.run_3d_app(show_all=args.show_all, threshold_percentile=args.threshold)
        elif args.mode == 'all':
            slice_port = 8050
            threed_port = 8051

            slice_process = multiprocessing.Process(target=_run_slice_app_process, args=(viewer.data, viewer.dim_size, viewer.element_spacing, viewer.statistics, slice_port, False))
            threed_process = multiprocessing.Process(target=run_3d_app_process, args=(viewer.data, viewer.dim_size, viewer.element_spacing, viewer.offset, viewer.statistics, args.show_all, args.threshold, threed_port, False))

            print("Launching Dash applications...")
            slice_process.start()
            threed_process.start()

            print(f"Slice viewer (Dash app) running on http://127.0.0.1:{slice_port}")
            print(f"3D viewer (Dash app) running on http://127.0.0.1:{threed_port}")
            print("Press CTRL+C to quit.")

            def signal_handler(sig, frame):
                print("\nTerminating processes...")
                slice_process.terminate()
                threed_process.terminate()
                slice_process.join()
                threed_process.join()
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            slice_process.join()
            threed_process.join()

    except (KeyboardInterrupt, SystemExit):
        print("\nExecution stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)