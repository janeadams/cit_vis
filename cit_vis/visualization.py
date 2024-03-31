import os
import dotenv
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cit_vis.parse import get_edgelist, find_relevant

def load_data(debug=True):
    """Load the data from the data directory.

    Args:
        debug (bool): Whether to print debug statements.

    Returns:
        tuple: A tuple containing the data DataFrame, traits, microbes, data directory, and port.
    """
    # Load environment variables
    if debug: print("Loading data...")
    dotenv.load_dotenv()
    port = os.getenv("PORT")
    if debug: print(f"PORT: {port}")
    data_dir = os.getenv("DATA_DIR")
    if debug: print(f"DATA_DIR: {data_dir}")

    # Load the experimental data
    trait_df = pd.read_csv(os.path.join(data_dir, "trait_scores.csv"), index_col=0)
    traits = trait_df.columns
    if debug: print(f"Found {len(traits)} Traits: {traits}")
    microbe_df = pd.read_csv(os.path.join(data_dir, "microbe_abundances.csv"), index_col=0)
    microbes = microbe_df.columns
    if debug: print(f"Found {len(microbes)} Microbes: {microbes}")

    # Combine the trait and microbe data
    df = pd.concat([trait_df, microbe_df], axis=1)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Mouse ID'}, inplace=True)
    if debug: print(f"Found {len(df)} Mice: {df['Mouse ID'].values}")
    return df, traits, microbes, data_dir, port

def make_microbes_dropdown_options(microbes, debug=True):
    """Create the dropdown options for the microbes.

    Args:
        microbes (list): The list of microbes.
        debug (bool): Whether to print debug statements.

    Returns:
        list: The dropdown options for the microbes.
    """
    if debug: print(f"Creating dropdown options for {len(microbes)} microbes...")

    # Create the dropdown options
    microbes_dropdown_options = [{'label': microbe, 'value': microbe} for microbe in microbes]

    # Add an option to select all microbes
    microbes_dropdown_options.insert(0, {'label': 'All', 'value': 'All'})
    return microbes_dropdown_options

def make_traits_dropdown_options(traits, debug=True):
    """Create the dropdown options for the traits.
    
    Args:
        traits (list): The list of traits.
        debug (bool): Whether to print debug statements.
        
        Returns:
        list: The dropdown options for the traits.
    """
    if debug: print(f"Creating dropdown options for {len(traits)} traits...")   

    # Create the dropdown options
    trait_dropdown_options = [{'label': trait, 'value': trait} for trait in traits]
    return trait_dropdown_options

def load_trait_specific_data(trait, data_dir, df, debug=True):
    """Load the data specific to the selected trait.

    Args:
        trait (str): The selected trait.
        data_dir (str): The path to the data directory.
        df (DataFrame): The data DataFrame.

    Returns:
        tuple: A tuple containing the groups, grid, color scale, and group IDs.
    """
    if debug: print(f"Loading data for trait: {trait}")
    groups = pd.read_pickle(os.path.join(data_dir, trait, "groups.pkl"))
    if debug: print(f"Found {len(groups)} groups for trait: {trait}")
    color_scale = create_color_scale(df[trait])
    if debug: print(f"Loading grid for trait: {trait}...")
    grid = pd.read_pickle(os.path.join(data_dir, trait, "grid.pkl"))
    if debug: print(f"Loading mouse data for trait: {trait}...")
    mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"), index_col=0).filter(["Group ID"], axis=1)
    group_ids = [mice.loc[mouse_id, 'Group ID'] for mouse_id in df['Mouse ID']]
    df['color'] = [get_color(value, color_scale) for value in df[trait]]
    return groups, grid, color_scale, group_ids

def create_color_scale(vector, debug=True):
    """Create a color scale for the vector.
    
    Args:
        vector (list): The vector to create a color scale for.
        debug (bool): Whether to print debug statements.
        
    Returns:
        tuple: A tuple containing the colormap and normalizer.
    """
    min_val = min(vector)
    max_val = max(vector)
    # Use Matplotlib's built-in 'coolwarm' colormap
    cmap = plt.get_cmap('coolwarm')
    # Normalize the colormap to fit the range of the results
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    # Return both colormap and normalizer to apply on values
    return cmap, norm

def get_color(value, color_scale, debug=True):
    """Get the color for the value.
    
    Args:
        value (float): The value to get the color for.
        color_scale (tuple): The color scale tuple.
        
    Returns:
        str: The color for the value.
    """
    cmap, norm = color_scale
    rgba_color = cmap(norm(value))
    # Convert RGBA to hexadecimal
    return mcolors.to_hex(rgba_color)

def create_sankey_diagram(trait, data_dir, color_scale, debug=True):
    """Create a Sankey diagram for the trait.

    Args:
        trait (str): The trait to create a Sankey diagram for.
        data_dir (str): The path to the data directory.
        color_scale (tuple): The color scale tuple.

    Returns:
        go.Figure: The Sankey diagram for the trait.
    """
    if debug: print(f"Creating Sankey diagram for trait: {trait}")
    sankey_df = get_edgelist(trait, data_dir)
    labels = list(set(sankey_df['source'].values) | set(sankey_df['target'].values))
    # Look up the color for each node in the labels group:
    colors = [get_color(sankey_df[sankey_df['target'] == label]['mean_trait'].mean(), color_scale) for label in labels]
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors,
            hovertemplate='%{label} has %{value:.0f} total mice',
        ),
        link=dict(
            source=[labels.index(x) for x in sankey_df['source']],
            target=[labels.index(x) for x in sankey_df['target']],
            value=sankey_df['value'],
            hovertemplate='%{value:.0f} mice',
        )
    )])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
    return fig

def create_stripplot(df, trait, color_scale, debug=True):
    """"Create a strip plot for the trait.

    Args:
        df (DataFrame): The data DataFrame.
        trait (str): The trait to create a strip plot for.
        color_scale (tuple): The color scale tuple.

    Returns:
        go.Figure: The strip plot for the trait.
    """
    if debug: print(f"Creating strip plot for trait: {trait}")

    # Order based on mean trait value for each Group ID:
    order = df.groupby('Group ID')[trait].mean().sort_values().index
    df['Group ID'] = pd.Categorical(df['Group ID'], categories=order, ordered=True)

    # Assign a color to each group
    color_lookup = dict(zip(df['Mouse ID'], df['color']))
    fig = px.strip(df, x='Group ID', y=trait, color='Mouse ID', color_discrete_map=color_lookup)
    fig.update_layout(template='plotly_white', showlegend=False)

    # Add a dark gray stroke to the strip plot points:
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGray')))

    # On hover, show the mouse ID and the trait score
    fig.update_traces(hovertemplate=trait+': %{y:.2f}')
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
    return fig

def create_empty_plot(debug=True):
    """Create an empty plot.

    Args:
        debug (bool): Whether to print debug statements.
    
    Returns:
        go.Figure: The empty plot.
    """
    fig = go.Figure()
    fig.update_layout(template='plotly_white', width=200, height=100, showlegend=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

def create_arrow_plot(debug=True):
    """Create an arrow plot.

    Args:
        debug (bool): Whether to print debug statements.

    Returns:
        go.Figure: The arrow plot.
    """
    fig = create_empty_plot()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text="↓",
        showarrow=False,
        font=dict(size=50, color='#333'),
        xref="paper",
        yref="paper"
    )
    return fig

def create_grid_plot(feature, mice, df, trait, color_scale, split, val, debug=True):
    """Create a grid plot for the feature.

    Args:
        feature (str): The feature to create a grid plot for.
        mice (list): The list of mice.
        df (DataFrame): The DataFrame for the experimental data.
        trait (str): The trait to create a grid plot for.
        color_scale (tuple): The color scale tuple.
        split (str): The split for the feature.
        val (float): The value for the split.

    Returns:
        go.Figure: The grid plot for the feature.
    """
    fig = go.Figure(
        go.Violin(x=df[feature], marker_color='lightgrey', showlegend=False, hoverinfo='skip',
                  points=False))
    subset = df[df['Mouse ID'].isin(mice)].copy()
    subset['jitter'] = [0.01 + np.random.normal(0, 0.05) for _ in range(len(subset[feature]))]
    for i, row in subset.iterrows():
        fig.add_trace(
            px.strip([row], x=feature, hover_name="Mouse ID",
                    hover_data={'Group ID':False, 'Mouse ID': False, feature: ':.2f', 'jitter': False},
                    color='Mouse ID', color_discrete_map={row['Mouse ID']: row['color']},
                    y='jitter').data[0])
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGray')))
    fig.update_traces(offsetgroup=0)
    # Add a vertical line to show the mean value
    fig.add_vline(x=df[feature].mean(), line_dash="dash", line_color="#999")
    fig.add_vline(x=val, line_color="red")
    fig.add_annotation(
        x=float(val)-30,
        y=0.5,
        text=f"{val}",
        showarrow=False,
        font=dict(size=20, color='red')
    )
    fig.update_layout(template='plotly_white', width=200, height=100, showlegend=False)
    # zero out margins:
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

def grid_handler(grid, df, trait, color_scale, debug=True):
    """Handle the grid for the trait.
    
    Args:
        grid (DataFrame): The grid for the trait.
        df (DataFrame): The DataFrame for the experimental data.
        trait (str): The trait to create the grid for.
        color_scale (tuple): The color scale tuple.
        debug (bool): Whether to print debug statements.
        
    Returns:
        html.Div: The grid container for the trait.
    """
    if debug: print("Creating grid...")
    grid_dict = grid.to_dict(orient='index')
    plot_grid = []

    # Enhanced cell style for better appearance
    cell_style = {
        'border': '2px solid #ddd',  # Lighter border color
        'minWidth': '150px',
        'height': '100px',
        'flexGrow': '1',  # Allows cell to grow
        'display': 'flex',
        'justifyContent': 'center',  # Center content horizontally
        'alignItems': 'center',  # Center content vertically
        'padding': '0'  # Add padding for text/content
    }

    header_style = cell_style.copy()
    header_style['height'] = 'auto'

    corner_style = header_style.copy()
    corner_style['width'] = 'auto'

    index_style = cell_style.copy()
    index_style['width'] = 'auto'

    header_row = [html.Div([], style=corner_style)]

    # Create column labels for each group
    for group in grid.columns:
        header_row.append(html.Div([html.H3(group)], style=header_style))

    # Flex container for header to ensure alignment
    plot_grid.append(html.Div(header_row, style={'display': 'flex', 'flexDirection': 'row'}))

    # Processing each row
    for col, cells in grid_dict.items():
        row_items = []
        row_items.append(html.Div([html.H3(col)], style=index_style))
        for group, celldata in cells.items():
            content_style = cell_style.copy()  # Copy cell style to modify individually if needed
            if celldata['Type'] == 'Arrow':
                row_items.append(html.Div([dcc.Graph(
                    figure=create_arrow_plot(),
                    config={'displayModeBar': False}
                    )], style=content_style))
            elif celldata['Type'] == 'Plot':
                split = celldata['Split']
                val = celldata['Value']
                row_items.append(html.Div([dcc.Graph(
                    figure=create_grid_plot(celldata['Feature'], celldata['Mouse IDs'], df, trait, color_scale, split, val),
                    config={'displayModeBar': False}
                    )], style=content_style))
            else:
                row_items.append(html.Div([dcc.Graph(
                    figure=create_empty_plot(),
                    config={'displayModeBar': False}
                    )], style=content_style))

        # Flex container for each row
        row_items = html.Div(row_items, style={'display': 'flex', 'flexDirection': 'row'})
        plot_grid.append(row_items)

    # Main container to ensure vertical stacking of rows
    plot_grid = html.Div(plot_grid, style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'stretch'})

    return plot_grid

def make_dashboard(debug=True, port=8050):
    """Create the dashboard and launch it
    
    Args:
        debug (bool): Whether to print debug statements.
        port (int): The port to run the dashboard on.
        
    Returns:
        None
    """
    if debug: print("Creating dashboard...")
    df, traits, microbes, data_dir, port = load_data(debug=debug)

    selected_microbe = 'All'
    selected_trait = traits[0]

    # Things that update when the trait changes
    groups, grid, color_scale, df['Group ID'] = load_trait_specific_data(selected_trait, data_dir, df, debug=debug)

    app = Dash(__name__)

    # Adjusted styles with flex wrap
    row_style = {
        'display': 'flex',
        'width': '100%',
        'margin': 'auto',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'flexWrap': 'wrap',  # Allow items to wrap
        'marginBottom': '20px'  # Add some space between rows for clarity
    }
    dropdown_cell_style = {
        'padding': '10px',  # Ensure padding has units
        'flexBasis': 'calc(50% - 20px)',  # Calculate width considering padding/margin
        'boxSizing': 'border-box'  # Include padding in the element's total width
    }
    overview_cell_style = dropdown_cell_style.copy()
    overview_cell_style.update({
        'height': '200px'
    })

    # Define the layout
    app.layout = html.Div([
        html.H1("CIT Visualization"),
        html.Div([
            html.Div([dcc.Dropdown(id='microbe-dropdown', options=make_microbes_dropdown_options(microbes, debug=debug), value=selected_microbe)],
                    style=dropdown_cell_style),
            html.Div([dcc.Dropdown(id='trait-dropdown', options=make_traits_dropdown_options(traits, debug=debug), value=selected_trait)],
                    style=dropdown_cell_style),
        ], style=row_style),

        html.Div([
            html.Div([dcc.Graph(id='sankey-diagram', figure=create_sankey_diagram(selected_trait, data_dir, color_scale, debug=debug), config={'displayModeBar': False})],
                    style=overview_cell_style),
            html.Div([dcc.Graph(id='stripplot', figure=create_stripplot(df, selected_trait, color_scale, debug=debug), config={'displayModeBar': False})],
                    style=overview_cell_style)
        ], style=row_style),
        
        html.Div(id='grid-container', children=grid_handler(grid, df, selected_trait, color_scale, debug=debug), style=row_style),
        dcc.Store(id='selected-trait', data=selected_trait)
    ], style={'width': '100%', 'margin': 'auto', 'padding': '20px', 'min-width': '1200px', 'font-family': 'Arial, sans-serif', 'color': '#333', 'background-color': '#ffffff'})

    # Define the callbacks
    @app.callback(
        [Output('trait-dropdown', 'options'),
         Output('trait-dropdown', 'value')],
        [Input('microbe-dropdown', 'value'),
         Input('selected-trait', 'data')]
    )
    def update_trait_dropdown(selected_microbe, selected_trait):
        # Update the trait dropdown based on the selected microbe
        if debug: print(f"Updating trait dropdown for microbe: {selected_microbe}")
        if selected_microbe == 'All':
            return [{'label': trait, 'value': trait} for trait in traits], selected_trait
        relevant_traits = find_relevant(selected_microbe, data_dir, debug=debug)
        if selected_trait not in relevant_traits:
            selected_trait = relevant_traits[0]
        return [{'label': trait, 'value': trait} for trait in relevant_traits], selected_trait
    
    @app.callback(
        [Output('sankey-diagram', 'figure'),
         Output('stripplot', 'figure'),
         Output('grid-container', 'children')],
        [Input('trait-dropdown', 'value')]
    )
    def update_charts(selected_trait):
        # Update the charts based on the selected trait
        if debug: print(f"Updating charts for trait: {selected_trait}")
        groups, grid, color_scale, df['Group ID'] = load_trait_specific_data(selected_trait, data_dir, df, debug=debug)
        sankey = create_sankey_diagram(selected_trait, data_dir, color_scale, debug=debug)
        stripplot = create_stripplot(df, selected_trait, color_scale, debug=debug)
        grid = grid_handler(grid, df, selected_trait, color_scale, debug=debug)
        return sankey, stripplot, grid

    # Run the app
    #app.run_server(debug=debug, port=port)
    
if __name__ == "__main__":
    make_dashboard(debug=True, port=8050)