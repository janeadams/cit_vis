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
from cit_vis.parse import aggregate_by_group, get_grid_structure, get_edgelist

def load_data():
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    trait_df = pd.read_csv(os.path.join(data_dir, "trait_scores.csv"), index_col=0)
    traits = trait_df.columns
    microbe_df = pd.read_csv(os.path.join(data_dir, "microbe_abundances.csv"), index_col=0)
    microbes = microbe_df.columns
    df = pd.concat([trait_df, microbe_df], axis=1)
    return df, traits, microbes, data_dir

def make_microbes_dropdown_options(microbes):
    microbes_dropdown_options = [{'label': microbe, 'value': microbe} for microbe in microbes]
    microbes_dropdown_options.insert(0, {'label': 'All', 'value': 'All'})
    return microbes_dropdown_options

def make_traits_dropdown_options(traits):
    trait_dropdown_options = [{'label': trait, 'value': trait} for trait in traits]
    return trait_dropdown_options

def load_trait_specific_data(trait, data_dir, df):
    groups = pd.read_pickle(os.path.join(data_dir, trait, "groups.pkl"))
    color_scale = create_color_scale(df[trait])
    grid = pd.read_pickle(os.path.join(data_dir, trait, "grid.pkl"))
    mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"), index_col=0).filter(["Group ID"], axis=1)
    group_ids = [mice.loc[mouse_id, 'Group ID'] for mouse_id in df.index]
    return groups, grid, color_scale, group_ids

def create_color_scale(vector):
    min_val = min(vector)
    max_val = max(vector)
    # Use Matplotlib's built-in 'coolwarm' colormap
    cmap = plt.get_cmap('coolwarm')
    # Normalize the colormap to fit the range of the results
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    # Return both colormap and normalizer to apply on values
    return cmap, norm

def get_color(value, color_scale):
    cmap, norm = color_scale
    rgba_color = cmap(norm(value))
    # Convert RGBA to hexadecimal
    return mcolors.to_hex(rgba_color)

def create_sankey_diagram(groups, color_scale):
    sankey_df = get_edgelist(groups)
    labels = list(set(sankey_df['source'].values) | set(sankey_df['target'].values))
    colors = [get_color(value, color_scale) for value in sankey_df['mean_trait']]
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=[labels.index(x) for x in sankey_df['source']],
            target=[labels.index(x) for x in sankey_df['target']],
            value=sankey_df['value'],
            #label=sankey_df['edge_label']
        )
    )])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
    return fig

def create_stripplot(df, trait, color_scale):
    df.sort_values(trait, inplace=True)
    # Assign a color to each group
    df['color'] = [get_color(t, color_scale) for t in df[trait]]
    color_lookup = dict(zip(df['Group ID'], df['color']))
    fig = px.strip(df, x='Group ID', y=trait, color='Group ID', color_discrete_map=color_lookup)
    fig.update_layout(template='plotly_white', showlegend=False)
    # Add a dark gray stroke to the strip plot points:
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGray')))
    # On hover, show the mouse ID and the trait score
    fig.update_traces(hovertemplate='Mouse ID: %{x}<br>Trait: %{y}')
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
    return fig

def create_empty_plot():
    fig = go.Figure()
    fig.update_layout(template='plotly_white', width=200, height=100, showlegend=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

def create_arrow_plot():
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

def create_grid_plot(feature, mice, df):
    fig = go.Figure(
        go.Violin(x=df[feature], marker_color='lightgrey', showlegend=False, hoverinfo='skip',
                  points=False))
    subset = df.loc[mice]
    subset.reset_index(inplace=True)
    subset.rename(columns={'index': 'Mouse ID'}, inplace=True)
    subset['jitter'] = [0.01 + np.random.normal(0, 0.01) for _ in range(len(subset[feature]))]
    fig.add_trace(
        px.strip(subset, x=feature, hover_name="Mouse ID",
                 hover_data={'Group ID':False, 'Mouse ID': False, feature: ':.2f', 'jitter': False},
                 color='Group ID', color_discrete_map={group: color for group, color in zip(subset['Group ID'], subset['color'])}, y='jitter').data[0])
    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGray')))
    # Add a vertical line to show the mean value
    fig.add_vline(x=df[feature].mean(), line_dash="dash", line_color="#999")
    fig.update_layout(template='plotly_white', width=200, height=100, showlegend=False)
    # zero out margins:
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig

def grid_handler(grid, df):
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
                row_items.append(html.Div([dcc.Graph(
                    figure=create_grid_plot(celldata['Feature'], celldata['Mouse IDs'], df),
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




def make_dashboard():
    df, traits, microbes, data_dir = load_data()

    selected_microbe = 'All'
    selected_trait = traits[0]

    # Things that update when the trait changes
    groups, grid, color_scale, df['Group ID'] = load_trait_specific_data(selected_trait, data_dir, df)

    app = Dash(__name__)

    # Adjusted styles with flex wrap
    row_style = {
        'display': 'flex',
        'width': '100%',
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

    # Define the layout with improved structure and styling
    app.layout = html.Div([
        html.H1("CIT Visualization"),
        html.Div([
            html.Div([dcc.Dropdown(id='microbe-dropdown', options=make_microbes_dropdown_options(microbes), value=selected_microbe)],
                    style=dropdown_cell_style),
            html.Div([dcc.Dropdown(id='trait-dropdown', options=make_traits_dropdown_options(traits), value=selected_trait)],
                    style=dropdown_cell_style),
        ], style=row_style),

        html.Div([
            html.Div([dcc.Graph(id='sankey-diagram', figure=create_sankey_diagram(groups, color_scale), config={'displayModeBar': False})],
                    style=overview_cell_style),
            html.Div([dcc.Graph(id='stripplot', figure=create_stripplot(df, selected_trait, color_scale), config={'displayModeBar': False})],
                    style=overview_cell_style)
        ], style=row_style),
        
        html.Div(id='grid-container', children=grid_handler(grid, df), style=row_style),
        dcc.Store(id='selected-trait', data=selected_trait)
    ], style={'width': '100%', 'margin': 'auto', 'padding': '20px', 'min-width': '1200px', 'font-family': 'Arial, sans-serif', 'color': '#333', 'background-color': '#ffffff'})


    @app.callback(
        [Output('trait-dropdown', 'options'),
         Output('trait-dropdown', 'value')],
        [Input('microbe-dropdown', 'value'),
         Input('selected-trait', 'data')]
    )
    def update_trait_dropdown(selected_microbe, selected_trait):
        if selected_microbe == 'All':
            return [{'label': trait, 'value': trait} for trait in traits], selected_trait
        relevant_traits = traits#get_relevant_traits(selected_microbe)
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
        groups, grid, color_scale, df['Group ID'] = load_trait_specific_data(selected_trait, data_dir, df)
        sankey = create_sankey_diagram(groups, color_scale)
        stripplot = create_stripplot(df, selected_trait, color_scale)
        grid = grid_handler(grid, df)
        return sankey, stripplot, grid

    app.run_server(debug=True)
    
if __name__ == "__main__":
    make_dashboard()