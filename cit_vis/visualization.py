import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cit_vis.parse import get_assay_scores, get_genus_abundances, group_by_nodes, get_relevant_assays, get_mice, create_matrix

def create_color_scale(df):
    min_val = min(df['Assay'])
    max_val = max(df['Assay'])
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

def get_mean_value(df, node):
    node_df = group_by_nodes(df)
    mean_val = node_df.loc[node]['Mean']
    return mean_val

def create_sankey_df(df):
    sankey_df = pd.DataFrame(columns=['source', 'target', 'value'])
    for i, row in df.iterrows():
        path = row['Path'].split('>')
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            if source not in sankey_df['source'].values:
                sankey_df = pd.concat([sankey_df, pd.DataFrame({'source': source, 'target': target, 'value': [1]})])
            else:
                if target in sankey_df[sankey_df['source'] == source]['target'].values:
                    sankey_df.loc[(sankey_df['source'] == source) & (sankey_df['target'] == target), 'value'] += 1
                else:
                    sankey_df = pd.concat([sankey_df, pd.DataFrame({'source': source, 'target': target, 'value': [1]})])
    # Create informative labels for the nodes
    sankey_df['edge_label'] = sankey_df['source'] + ' to ' + sankey_df['target']
    return sankey_df

def create_sankey_diagram(df, color_scale):
    sankey_df = create_sankey_df(df)
    labels = list(set(sankey_df['source'].values) | set(sankey_df['target'].values))
    color = [get_color(get_mean_value(df, node), color_scale) if 'Node' in node else 'black' for node in labels]
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=color
        ),
        link=dict(
            source=[labels.index(x) for x in sankey_df['source']],
            target=[labels.index(x) for x in sankey_df['target']],
            value=sankey_df['value'],
            label=sankey_df['edge_label']
        )
    )])
    return fig

def create_stripplot(df, color_scale):
    # Sort by the mean node value
    node_df = group_by_nodes(df)
    node_df = node_df.sort_values(by='Mean')
    node_order = node_df.index
    df = df.set_index('Node_ID').loc[node_order].reset_index()
    # Assign a color to each node group
    df['color'] = [get_color(value, color_scale) for value in df['Assay']]
    color_lookup = dict(zip(df['Node_ID'], df['color']))
    fig = px.strip(df, x='Node_ID', y='Assay', color='Node_ID', color_discrete_map=color_lookup)
    fig.update_layout(template='plotly_white', showlegend=False, title='Assay scores by node')
    # Add a dark gray stroke to the strip plot points:
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGray')))
    # On hover, show the mouse ID and the assay score
    fig.update_traces(hovertemplate='Mouse ID: %{x}<br>Assay: %{y}')
    return fig

def create_grid_plot(assay, node_id, genus):
    mice = get_mice(assay, node_id)
    df = get_genus_abundances()
    df['group'] = [node_id if x else 'other' for x in df.index.isin(mice)]
    # Create a histogram with the genus abundance
    fig = px.histogram(df, x=genus, color='group', color_discrete_sequence=['lightgrey', 'blue'], nbins=20, histnorm='probability density', barmode='overlay', title=f'{genus} abundance in node {node_id}')
    # Add a vertical line to show the mean value
    fig.add_vline(x=df[genus].mean(), line_dash="dash", line_color="black")
    fig.update_layout(template='plotly_white', width=400, height=400)
    return fig

def grid_handler(df, assay):
    matrix = create_matrix(df, assay)
    plot_grid = [html.Div(f'{node}') for node in matrix.index]
    for genus in matrix.columns:
        row_items = [html.Div(f'{genus}')]
        for node in matrix.index:
            if matrix.loc[node, genus] == 0:
                row_items.append(html.Div(style={'width': '400px', 'height': '400px'}))
            else:
                row_items.append(html.Div([dcc.Graph(figure=create_grid_plot(assay, node, genus))]))
        row_items = html.Div(row_items, style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap'})
        plot_grid.append(row_items)
    plot_grid = html.Div(plot_grid, style={'display': 'flex', 'flex-direction': 'column'})
    return plot_grid

def make_dashboard():
    assay_df = get_assay_scores()
    assays = assay_df.columns
    genus_df = get_genus_abundances()
    genera = genus_df.columns
    genera_dropdown_options = [{'label': genus, 'value': genus} for genus in genera]
    genera_dropdown_options.insert(0, {'label': 'All', 'value': 'All'})

    # Things that update when the genus changes
    selected_genus = 'All'
    assay_dropdown_options = [{'label': assay, 'value': assay} for assay in assays]
    selected_assay = assays[0]

    # Things that update when the assay changes
    df = get_assay_scores(selected_assay)
    color_scale = create_color_scale(df)

    app = dash.Dash(__name__)
    
    # Define the layout
    app.layout = html.Div([
        html.H1("CIT Visualization"),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='genus-dropdown',
                    options=genera_dropdown_options,
                    value=selected_genus
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='assay-dropdown',
                    options=assay_dropdown_options,
                    value=selected_assay
                )
            ]),
        ]),
        html.Div([
            html.Div([
                dcc.Graph(id='sankey-diagram', figure=create_sankey_diagram(df, color_scale))
            ], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='stripplot', figure=create_stripplot(df, color_scale))
            ], style={'width': '49%', 'display': 'inline-block'})
        ]),
        html.Div(id='grid-container', children=grid_handler(df, selected_assay), style={'width': '100%'}),
        dcc.Store(id='selected-assay', data=selected_assay)
    ])

    @app.callback(
        [Output('assay-dropdown', 'options'),
         Output('assay-dropdown', 'value')],
        [Input('genus-dropdown', 'value'),
         Input('selected-assay', 'data')]
    )
    def update_assay_dropdown(selected_genus, selected_assay):
        if selected_genus == 'All':
            return [{'label': assay, 'value': assay} for assay in assays], selected_assay
        relevant_assays = get_relevant_assays(selected_genus)
        if selected_assay not in relevant_assays:
            selected_assay = relevant_assays[0]
        return [{'label': assay, 'value': assay} for assay in relevant_assays], selected_assay

    @app.callback(
        [Output('sankey-diagram', 'figure'),
         Output('stripplot', 'figure')],
        [Input('assay-dropdown', 'value')]
    )
    def update_sankey_diagram(selected_assay):
        df = get_assay_scores(selected_assay)
        color_scale = create_color_scale(df)
        return create_sankey_diagram(df, color_scale), create_stripplot(df, color_scale)

    @app.callback(
        Output('grid-container', 'children'),
        [Input('assay-dropdown', 'value')]
    )
    def update_grid(selected_assay):
        df = get_assay_scores(selected_assay)
        return grid_handler(df, selected_assay)

    app.run_server(debug=True)