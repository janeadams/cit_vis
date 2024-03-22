import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import dash
import os
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cit_vis.parse import get_assay_scores, get_genus_abundances, group_by_nodes, get_relevant_assays

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

def get_subset(row, current_bug):
    rules = row['path']
    subset = pd.read_csv('data/genus_abundances.csv')
    rules = rules.split(' & ')
    for rule in rules:
        bug, sign, val = rule.split(' ')
        if current_bug != bug:
            if sign == '<=':
                subset = subset[subset[bug] <= float(val)]
            elif sign == '>':
                subset = subset[subset[bug] > float(val)]
    print(subset.head())
    print(subset.shape)
    print()
    return subset

def make_charts(assay):
    """
    Adjust this function to fit your new data structure and requirements.

    Parameters:
    - assay: The assay to analyze.

    Returns:
    - depths: Dictionary mapping depths to charts.
    """
    A_DF = pd.read_csv('data/assay_scores.csv').reset_index().rename(columns={'index': 'Mouse_ID'})
    split_df = pd.read_csv(f'data/{assay}/tree_summary.csv')
    # Ensure output directories exist
    os.makedirs(f'output/{assay}', exist_ok=True)
    
    # Create assay lookup and prepare split_df
    assay_lookup = dict(zip(A_DF['Mouse_ID'], A_DF[assay]))
    split_df['depth'] = [len(str(s).split(' ')) for s in split_df['path']]
    split_df['depth'] = split_df['depth'].rank(method='dense')
    
    # Filter for leaf nodes
    leaf_df = split_df[split_df['is_leaf'] == True]
    
    # Initialize tracking variables
    seen_bugs = {'root': 0}
    depths = {l: [] for l in set(split_df['depth'].values)}
    
    # Loop through the DataFrame rows
    for index, row in split_df.iterrows():
        g = row['genus']  # Adjust for new column name if changed
        v = row['split_value']  # Adjust if the format or logic has changed
        
        compute = False
        # Check if you need to update the computation logic based on new data
        if g not in seen_bugs or (g != 'root' and seen_bugs[g] != v):
            compute = True
        
        if compute:
            # Update how you slice your data if necessary
            subset = get_subset(row, g)
            subset[assay] = [assay_lookup[m] for m in slice['Mouse_ID']]
            subset['group'] = slice[g].apply(lambda x: 'below' if x < v else 'above')
            
            # Create your plotly chart here (adjust as needed)
            fig = create_plotly_chart(slice, g, v, assay, get_color)
            
            # Save the chart and update depths dictionary
            fig.write_image(f"output/{assay}/{row['node']}_{g}.svg")
    
    return depths

def create_plotly_chart(slice, g, v, assay, get_color):
    """
    Create and return a Plotly chart based on the given parameters.
    Adjust the function parameters and logic as needed.
    """
    # Placeholder function for creating a plotly chart
    # Replace this with your actual chart creation logic
    fig = px.histogram(slice, x=g)
    # Customize fig as needed
    return fig

def make_grid(assay):
    content = []
    depths = make_charts(assay)
    for depth, figs in depths.items():
        figs = [f for f in figs if f is not None]
        content.append(html.Div(figs, style={'margin': '0 auto', 'background-color': '#666'}))
    return content

def make_dashboard():
    assay_df = get_assay_scores()
    assays = assay_df.columns
    genus_df = get_genus_abundances()
    genera = genus_df.columns
    genera_dropdown_options = [{'label': genus, 'value': genus} for genus in genera]
    genera_dropdown_options.insert(0, {'label': 'All', 'value': 'All'})

    selected_genus = 'All'
    assay_dropdown_options = [{'label': assay, 'value': assay} for assay in assays]
    selected_assay = assays[0]

    df = get_assay_scores(selected_assay)
    color_scale = create_color_scale(df)

    grid = make_grid(selected_assay)

    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("CIT Visualization"),
        html.Div([
            dcc.Dropdown(
                id='genus-dropdown',
                options=genera_dropdown_options,
                value=selected_genus
            ),
            dcc.Dropdown(
                id='assay-dropdown',
                options=assay_dropdown_options,
                value=selected_assay
            )
        ], style={'display': 'flex', 'width': '100%'}),
        html.Div([
            dcc.Graph(id='sankey-diagram'),
            dcc.Graph(id='stripplot')
        ], style={'display': 'flex', 'width': '100%'}),
        html.Div(dcc.Graph(id='grid', figure=grid)),
        dcc.Store(id='selected-assay', data=selected_assay)
    ])

    @app.callback(
        [Output('assay-dropdown', 'options'),
         Output('assay-dropdown', 'value')],
        [Input('genus-dropdown', 'value')]
    )
    def update_assay_dropdown(selected_genus):
        if selected_genus == 'All':
            options = [{'label': assay, 'value': assay} for assay in assays]
            return options, assays[0]
        relevant_assays = get_relevant_assays(selected_genus)
        options = [{'label': assay, 'value': assay} for assay in relevant_assays]
        return options, relevant_assays[0] if relevant_assays else assays[0]

    @app.callback(
        [Output('sankey-diagram', 'figure'),
         Output('stripplot', 'figure'),
         Output('grid', 'figure')],
        [Input('assay-dropdown', 'value')]
    )
    def update_figures(selected_assay):
        df = get_assay_scores(selected_assay)
        color_scale = create_color_scale(df)
        sankey_fig = create_sankey_diagram(df, color_scale)
        stripplot_fig = create_stripplot(df, color_scale)
        grid = make_grid(selected_assay)
        return sankey_fig, stripplot_fig, grid

    app.run_server(debug=True, use_reloader=False)