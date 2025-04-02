import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import colorsys
import os
import base64
import io

# Initialiser l'application Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Nécessaire pour le déploiement sur Render

def create_all_sequences_sankey(df, max_species_in_sequence=5):
    """
    Crée un diagramme de Sankey pour toutes les séquences de tous les échantillons.
    La couleur est basée sur la première espèce.
    """
    # Identifier les colonnes d'espèces
    species_cols = [col for col in df.columns if col.startswith('SP')]
    
    # Préparer les données pour le diagramme
    links = []
    color_map = {}  # Pour associer une couleur à chaque première espèce
    
    # Parcourir chaque ligne du dataframe (chaque combinaison unique de sample_index et sample_num)
    for idx, row in df.iterrows():
        # Identifier les espèces présentes et leur abondance
        species_present = []
        for col in species_cols:
            if row[col] > 0:
                species_present.append((col, row[col]))
        
        # Trier les espèces par abondance décroissante
        species_present.sort(key=lambda x: x[1], reverse=True)
        
        # Limiter à max_species_in_sequence pour la visualisation
        species_present = species_present[:max_species_in_sequence]
        
        # Créer les liens pour cette séquence
        if species_present:
            # Identifier la première espèce pour la couleur
            first_species = species_present[0][0]
            
            # Attribuer une couleur si elle n'existe pas encore
            if first_species not in color_map:
                # Créer une couleur HSV et la convertir en RGB
                hue = len(color_map) * 0.1 % 1  # Distribuer les teintes
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                hex_color = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.7)"
                color_map[first_species] = hex_color
            
            # Ajouter lien du début à la première espèce
            sample_id = f"Site {row['sample_index']}, Rep {row['sample_num']}"
            links.append({
                "source": "Début", 
                "target": f"Rang 1: {first_species}", 
                "value": 1,
                "color": color_map[first_species]
            })
            
            # Ajouter les liens entre espèces consécutives
            for i in range(len(species_present) - 1):
                source_species = species_present[i][0]
                target_species = species_present[i+1][0]
                links.append({
                    "source": f"Rang {i+1}: {source_species}", 
                    "target": f"Rang {i+2}: {target_species}", 
                    "value": 1,
                    "color": color_map[first_species]  # Même couleur basée sur la première espèce
                })
    
    # Construire le diagramme Sankey
    return build_sankey_diagram(
        links, 
        title="Séquences d'espèces pour tous les échantillons"
    )

def create_random_sites_sankey(df, num_sites=20, max_species_in_sequence=5):
    """
    Crée un diagramme de Sankey pour 20 sites aléatoires montrant les 30 répétitions.
    Utilise 30 couleurs différentes pour les répétitions.
    """
    # Identifier les colonnes d'espèces
    species_cols = [col for col in df.columns if col.startswith('SP')]
    
    # Sélectionner aléatoirement les sites
    all_sites = df['sample_index'].unique()
    if len(all_sites) <= num_sites:
        random_sites = all_sites
    else:
        random_sites = random.sample(list(all_sites), num_sites)
    
    # Filtrer le dataframe pour ces sites
    filtered_df = df[df['sample_index'].isin(random_sites)]
    
    # Créer une palette de 30 couleurs pour les répétitions
    repetition_colors = {}
    for i in range(30):  # 0 à 29 pour sample_num
        hue = i / 30
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        hex_color = f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.7)"
        repetition_colors[i] = hex_color
    
    # Préparer les données pour le diagramme
    links = []
    
    # Parcourir chaque ligne du dataframe filtré
    for idx, row in filtered_df.iterrows():
        # Identifier les espèces présentes et leur abondance
        species_present = []
        for col in species_cols:
            if row[col] > 0:
                species_present.append((col, row[col]))
        
        # Trier les espèces par abondance décroissante
        species_present.sort(key=lambda x: x[1], reverse=True)
        
        # Limiter à max_species_in_sequence pour la visualisation
        species_present = species_present[:max_species_in_sequence]
        
        # Créer les liens pour cette séquence
        if species_present:
            site_id = row['sample_index']
            rep_num = row['sample_num']
            
            # Ajouter lien du début à la première espèce
            links.append({
                "source": f"Site {site_id}", 
                "target": f"Rang 1: {species_present[0][0]}", 
                "value": 1,
                "color": repetition_colors[rep_num]
            })
            
            # Ajouter les liens entre espèces consécutives
            for i in range(len(species_present) - 1):
                source_species = species_present[i][0]
                target_species = species_present[i+1][0]
                links.append({
                    "source": f"Rang {i+1}: {source_species}", 
                    "target": f"Rang {i+2}: {target_species}", 
                    "value": 1,
                    "color": repetition_colors[rep_num]  # Couleur basée sur le numéro de répétition
                })
    
    # Construire le diagramme Sankey
    return build_sankey_diagram(
        links, 
        title=f"Séquences d'espèces pour {num_sites} sites aléatoires (30 répétitions par site)"
    )

def build_sankey_diagram(links, title):
    """
    Construit un diagramme Sankey à partir des liens fournis.
    """
    # Extraire les nœuds uniques
    unique_nodes = set()
    for link in links:
        unique_nodes.add(link["source"])
        unique_nodes.add(link["target"])
    
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    
    # Préparer les données pour Plotly
    sources = [node_to_idx[link["source"]] for link in links]
    targets = [node_to_idx[link["target"]] for link in links]
    values = [link["value"] for link in links]
    colors = [link["color"] for link in links]
    labels = list(unique_nodes)
    
    # Créer le diagramme
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])
    
    # Ajouter ces options avancées
    fig.update_layout(
        title_text=title,
        font_size=10,
        height=900,
        width=1200,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(label="Réinitialiser la vue", method="relayout", args=[{"xaxis.range": [None, None], "yaxis.range": [None, None]}]),
                    dict(label="Zoom avant", method="relayout", args=[{"xaxis.range[0]": 0.1, "xaxis.range[1]": 0.9, "yaxis.range[0]": 0.1, "yaxis.range[1]": 0.9}])
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                y=1.1
            )
        ],
        dragmode="zoom",
        modebar_add=["zoomIn2d", "zoomOut2d", "resetScale2d", "toImage"]
    )
    
    return fig

# Définir la mise en page de l'application
app.layout = html.Div([
    html.H1("Visualisation des Diagrammes de Sankey"),
    
    html.Div([
        html.H3("Charger vos données"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Glissez-déposez ou ',
                html.A('sélectionnez un fichier CSV')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ]),
    
    html.Div([
        html.H3("Paramètres de visualisation"),
        html.Label("Nombre maximum d'espèces par séquence :"),
        dcc.Slider(
            id='max-species-slider',
            min=2,
            max=10,
            step=1,
            value=5,
            marks={i: str(i) for i in range(2, 11)},
        ),
        
        html.Label("Nombre de sites (pour le 2ème diagramme) :"),
        dcc.Slider(
            id='num-sites-slider',
            min=5,
            max=30,
            step=5,
            value=20,
            marks={i: str(i) for i in range(5, 31, 5)},
        ),
        
        html.Button('Générer les diagrammes', id='generate-button', n_clicks=0)
    ], style={'margin': '20px'}),
    
    html.Div([
        html.H3("Diagramme 1: Toutes les séquences"),
        dcc.Graph(id='sankey-all-sequences')
    ]),
    
    html.Div([
        html.H3("Diagramme 2: Sites aléatoires"),
        dcc.Graph(id='sankey-random-sites')
    ])
])

# Fonction pour parser le contenu du fichier chargé
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Suppose que c'est un csv utf-8
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.Div([
                'Veuillez charger un fichier CSV.'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'Une erreur est survenue lors du traitement du fichier.'
        ])
    
    return html.Div([
        html.H5(f'Fichier chargé: {filename}'),
        html.Hr(),
        html.Div('Données chargées avec succès. Dimensions: {}x{}'.format(df.shape[0], df.shape[1])),
        html.Div('Colonnes: ' + ', '.join(df.columns[:10]) + ('...' if len(df.columns) > 10 else '')),
        
        # Stocker les données dans une div invisible
        html.Div(df.to_json(date_format='iso', orient='split'), id='stored-data', style={'display': 'none'})
    ])

# Callback pour afficher les informations du fichier chargé
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        return parse_contents(contents, filename)
    return html.Div('Aucun fichier chargé')

# Callback pour générer les diagrammes
@app.callback(
    [Output('sankey-all-sequences', 'figure'),
     Output('sankey-random-sites', 'figure')],
    [Input('generate-button', 'n_clicks')],
    [State('stored-data', 'children'),
     State('max-species-slider', 'value'),
     State('num-sites-slider', 'value')]
)
def update_graphs(n_clicks, stored_data, max_species, num_sites):
    if n_clicks == 0 or stored_data is None:
        # Retourner des graphiques vides
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Chargez vos données et cliquez sur 'Générer'")
        return empty_fig, empty_fig
    
    # Charger les données depuis le stockage
    df = pd.read_json(stored_data, orient='split')
    
    # Générer les diagrammes
    fig1 = create_all_sequences_sankey(df, max_species_in_sequence=max_species)
    fig2 = create_random_sites_sankey(df, num_sites=num_sites, max_species_in_sequence=max_species)
    
    return fig1, fig2

# Point d'entrée pour Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)

# S'assurer que la variable server est accessible
# Cette ligne est nécessaire pour que gunicorn trouve l'objet server
server = app.server