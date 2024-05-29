
import plotly.graph_objects as go
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State,no_update
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import requests

# "C:\Users\akkus\OneDrive\Masaüstü\PROJ201\Final\Code\Forklore_final.py"

df = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/Finalized_KamerHoca_dataset.csv")
ob1 = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/Obese%20Data.csv")
ob2 = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/obsdfinal.csv", sep=';')
co2_df = pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/historical_emissions2.csv")
country_to_region_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/final.csv",  delimiter=';')
protein_df = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/protein.csv')
carbohydrate_df = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/carbohydrate.csv')
fat_df = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/saturated_fat.csv')
calcium_dfo = pd.read_csv('https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/calcium.csv')
number_of_ing_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/top_5_ingredients_by_region.csv")
region_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/country_region.csv")
affordibility_df=pd.read_csv("https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/Datasets/faostatnewv.csv" ,  delimiter=';')

logo="https://raw.githubusercontent.com/EnesAkkusci/Proj201-Datasets/main/proj.jpg"

obeseData = ob1
obeseDataGN = obeseData[obeseData['Gender'] == 'Total']
obeseDataGNonlyCount = obeseDataGN.dropna(subset=['Country Name'])

obeseDataGNonylCountAdult = obeseDataGNonlyCount[obeseDataGNonlyCount['Indicator Name'] == 'Obesity, adults aged 18+']


obeseDataWRegion = ob2

obeseDataWRegionGN = obeseDataWRegion[obeseDataWRegion['Gender'] == 'Total']


obeseDataWRegionAdult = obeseDataWRegion[obeseDataWRegion['Indicator Name'] == 'Obesity, adults aged 18+']

obeseDataWRegionAdultGendered = obeseDataWRegionAdult[obeseDataWRegionAdult['Gender'] != 'Total']

calcium_df = calcium_dfo.copy()


calcium_df["Median"] = calcium_df["Median"] / 10


protein_df['Nutrient'] = 'Protein (g)'
carbohydrate_df['Nutrient'] = 'Carbohydrate (g)'
fat_df['Nutrient'] = 'Fat (g)'
calcium_df['Nutrient'] = 'Calcium (0,1mg)'


combined_df = pd.concat([protein_df, carbohydrate_df, fat_df, calcium_df])

fig = go.Figure(data=go.Choropleth(
    locations=co2_df['Country'][1:],
    locationmode='country names',
    z=co2_df['2020'][1:],
    text="",
    colorscale='fall',
    autocolorscale=False,
    reversescale=False, 
    marker_line_color='darkgray',
    marker_line_width=1,
    colorbar_tickprefix='',
    colorbar_title='CO2 release (kg/L)',
))

fig.update_layout(
    width=1500,  
    height=500,  
    margin={"r":0,"t":0,"l":0,"b":0}  
)



first = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = first.server

#----------------------------------------------------------------------------------------------------------------------------------------


iceFig = px.icicle(
    obeseDataGNonylCountAdult,
    path=[px.Constant("World"), 'Region', 'Country Name'],
    values='Numeric',
    color='Numeric',
    color_continuous_scale='RdBu_r',
    title='% of obesity among adults',
    labels={'Numeric': ''},
)

iceFig.update_layout(font=dict(size=16))
iceFig.update_traces(hovertemplate='', hoverinfo='skip')


def create_age_histogram(data):
    fig = px.histogram(data, x='Indicator Name', y='Numeric', histfunc='avg', labels={'Numeric': 'percentage', 'Indicator Name': ''})

    fig.update_layout(font=dict(size=16))


    values = data['Numeric'].values
    sorted_values = sorted(values)
    color_map = {sorted_values[0]: '#2971b1', sorted_values[len(sorted_values) // 2]: '#fac8af', sorted_values[-1]: '#b6212f'}
    colors = [color_map[val] for val in values]

    fig.update_traces(hovertemplate='', hoverinfo='skip', marker=dict(color=colors))
    return fig


ageFig = create_age_histogram(obeseDataWRegionGN[obeseDataWRegionGN['Country Name'] == 'World'])


def create_gender_histogram(data):
    fig = px.histogram(data, x='Gender', y='Numeric', histfunc='avg', labels={'Numeric': 'percentage', 'Gender': ''})

    fig.update_layout(font=dict(size=16))


    values = data['Numeric'].values
    sorted_values = sorted(values)
    color_map = {sorted_values[0]: '#2971b1', sorted_values[-1]: '#b6212f'}  
    colors = [color_map[val] for val in values]

    fig.update_traces(hovertemplate='', hoverinfo='skip', marker=dict(color=colors))
    return fig


genderFig = create_gender_histogram(obeseDataWRegionAdultGendered[obeseDataWRegionAdultGendered['Country Name'] == 'World'])

#----------------------------------------------------------------------------------------------------------------------------------------

items = [
    'Obesity', 'Overweight'
]

CSE_ID = "87b06a24b639a4481"
API_KEY = "AIzaSyCPujz53FZgvdOpSZ8Xk81rKt1qYO8wdp0"

#----------------------------------------------------------------------------------------------------------------------------------------

first.layout = dbc.Container([
    dbc.Col([
        dbc.Row([
            dbc.Col([
                    html.H1("Forklore", style={'text-align': 'left', 'color': 'black'}),
                    html.H3("A nutritional data visualization web-app.", style={'text-align': 'left', 'color': 'black'})
        ]),
            dbc.Col(html.Img(src=logo, style={'width': '100px', 'height': '100px'}), width=2)
                 ]),      
        html.Hr(),
        dbc.Row([
            html.Div(
                children=[
                    html.H2(children="Get Links for your Recipes using AI!"),
                    dcc.Input(
                    id="search-input",
                    placeholder="Enter your recipe query...",
                    type="text",
                    value="",
                    style={'width': '75%', 'padding': '10px'}
                    ),
                    html.Button(
                    id="search-button",
                    children="Search",
                    style={'padding': '10px'}
                    ),
                    html.Div(
                    id="linkresults",
                    children=[]
                    )
                ]
            )
        ], style={'padding-bottom': '50px'}),
        dbc.Row([
            html.H2(children='Match Foods with their regions!'),
            dcc.Input(id='search-bar', type='text', placeholder='Search for a recipe or region...', style={'width': '75%', 'padding': '10px'}),
            html.Div(id='results')
        ], style={'padding-bottom': '50px'}),
        dbc.Row([
            html.H2(children='Obesity Data from around the World'),

        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=iceFig, id="iceGraph", style={"height": "1000px"}), width=6),
            dbc.Col([
                dcc.Graph(figure=ageFig, id="ageGraph", style={"padding-top": "50px"}),
                dcc.Graph(figure=genderFig, id="genderGraph", style={'padding-top': '50px'})
            ], width=6),
        ]),
            html.Hr(),
        dbc.Row([
            html.H2(children='CO2 Emissions of the Countries Around the World in 2020'),
        ]),
        dbc.Row([
            dbc.Col(html.Div(
                [
                dcc.Graph(figure=fig,id="co2map")
                ]
            ))
        ], style={}),
        
        dbc.Row([
            dbc.Col(html.Div([dcc.Graph(id="fruit"),
                            dcc.Slider(
            protein_df['Year'].min(),
            protein_df['Year'].max(),
            step=None,
            id='year-slider',
            value=protein_df['Year'].max(),
            marks={str(year): str(year) for year in protein_df['Year'].unique()})])),
            dbc.Col(dcc.Graph(id="co2")),
            dbc.Col(dcc.Graph(id="ingridient"))
        ]),
        dbc.Row([
            html.Div(id="economy" , style={'text-align': 'right'})
        ]),
        html.Hr()  
    ])
])

#----------------------------------------------------------------------------------------------------------------------------------------

@callback(
    Output(component_id='ageGraph', component_property='figure'),
    Input(component_id='iceGraph', component_property='clickData'),
)
def updateAgeGraph(regionChosen):
    if regionChosen is not None:
        region = regionChosen['points'][0]['label']
        filtered_data = obeseDataWRegionGN[obeseDataWRegionGN['Country Name'] == region]
    else:
        filtered_data = obeseDataWRegionGN[obeseDataWRegionGN['Country Name'] == 'World']

    return create_age_histogram(filtered_data)


@callback(
    Output(component_id='genderGraph', component_property='figure'),
    Input(component_id='iceGraph', component_property='clickData'),
)
def updateGenderGraph(regionChosen):
    if regionChosen is not None:
        region = regionChosen['points'][0]['label']
        filtered_data = obeseDataWRegionAdultGendered[obeseDataWRegionAdultGendered['Country Name'] == region]
    else:
        filtered_data = obeseDataWRegionAdultGendered[obeseDataWRegionAdultGendered['Country Name'] == 'World']

    return create_gender_histogram(filtered_data)


@callback(
    Output('results', 'children'),
    [Input('search-bar', 'value')]
)
def update_results(search_value):
    if not search_value:
        return []
    
    search_value = search_value.lower()


    if 'Recipe Name' not in df.columns or 'Region' not in df.columns:
        return [html.Div("Columns 'Recipe Name' or 'Region' not found in the dataset.")]

    filtered_df = df[df.apply(lambda row: search_value in str(row['Recipe Name']).lower() or search_value in str(row['Region']).lower(), axis=1)]
    if filtered_df.empty:
        return [html.Div("No results found.")]

    results_to_display = filtered_df.head(10)

    results = [html.Div(f"{row['Recipe Name']} - {row['Region']}") for _, row in results_to_display.iterrows()]

    if len(filtered_df) > 10:
        results.append(html.Div('More results available...'))

    return results

@callback(
    Output("linkresults", "children"),
    [Input("search-button", "n_clicks")],
    [State("search-input", "value")]
)
def update_results(n_clicks, search_query):
    if n_clicks is None:
        return []
    else:
        search_query = search_query + " recipe" 
        url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CSE_ID}&q={search_query}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "items" in data:  
                first_result = data["items"][0]
                return [html.A(href=first_result["link"], children=first_result["title"])]
            else:
                return [html.P("No results found")]
        else:
            return [html.P("Error: Could not retrieve search results")]

@callback(
    Output(component_id='fruit', component_property='figure'),
    Output(component_id='co2', component_property='figure'),
    Output(component_id='ingridient', component_property='figure'),
    Output(component_id='economy', component_property='children'),
    Input(component_id='year-slider', component_property='value'),
    Input(component_id='co2map', component_property='clickData'),
    prevent_initial_call=True
)
def update_children(year_selected,click):
    if click is None:
        return no_update,no_update,no_update,no_update
    else:
        country=click["points"][0]["location"]
        region_row=country_to_region_df[country_to_region_df["Country"]==country]
        region = region_row["Subregion"].iloc[0]
        filtered_data = combined_df[(combined_df['Year'] == year_selected) & (combined_df['Region'] == region)]
        row = co2_df[co2_df['Country'] == country]
        
        continent_row=region_df[region_df["Unnamed: 2"]==country]


        econ_df = affordibility_df[affordibility_df["Area"] == country]


        
        cons = row.iloc[0]['2020']  

        fig_nutrition = px.bar(filtered_data, x='Nutrient', y='Median', title=f'Nutrient Consumption in {region} for {year_selected}')
        
        co2_little = pd.DataFrame({"Country": ["World", country], "2020": [47513.15/195, cons]})
        co2_fig=px.histogram(co2_little,x="Country",y="2020", histfunc='avg', title=f'CO2 release of {country} and worlds avg')



        if econ_df.empty:
            econ_child=html.Div(["",
                                 html.Br(),
                                 ""])
            
        else:
            cost = econ_df[econ_df["Item"] == "Cost of a healthy diet (PPP dollar per person per day)"]["Value"].values[0]
            percentage = econ_df[econ_df["Item"] == "Percentage of the population unable to afford a healthy diet (percent)"]["Value"].values[0]
            econ_child = html.Div([
                f"Cost of a healthy diet (PPP dollar per person per day) in {country} is {cost}$ in 2021.",
                html.Br(),
                f"Percentage of the population unable to afford a healthy diet (percent) in {country} is {percentage}% in 2021."
            ])
        if continent_row.empty:
            return fig_nutrition,co2_fig,no_update,econ_child
        else:
            subregion = continent_row["Unnamed: 1"].iloc[0]
            top_five_df=number_of_ing_df[number_of_ing_df["Region"]==subregion]
            top_fig=px.histogram(top_five_df, x="Count",y="Ingredient", title=f"Top 5 ingridients used in {subregion}")            
            return fig_nutrition,co2_fig,top_fig,econ_child

if __name__ == '__main__':
    first.run(debug=True)
