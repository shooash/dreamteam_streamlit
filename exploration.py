import pandas as pd
import numpy as np
from IPython.display import display
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


## Mettons les chemins pour nos fichier dans des variable
ENTREPRISES = 'input/base_etablissement_par_tranche_effectif.csv'
VILLESDEFRANCE = 'input/cities.csv'
REFVILLESDEFRANCE = 'input/referentiel_cities_data_gouv.csv'

GEO = 'input/name_geographic_information.csv'
SALAIRE = 'input/net_salary_per_town_categories.csv'
POPULATION_ORIG = 'input/population.csv'
POPULATION = 'input/population_dataset.csv'
OUTPUT = 'output/dataset.csv'


def exploration_enterprises_and():
    entreprises = pd.read_csv(ENTREPRISES)
    entreprises.columns = entreprises.columns.str.lower() #pas besoin de caps lock ici
    entreprises['reg'] = entreprises.reg.astype(str) #les regions c'est catégoriel

    entreprises.info() #tout semble bien propre
    display(entreprises.isna().sum()) #pas de na a traiter
    display(entreprises.dep.unique()) #liste departement a des anomalie?
    entreprises[entreprises.dep.isin(['2A', '2B'])] #Non, c'est la Corse
    display(entreprises.codgeo.str[:2].unique())
    return entreprises

def exploration_population_and():
    ## attention! memory usage: 455.9+ MB
    population = pd.read_csv(POPULATION_ORIG)
    population.columns = population.columns.str.lower() #pas besoin de caps lock ici
    display(population.head())
    population.info()
    #les valeurs codgeo on 'mixed type'. Comme ce sont les numeros insee avec les codes 2A et 2B il vaut mieux les avoirs en str
    population['codgeo'] = population.codgeo.apply(lambda a: '0' + str(a) if not isinstance(a, str) and a < 10000 else str(a) )
    display(population.isna().sum())
    display(population.codgeo.str[:2].unique()) #au niveau des departement tout est pertinant
    return population

def exploration_salaire_and():
    salaires = pd.read_csv(SALAIRE)
    salaires.columns = salaires.columns.str.lower() #pas besoin de caps lock ici
    display(salaires.head())
    salaires.info()
    display(salaires.isna().sum())
    display(salaires.codgeo.str[:2].unique()) #au niveau des departement tout est pertinant
    display(salaires.iloc[:, 2:25].eq(0).sum()) #pas de salaire zero
    return salaires

'''
Pour chaque cologne on demande:
Type de variable	La variable est-elle une variable explicative ou la variable cible ? (Applicable uniquement aux projets d'apprentissage supervisé)

Description	Que représente cette variable en quelques mots ?

Disponibilité de la variable a priori	Pouvez vous connaitre ce champ en amont d'une prédiction ?

Type informatique	Aurez vous accès à cette variable en environnement de production ?"	"int64, float etc... Si ""object"", détaillez.

Taux de NA en %

Gestion des NA	Quelle mode de (non) - gestion des NA favorisez vous ?

Distribution des valeurs 	Pour les variables catégorielles comportant moins de 10 catégories, énumérez toutes les catégories. Pour les variables quantitatives, détaillez la distribution (statistiques descriptives de base).

Remarques sur la colonne champs libre à renseigner

'''

descr = {}
descr['entreprises'] = {
    'CODGEO' : "ID géographique de la ville",
    'LIBGEO' : "nom de la ville",
    'REG' : "numéro de région",
    'DEP' : "numéro de département",
    'E14TST' : "nombre total d'entreprises dans la ville",
    'E14TS0ND' : "nombre d'entreprises de taille inconnue ou nulle dans la ville",
    'E14TS1' : "nombre d'entreprises de 1 à 5 employés dans la ville",
    'E14TS6' : "nombre d'entreprises de 6 à 9 employés dans la ville",
    'E14TS10' : "nombre d'entreprises de 10 à 19 employés dans la ville",
    'E14TS20' : "nombre d'entreprises de 20 à 49 employés dans la ville",
    'E14TS50' : "nombre d'entreprises de 50 à 99 employés dans la ville",
    'E14TS100' : " nombre d'entreprises de 100 à 199 employés dans la ville",
    'E14TS200' : "nombre d'entreprises de 200 à 499 employés dans la ville",
    'E14TS500' : "nombre d'entreprises de plus de 500 employés dans la ville"
    }
descr['salaires'] = {
    "CODGEO" : "ID géographique de la ville",
    "LIBGEO" : "nom de la ville",
    "SNHM14" : "salaire net moyen",
    "SNHMC14" : "salaire net moyen par heure pour les cadres",
    "SNHMP14" : "salaire net moyen par heure pour un cadre moyen",
    "SNHME14" : "salaire net moyen par heure pour l'employé",
    "SNHMO14" : " salaire net moyen par heure pour le travailleur",
    "SNHMF14" : "salaire net moyen pour les femmes",
    "SNHMFC14" : "salaire net moyen par heure pour les cadres féminins",
    "SNHMFP14" : "salaire net moyen par heure pour les cadres moyens féminins",
    "SNHMFE14" : "salaire net moyen par heure pour une employée ",
    "SNHMFO14" : "salaire net moyen par heure pour une travailleuse ",
    "SNHMH14" : "salaire net moyen pour un homme",
    "SNHMHC14" : "salaire net moyen par heure pour un cadre masculin",
    "SNHMHP14" : "salaire net moyen par heure pour les cadres moyens masculins",
    "SNHMHE14" : "salaire net moyen par heure pour un employé masculin",
    "SNHMHO14" : "salaire net moyen par heure pour un travailleur masculin",
    "SNHM1814" : "salaire net moyen par heure pour les 18-25 ans",
    "SNHM2614" : "salaire net moyen par heure pour les 26-50 ans",
    "SNHM5014" : "salaire net moyen par heure pour les >50 ans",
    "SNHMF1814" : "salaire net moyen par heure pour les femmes âgées de 18 à 25 ans",
    "SNHMF2614" : "salaire net moyen par heure pour les femmes âgées de 26 à 50 ans",
    "SNHMF5014" : "salaire net moyen par heure pour les femmes de plus de 50 ans",
    "SNHMH1814" : "salaire net moyen par heure pour les hommes âgés de 18 à 25 ans",
    "SNHMH2614" : "salaire net moyen par heure pour les hommes âgés de 26 à 50 ans",
    "SNHMH5014" : "salaire net moyen par heure pour les hommes de plus de 50 ans"
    }
descr['population'] = {
    "NIVGEO" : "geographic level (arrondissement, communes…)",
    "CODGEO" : "unique code for the town",
    "LIBGEO" : "name of the town",
    "MOCO" : "mode de cohabitation (11 = enfants vivant avec deux parents,12 = enfants vivant avec un seul parent, 21 = adultes vivant en couple sans enfant, 22 = adultes vivant en couple avec enfants, 23 = adultes vivant seuls avec enfants, 31 = personnes étrangères à la famille vivant au foyer, 32 = personnes vivant seules",
    "AGE80_17" : "catégorie d'âge (tranche de 5 ans) | ex : 0 -> personnes âgées de 0 à 4 ans",
    "SEXE" : "sexe, 1 pour homme | 2 pour femme",
    "NB" : "Nombre de personnes dans la catégorie"
}
descr['geo'] = { #https://www.data.gouv.fr/fr/datasets/listes-des-communes-geolocalisees-par-regions-departements-circonscriptions-nd/
    "EU_CIRCO" : "Circonscriptions françaises aux élections européennes", 
    "CODE_RÉGION" : "Numéro de la région",
    "NOM_RÉGION" : "Nom de la région",
    "CHEF.LIEU_RÉGION" : "Chef-lieu de la région",
    "NUMÉRO_DÉPARTEMENT" : "Numéro du département",
    "NOM_DÉPARTEMENT" : "Nom du département",
    "PRÉFECTURE" : "Ville de la préfecture",
    "NUMÉRO_CIRCONSCRIPTION" : "Numéro de circonscription",
    "NOM_COMMUNE" : "Nom de la commune",
    "CODES_POSTAUX" : "Code postale",
    "CODE_INSEE" : "Code INSEE de la commune",
    "LATITUDE" : "Latitude",
    "LONGITUDE" : "Longitude",
    "ÉLOIGNEMENT" : "NA"
}

def rapport_table(df: pd.DataFrame, description):
    '''
    Generer le rapport standard. Description est le dictionnaire descriptive 'col' : 'description'.
    '''
    cols = df.columns.tolist()
    result = pd.DataFrame(columns = ['name', 'type', 'description', 'type_info', 'taux NA', 'gestion NA', 'distibution valeur'])
    def valeur_distrib(col):
        '''
        Pour les variables catégorielles comportant moins de 10 catégories, 
        énumérez toutes les catégories. 
        Pour les variables quantitatives, 
        détaillez la distribution (statistiques descriptives de base).
        '''
        ret = ''
        if df[col].dtype == 'O': #c'est categoriel
            if len(df[col].unique()) < 10:
                for k, v in df[col].value_counts(normalize = True).to_dict():
                    ret += k + ' : ' + str(eval("%.1e" % (v * 100))) + '\n' #il faut enumérer les variables
                return ret
            return ''
        #c'est quantitatif
        ret = str(df[col].describe())
        return ret[:ret.rfind('\n')]
    for c in cols:
        ln = [c, #nom
              '', # explicative ou cible? à remplir manuellement
              description.get(c.upper(),''), # description de meta
              'str' if df[c].dtype == 'O' else str(df[c].dtype), #type informatique
              str(eval("%.1e" % (df[c].isna().value_counts(normalize = True)[1] * 100))) if True in df[c].isna().value_counts(normalize = True) else '0%', #taux NA
              '', #gestion NA à faire à la main
              valeur_distrib(c) #distribution valeur
              ]
        result.loc[len(result)] = ln #on ajoute la ligne dans le tableau
    return result

def get_google_coordinates(entreprises : pd.DataFrame):
    '''from dhelpers import get_location
    # cela prend > 40 min, c'est payant
    geo_par_codgeo = pd.DataFrame(get_location(entreprises.codgeo.to_list()))
    geo_par_codgeo.to_csv('entreprises_geo.csv', index = False)'''
    pass

def get_corr_matrix(df, title, angle=None):
    corr = df.select_dtypes(exclude='O').corr()
    mask = np.triu_indices(corr.shape[0], 1)
    corr.values[mask] = None
    # Voyons le heatmap normalisé
    fig = px.imshow(corr,
                    range_color=[-1, 1], 
                    color_continuous_scale=[(0, 'blue'), (0.25, 'lightgreen'), (0.5, 'white'), (0.75, 'orange'), (1, 'red') ],
                    text_auto='0.1f',
                    title=title,
                    width=800,
                    height=800)
    if angle is not None:
        fig.update_xaxes(tickangle=angle)
    return fig
def show_corr_matrix(df, title, angle=None):
    get_corr_matrix(df, title, angle).show()

def get_industrial_graph(df_deps, title):
    # Le niveau d'industrialisation par population
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_deps.population, y=df_deps['eti_ge'], name='ETI + GE', mode='markers'))
    fig.add_trace(go.Scatter(x=df_deps.population, y=df_deps['pme'], name='PME', mode='markers'))
    fig.add_trace(go.Scatter(x=df_deps.population, y=df_deps['mic'], name='Micro', mode='markers'))
    fig.update_yaxes(title_text='Entreprises')
    fig.update_xaxes(title_text='Population')
    fig.update_layout(title_text = title, height=500)
    return fig
def show_industrial_graph(df_deps, title):
    get_industrial_graph(df_deps, title).show()

def get_commune_distrib_bars(data : pd.Series, name = 'ETI et GE'):
    total = data.sum()
    top50 = data.sort_values(ascending=False).head(len(data) // 2).sum() / total * 100
    top25 = data.sort_values(ascending=False).head(len(data) // 4).sum() / total * 100
    top10 = data.sort_values(ascending=False).head(len(data) // 10).sum() / total * 100
    fig = px.bar(x = ['Top 10%', 'Top 25%', 'Top 50%'], y = [top10, top25, top50], text_auto='0.2f')
    fig.update_yaxes(ticksuffix='%', title = "Taux d'entreprises")
    fig.update_xaxes(title = "Communes les plus industrialisées")
    fig.update_layout(title='Distribution de nombre de {} par communes où ils sont présentes'.format(name))
    return fig

def get_ent_hist(data : pd.Series, name = 'ETI et GE'):
    fig2 = px.histogram(data, marginal='violin', labels = {'x' : "Entreprises"}, title = "Nombre d'entreprises de catégories {} par commune où ils sont présentes".format(name))
    fig2.update_yaxes(title_text='Communes')
    return fig2


def show_commune_distrib_bars(data : pd.Series, name = 'ETI et GE'):
    q = data.quantile(q=[0.25, 0.5, 0.75, 0.9, 0.95]).to_list()
    eti_ge_margin = q[2] + (q[2] - q[1]) * 1.5
    q = pd.qcut(data, q=[0.75, 1], labels=False)
    q = q.dropna()
    q4_count = q.count()
    print("Dernière quartile de {} est distribué dans {:0.2f}% ({}) communes.".format(name, q4_count / len(data) * 100, q4_count))
    total = data.sum()
    top50 = data.sort_values(ascending=False).head(len(data) // 2).sum() / total * 100
    top25 = data.sort_values(ascending=False).head(len(data) // 4).sum() / total * 100
    top10 = data.sort_values(ascending=False).head(len(data) // 10).sum() / total * 100
    print("Top 10% de communes avec ETI/GE en France ont {:0.2f}% de {}".format(top10, name))
    print("Top 25% de communes avec ETI/GE en France ont {:0.2f}% de {}".format(top25, name))
    print("Top 50% de communes avec ETI/GE en France ont {:0.2f}% de {}".format(top50, name))
    fig = px.bar(x = ['Top 10%', 'Top 25%', 'Top 50%'], y = [top10, top25, top50], text_auto='0.2f')
    fig.update_yaxes(ticksuffix='%', title = "Taux d'entreprises")
    fig.update_xaxes(title = "Communes les plus industrialisées")
    fig.update_layout(title='Distribution de nombre de {} par communes où ils sont présentes'.format(name))
    fig.show()
    fig = px.histogram(data, marginal='violin', labels = {'x' : "Entreprises"}, title = "Nombre d'entreprises de catégories {} par commune où ils sont présentes".format(name))
    fig.update_yaxes(title_text='Communes')
    fig.show()
    fig = px.ecdf(data, labels = {'x' : 'Entreprises de catégories {} dans les communes de France'.format(name)})
    fig.show()

def get_geo_map(df_communes, df_deps):
    df_eti_ge = df_communes[df_communes.eti_ge > 0]
    q = df_eti_ge.eti_ge.quantile(q=[0.25, 0.5, 0.75, 0.9, 0.95]).to_list()
    eti_ge_margin = q[2] + (q[2] - q[1]) * 1.5
    ## on charge le GeoJSON du data.gouv.fr pour les département
    import json
    with open('input/a-dep2021.json') as f:
        deps = json.load(f)
    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(geojson=deps,
                                    featureidkey='properties.dep',
                                    locations=df_deps.dep.str.rjust(2, '0'), #attention! toujours '01' etc pour les dept 
                                    z = df_deps.eti_ge,
                                    zmin = df_deps.eti_ge.min(),
                                    zmax = df_deps.eti_ge.max(),
                                    zmid = df_deps.eti_ge.median(),
                                    text = df_deps.libgeo + ' : ' + df_deps.eti_ge.astype(str) + ' entreprises',

                                    colorscale = [[0,'white'],[0.25,'red'],[1,'orange']],
                                    name = 'Par departements',
                                    showscale = True,
                                    colorbar=dict(len=0.5,
                                                bordercolor='black',
                                                title='ETI/GE en départements',
                                                title_side='right',
                                                ),
                                    showlegend = True,

                        ))
    fig.add_trace(go.Scattermapbox(lon = df_communes[df_communes.eti_ge > eti_ge_margin].longitude, 
                                lat = df_communes[df_communes.eti_ge > eti_ge_margin].latitude, 
                                text = df_communes[df_communes.eti_ge > eti_ge_margin].libgeo + 
                                    ' : ' + df_communes[df_communes.eti_ge > eti_ge_margin].eti_ge.astype(str) 
                                    + ' entreprises',
                                marker = {
                                    'size' : np.sqrt(df_communes[df_communes.eti_ge > eti_ge_margin].eti_ge * 50), #, a_min=0, a_max=150),
                                    'color' : df_communes[df_communes.eti_ge > eti_ge_margin].eti_ge,
                                    'showscale' : True,
                                    'colorscale' : [[0,'blue'],[0.10,'red'],[0.25, 'orange'], [1,'yellow']],
                                    'colorbar' : dict(len=0.5, 
                                                    xpad=100, 
                                                    bordercolor='black',
                                                    borderwidth=1, 
                                                    title='ETI/GE en communes',
                                                    title_side='right')
                                },
                                name='Par communes (> {} entreprises)'.format(int(eti_ge_margin)),
                                ))
    fig.update_layout(mapbox_style="carto-positron",
                    mapbox = dict(center=dict(lat=46.60, lon=1.98),            
                                zoom=5
                                ))
    fig.update_layout(
        width = 800,
        height = 800,
        title =  'Distribution géographique des ETI et GE en France (sauf DOM/TOM)',
        )
    return fig

def show_geo_map(df_communes, df_deps):
    get_geo_map(df_communes, df_deps).show()
    
def get_df_communes():
    # Charger la base des entreprises
    entreprises = pd.read_csv(ENTREPRISES)
    entreprises.columns = entreprises.columns.str.lower() #pas besoin de caps lock ici
    entreprises['reg'] = entreprises.reg.astype(str) #les regions c'est catégoriel
    #reunir pme et mic
    #pme : E14TS10 + E14TS20 + E14TS50 + E14TS100
    #pour des raisons de collecte de données on considera comme PME les entreprises avec moins de 200 employés
    #mic : E14TS1 + E14TS6
    #eti_ge: E14TS200, E14TS500
    entreprises['pme'] = entreprises[['e14ts10', 'e14ts20', 'e14ts50', 'e14ts100']].sum(axis=1)
    entreprises['mic'] = entreprises[['e14ts1','e14ts6']].sum(axis=1)
    entreprises['eti_ge'] = entreprises[['e14ts200','e14ts500']].sum(axis=1)
    entreprises = entreprises.drop(columns=['e14ts1','e14ts6', 'e14ts10', 'e14ts20', 'e14ts50', 'e14ts100', 'e14ts200','e14ts500']) 
    entreprises = entreprises.rename(columns = {'e14tst' : 'brut', 'e14ts0nd' : 'zero'})
    entreprises['net'] = entreprises.brut - entreprises.zero
    entreprises.dep = entreprises.dep.astype(str)
    entreprises.dep = entreprises.dep.str.replace('2A', '200').str.replace('2B', '300').astype('int')
    #on devra ignorer les DOM/TOM
    entreprises = entreprises.drop(index=entreprises[entreprises.dep > 900].index)
    #restaurer les noms
    entreprises.dep = entreprises.dep.astype(str)
    entreprises.dep = entreprises.dep.str.replace('200', '2A').str.replace('300', '2B')
    # chargement des données géographiques
    cities = pd.read_csv(VILLESDEFRANCE) #c'est mieux que Google)
    # merge
    df_communes = pd.merge(left = entreprises, right = cities[['insee_code', 'latitude', 'longitude']], left_on='codgeo', right_on='insee_code', how='left')
    df_communes = df_communes.drop('insee_code', axis=1)
    df_communes = df_communes.drop_duplicates()
    #Des valeurs manquantes on va essayer de les recupere de la base moins fiable name_geographic_information.csv
    df_geo = pd.read_csv(GEO)
    def restore_data(d):
        codgeo = d.codgeo
        info = df_geo[df_geo.code_insee == int(codgeo)].iloc[0]
        try:
            d.latitude = float(info.latitude)
            d.longitude = float(info.longitude)
        except ValueError:
            pass
        return d
    df_communes[df_communes.latitude.isna()] = df_communes[df_communes.latitude.isna()].apply(restore_data, axis=1)
    population = pd.read_csv(POPULATION)
    population.codgeo = population.codgeo.astype(str).str.rjust(5, '0')
    df_communes = pd.merge(left=df_communes, right=population, on='codgeo', how='left')
    base_size = df_communes.shape[0]
    df_communes = df_communes.dropna()
    new_size = df_communes.shape[0]
    print('On a rejeté {}% de la base à cause de NA soit {} lignes.'.format(round((base_size - new_size) / base_size, 2), (base_size - new_size)))
    return df_communes

def get_df_dep(df_communes):
    df_deps = df_communes.groupby('dep', as_index=False).agg('sum')
    df_geo = pd.read_csv(GEO)
    #il nous faudra recuperer les noms des departements dans la col 'libgeo'
    def add_dep(a):
        label = a.dep.rjust(2, '0') #format '01','02' etc
        a['libgeo'] = df_geo[df_geo.numéro_département == label].nom_département.iloc[0]
        return a
    df_deps = df_deps.apply(add_dep, axis = 1)
    return df_deps

def normalize_stats(data_, cols, total_col):
    # Les données absoluts soient en corrélation avec leurs sommes. Normalizons-les.
    data = data_.copy()
    renamer = dict()
    for i, c in enumerate(cols):
        data[c] = data[c] / data[total_col]
        cols[i] = cols[i] + '*'
        renamer[c] = c + '*'
    data.rename(renamer, axis=1, inplace=True)
    return data

def normalize_names(l : [str]):
    return [c + '*' for c in l]

def prepare_vin():
    #chargement des fichiers de données
    #données référentielles data gouv (https://www.data.gouv.fr/fr/datasets/villes-de-france/#/resources)
    referentiel_city = pd.read_csv(REFVILLESDEFRANCE, sep =";",encoding='ISO-8859-1') 
    referentiel_city.info()

    #chargement du fichier name_geographic_information.csv
    geographic_information =pd.read_csv(GEO) 
    #on constate que le fichier name_geographic_information possède beaucoup plus de valeur manquante en % sur les colonnes lagitude et longitude que le fichier referentiel_cities_data_gouv
    #l'idée sera de récupérer les longitudes et latitudes et code insee du référentiel pour l'ajouter dans le fichier avec la clé de jointure sur le code insee
    #on garde que ces 3 colonnes dans le dataset referentiel_city
    colonnes_a_verifier = ['latitude', 'longitude', 'insee_code']
    referentiel_city_filtre = referentiel_city[colonnes_a_verifier]
    referentiel_city_filtre = referentiel_city_filtre.dropna(subset=['insee_code'], how='all')
    referentiel_city_filtre = referentiel_city_filtre.dropna(subset=['longitude'], how='all')
    referentiel_city_filtre_clean_na = referentiel_city_filtre.dropna(subset=['latitude'], how='all')
    display("vérification de présence des NA", referentiel_city_filtre_clean_na.isna().sum())

    #réaffectation du nom referentiel_city sur le dataset nettoyé
    referentiel_city = referentiel_city_filtre_clean_na
    #changement du typage des données dans les datasets
    dictionnaire_typage_referentiel_city = {'insee_code': int,  "latitude" : float, "longitude" : float  }
    dictionnaire_typage_geographic_information ={'EU_circo': str, 'code_région': int, 'nom_région': str , "chef.lieu_région" : str , "numéro_département" : str , "nom_département" : str , "préfecture" : str , "numéro_circonscription" : int , "nom_commune" : str, "codes_postaux" : str, "code_insee": int , "latitude" : float, "longitude" : float, "éloignement":float  }
    #application des dictionnaires aux données
    referentiel_city = referentiel_city.astype(dictionnaire_typage_referentiel_city)
    geographic_information['longitude'] = geographic_information['longitude'].str.replace(',', '.') #ValueError: could not convert string to float: '5,83': Error while type casting for column 'longitude'
    geographic_information['longitude'] = geographic_information['longitude'].replace('-', np.nan) #ValueError: could not convert string to float: '-': Error while type casting for column 'longitude'
    geographic_information = geographic_information.astype(dictionnaire_typage_geographic_information) 
    colonnes_a_supprimer = ['longitude', 'latitude']
    geographic_information = geographic_information.drop(colonnes_a_supprimer, axis=1)
    #typage des données sur le champs CODEGEO et LIBGEO
    dictionnaire_typage_pop = {"CODGEO": int, "LIBGEO": str}
    #chargement du fichier de population

    pop = pd.read_csv(SALAIRE)
    # Convertir la colonne "CODGEO" en nombres avec gestion des erreurs
    pop['CODGEO'] = pd.to_numeric(pop['CODGEO'], errors='coerce')
    #calcul du salaire net annuel basé sur le salaire net par heure
    pop['SNHM14'] = pop['SNHM14'] * 151.67 * 12
    #on retire les champs avec un codgeo non numérique
    pop_filter = pop.dropna(subset=['CODGEO'])

    pop_type = pop_filter.astype(dictionnaire_typage_pop)

    resultat_fusion = pd.merge(referentiel_city, geographic_information, left_on='insee_code', right_on='code_insee', how='left')
    merged = pd.merge(resultat_fusion, pop_type, left_on='insee_code', right_on='CODGEO', how='left')
    #purge des données référentielles sur les villes où on a pas les salaires
    merged_clean_missing_salary = merged.dropna(subset=['SNHM14'])
    return merged_clean_missing_salary
    
def map_salaire(merged_clean_missing_salary : pd.DataFrame, show=True):
    from branca.colormap import linear

    # Agrégation par département (calcul de la moyenne ici)
    stats_departement = merged_clean_missing_salary.groupby('nom_département')['SNHM14'].mean().reset_index()
    display(stats_departement.head())
    # Charger votre GeoJSON représentant les limites des départements
    geojson_path = 'input/a-dep2021.json'
    gdf_departements = gpd.read_file(geojson_path)
    gdf_merged = gdf_departements.merge(stats_departement, left_on='libgeo', right_on='nom_département')
    gdf_merged['dep'] = gdf_merged['dep'].astype("string")
    gdf_merged['reg'] = gdf_merged['reg'].astype("string")
    gdf_merged['libgeo'] = gdf_merged['libgeo'].astype("string")
    gdf_merged['nom_département'] = gdf_merged['nom_département'].astype("string")
    fig = px.choropleth_mapbox(gdf_merged, 
                            geojson=gdf_merged.geometry, 
                            locations=gdf_merged.index,
                            color='SNHM14',
                            color_continuous_scale="Viridis",
                            mapbox_style="carto-positron",
                            zoom=5, center={"lat": 46.603354, "lon": 1.888334},
                            opacity=0.5,
                            hover_data=['nom_département', 'SNHM14'],
                            labels={'SNHM14': 'Salaire moyen net annuel'})
    fig.update_layout(mapbox_style="carto-positron",
                    mapbox = dict(center=dict(lat=46.60, lon=1.98),            
                                zoom=5
                                ))
    fig.update_layout(
        width = 800,
        height = 800,
        title =  'Salaire moyen par département',
        )
    if not show:
        return fig
    # Afficher la carte
    fig.show()
    
def map_salaire_gini(merged_clean_missing_salary : pd.DataFrame, show=True):
    #application du coefficient de gini dans les salaires

    def gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))

    # Test du calcul de Gini sur un département
    # incomes = np.array([50, 50, 70, 70, 70, 90, 150, 150, 150, 150])

    #calculate Gini coefficient for array of incomes
    # display(gini(incomes))

    # dep_test = merged_clean_missing_salary[merged_clean_missing_salary['nom_département'] == "Calvados"].SNHM14
    # display("gini function " , gini(dep_test))


    gini_values = merged_clean_missing_salary.groupby('nom_département')['SNHM14'].apply(gini).reset_index()
    gini_values.columns = ['nom_département', 'gini_coefficient']


    # Charger votre GeoJSON représentant les limites des départements
    geojson_path = 'input/a-dep2021.json'
    gdf_departements = gpd.read_file(geojson_path)


    # display(gini_values[gini_values['gini_coefficient']> 1])
    gdf_merged_gini_dep = gdf_departements.merge(gini_values, left_on='libgeo', right_on='nom_département')


    gdf_merged_gini_dep['dep'] = gdf_merged_gini_dep['dep'].astype("string")
    gdf_merged_gini_dep['reg'] = gdf_merged_gini_dep['reg'].astype("string")
    gdf_merged_gini_dep['libgeo'] = gdf_merged_gini_dep['libgeo'].astype("string")
    gdf_merged_gini_dep['nom_département'] = gdf_merged_gini_dep['nom_département'].astype("string")


    # Créer une carte choroplèthe
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf_merged_gini_dep.plot(column='gini_coefficient', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    ax.set_title('Coefficient de Gini du Salaire mensuel Net par département')
    ax.set_axis_off()
    if not show:
        return fig
    plt.show()

def plot_region_correlation(data, target_variable, show=True):
    """
    Plot a heatmap of correlations between 'nom_région' and the specified target variable.

    Parameters:
    - data (pd.DataFrame)
    - target_variable 

    Returns:
    - displays the heatmap
    - return a pyplot Figure if show=False 
    """
    # Vérifier si la variable cible est valide
    valid_variables = ['SNHM14', 'SNHMC14', 'SNHMP14', 'SNHME14', 'SNHMO14', 'SNHMF14', 'SNHMH14']
    if target_variable not in valid_variables:
        print(f"Erreur : Variable cible invalide. Les options valides sont {valid_variables}.")
        return
    # Sélectionner les colonnes pertinentes
    selected_columns = ['nom_région', target_variable]
    # Créer un sous-ensemble du DataFrame avec ces colonnes
    selected_data = data[selected_columns]
    # Effectuer le One-Hot Encoding
    selected_data_encoded = pd.get_dummies(selected_data, columns=['nom_région'])
    # Calculez la matrice de corrélation
    correlation_matrix_encoded = selected_data_encoded.corr()
    # Sélectionnez la moitié basse de la matrice de corrélation
    mask = np.triu(np.ones_like(correlation_matrix_encoded, dtype=bool))
    # Créez une heatmap avec Seaborn en utilisant le masque
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_encoded * 100, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5, mask=mask, square=True)
    plt.title(f"Corrélations entre 'Région' et '{target_variable}'")
    if not show:
        return fig
    plt.show()

def get_salaire_data():
    if not check_restore_rom_data():
        print("Les fichiers output/merged_data.csv ou output/merged_data.zip sont pas présents! Impossible de proceder avec le rendu sur les salaires")
        return
    # Load your dataset
    data = pd.read_csv('output/merged_data.csv', low_memory=False)
    # Filter out rows where NB is 0
    data = data[data['NB'] != 0]
    return data
            
def get_ols(data):
    import statsmodels.api as sm

    # Perform linear regression
    X = sm.add_constant(data['éloignement'])
    y = data['SNHM5014']
    model = sm.OLS(y, X).fit()
    return model, X

def get_eloignement(data, model, X):
    # Visualize the regression line
    sns.scatterplot(x='éloignement', y='SNHM14', data=data)
    sns.lineplot(x=data['éloignement'], y=model.predict(X), color='red')
    # Set plot labels
    plt.xlabel('éloignement')
    plt.ylabel('SNHM5014 (Salaire Net Moyen)')
    return plt
    
def get_salaire_residus(model):
    from scipy import stats

    # Calculer les résidus du modèle
    residus = model.resid

    # Tracer un histogramme des résidus
    plt.figure(figsize=(8, 6))
    sns.histplot(residus, kde=True, color='blue', stat='density', bins=50)

    # Ajouter une ligne pour la distribution normale
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, residus.mean(), residus.std())
    plt.plot(x, p, 'k', linewidth=2)

    # Ajouter des légendes et des titres
    plt.title('Normalité des Résidus')
    plt.xlabel('Résidus')
    plt.ylabel('Densité')
    return plt

def get_salaire_bokeh(data, show_plot=True):
    from bokeh.plotting import figure
    from bokeh.io import output_notebook,reset_output, show
    try:    
        from bokeh.models import Panel as TabPanel
    except ImportError:
        from bokeh.models import TabPanel
    from bokeh.models import HoverTool, LegendItem, Legend, Tabs
    from bokeh.layouts import row    
    # Agréger les données pour calculer les moyennes
    grouped_data = data.groupby('nom_région')['SNHM14'].mean().reset_index()
    # Trier les données par ordre croissant
    grouped_data = grouped_data.sort_values(by='SNHM14')
    # Définir une fonction pour la coloration conditionnelle
    def color_condition(value):
        if value < 13:
            return 'yellow'
        elif value < 14:
            return 'orange'
        else:
            return 'red'
    # Appliquer la fonction de coloration conditionnelle à la colonne 'SNHM14'
    grouped_data['color'] = grouped_data['SNHM14'].apply(color_condition)
    # Créer la figure
    p = figure(x_range=grouped_data['nom_région'], height=400, title='Barplot interactif avec moyennes de SNHM14',
            toolbar_location='right')
    # Ajouter les barres
    source = dict(x=grouped_data['nom_région'], top=grouped_data['SNHM14'], color=grouped_data['color'])
    bars = p.vbar(x='x', top='top', width=0.9, source=source, line_color="white", fill_color='color')
    # Ajouter l'outil de survol
    hover = HoverTool()
    hover.tooltips = [("Nom de la région", "@x"), ("Moyenne SNHM14", "@top")]
    p.add_tools(hover)
    # Personnaliser les axes
    p.xaxis.axis_label = 'Nom de la région'
    p.yaxis.axis_label = 'Moyenne SNHM14'
    # Ajuster manuellement la position des étiquettes de l'axe x
    p.xaxis.major_label_standoff = 12
    p.xaxis.major_label_text_font_size = '10pt'
    p.xaxis.major_label_orientation = 3.14/4  # Rotation de 45 degrés
    # Ajouter une légende avec des carrés de couleur
    legend_items = [
        LegendItem(label="SNHM14 < 13e net", renderers=[bars], index=0),
        LegendItem(label="13e net <= SNHM14 < 14e net", renderers=[bars], index=6),
        LegendItem(label="SNHM14 >= 14e net", renderers=[bars], index=23)
    ]
    legend = Legend(items=legend_items, location="top_left")
    p.add_layout(legend, 'above')

    if show_plot:
        # Afficher la figure
        show(p)    
        # # Reset Bokeh output state
        reset_output()
        # # Activer la sortie dans le notebook
        output_notebook()

    # Calculer les quartiles et l'écart-type
    data['SNHM14_q1'] = data.groupby('nom_région')['SNHM14'].transform(lambda x: x.quantile(0.25))
    data['SNHM14_q2'] = data.groupby('nom_région')['SNHM14'].transform('median')
    data['SNHM14_q3'] = data.groupby('nom_région')['SNHM14'].transform(lambda x: x.quantile(0.75))
    data['SNHM14_std_dev'] = data.groupby('nom_région')['SNHM14'].transform('std')
    # Agréger les données pour calculer les moyennes
    grouped_data = data.groupby('nom_région').agg(
        SNHM14=('SNHM14', 'mean'),
        color=('SNHM14', lambda x: color_condition(x.mean())),
        SNHM14_q1=('SNHM14_q1', 'first'),
        SNHM14_q2=('SNHM14_q2', 'first'),
        SNHM14_q3=('SNHM14_q3', 'first'),
        SNHM14_std_dev=('SNHM14_std_dev', 'first')
    ).reset_index()
    # Trier les données par ordre croissant
    grouped_data = grouped_data.sort_values(by='SNHM14')

    # Créer la figure
    p = figure(x_range=grouped_data['nom_région'], height=400, title='Barplot interactif avec moyennes de SNHM14',
            toolbar_location='right')
    # Ajouter les barres
    source = dict(
        x=grouped_data['nom_région'],
        top=grouped_data['SNHM14'],
        color=grouped_data['color'],
        q1=grouped_data['SNHM14_q1'],
        q2=grouped_data['SNHM14_q2'],
        q3=grouped_data['SNHM14_q3'],
        std_dev=grouped_data['SNHM14_std_dev']
    )
    bars = p.vbar(x='x', top='top', width=0.9, source=source, line_color="white", fill_color='color')
    # Ajouter l'outil de survol
    hover = HoverTool()
    hover.tooltips = [
        ("Nom de la région", "@x"),
        ("Moyenne SNHM14", "@top"),
        ("1er Quartile", "@q1"),
        ("Médiane", "@q2"),
        ("3e Quartile", "@q3"),
        ("Écart-type", "@std_dev")
    ]
    hover.formatters = {'@top': 'printf', '@q1': 'printf', '@q2': 'printf', '@q3': 'printf', '@std_dev': 'printf'}
    hover.mode = 'vline'
    p.add_tools(hover)
    # Personnaliser les axes
    p.xaxis.axis_label = 'Nom de la région'
    p.yaxis.axis_label = 'Moyenne SNHM14'
    # Ajuster manuellement la position des étiquettes de l'axe x
    p.xaxis.major_label_standoff = 12
    p.xaxis.major_label_text_font_size = '10pt'
    p.xaxis.major_label_orientation = 3.14/4  # Rotation de 45 degrés
    # Ajouter une légende avec des carrés de couleur
    legend_items = [
        LegendItem(label="SNHM14 < 13e net", renderers=[bars], index=0),
        LegendItem(label="13e net <= SNHM14 < 14e net", renderers=[bars], index=6),
        LegendItem(label="SNHM14 >= 14e net", renderers=[bars], index=23)
    ]
    legend = Legend(items=legend_items, location="top_left")
    p.add_layout(legend, 'above')

    if show_plot:
        # Afficher la figure
        show(p)
    
    # Calculer les quartiles et l'écart-type pour SNHMF14
    data['SNHMF14_q1'] = data.groupby('nom_région')['SNHMF14'].transform(lambda x: x.quantile(0.25))
    data['SNHMF14_q2'] = data.groupby('nom_région')['SNHMF14'].transform('median')
    data['SNHMF14_q3'] = data.groupby('nom_région')['SNHMF14'].transform(lambda x: x.quantile(0.75))
    data['SNHMF14_std_dev'] = data.groupby('nom_région')['SNHMF14'].transform('std')
    # Calculer les quartiles et l'écart-type pour SNHMH14
    data['SNHMH14_q1'] = data.groupby('nom_région')['SNHMH14'].transform(lambda x: x.quantile(0.25))
    data['SNHMH14_q2'] = data.groupby('nom_région')['SNHMH14'].transform('median')
    data['SNHMH14_q3'] = data.groupby('nom_région')['SNHMH14'].transform(lambda x: x.quantile(0.75))
    data['SNHMH14_std_dev'] = data.groupby('nom_région')['SNHMH14'].transform('std')
    # Agréger les données pour SNHMF14
    grouped_data_female = data.groupby('nom_région').agg(
        SNHMF14=('SNHMF14', 'mean'),
        color=('SNHMF14', lambda x: color_condition(x.mean())),
        SNHMF14_q1=('SNHMF14_q1', 'first'),
        SNHMF14_q2=('SNHMF14_q2', 'first'),
        SNHMF14_q3=('SNHMF14_q3', 'first'),
        SNHMF14_std_dev=('SNHMF14_std_dev', 'first')
    ).reset_index()
    # Agréger les données pour SNHMH14
    grouped_data_male = data.groupby('nom_région').agg(
        SNHMH14=('SNHMH14', 'mean'),
        color=('SNHMH14', lambda x: color_condition(x.mean())),
        SNHMH14_q1=('SNHMH14_q1', 'first'),
        SNHMH14_q2=('SNHMH14_q2', 'first'),
        SNHMH14_q3=('SNHMH14_q3', 'first'),
        SNHMH14_std_dev=('SNHMH14_std_dev', 'first')
    ).reset_index()
    # Trier les données par ordre croissant pour SNHMF14
    grouped_data_female = grouped_data_female.sort_values(by='SNHMF14')
    # Trier les données par ordre croissant pour SNHMH14
    grouped_data_male = grouped_data_male.sort_values(by='SNHMH14')
    # Créer la figure pour SNHMF14
    p_female = figure(x_range=grouped_data_female['nom_région'], height=400, title='Salaire net moyen pour les femmes',
                    toolbar_location='right')
    # Ajouter les barres pour SNHMF14
    source_female = dict(
        x=grouped_data_female['nom_région'],
        top=grouped_data_female['SNHMF14'],
        color=grouped_data_female['color'],
        q1=grouped_data_female['SNHMF14_q1'],
        q2=grouped_data_female['SNHMF14_q2'],
        q3=grouped_data_female['SNHMF14_q3'],
        std_dev=grouped_data_female['SNHMF14_std_dev']
    )
    bars_female = p_female.vbar(x='x', top='top', width=0.9, source=source_female, line_color="white", fill_color='color')
    # Ajouter l'outil de survol pour SNHMF14
    hover_female = HoverTool()
    hover_female.tooltips = [
        ("Nom de la région", "@x"),
        ("Moyenne SNHMF14", "@top"),
        ("1er Quartile", "@q1"),
        ("Médiane", "@q2"),
        ("3e Quartile", "@q3"),
        ("Écart-type", "@std_dev")
    ]
    hover_female.formatters = {'@top': 'printf', '@q1': 'printf', '@q2': 'printf', '@q3': 'printf', '@std_dev': 'printf'}
    hover_female.mode = 'vline'
    p_female.add_tools(hover_female)
    # Personnaliser les axes pour SNHMF14
    p_female.xaxis.axis_label = 'Nom de la région'
    p_female.yaxis.axis_label = 'Salaire net moyen pour les femmes'
    # Ajuster manuellement la position des étiquettes de l'axe x
    p_female.xaxis.major_label_standoff = 12
    p_female.xaxis.major_label_text_font_size = '10pt'
    p_female.xaxis.major_label_orientation = 3.14/4  # Rotation de 45 degrés
    # Ajouter une légende avec des carrés de couleur pour SNHMF14
    legend_items_female = [
        LegendItem(label="SNHMF14 < 13e net", renderers=[bars_female], index=0),
        LegendItem(label="13e net <= SNHMF14 < 14e net", renderers=[bars_female], index=24),
        LegendItem(label="SNHMF14 >= 14e net", renderers=[bars_female], index=25)
    ]
    legend_female = Legend(items=legend_items_female, location="top_left")
    p_female.add_layout(legend_female, 'above')

    # Créer la figure pour SNHMH14
    p_male = figure(x_range=grouped_data_male['nom_région'], height=400, title='Salaire net moyen pour les hommes',
                    toolbar_location='right')
    # Ajouter les barres pour SNHMH14
    source_male = dict(
        x=grouped_data_male['nom_région'],
        top=grouped_data_male['SNHMH14'],
        color=grouped_data_male['color'],
        q1=grouped_data_male['SNHMH14_q1'],
        q2=grouped_data_male['SNHMH14_q2'],
        q3=grouped_data_male['SNHMH14_q3'],
        std_dev=grouped_data_male['SNHMH14_std_dev']
    )
    bars_male = p_male.vbar(x='x', top='top', width=0.9, source=source_male, line_color="white", fill_color='color')
    # Ajouter l'outil de survol pour SNHMH14
    hover_male = HoverTool()
    hover_male.tooltips = [
        ("Nom de la région", "@x"),
        ("Moyenne SNHMH14", "@top"),
        ("1er Quartile", "@q1"),
        ("Médiane", "@q2"),
        ("3e Quartile", "@q3"),
        ("Écart-type", "@std_dev")
    ]
    hover_male.formatters = {'@top': 'printf', '@q1': 'printf', '@q2': 'printf', '@q3': 'printf', '@std_dev': 'printf'}
    hover_male.mode = 'vline'
    p_male.add_tools(hover_male)
    # Personnaliser les axes pour SNHMH14
    p_male.xaxis.axis_label = 'Nom de la région'
    p_male.yaxis.axis_label = 'Salaire net moyen pour les hommes'
    # Ajuster manuellement la position des étiquettes de l'axe x
    p_male.xaxis.major_label_standoff = 12
    p_male.xaxis.major_label_text_font_size = '10pt'
    p_male.xaxis.major_label_orientation = 3.14/4  # Rotation de 45 degrés
    # Ajouter une légende avec des carrés de couleur pour SNHMH14
    legend_items_male = [
        LegendItem(label="SNHMH14 < 13e net", renderers=[bars_male], index=0),
        LegendItem(label="13e net <= SNHMH14 < 14e net", renderers=[bars_male], index=6),
        LegendItem(label="SNHMH14 >= 14e net", renderers=[bars_male], index=23)
    ]

    legend_male = Legend(items=legend_items_male, location="top_left")

    p_male.add_layout(legend_male, 'above')

    if show_plot:
        # Afficher les deux figures
        show(row(p_female, p_male))
    # Calculer les quartiles et l'écart-type pour SNHM5014
    data['SNHM5014_q1'] = data.groupby('nom_région')['SNHM5014'].transform(lambda x: x.quantile(0.25))
    data['SNHM5014_q2'] = data.groupby('nom_région')['SNHM5014'].transform('median')
    data['SNHM5014_q3'] = data.groupby('nom_région')['SNHM5014'].transform(lambda x: x.quantile(0.75))
    data['SNHM5014_std_dev'] = data.groupby('nom_région')['SNHM5014'].transform('std')
    # Agréger les données pour SNHM5014
    grouped_data_5014 = data.groupby('nom_région').agg(
        SNHM5014=('SNHM5014', 'mean'),
        color=('SNHM5014', lambda x: color_condition(x.mean())),
        SNHM5014_q1=('SNHM5014_q1', 'first'),
        SNHM5014_q2=('SNHM5014_q2', 'first'),
        SNHM5014_q3=('SNHM5014_q3', 'first'),
        SNHM5014_std_dev=('SNHM5014_std_dev', 'first')
    ).reset_index()

    # Trier les données par ordre croissant pour SNHM5014
    grouped_data_5014 = grouped_data_5014.sort_values(by='SNHM5014')

    # Créer la figure pour SNHM5014
    p_5014 = figure(x_range=grouped_data_5014['nom_région'], height=400, title='Salaire net moyen par heure pour >50 ans',
                    toolbar_location='right')
    # Ajouter les barres pour SNHM5014
    source_5014 = dict(
        x=grouped_data_5014['nom_région'],
        top=grouped_data_5014['SNHM5014'],
        color=grouped_data_5014['color'],
        q1=grouped_data_5014['SNHM5014_q1'],
        q2=grouped_data_5014['SNHM5014_q2'],
        q3=grouped_data_5014['SNHM5014_q3'],
        std_dev=grouped_data_5014['SNHM5014_std_dev']
    )
    bars_5014 = p_5014.vbar(x='x', top='top', width=0.9, source=source_5014, line_color="white", fill_color='color')
    # Ajouter l'outil de survol pour SNHM5014
    hover_5014 = HoverTool()
    hover_5014.tooltips = [
        ("Nom de la région", "@x"),
        ("Moyenne SNHM5014", "@top"),
        ("1er Quartile", "@q1"),
        ("Médiane", "@q2"),
        ("3e Quartile", "@q3"),
        ("Écart-type", "@std_dev")
    ]
    hover_5014.formatters = {'@top': 'printf', '@q1': 'printf', '@q2': 'printf', '@q3': 'printf', '@std_dev': 'printf'}
    hover_5014.mode = 'vline'

    p_5014.add_tools(hover_5014)
    # Personnaliser les axes pour SNHM5014
    p_5014.xaxis.axis_label = 'Nom de la région'
    p_5014.yaxis.axis_label = 'Salaire net moyen par heure pour >50 ans'
    # Ajuster manuellement la position des étiquettes de l'axe x
    p_5014.xaxis.major_label_standoff = 12
    p_5014.xaxis.major_label_text_font_size = '10pt'
    p_5014.xaxis.major_label_orientation = 3.14/4  # Rotation de 45 degrés
    # Ajouter une légende avec des carrés de couleur pour SNHM5014
    legend_items_5014 = [
        LegendItem(label="SNHM5014 >= 14e net", renderers=[bars_5014], index=23)
    ]

    legend_5014 = Legend(items=legend_items_5014, location="top_left")
    p_5014.add_layout(legend_5014, 'above')

    if show_plot:
        # Afficher la figure
        show(p_5014)
    
    # Calculer les quartiles et l'écart-type pour SNHM2614
    data['SNHM2614_q1'] = data.groupby('nom_région')['SNHM2614'].transform(lambda x: x.quantile(0.25))
    data['SNHM2614_q2'] = data.groupby('nom_région')['SNHM2614'].transform('median')
    data['SNHM2614_q3'] = data.groupby('nom_région')['SNHM2614'].transform(lambda x: x.quantile(0.75))
    data['SNHM2614_std_dev'] = data.groupby('nom_région')['SNHM2614'].transform('std')
    # Agréger les données pour SNHM2614
    grouped_data_2614 = data.groupby('nom_région').agg(
        SNHM2614=('SNHM2614', 'mean'),
        color=('SNHM2614', lambda x: color_condition(x.mean())),
        SNHM2614_q1=('SNHM2614_q1', 'first'),
        SNHM2614_q2=('SNHM2614_q2', 'first'),
        SNHM2614_q3=('SNHM2614_q3', 'first'),
        SNHM2614_std_dev=('SNHM2614_std_dev', 'first')
    ).reset_index()
    # Trier les données par ordre croissant pour SNHM2614
    grouped_data_2614 = grouped_data_2614.sort_values(by='SNHM2614')
    # Créer la figure pour SNHM2614
    p_2614 = figure(x_range=grouped_data_2614['nom_région'], height=400, title='Salaire net moyen par heure pour 26-50 ans',
                    toolbar_location='right')
    # Ajouter les barres pour SNHM2614
    source_2614 = dict(
        x=grouped_data_2614['nom_région'],
        top=grouped_data_2614['SNHM2614'],
        color=grouped_data_2614['color'],
        q1=grouped_data_2614['SNHM2614_q1'],
        q2=grouped_data_2614['SNHM2614_q2'],
        q3=grouped_data_2614['SNHM2614_q3'],
        std_dev=grouped_data_2614['SNHM2614_std_dev']
    )
    bars_2614 = p_2614.vbar(x='x', top='top', width=0.9, source=source_2614, line_color="white", fill_color='color')
    # Ajouter l'outil de survol pour SNHM2614
    hover_2614 = HoverTool()
    hover_2614.tooltips = [
        ("Nom de la région", "@x"),
        ("Moyenne SNHM2614", "@top"),
        ("1er Quartile", "@q1"),
        ("Médiane", "@q2"),
        ("3e Quartile", "@q3"),
        ("Écart-type", "@std_dev")
    ]
    hover_2614.formatters = {'@top': 'printf', '@q1': 'printf', '@q2': 'printf', '@q3': 'printf', '@std_dev': 'printf'}
    hover_2614.mode = 'vline'
    p_2614.add_tools(hover_2614)
    # Personnaliser les axes pour SNHM2614
    p_2614.xaxis.axis_label = 'Nom de la région'
    p_2614.yaxis.axis_label = 'Salaire net moyen par heure pour 26-50 ans'
    # Ajuster manuellement la position des étiquettes de l'axe x
    p_2614.xaxis.major_label_standoff = 12
    p_2614.xaxis.major_label_text_font_size = '10pt'
    p_2614.xaxis.major_label_orientation = 3.14/4  # Rotation de 45 degrés
    # Ajouter une légende avec des carrés de couleur pour SNHM2614
    legend_items_2614 = [
        LegendItem(label="SNHM2614 < 13e net", renderers=[bars_2614], index=0),
        LegendItem(label="13e net <= SNHM2614 < 14e net", renderers=[bars_2614], index=10),
        LegendItem(label="SNHM2614 >= 14e net", renderers=[bars_2614], index=25)
    ]
    legend_2614 = Legend(items=legend_items_2614, location="top_left")
    p_2614.add_layout(legend_2614, 'above')

    if show_plot:
        # Afficher la figure
        show(p_2614)
    
    # Calculer les quartiles et l'écart-type pour SNHM1814
    data['SNHM1814_q1'] = data.groupby('nom_région')['SNHM1814'].transform(lambda x: x.quantile(0.25))
    data['SNHM1814_q2'] = data.groupby('nom_région')['SNHM1814'].transform('median')
    data['SNHM1814_q3'] = data.groupby('nom_région')['SNHM1814'].transform(lambda x: x.quantile(0.75))
    data['SNHM1814_std_dev'] = data.groupby('nom_région')['SNHM1814'].transform('std')
    # Agréger les données pour SNHM1814
    grouped_data_1814 = data.groupby('nom_région').agg(
        SNHM1814=('SNHM1814', 'mean'),
        color=('SNHM1814', lambda x: color_condition(x.mean())),
        SNHM1814_q1=('SNHM1814_q1', 'first'),
        SNHM1814_q2=('SNHM1814_q2', 'first'),
        SNHM1814_q3=('SNHM1814_q3', 'first'),
        SNHM1814_std_dev=('SNHM1814_std_dev', 'first')
    ).reset_index()
    # Trier les données par ordre croissant pour SNHM1814
    grouped_data_1814 = grouped_data_1814.sort_values(by='SNHM1814')
    # Créer la figure pour SNHM1814
    p_1814 = figure(x_range=grouped_data_1814['nom_région'], height=400, title='Salaire net moyen par heure pour 18-25 ans',
                    toolbar_location='right')
    # Ajouter les barres pour SNHM1814
    source_1814 = dict(
        x=grouped_data_1814['nom_région'],
        top=grouped_data_1814['SNHM1814'],
        color=grouped_data_1814['color'],
        q1=grouped_data_1814['SNHM1814_q1'],
        q2=grouped_data_1814['SNHM1814_q2'],
        q3=grouped_data_1814['SNHM1814_q3'],
        std_dev=grouped_data_1814['SNHM1814_std_dev']
    )
    bars_1814 = p_1814.vbar(x='x', top='top', width=0.9, source=source_1814, line_color="white", fill_color='color')
    # Ajouter l'outil de survol pour SNHM1814
    hover_1814 = HoverTool()
    hover_1814.tooltips = [
        ("Nom de la région", "@x"),
        ("Moyenne SNHM1814", "@top"),
        ("1er Quartile", "@q1"),
        ("Médiane", "@q2"),
        ("3e Quartile", "@q3"),
        ("Écart-type", "@std_dev")
    ]
    hover_1814.formatters = {'@top': 'printf', '@q1': 'printf', '@q2': 'printf', '@q3': 'printf', '@std_dev': 'printf'}
    hover_1814.mode = 'vline'
    p_1814.add_tools(hover_1814)
    # Personnaliser les axes pour SNHM1814
    p_1814.xaxis.axis_label = 'Nom de la région'
    p_1814.yaxis.axis_label = 'Salaire net moyen par heure pour 18-25 ans'
    # Ajuster manuellement la position des étiquettes de l'axe x
    p_1814.xaxis.major_label_standoff = 12
    p_1814.xaxis.major_label_text_font_size = '10pt'
    p_1814.xaxis.major_label_orientation = 3.14/4  # Rotation de 45 degrés
    # Ajouter une légende avec des carrés de couleur pour SNHM1814
    legend_items_1814 = [
        LegendItem(label="SNHM1814 < 13e net", renderers=[bars_1814], index=0),
    ]
    legend_1814 = Legend(items=legend_items_1814, location="top_left")
    p_1814.add_layout(legend_1814, 'above')

    if show_plot:
        # Afficher la figure
        show(p_1814)
    
    # Créer des onglets avec les trois graphiques
    tab_snhm14 = TabPanel(child=p, title="SNHM14")
    tab_male = TabPanel(child=p_male, title="Hommes (SNHMH14)")
    tab_female = TabPanel(child=p_female, title="Femmes (SNHMF14)")
    tab_18ans = TabPanel(child=p_1814,title = "18-25 ans (SNHM1814)")
    tab_26ans = TabPanel(child=p_2614,title="26-50 ans (SNHM2614)")
    tab_50ans = TabPanel(child=p_5014, title = ">50 ans (SNHM5014)")
    # Créer l'objet Tabs
    tabs = Tabs(tabs=[tab_snhm14, tab_male, tab_female,tab_18ans,tab_26ans,tab_50ans])
    return tabs       
     
def salaire_report():
    from bokeh.io import output_notebook,reset_output, show


    data = get_salaire_data()
    model, X = get_ols(data)
    # Display regression results
    print(model.summary())
    get_eloignement(data, model, X).show()
    get_salaire_residus(model).show()
    #Variable éloignement ne révèle aucun d'insight intéressant, l'éloignement ne semble pas avoir de relation linéaire avec les salaires moyens peu importe age ou classe
    plot_region_correlation(data, 'SNHM14')
    plot_region_correlation(data, 'SNHMC14')

    # Reset Bokeh output state
    reset_output()
    # Activer la sortie dans le notebook
    output_notebook()
    tabs = get_salaire_bokeh(data)
    # Afficher les onglets
    show(tabs)
    
def check_restore_rom_data():
    from pathlib import Path
    import zipfile
    csv_path = 'output/merged_data.csv'
    zip_path = 'output/merged_data.zip'
    if Path(csv_path).is_file():
        return True
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('output/')
    except FileNotFoundError:
        return False
    if Path(csv_path).is_file():
        return True
    return False

def show_boxplots(show=True):
    df = pd.read_csv(SALAIRE)
    # Changement de nom des colonnes pour plus de compréhension
    df = df.rename(columns = {"CODGEO" : "ID géographique de la ville",
        "LIBGEO" : "nom de la ville",
        "SNHM14" : "salaire net moyen",
        "SNHMC14" : "salaire net moyen par heure pour les cadres",
        "SNHMP14" : "salaire net moyen par heure pour un cadre moyen",
        "SNHME14" : "salaire net moyen par heure pour l'employé",
        "SNHMO14" : " salaire net moyen par heure pour le travailleur",
        "SNHMF14" : "salaire net moyen pour les femmes",
        "SNHMFC14" : "salaire net moyen par heure pour les cadres féminins",
        "SNHMFP14" : "salaire net moyen par heure pour les cadres moyens féminins",
        "SNHMFE14" : "salaire net moyen par heure pour une employée ",
        "SNHMFO14" : "salaire net moyen par heure pour une travailleuse ",
        "SNHMH14" : "salaire net moyen pour un homme",
        "SNHMHC14" : "salaire net moyen par heure pour un cadre masculin",
        "SNHMHP14" : "salaire net moyen par heure pour les cadres moyens masculins",
        "SNHMHE14" : "salaire net moyen par heure pour un employé masculin",
        "SNHMHO14" : "salaire net moyen par heure pour un travailleur masculin",
        "SNHM1814" : "salaire net moyen par heure pour les 18-25 ans",
        "SNHM2614" : "salaire net moyen par heure pour les 26-50 ans",
        "SNHM5014" : "salaire net moyen par heure pour les >50 ans",
        "SNHMF1814" : "salaire net moyen par heure pour les femmes âgées de 18 à 25 ans",
        "SNHMF2614" : "salaire net moyen par heure pour les femmes âgées de 26 à 50 ans",
        "SNHMF5014" : "salaire net moyen par heure pour les femmes de plus de 50 ans",
        "SNHMH1814" : "salaire net moyen par heure pour les hommes âgés de 18 à 25 ans",
        "SNHMH2614" : "salaire net moyen par heure pour les hommes âgés de 26 à 50 ans",
        "SNHMH5014" : "salaire net moyen par heure pour les hommes de plus de 50 ans"
        })
    numeric_cols = df.select_dtypes(exclude='O')
    # Créer un boxplot pour visualiser l'ensemble des variables numériques
    fig = plt.figure(figsize=(8, 16))
    sns.boxplot(data=numeric_cols, orient='h', palette='Set2')
    plt.title('Boxplot pour l\'ensemble des variables numériques')
    plt.xlabel('Valeurs')
    if not show:
        return fig
    plt.show()
    # Focus sur les données des salaire moyen(hommes + femmes) / par CSP

    cols_to_plot = df.iloc[:, :7]

    # Créer un boxplot en salaire moyen / par statut
    plt.figure(figsize=(12, 8))
    sns.boxplot(cols_to_plot, orient='h', palette='Set2')
    plt.title('salaire moyen / par CSP ')
    plt.xlabel('Valeurs')
    plt.show()

    # Focus sur les données des salaire moyens pour les femmes par CSP

    cols_to_plot = df.iloc[:, 7:12]

    # Créer un boxplot en salaire moyen / par statut
    plt.figure(figsize=(12, 8))
    sns.boxplot(cols_to_plot, orient='h', palette='Set2')
    plt.title('salaire moyen pour les femmes / par CSP ')
    plt.xlabel('Valeurs')
    plt.show()
    
        
    # Focus sur les données des salaires moyens pour les hommes par CSP

    cols_to_plot = df.iloc[:, 12: 17]

    # Créer un boxplot en salaire moyen / par statut
    plt.figure(figsize=(12, 8))
    sns.boxplot(cols_to_plot, orient='h', palette='Set2')
    plt.title('salaire moyen pour les hommes / par CSP ')
    plt.xlabel('Valeurs')
    plt.show()
    
    # Focus sur les données des salaires moyen (hommes + femmes) en fonction de lâge

    cols_to_plot = df.iloc[:, 17: 20]

    # Créer un boxplot en salaire moyen / par statut
    plt.figure(figsize=(12, 8))
    sns.boxplot(cols_to_plot, orient='h', palette='Set2')
    plt.title('salaire moyen / Âge ')
    plt.xlabel('Valeurs')
    plt.show()