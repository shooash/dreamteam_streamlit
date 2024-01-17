from zipfile import ZIP_DEFLATED, ZipFile
import streamlit as st
import exploration as ex
import pandas as pd
import pickle

st.set_page_config(page_title="Visualisation de données", page_icon="📊")
last_key=0
st.sidebar.write('## **Visualisation de données**')
pages = ['Entreprises', 'Salaires']
page = st.sidebar.radio('Presenter les graphiques sur:', pages)

@st.cache_data
def charger_dataset():
    df_communes = ex.get_df_communes() 
    df_deps = ex.get_df_dep(df_communes)
    return df_communes, df_deps

@st.cache_resource
def charger_carte(df_communes, df_deps):
    result = ex.get_geo_map(df_communes, df_deps)
    return result

@st.cache_resource
def charger_carte_gini():
    merged_clean_missing_salary = ex.prepare_vin()
    '''
    ### Cartographie du coefficient de Gini sur le salaire mensuel net par département
    
    L'objectif de ce coefficient est de visualiser s'il y a une forte disparité de salaire au sein d'un même département. À la vue de la carte, ce sont les départements proches de la région d'île de France qui ont une forte disparité de salaire mensuel moyen entre les villes. Le même constat est fait pour quelques départements situés dans le sud-est de la France et en Rhône-Alpes. Deux départements ont une répartition très homogène du salaire moyen. Il s'agit de la Lozère et du département Haute-Marne. De façon globale ,la disparité de salaire moyen par ville reste assez faible car le coefficient de gini est borné sur l'intervalle [0,1]. Plus ce coefficient tend vers 1 et plus la disparité est forte dans le regroupement de données. Plus ce coefficient tend vers 0 est moins l'inégalité est forte.
    '''
    st.write(ex.map_salaire_gini(merged_clean_missing_salary, show=False))
    '''
    ### Cartographie représentant le salaire moyen par département
    
    Ici nous pouvons voir qu'il y a une forte concentration des salaires élevés dans les départements composant l'île de France. Il y a une concentration des hauts salaires aussi située dans le sud-est de la France ainsi que certains départements de Rhône-Alpes. Le reste des départements ont un salaire annuel faible.
    '''
    st.write(ex.map_salaire(merged_clean_missing_salary, show=False))

if page == pages[0]:
    df_communes, df_deps = charger_dataset() 
    col_entreprises = ['brut', 'zero', 'pme', 'mic', 'eti_ge']
    col_couples = [i for i in df_communes.columns.to_list() if i.startswith('couple')]
    col_seuls = [i for i in df_communes.columns.to_list() if i.startswith('seul')]
    col_enfants = [i for i in df_communes.columns.to_list() if i.startswith('enfant')]
    col_etrangers = [i for i in df_communes.columns.to_list() if i.startswith('etranger')]
    col_population = col_couples + col_seuls + col_enfants + col_etrangers
    col_base = ['net', 'population']
    # On veut pas garder certain chiffre en absolut
    df_communes_norm = ex.normalize_stats(df_communes, col_entreprises, 'net')
    df_communes_norm = ex.normalize_stats(df_communes_norm, col_population, 'population')
    df_communes_norm = df_communes_norm.drop(['longitude', 'latitude'], axis = 1)
    col_entreprises = ['brut', 'zero', 'pme', 'mic', 'eti_ge']
    col_population = col_couples + col_seuls + col_enfants + col_etrangers
    df_deps_norm = ex.normalize_stats(df_deps, col_entreprises, 'net')
    df_deps_norm = ex.normalize_stats(df_deps_norm, col_population, 'population')
    df_deps_norm = df_deps_norm.drop(['longitude', 'latitude'], axis = 1)
    # ajouter les '*' aux noms des colonnes normalizées
    col_couples = ex.normalize_names(col_couples)
    col_seuls = ex.normalize_names(col_seuls)
    col_enfants = ex.normalize_names(col_enfants)
    col_etrangers = ex.normalize_names(col_etrangers)

    '# _French Industry_: preuves d\'inégalité en France'
    '## Visualisation de données : Entreprises'
    '''
    Le dataset *base_etablissement_par_tranche_effectif* présente des groupes très différents de données. Après la phase de préparation on assemble dans un même tableau les chiffres relatifs à la population, le nombre d'entreprises total («net»), micro entreprises, PME, ETI et GE. Cela nous permet de vérifier s'il existe une corrélation entre ces valeurs par communes et par départements.
    '''
    if st.checkbox("Afficher les matrix de correlation pour les **entreprises**", key = (last_key := last_key+1)):
        title_matrix_com = "Correlation des données sur les entreprises dans les commune françaises sauf DOM/TOM"
        st.write(ex.get_corr_matrix(df_communes_norm[col_entreprises + col_base + col_couples], title_matrix_com, angle=-45))
        st.write(ex.get_corr_matrix(df_communes_norm[col_entreprises + col_base + col_seuls], title_matrix_com, angle=-45))
        #st.write(ex.get_corr_matrix(df_communes_norm[col_entreprises + col_base + col_enfants], title_matrix_com, angle=-45))
        #st.write(ex.get_corr_matrix(df_communes_norm[col_entreprises + col_base + col_etrangers], title_matrix_com, angle=-45))
        # Au niveau departementale:
        title_matrix_dep = "Correlation des données sur les entreprises dans les départements français sauf DOM/TOM"
        st.write(ex.get_corr_matrix(df_deps_norm[col_entreprises + col_base + col_couples], title_matrix_dep, angle=-45))
        st.write(ex.get_corr_matrix(df_deps_norm[col_entreprises + col_base + col_seuls], title_matrix_dep, angle=-45))
        #st.write(ex.get_corr_matrix(df_deps_norm[col_entreprises + col_base + col_enfants], title_matrix_dep, angle=-45))
        #st.write(ex.get_corr_matrix(df_deps_norm[col_entreprises + col_base + col_etrangers], title_matrix_dep, angle=-45))
    '''
    L'analyse par quartile démontre qu'il s'agit des données avec une distribution fortement biaisée car la plupart des communes n'ont pas d'entreprises de certains types. Même parmi ceux où on a enregistré cette activité économique on peut constater que la distribution est très disproportionnée.
    '''
    if st.checkbox("Afficher la distribution des entreprises parmi les communes industrialisées", key = (last_key := last_key+1)):
        '''
        93.19% de communes n'ont pas d'ETI et GE, parmi les autres:
        '''
        st.write(ex.get_commune_distrib_bars(df_communes[df_communes.eti_ge > 0].eti_ge, "ETI/GE"))
        st.write(ex.get_ent_hist(df_communes[df_communes.eti_ge > 0].eti_ge, "ETI/GE"))
        
        '''
        53.63% de communes n'ont pas de PME, parmi les autres:
        
        '''
        st.write(ex.get_commune_distrib_bars(df_communes[df_communes.pme > 0].pme, "PME"))
        st.write(ex.get_ent_hist(df_communes[df_communes.pme > 0].pme, "PME"))
        '''
        14.89% de communes n'ont pas de micro entreprises, parmi les autres:
        
        '''
        st.write(ex.get_commune_distrib_bars(df_communes[df_communes.mic > 0].mic, "micro entreprises"))
        st.write(ex.get_ent_hist(df_communes[df_communes.mic > 0].mic, "micro entreprises"))
    '''
    L'analyse de la distribution géographique d'entreprises dévoiles des inégalités régionales dans la structure d'économie française.
    '''
    st.write(charger_carte(df_communes, df_deps))
    '''
    Sur les cartes on peut bien distinguer plusieurs clastères industrielles, notamment Île-de-France, Vallée du Rhône, Marseille, Toulouse et Lille. La région parisienne est assez évidemment dans une situation exclusive, ayant des chiffres d'industrialisation exceptionnels.
    '''
    
if page == pages[1]:
    data_name = 'models/dataviz_salaire'
    try:
        z = ZipFile(data_name, 'r')
        data = pickle.load(z.open(data_name, 'r'))
        z.close()
    except:
        data = ex.get_salaire_data()
        z = ZipFile(data_name, 'w', compression=ZIP_DEFLATED, compresslevel=5)
        pickle.dump(data, z.open(data_name, 'w'))
        z.close()
    '# _French Industry_: preuves d\'inégalité en France'
    '## Visualisation de données : Salaires'
    
    @st.cache_data
    def corr_plot(col):
        result = ex.plot_region_correlation(data, col, show=False)
        return result 

    @st.cache_resource
    def bokeh_tabs():
        result = ex.get_salaire_bokeh(data, show_plot=False)
        return result 
    @st.cache_data
    def boxplots():
        result = ex.show_boxplots(show=False)
        return result
    
    st.pyplot(corr_plot('SNHM14'))
    st.pyplot(corr_plot('SNHMC14'))
    st.bokeh_chart(bokeh_tabs())
    '''
    Les boxplots nous montrent que les données sont en général compactées. Elles ont tendance à s'élargir et à avoir une plus grande amplitude lorsqu'on est âgé de plus de 50 ans et que l'on appartient à une CSP supérieure. Enfin, nous noterons qu'il y a une différence notable sur la répartition des données et la valeur des données entre les hommes et les femmes.
    '''
    st.write(boxplots())
    charger_carte_gini()
    '''
    ## Analyse statistique
    '''
    '''
    Toutes les observations graphiques faites se sont avérées significatives après vérification par Two-sample T-test. Tous les tests ont été effectués avec des échantillons aléatoires de 1000 individus de chaque population pour simplifier les analyses. Les échantillons étaient tous normalement distribués et les variances des échantillons étaient homogènes. Ces tailles d'échantillonnages sont suffisantes pour ignorer les potentielles erreurs de type I et II. 
    
    Les salaires net moyen par heure des femmes et des hommes ont été comparés. Les hommes ont en moyenne en France dépendamment et indépendamment des régions un salaire plus élevé y compris pour la Martinique et la Guyane qui semblaient moins concernées. Cependant, relativement aux autres régions, le salaire des femmes dans ces régions semble plus intéressant que celui des hommes en observant le classement des régions pour les deux catégories. Ces différences significatives s'observent aussi entre les cadres féminins et masculins ( T= -7.69, p = 8.31e-27) ainsi que pour les travailleurs et travailleuses (T = -11.8, p = 1.23e-24). Enfin des différences significatives sont aussi observées entre les tranches d'âges et entre les sexes pour ces différentes tranches d'âge. Globalement, pour chaque tranche d'âge, les hommes gagnent mieux leur vie que les femmes. Aussi, les salaires moyens sont significativement supérieurs à chaque catégorie d'âge pour les femmes et pour les hommes. Avec pour résultat de chacun des tests des p-value très nettement inférieures au seuil fixé de 0.05.

    '''
