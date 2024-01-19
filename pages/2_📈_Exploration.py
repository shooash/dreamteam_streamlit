import streamlit as st
import exploration as ex
import pandas as pd

st.set_page_config(page_title="Exploration", page_icon="📈")
last_key = 0
def show_table_stats(path):
    data = pd.read_csv(path)
    st.write("Premières lignes du dataset qui a les dimensions {}:".format(data.shape))
    st.dataframe(data.head(5))
    st.dataframe(data.describe())
    taux_na = pd.DataFrame({'NA': data.isna().sum(), 'Non NA' : data.count(), 'Type' : data.dtypes})
    taux_na

'# _French Industry_: preuves d\'inégalités en France'
'## Exploration'
'''
Dans le cadre de ce projet, nous disposons du jeu de données «French Industry», composé de quatre tableaux qui dépeignent la réalité de l'année 2014:
'''
invitation = "> > Afficher les informations détaillées"
"### base_etablissement_par_tranche_effectif"
"Liste des communes avec les nombres d'entreprises de différents types classés par effectifs."
if st.checkbox(invitation,  key = (last_key:= last_key+1)):
    '''
    > Ce tableau détaille la structure des établissements au 1er janvier 2014 dans le domaine des activités marchandes hors agriculture, classée par tranche d'effectifs salariés. Comprenant 36 681 lignes et 14 colonnes, il fusionne les données géographiques avec les index officiels du répertoire d'entreprises, offrant ainsi une vision complète du nombre total d'entreprises et de leur répartition par tranches d'effectifs.
    '''
    show_table_stats(ex.ENTREPRISES)
"### net_salary_per_town_categories"
'''
Données sur les salaires par heures pour différentes catégories sociales dans les communes.
'''
if st.checkbox(invitation,  key = (last_key:= last_key+1)):
    '''
    > Présentant le salaire net horaire moyen par commune, ce tableau divise les données selon la catégorie socioprofessionnelle, le sexe et l'âge. Composé de 5 136 lignes, il permet d'explorer les inégalités salariales à travers une segmentation fine des informations, bien que la date de collecte précise ne soit pas spécifiée.
    '''
    show_table_stats(ex.SALAIRE)
"### population"
"Informations sur la population des communes, avec la distribution par différents groupes familiaux."
if st.checkbox(invitation,  key = (last_key:= last_key+1)):
    '''
    > Le tableau le plus volumineux, avec 8 536 584 lignes, offre une perspective sociale en présentant le nombre d'habitants enregistrés dans les communes françaises. Classées par catégorie d'âge, sexe et mode de cohabitation, ces données enrichissent notre analyse économique en ajoutant une dimension démographique.
    '''
    show_table_stats(ex.POPULATION_ORIG)
"### name_geographic_information"
"Données sur les salaires par heures pour différentes catégories sociales dans les communes."
if st.checkbox(invitation,  key = (last_key:= last_key+1)):
    '''
    > Compilées en 2014, ces données offrent une base de 36 840 lignes, fournissant des informations cruciales sur la localisation des communes françaises. Toutefois, des lacunes et erreurs ont nécessité l'intégration du jeu de données « Villes de France » de Mickaël Andrieu [publié sur data.gouv.fr](https://www.data.gouv.fr/fr/datasets/villes-de-france/#/resources) pour corriger et compléter les informations initiales.
    '''
    show_table_stats(ex.GEO)
'''
En outre, un dataset supplémentaire, [**cities**](https://www.data.gouv.fr/en/datasets/villes-de-france/), a été inclus, offrant des coordonnées plus précises des communes en France. Ces bases constituent le socle sur lequel repose notre exploration et analyse.
'''
if st.checkbox("Afficher les informations détaillées sur **cities**",  key = (last_key:= last_key+1)):
    show_table_stats(ex.VILLESDEFRANCE)
