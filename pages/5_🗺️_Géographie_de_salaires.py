import streamlit as st

st.set_page_config(page_title="Géographie de salaires", page_icon="🗺️")

'# _French Industry_: preuves d\'inégalités en France'
'## Distribution géographique des catégories salariales prédites'


'''
La classification choisie permet de distinguer les villes et les départements les plus pauvres et les plus riches. La cartographie montre une concentration de communes à faible niveau de vie moyen près des frontières, notamment dans le nord de la France. On observe également deux groupes de communes dans cette catégorie, en Pays de la Loire et en Occitanie.
'''

import dreamteam as dt
df = dt.add_sexe_col(dt.df)
df['codgeo'] = df.codgeo.apply(lambda a: '0' + str(a) if not isinstance(a, str) and a < 10000 else str(a) )
def cat_man(y):
    seuls = [1622/151.67, #Minimum pour vivre en 2014 selon le Baromètre de DREES https://drees.solidarites-sante.gouv.fr/sites/default/files/2021-01/principaux_enseignements_barometre_2015.pdf P.22
             df.salaire.median(),
             37250/52/35 # Dernier décile de richèsse aisé par Insee en 2014 
             ]
    if y < seuls[0]: return 0
    if y < seuls[1]: return 1
    if y < seuls[2]: return 2
    return 3
y_cat = df.salaire.apply(cat_man)
x = df
x['cat'] = y_cat
x_dep = dt.df.copy()[['codgeo_dep', 'departement', 'salaire']]
x_dep = x_dep.groupby(['codgeo_dep', 'departement']).mean().reset_index()
y_dep_cat = x_dep.salaire.apply(cat_man)
x_dep['cat'] = y_dep_cat

@st.cache_data
def get_map():
    result = dt.show_map_class(x, x_dep, show=False)
    return result

st.write(get_map())

'''
La distribution des communes appartenant à la classe de richesse la plus élevée est similaire à la carte des clusters industriels, qui représente la distribution géographique des grandes entreprises. La région parisienne, Toulouse, la vallée du Rhône et Lille sont caractérisées par des concentrations de communes aux salaires moyens très élevés. Paris, Hauts-de-Seine et Yvelines ont des salaires moyens par département plus élevés que le seuil de richesse.
'''