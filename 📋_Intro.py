import numpy as np
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import exploration as ex
from streamlit.components.v1 import html

st.set_page_config(page_title='"French Industry": preuves d\'inégalité en France', 
                   page_icon="📋",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'About': 
                           '''
                    Projet "French Industry" accompli dans le cadre de la formation Data Analyst chez [DataScientest](https://datascientest.com/).
                    
                    *Par [Romain Biancato](https://www.linkedin.com/in/romain-biancato-data-analyst/), [Vincent Louison](https://www.linkedin.com/in/vincent-louison/), [Andrey Poznyakov](https://www.linkedin.com/in/andrey-poznyakov/) et [Guillaume Zighmi](https://www.linkedin.com/in/guillaume-zighmi-05aa3b28/)*
                    '''
                    },
                   )

# Add some content to the main area of the app

st.sidebar.header('About')
st.sidebar.write('''
                    Projet "French Industry" accompli dans le cadre de la formation Data Analyst chez [DataScientest](https://datascientest.com/).
                    
                    *Par [Romain Biancato](https://www.linkedin.com/in/romain-biancato-data-analyst/), [Vincent Louison](https://www.linkedin.com/in/vincent-louison/), [Andrey Poznyakov](https://www.linkedin.com/in/andrey-poznyakov/) et [Guillaume Zighmi](https://www.linkedin.com/in/guillaume-zighmi-05aa3b28/)*
                    ''')

'# _French Industry_: preuves d\'inégalité en France'
'## Contexte'
'''
En 2013, les secteurs marchands en France ont contribué 986 milliards d'euros, provenant de 3,3 millions d'entreprises, [selon l'Insee](https://www.insee.fr/fr/statistiques/1908497). Cela représentait plus de la moitié de la valeur ajoutée totale de l'économie.  L'analyse des données sur l'industrie permettra de dévoiler la structure et les particularités de cette composante essentielle de la vie économique, fournissant un aperçu approfondi de la période antérieure à la pandémie et soulignant les inégalités sociales et salariales des différentes catégories socio-professionnelles.
'''
'### Objectifs'
'''
Les données fournies dans le cadre de ce projet offrent un large éventail de possibilités d'études, notamment :
- *Analyse des tendances industrielles en France* : Exploration des dynamiques globales de l'industrie, avec un accent sur la distribution géographique des différentes catégories d'entreprises. Cette analyse permettra de confirmer l'existence de clusters industriels et d'examiner les inégalités économiques régionales.
- *État des lieux de la rémunération en France* : Investigation approfondie sur les disparités salariales entre hommes et femmes. Nous chercherons également à prédire les salaires en fonction de variables telles que l'âge, le sexe et la localisation à l’aide de modèles de machine learning.
- *Préconisations salariales par Machine Learning* : A l'aide d'un modèle de machine learning nous devrions pouvoir prédire une estimation de salaire en fonction de votre genre, de votre âge, de votre lieu de résidence ainsi que de votre échelon professionnel.

Ainsi, notre étude se déploie autour de trois pôles majeurs : l'industrie nationale, les salaires, et l'inégalité sociale.
'''


#st.sidebar.subheader('Subheader')
#st.sidebar.title('Sommaire')


