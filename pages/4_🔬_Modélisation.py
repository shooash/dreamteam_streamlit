from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st
import dreamteam as dt
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from zipfile import ZipFile, ZIP_DEFLATED

st.set_page_config(page_title="Modélisation", page_icon="🔬")
last_key=0
st.sidebar.write('## **Modélisation**')
pages = ['Regression', 'Classification']
page = st.sidebar.radio('Afficher les modèles de:', pages)

data_selection = ['Tous', '4 valeurs salariales', '2 valeurs salariales', 'Top 6 Bayesian Ridge']
top_6_br = ['salaire', 'salaire_hommes', 'salaire_26_50', 'salaire_femmes', 'salaire_50+', 'salaire_travailleur', 'salaire_travailleur_hommes']

@st.cache_data
def load_data_regression(selection = 0):
    select = [
        dt.selection_max,
        ['salaire', 'salaire_employe_hommes', 'salaire_employe_femmes', 'salaire_femmes', 'salaire_travailleur_femmes'],
        ['salaire', 'salaire_employe_hommes', 'salaire_employe'],
        top_6_br
    ]
    if selection in [0, 3]:
        sexe = False
        age = False
        gender = False
        population_drop=True
    else:
        sexe = True
        age = False
        gender = False
        population_drop=False,

    x = dt.set_x(dt.df,
             selection=select[selection],
             sexe=sexe,
             gender_stats=gender,
             age_stats=age,    
             population_drop=population_drop,
             sans_extremites=False, 
             sans_paris=False,
             norm='', #aucune normalisation horizontale ni scale
             )
    return x
    
def regress(data, regressor=BayesianRidge()):
    # model_name = 'models/model_' + regressor.__class__.__name__
    # test_name = 'models/test_' + regressor.__class__.__name__
    # pred_name = 'models/pred_' + regressor.__class__.__name__
    # try:
    #     model = pickle.load(open(model_name, 'rb'))
    #     y_test = pickle.load(open(test_name, 'rb'))
    #     y_pred = pickle.load(open(pred_name, 'rb'))
    # except:
    y_test, y_pred, model = dt.regress(data.drop('salaire', axis=1), data.salaire, regressor=regressor, show=False)
    # pickle.dump(model, open(model_name, 'wb'))
    # pickle.dump(y_test, open(test_name, 'wb'))
    # pickle.dump(y_pred, open(pred_name, 'wb'))
    return y_test, y_pred, model

def get_scores(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    error_rate = np.sqrt(mse)
    return mse, r2, error_rate

if page == pages[0]:
    '# _French Industry_: preuves d\'inégalité en France'
    '## Modélisation : Regression'
    '''
    **DataSet à utiliser:**
    '''
    data_formater = lambda d: data_selection.index(d)
    data_index = data_selection.index(st.selectbox(' ', data_selection, label_visibility='collapsed'))
    '''
    **Faire la regression avec:**
    '''
    models = [LinearRegression(), Ridge(), BayesianRidge()]
    formater = lambda m: m.__class__.__name__
    regressor = st.selectbox(' ', models, label_visibility='collapsed', format_func=formater)
    data = load_data_regression(data_index)
    y_test, y_pred, model = regress(data, regressor=regressor)
    mse, r2, error_rate = get_scores(y_test, y_pred)
    st.write(f'Mean squared error: {mse}')
    st.write(f'Score r2: {r2}')
    st.write(f'Error rate: {error_rate}')
    st.write(dt.show_regression_results(y_test, y_pred, show=False))
    st.write(dt.show_regression_features(data.drop('salaire', axis=1), model, show=False))
    '''
    Ces différents tests confirment de nombreuses observations faites lors de la phase d'exploration de nos jeux de données. Les salaires de la catégorie employé portent la majorité de l'explication des modèles de part la proportion d'employés en France nettement supérieure à la proportion de cadres ainsi que de leurs salaires majoritairement proches du smic horaire.
    
    Les salaires importants (ou outliers) sont quant à eux majoritairement expliqués par les salaires de cadres d'où la nécessité d'inclure une variable cadre pour réduire l'erreur sur ces valeurs. L'ajout de la variable salaire cadre homme donne des résultats plus concluant par rapport à l'ajout de la variable salaire cadre femme mettant en avant les inégalités salariales homme et femme cadres.
    
    La limite principale de ces méthodes de régression est l'overfitting dans notre cas. Nous ne parvenons pas à obtenir de score concluants en retirant les données salariales de nos variables explicatives. A l'inverse, en incluant les variables salariales, nous excluons nos autres variables qui n'ont alors plus aucune importance et nous arrivons rapidement sur un sur-apprentissage des modèles.
    '''

#########################################################
#####               CLASSIFICATION                  #####
#########################################################
data_selection_class = ['Tous', 'Sans valeurs salariales']
SVC_params = {'kernel': 'rbf', 'gamma': 0.01, 'C': 10}
RandomForest_params = {'n_estimators': 1000, 'min_samples_leaf': 5, 'max_features': 'log2', 'max_depth': 15, 'criterion': 'gini'}
logreg_params = {}

#@st.cache_data
def load_data_classification(selection = 0):
    select = [
        dt.selection_max,
        dt.selection_sans_salaire
    ]
    x = dt.default_x(select[selection])
    return x
def make_cat(y, n_cat=2, fun_cat=None):
    if fun_cat is None:
        return pd.qcut(y, q=n_cat, labels=range(n_cat))
    return y.apply(fun_cat)

def get_forest(n_cat, x_train, y_train, x_test, y_test, fun_cat=None):

    title = 'Random Forest Classifier pour %d catégories' % n_cat
    #st.write('### ' + title)
    model_name = 'models/model_forest_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    test_name = 'models/test_forest_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    pred_name = 'models/pred_forest_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    try:
        # Comme les modèles Forest sont enorme on va les archiver
        z = ZipFile(model_name, 'r')
        model = pickle.load(z.open(model_name, 'r'))
        z.close()
        #model = pickle.load(open(model_name, 'rb'))
        y_test_cat = pickle.load(open(test_name, 'rb'))
        y_pred = pickle.load(open(pred_name, 'rb'))
    except:
        y_test_cat = make_cat(y_test, n_cat, fun_cat)
        y_train_cat = make_cat(y_train, n_cat, fun_cat)
        y_pred, model = dt.grow_random_forest(x_train, x_test, y_train_cat, y_test_cat, RandomForest_params)
        # Comme les modèles Forest sont enorme on va les archiver
        z = ZipFile(model_name, 'w', compression=ZIP_DEFLATED, compresslevel=5)
        pickle.dump(model, z.open(model_name, 'w'))
        z.close()
        #pickle.dump(model, open(model_name, 'wb'))
        pickle.dump(y_test_cat, open(test_name, 'wb'))
        pickle.dump(y_pred, open(pred_name, 'wb'))
    dat = dt.get_scores(y_test_cat, y_pred, "RandomForest %d catégories" % n_cat)
    st.write(dt.show_class_matrix(y_test_cat, y_pred, title, show=False))
    st.dataframe(pd.DataFrame(classification_report(y_test_cat, y_pred, output_dict=True)).transpose(), width=600)
    features = dt.get_top_features_forest(x_test, model)
    roc = dt.get_auc(x_test, y_test_cat, model)
    return dat, features, roc

def get_svc(n_cat, x_train, y_train, x_test, y_test, fun_cat=None):
    title = 'SVC Classifier pour %d catégories' % n_cat
    #st.write('### ' + title)
    model_name = 'models/model_svc_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    test_name = 'models/test_svc_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    pred_name = 'models/pred_svc_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    scaler_name = 'models/scale_svc_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    x_name = 'models/x_svc_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    try:
        model = pickle.load(open(model_name, 'rb'))
        y_test_cat = pickle.load(open(test_name, 'rb'))
        y_pred = pickle.load(open(pred_name, 'rb'))
        scaler = pickle.load(open(scaler_name, 'rb'))
        x_test = pickle.load(open(x_name, 'rb'))
    except:
        y_test_cat = make_cat(y_test, n_cat, fun_cat)
        y_train_cat = make_cat(y_train, n_cat, fun_cat)
        y_pred, model, scaler = dt.make_svc(x_train, x_test, y_train_cat, y_test_cat, SVC_params)
        pickle.dump(model, open(model_name, 'wb'))
        pickle.dump(y_test_cat, open(test_name, 'wb'))
        pickle.dump(y_pred, open(pred_name, 'wb'))
        pickle.dump(scaler, open(scaler_name, 'wb'))
        pickle.dump(x_test, open(x_name, 'wb'))
    dat = dt.get_scores(y_test_cat, y_pred, "SVC %d catégories" % n_cat)
    st.write(dt.show_class_matrix(y_test_cat, y_pred, title, show=False))
    st.dataframe(pd.DataFrame(classification_report(y_test_cat, y_pred, output_dict=True)).transpose(), width=600)
    features_name = 'models/features_svc_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    try:
        features = pickle.load(open(features_name, 'rb'))
    except:
        features = dt.get_top_features_svc(x_test, y_test_cat, model, scaler)
        pickle.dump(features, open(features_name, 'wb'))
    #pas de roc pour svc si pas kernel linear
    #roc = dt.get_auc(x_test_scaled, y_test_cat, model)
    return dat, features

def get_logreg(n_cat, x_train, y_train, x_test, y_test, fun_cat=None):
    title = 'LogisticRegression Classifier pour %d catégories' % n_cat
    #st.write('### ' + title)
    model_name = 'models/model_logreg_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    test_name = 'models/test_logreg_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    pred_name = 'models/pred_logreg_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    scaler_name = 'models/scale_logreg_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    x_name = 'models/x_logreg_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    try:
        model = pickle.load(open(model_name, 'rb'))
        y_test_cat = pickle.load(open(test_name, 'rb'))
        y_pred = pickle.load(open(pred_name, 'rb'))
        scaler = pickle.load(open(scaler_name, 'rb'))
        x_test = pickle.load(open(x_name, 'rb'))
    except:
        y_test_cat = make_cat(y_test, n_cat, fun_cat)
        y_train_cat = make_cat(y_train, n_cat, fun_cat)
        y_pred, model, scaler = dt.make_logreg(x_train, x_test, y_train_cat, y_test_cat, logreg_params)
        pickle.dump(model, open(model_name, 'wb'))
        pickle.dump(y_test_cat, open(test_name, 'wb'))
        pickle.dump(y_pred, open(pred_name, 'wb'))
        pickle.dump(scaler, open(scaler_name, 'wb'))
        pickle.dump(x_test, open(x_name, 'wb'))
    # x_test_scaled = x_test.copy()
    # x_test_scaled[:] = scaler.transform(x_test)
    dat = dt.get_scores(y_test_cat, y_pred, "SVC %d catégories" % n_cat)
    st.write(dt.show_class_matrix(y_test_cat, y_pred, title, show=False))
    st.dataframe(pd.DataFrame(classification_report(y_test_cat, y_pred, output_dict=True)).transpose(), width=600)
    features_name = 'models/features_logreg_{}_{}'.format(str(n_cat) if fun_cat is None else 'fun', data_index)
    try:
        features = pickle.load(open(features_name, 'rb'))
    except:
        features = dt.get_top_features_svc(x_test, y_test_cat, model, scaler)
        pickle.dump(features, open(features_name, 'wb'))
    #pas de roc pour svc si pas kernel linear
    #roc = dt.get_auc(x_test_scaled, y_test_cat, model)
    return dat, features

def cat_man(y):
    seuls = [1622/151.67, #Minimum pour vivre en 2014 selon le Baromètre de DREES https://drees.solidarites-sante.gouv.fr/sites/default/files/2021-01/principaux_enseignements_barometre_2015.pdf P.22
             data.salaire.median(),
             37250/52/35 # Dernier décile de richèsse aisé par Insee en 2014 
             ]
    if y < seuls[0]: return 0
    if y < seuls[1]: return 1
    if y < seuls[2]: return 2
    return 3
def classify(data, model, num_cat, fun_cat):
    x_train, x_test, y_train, y_test = train_test_split(data.drop('salaire', axis=1), data.salaire, random_state=7)
    if model.__class__ is RandomForestClassifier:
        dat, features, roc = get_forest(num_cat, x_train, y_train, x_test, y_test, fun_cat)
        st.write(px.bar(features, orientation='h', 
            text_auto='.4f', labels={'index' : 'Valeurs', 'value' : 'Importance', 'variable' : 'Selection'},
            title='Variables les plus importantes pour le modèle RandomForestClassifier'))
        # Courbe ROC
        fig = go.Figure()
        fig.update_layout(title='ROC curve RandomForestClassifier', width=800)
        fig.add_scatter(x=roc[0], y=roc[1], name='Modèle (auc = {:0.2f})'.format(roc[2]), mode='lines')
        fig.add_scatter(x=[0,1], y=[0,1], name='Aléatoire (auc = 0.5)',  mode='lines', line_dash='dash')
        st.write(fig)
    if model.__class__ is SVC:
        dat, features = get_svc(num_cat, x_train, y_train, x_test, y_test, fun_cat)
        st.write(px.bar(features, orientation='h', 
            text_auto='.4f', labels={'index' : 'Valeurs', 'value' : 'Importance', 'variable' : 'Selection'},
            title='Variables les plus importantes pour le modèle SVC'))
    if model.__class__ is LogisticRegression:
        dat, features = get_logreg(num_cat, x_train, y_train, x_test, y_test, fun_cat)
        st.write(px.bar(features, orientation='h', 
            text_auto='.4f', labels={'index' : 'Valeurs', 'value' : 'Importance', 'variable' : 'Selection'},
            title='Variables les plus importantes pour le modèle LogisticRegression'))

if page == pages[1]:
    '# _French Industry_: preuves d\'inégalité en France'
    '## Modélisation : Classification'
    '''
    Le type de données du dataset et la nature de la variable cible (salaire) permettent l'utilisation de méthodes de classification par apprentissage non supervisé. Cette approche nécessite de reformater la variable cible en catégories. En outre, un prétraitement adapté est nécessaire pour améliorer la qualité des prédictions.
    
    En conséquence, nous avons obtenu un tableau de 85 colonnes, y compris la variable cible. La variable sexe a été ajoutée ainsi que la moyenne des salaires par genre qui servent comme variable cible 	(snhmf14 et snhmh14 du dataset “net_salary_per_town_categories”). Des colonnes contenant les proportions d'hommes, de femmes et d'enfants ont également été ajoutées, ainsi que le nombre d'habitants par tranche d'âge pour chaque commune étudiée.
    
    Bien que dans la réalité, il soit peu probable d'avoir ce type de données pour prédire le salaire des hommes et des femmes, la modélisation s'avère utile dans le cadre de notre étude pour démontrer l'importance des valeurs présentes pour la variable cible.
    '''
    '''
    **DataSet à utiliser:**
    '''
    data_index = data_selection_class.index(st.selectbox(' ', data_selection_class, label_visibility='collapsed'))
    '''
    **Faire la classificaton pour:**
    '''
    def cat_label(x):
        if x == 1:
            return "Groupes par niveau de vie"
        return f"{x} quantiles"
    num_cat = st.selectbox(' ', range(1,5), label_visibility='collapsed', format_func=cat_label)
    '''
    **Modèle à utiliser:**
    '''
    models = [LogisticRegression(), SVC(), RandomForestClassifier()]
    formater = lambda m: m.__class__.__name__
    classifier = st.selectbox(' ', models, label_visibility='collapsed', format_func=formater)
    # Action!
    if num_cat == 1:
        num_cat = 4
        fun_cat = cat_man
    else:
        fun_cat = None
    data = load_data_classification(data_index)
    classify(data, classifier, num_cat, fun_cat)
    '''
    Les taux de salaires de différents groupes sociaux jouent un rôle clé pour la prédiction des salaires des hommes et des femmes. Dans les modèles sans ces valeurs la qualité de prédiction diminue drastiquement avec l'augmentation du nombre de classes. Mais les deux catégories sont toujours bien prédites.
    
    Il est étonnant que les modèles utilise très peu la catégorie "sexe" pour prédire le niveau de salaires moyen pour les hommes et les femmes. Les facteurs les plus souvent mentionnés parmi les plus importants sont les suivants :
    - Le taux de grandes et moyennes entreprises.
    - Le taux d'enfants de plus de 50 ans vivants avec un/deux parents. 
    - Le taux d'hommes de 15-24 ans vivant seuls avec des enfants.
    - Le taux de couples de plus de 50 ans.
    
    La manque de valeur “sexe” parmi les components les plus important pourrait s'expliquer par le fait que plus souvent les salaire de deux genres dans une même ville tombe dans une seule catégorie. Or, on sait très bien que les catégories de niveau de vie faible (0) et de la richesse (3) sont représentées quasi exclusivement par les revenus des femmes et des hommes. En plus si on supprime cette séparation et si on revient vers le salaire moyen par commune la qualité de prédiction devient trop faible.
    '''