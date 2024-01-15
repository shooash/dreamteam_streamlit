import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('output/dataset.csv', low_memory=False)
# Pour une version sans separation par genre
#df = pd.read_csv('output/dataset_nosex.csv', low_memory=False)

df = df.dropna()


### Variables globales
# Listes de colonnes
col_geo = ['codgeo', 'codgeo_reg']
col_geo_names = ['region', 'departement', 'commune']
col_salaire = [c for c in df.columns if c.startswith('salaire_')]
col_salaire_limited = [c for c in col_salaire if 'cadre' not in c]
col_hommes = [ i for i in df.columns if ('homme' in i) and ('salaire' not in i) and not i.startswith('enfant')]
col_femmes = [ i for i in df.columns if ('femme' in i) and ('salaire' not in i) and not i.startswith('enfant')]
col_enfants = [i for i in df.columns if i.startswith('enfant')]
col_sex = [ i for i in df.columns if i.endswith('sex')]
col_index_age = ['0_14', '15_24', '25_49', '50_00']
col_population = col_hommes + col_femmes + col_enfants
col_entreprises = ['pme', 'mic', 'eti_ge', 'ent_brut', 'ent_zero']
col_totals = ['salaire', 'ent_net', 'population']
# Total des entreprises
ENTREPRISES = 'ent_net'
## Classification avec max d'elements
selection_max = col_totals + col_salaire + col_population + col_entreprises
## Classification sans groupes salairiales
selection_salaire_limited = col_totals + col_salaire_limited + col_population + col_entreprises

## Classification sans groupes salairiales
selection_sans_salaire = col_totals + col_population + col_entreprises

#Fonctions de preprocessing catégoriel

def norm_names(l : [str]):
    return [c + '*' for c in l]

def denorm_names(l : [str]):
    return [c.rstrip('*') for c in l]

def norm_horiz(data: pd.DataFrame, selection: [str], ref_col: str):
    '''
    Normalizer la série selon le chiffre de référence (eg % de salaire homme en salaire moyen)
    '''
    data = data.copy()
    renamer = dict()
    cols = data.columns.to_list()
    selection = [s for s in selection if s in cols]
    for c in selection:
        data[c] = (data[c] / data[ref_col]).replace(np.inf, 0).fillna(0)
        renamer[c] = c + '*'
    data.rename(renamer, axis=1, inplace=True)
    return data

#Fonctions de chargement de X pour l'analyse catégoriel
def add_gender_stats(data, drop=True):
    '''
    Il serait mieux d'avoir la statistique sur les hommes / femmes / enfants sans séparation par groupes sociales
    '''
    data.loc[:,'hommes'] = data[col_hommes].sum(axis=1)
    data.loc[:,'femmes'] = data[col_femmes].sum(axis=1)
    data.loc[:,'enfants'] = data[col_enfants].sum(axis=1)
    if drop:
        data = data.drop(col_hommes + col_femmes + col_enfants, axis=1)
    return data
def add_ages_stats(data, drop=True):
    for i in col_index_age:
        cols = [c for c in data.columns if (i in c)]
        data.loc[:, i] = data[cols].sum(axis=1)
        if drop:
            data = data.drop(cols, axis=1)
    return data
def add_sexe_col(data, sexe=True):
    '''
    # On remplace le salaire moyen par salaire moyen pour des hommes et des femmes: c'est l'inégalité qui nous interrèsse.
    sexe = True, 'm' ou 'f' (tous, seulement les hommes ou le femmes)
    '''
    if sexe != 'm':
        x = data.copy()
        x = x.drop('salaire', axis=1, errors='ignore')
        x.insert(0, 'salaire',  data['salaire_femmes'])
        x.insert(x.shape[1], 'sexe', 0)
        x = x.drop('salaire_hommes', axis=1, errors='ignore')
        x = x.drop('salaire_femmes', axis=1, errors='ignore')
    else:
        x = pd.DataFrame()
    if sexe != 'f':
        x_2 = data.copy()
        x_2 = x_2.drop('salaire', axis=1, errors='ignore')
        x_2.insert(0, 'salaire',  data['salaire_hommes'])
        x_2.insert(x_2.shape[1], 'sexe', 1)
        x_2 = x_2.drop('salaire_femmes', axis=1, errors='ignore')
        x_2 = x_2.drop('salaire_hommes', axis=1, errors='ignore')
    else:
        x_2 = pd.DataFrame()
    return pd.concat([x, x_2]).reset_index(drop=True)

def get_density():
    DENSITE = 'input/grille_densite_7_niveaux_2015.csv'
    df_densite = pd.read_csv(DENSITE, sep=';')
    densite = df_densite[['CODGEO', 'DENS']]
    densite = densite.rename({'DENS' : 'density', 'CODGEO' : 'codgeo'}, axis=1)
    densite.codgeo = densite.codgeo.apply(lambda i: i.replace('B', '0').replace('A', '0') if not i.isnumeric() else i).astype(int)

    return densite.drop_duplicates().dropna()

def add_density(data : pd.DataFrame):
    return pd.merge(left=data, right=get_density(), how='left', left_on='codgeo', right_on='codgeo').fillna(0)


def set_x(x, #DataFrame 
          norm='', # Pour norm_horiz: population, gender, entreprises, age; pour StandardScaler: all; format: "population+gender+all"
          sexe=True, # ajoute de colonne sexe, salaire = salaire_homme ou salaire_femme
          selection = False, # limiter la selection par liste de colonnes: False ou [str] 
          gender_stats = True, # Ajouter les colonne hommes, femmes, enfants
          age_stats = False, #Ajouter les colonnes de groupes par age?
          population_drop = False, # Faut-il dropper les colonnes d'origine de groupes de population après les normalisations?
          scale_normalized = False, # Peut-on appliquer StandardScaler au colonnes subis la norm_horiz?
          sans_extremites = True, #Supprimer les valeurs extreme?
          sans_paris = True, #supprimer les villes plus gros que 2mln?
          density = False, #grille densité 2015 https://www.insee.fr/fr/information/2114627
          ):
    '''
    Chargement de dataframe.
    x: DataFrame pour avoir les données 
    norm: colonnes à normaliser horizontalement et/ou all pour StandardScaler:
    norm='population'|'salaire'|'population+salaire' etc, default: ''
    selection: liste de colonnes a utiliser ou False s'il faut utiliser tout
    sexe: faut-il remplacer le salaire moyen par salaire moyen par gendre et ajouter le col. sexe
    sexe=True|False default:False
    '''
    x = x.copy()
    if sans_paris:
        x = x[x.population<2000000]
    if sans_extremites:
        q1, q3 = x.salaire.quantile(q=[0.25,0.75])
        x = x[x.salaire <= q3 + (q3 - q1) * 1.5]
    drop_col = [] + col_geo
    non_scale_cols = ['salaire']
    if isinstance(selection, list):
        cols = x.columns.to_list()
        selection_drop = [s for s in cols if s not in selection]
        selection_drop += norm_names(selection_drop)
        drop_col += selection_drop
    if gender_stats:
        x = add_gender_stats(x, False)
    if age_stats:
        x = add_ages_stats(x, False)
    if sexe:
        x = add_sexe_col(x, sexe=sexe)
        non_scale_cols += ['sexe'] #c'est binaire
    if 'entreprises' in norm:
        x = norm_horiz(x, col_entreprises, ENTREPRISES)
        if not scale_normalized:
            non_scale_cols += norm_names(col_entreprises)
    if 'population' in norm:
        x = norm_horiz(x, col_population, 'population')
        if not scale_normalized:
            non_scale_cols += norm_names(col_population)
    if 'gender' in norm:
        x = norm_horiz(x, ['hommes', 'femmes', 'enfants'], 'population')
        if not scale_normalized:
            non_scale_cols += norm_names(col_population)
    if 'age' in norm:
        x = norm_horiz(x, col_index_age, 'population')
        if not scale_normalized:
            non_scale_cols += norm_names(col_index_age)
    if 'salaire' in norm:
        x = norm_horiz(x, col_salaire, 'salaire')
        x = x.drop(col_sex, axis=1, errors='ignore') #normalization des pourcentage n'a aucun sense
    if population_drop:
        drop_col += col_hommes + col_femmes + col_enfants
    if density:
        x = add_density(x)
        non_scale_cols += ['density']

    x = x.drop(drop_col, axis=1, errors='ignore')
    
    x = x.select_dtypes(exclude='O')
    if 'all' in norm:
        num_cols = x.drop(non_scale_cols, axis=1, errors='ignore').columns.to_list() #les colonnes qui necessite la normalization
        scaler = StandardScaler()
        x.loc[:, num_cols] = scaler.fit_transform(x[num_cols])
        #x[:] = scaler.fit_transform(x)
    return x

def select_classifier_forest(x, y):
    '''
    Presentation de top10 classifier pour les x et y données
    '''
    # params = dict(
    #     n_estimators = [1000],
    #     criterion = ['gini', 'entropy'], #on a rejeté les autre avec les scores moins pértinents pour accélérer la fonction
    #     #max_features = ['sqrt','log2'], #presque pas d'impacte
    #     min_samples_leaf = [1, 10, 100],
    #     max_depth = [10, 15, None]        
    # )
    params = {'bootstrap': [True, False],
        'max_depth': [10, 20, 40],
        'min_samples_leaf': [1, 5, 8],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [200, 600, 1000, 2000]}
    grid = RandomizedSearchCV(RandomForestClassifier(), params, scoring='accuracy')
    rfc_result = grid.fit(x, y)
    top10 = pd.DataFrame.from_dict(rfc_result.cv_results_)[['params','mean_test_score']].sort_values('mean_test_score', ascending=False).head(10)
    display(rfc_result.best_params_)
    display(top10)
    return rfc_result

def select_classifier_svc(x, y):
    '''
    Presentation de top10 classifier pour les x et y données
    '''
    params = {'C': [0.1,1, 10, 100], 
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']}

    grid = RandomizedSearchCV(SVC(), params, scoring='accuracy')
    rfc_result = grid.fit(x, y)
    top10 = pd.DataFrame.from_dict(rfc_result.cv_results_)[['params','mean_test_score']].sort_values('mean_test_score', ascending=False).head(10)
    display(rfc_result.best_params_)
    display(top10)
    return rfc_result

def confirm_classifier_svc(x, y, params):
    '''
    Presentation de top10 classifier pour les x et y données
    '''
    grid = GridSearchCV(SVC(), params, scoring='accuracy')
    rfc_result = grid.fit(x, y)
    top10 = pd.DataFrame.from_dict(rfc_result.cv_results_)[['params','mean_test_score']].sort_values('mean_test_score', ascending=False).head(10)
    display(rfc_result.best_params_)
    display(top10)
    return rfc_result


class DTDouble:
    '''
    Un expériment pour voir si découper l'échantillon pas les quantiles et puis modéliser la regression sur chaque partie améliore le résultat. 
    '''
    def __init__(self, regressor=BayesianRidge(
                     alpha_init=3), 
                 classifier=RandomForestClassifier(
                     n_estimators = 1000,
                     min_samples_leaf = 5,
                     max_features = 'log2',
                     max_depth = 15,
                     criterion = 'gini')
                 ):
        self.regressor = regressor
        self.classifier = classifier
        #On fait la coupe par niveau de vie median
        niveau_vie_mediane_insaa_2014 = 20150 #donnée INSAA
        salaire_limite = niveau_vie_mediane_insaa_2014 / 52 / 35 #on compte le salaire horaire
        self.fun_categories = lambda x: 0 if x < salaire_limite else 1
    def fit(self, x, y, limit_x=True):
        self.x = x.copy()
        self.x.insert(0, 'salaire', y.copy())
        self.y = y.apply(self.fun_categories)
        self._rebalance_2_cat(method='maximize')
        if limit_x:
            self.limit_x()
        return self.classifier.fit(self.x.drop('salaire', axis=1), self.y)
    def predict(self, x):
        x_test = x.copy()
        y_pred_class = self.classifier.predict(x_test)
        y_pred = self.regress_cats(x_test, y_pred_class)
        return y_pred
    def limit_x(self):
        '''
        Enlever les valeurs extrèmes pour le jeu d'entrainement
        '''
        q1, q3 = self.x.salaire.quantile(q=[0.25,0.75])
        self.y = self.y[self.x.salaire <= q3 + (q3 - q1) * 1.5]
        self.y = self.y[self.x.salaire >= q1 - (q3 - q1) * 1.5]
        self.x = self.x[self.x.salaire <= q3 + (q3 - q1) * 1.5]
        self.x = self.x[self.x.salaire >= q1 - (q3 - q1) * 1.5]        
    def regress(self, x_train, x_test, regressor=BayesianRidge(alpha_init=3)):
        '''
        Fonction pour automatiser regress_cats()
        '''
        self.regressor.fit(x_train.drop('salaire', axis=1), x_train.salaire)
        target = x_test.iloc[:, 0].copy()
        target.iloc[:] = regressor.predict(x_test)
        return target
    def regress_cats(self, x_test, y_test):
        '''
        x_test = DataFrame
        y_test = Catégories prédites ou connues pour x_test        
        '''
        scaler = StandardScaler()
        uniques = self.y.unique()
        targets = []
        x_train_reg = self.x.copy()
        x_train_reg.iloc[:,1:] = scaler.fit_transform(x_train_reg.iloc[:,1:])
        x_test_reg = x_test.copy()
        x_test_reg.iloc[:] = scaler.transform(x_test_reg.iloc[:])
        # On va compter chaque catégorie séparémment
        for i in uniques:
            targets.append(self.regress(x_train=x_train_reg[self.y==i], x_test=x_test_reg[y_test==i]))
        return pd.concat(targets).reindex(x_test.index)
    @staticmethod
    def default_x():
        '''
        La meilleure solution selon les expériments
        '''
        return set_x(df.copy(), 
             norm='age+gender+population+enterprises+all',
             sexe=True, 
             selection=selection_sans_salaire,
             age_stats=True,
             gender_stats=True,
             scale_normalized=True,
             sans_extremites=False,
             sans_paris=False
             )
    def _rebalance_2_cat(self, method='minimize', n=5000):
        '''
        Si le disbalance est trop important on peut utiliser cela pour balance le jeu de donnée artificiellement.
        '''
        x = self.x
        y = self.y
        min_cat = y.value_counts().index[1]
        max_cat = y.value_counts().index[0]
        min_selection_x = x[y==min_cat]
        min_selection_y = y[y==min_cat].rename('salaire_y')
        max_selection_x = x[y==max_cat]
        max_selection_y = y[y==max_cat].rename('salaire_y')
        if method == 'minimize':
            joint = pd.concat([max_selection_y, max_selection_x], axis=1)
            joint = joint.sample(n=len(min_selection_x))
            max_selection_x = joint.drop('salaire_y', axis=1)
            max_selection_y = joint.salaire_y
        if method == 'maximize':
            joint = pd.concat([min_selection_y, min_selection_x], axis=1)
            joint = joint.sample(n=len(max_selection_x), replace=True)
            min_selection_x = joint.drop('salaire_y', axis=1)
            min_selection_y = joint.salaire_y            
        if method == 'increase':
            joint = pd.concat([min_selection_y, min_selection_x], axis=1)
            joint = joint.sample(n=n, replace=True)
            min_selection_x = joint.drop('salaire_y', axis=1)
            min_selection_y = joint.salaire_y            
            joint = pd.concat([max_selection_y, max_selection_x], axis=1)
            joint = joint.sample(n=n)
            max_selection_x = joint.drop('salaire_y', axis=1)
            max_selection_y = joint.salaire_y        #print([min_selection_y, max_selection_y, y])
        self.x = pd.concat([min_selection_x, max_selection_x]).reset_index(drop=True)
        self.y = pd.concat([min_selection_y, max_selection_y]).reset_index(drop=True)
    @staticmethod
    def rebalance_2_cat(x, y, method='minimize', n=5000):
        '''
        Si le disbalance est trop important on peut utiliser cela pour balance le jeu de donnée artificiellement. Methode statique.
        '''
        min_cat = y.value_counts().index[1]
        max_cat = y.value_counts().index[0]
        min_selection_x = x[y==min_cat]
        min_selection_y = y[y==min_cat].rename('salaire_y')
        max_selection_x = x[y==max_cat]
        max_selection_y = y[y==max_cat].rename('salaire_y')
        if method == 'minimize':
            joint = pd.concat([max_selection_y, max_selection_x], axis=1)
            joint = joint.sample(n=len(min_selection_x), random_state=7)
            max_selection_x = joint.drop('salaire_y', axis=1)
            max_selection_y = joint.salaire_y
        if method == 'maximize':
            joint = pd.concat([min_selection_y, min_selection_x], axis=1)
            joint = joint.sample(n=len(max_selection_x), replace=True, random_state=7)
            min_selection_x = joint.drop('salaire_y', axis=1)
            min_selection_y = joint.salaire_y            
        if method == 'increase': #faire un resample sur les deux catégories
            joint = pd.concat([min_selection_y, min_selection_x], axis=1)
            joint = joint.sample(n=n, replace=True, random_state=7)
            min_selection_x = joint.drop('salaire_y', axis=1)
            min_selection_y = joint.salaire_y            
            joint = pd.concat([max_selection_y, max_selection_x], axis=1)
            joint = joint.sample(n=n, replace=True, random_state=7)
            max_selection_x = joint.drop('salaire_y', axis=1)
            max_selection_y = joint.salaire_y        #print([min_selection_y, max_selection_y, y])
        #print([min_selection_y, max_selection_y, y])
        x = pd.concat([min_selection_x, max_selection_x]).reset_index(drop=True)
        y = pd.concat([min_selection_y, max_selection_y]).reset_index(drop=True)
        return x, y


def default_x(selection=selection_sans_salaire, density=False):
    '''
    La meilleure solution sans les colonnes salariales selon les expériments.
    Toutes les colonnes ajouté, normalize horizontalement, scaled.
    '''
    return set_x(df.copy(), 
            norm='age+salaire+gender+population+enterprises+all',
            sexe=True, 
            selection=selection,
            age_stats=True,
            gender_stats=True,
            scale_normalized=False,
            sans_extremites=False,
            sans_paris=False,
            density=density,
            )
def default_x_salaire(density=False):
    '''
    La meilleure solution avec les colonnes salariales.
    Toutes les colonnes ajouté, normalize horizontalement, scaled.
    '''
    return set_x(df.copy(), 
            norm='age+gender+salaire+population+enterprises+all',
            sexe=True, 
            selection=selection_max,
            age_stats=True,
            gender_stats=True,
            scale_normalized=False,
            sans_extremites=False,
            sans_paris=False,
            density=density
            )

## Fonctions de classification

def grow_random_forest(x_train, x_test, y_train, y_test, params):
    '''
    Créer le modèle RandomForestClassifier et la prédiction.
    '''
    print('Calcule RandomForest', params)
    model = RandomForestClassifier(**params, random_state=7)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print('Accuracy train et test:', accuracy_score(model.predict(x_train), y_train), accuracy_score(y_pred, y_test))
    return y_pred, model

def show_forest_graphs(x_test, y_test, y_pred, model, tag = ''):
    '''
    Rapport standart des graphiques sur le modèle RandomForestClassifier.
    '''
    display(pd.crosstab(y_test, y_pred))
    px.imshow(pd.crosstab(y_test, y_pred), zmin=0, text_auto=True, 
                labels={'x' : 'Prediction', 'y' : 'Test'},
                color_continuous_scale='OrRd').show()
    print(classification_report(y_test, y_pred))
    dat = pd.Series(model.feature_importances_, index=x_test.columns)
    dat = dat.sort_values(ascending=True).head(10)
    px.bar(y=dat.index, x=dat, labels={'y' : 'Features', 'x' : 'Importance'},
            width=800, height=1000, 
        title='Importance de top10 valeurs selon RandomForest ' + tag).show()
    show_auc(x_test, y_test, model)
    return dat

def make_svc(x_train, x_test, y_train, y_test, params):
    '''
    Créer le modèle SVC et la prédiction.
    '''
    #params['kernel'] = 'rbf'
    print('Calcule SVC', params)
    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_train_scaled[:] = scaler.fit_transform(x_train)
    x_test_scaled = x_test.copy()
    x_test_scaled[:] = scaler.transform(x_test)
    model = SVC(**params, random_state=7)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    print('Accuracy train et test:', accuracy_score(model.predict(x_train), y_train), accuracy_score(y_pred, y_test))
    return y_pred, model, scaler

def make_logreg(x_train, x_test, y_train, y_test, params):
    '''
    Créer le modèle LogisticRegression et la prédiction.
    '''
    #params['kernel'] = 'rbf'
    print('Calcule LogisticRegression', params)
    scaler = StandardScaler()
    x_train_scaled = x_train.copy()
    x_train_scaled[:] = scaler.fit_transform(x_train)
    x_test_scaled = x_test.copy()
    x_test_scaled[:] = scaler.transform(x_test)
    model = LogisticRegression(**params, random_state=7)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    print('Accuracy train et test:', accuracy_score(model.predict(x_train), y_train), accuracy_score(y_pred, y_test))
    return y_pred, model, scaler

def show_svc_graph(x_test, y_test, y_pred, model, scaler, tag=''):
    '''
    Rapport standart des graphiques sur le modèle SVC. C'est différent de RandomForestClassifier.
    '''
    x_test_scaled = x_test.copy()
    x_test_scaled[:] = scaler.transform(x_test)
    display(pd.crosstab(y_test, y_pred))
    px.imshow(pd.crosstab(y_test, y_pred), zmin=0, text_auto=True, 
                labels={'x' : 'Prediction', 'y' : 'Test'},
                color_continuous_scale='OrRd').show()
    print(classification_report(y_test, y_pred))
    # Merci https://stackoverflow.com/questions/41592661/determining-the-most-contributing-features-for-svm-classifier-in-sklearn
    perm_importance = permutation_importance(model, x_test_scaled, y_test)
    dat = pd.Series(perm_importance.importances_mean, index=x_test.columns)
    dat = dat.reindex(dat.abs().sort_values(ascending=True).head(10).index)
    px.bar(y=dat.index, 
            x=dat,
            labels={'y' : 'Features', 'x' : 'Importance'},
            width=800, height=1000,
            title='Importance de top10 valeurs selon SVC ' + tag).show()
    return dat

def get_top_features_svc(x_test, y_test, model, scaler):
    '''
    Faire une série avec les features les plus important selon SVC.
    '''
    x_test_scaled = x_test.copy()
    x_test_scaled[:] = scaler.transform(x_test)
    perm_importance = permutation_importance(model, x_test_scaled, y_test)
    dat = pd.Series(perm_importance.importances_mean, index=x_test.columns)
    dat = dat.reindex(dat.abs().sort_values(ascending=True).head(10).index)
    return dat

def get_top_features_forest(x_test, model):
    '''
    Faire une série avec les features les plus important selon RandomForestClassifier.
    '''
    dat = pd.Series(model.feature_importances_, index=x_test.columns)
    dat = dat.sort_values(ascending=True).head(10)
    return dat

def get_scores(y_test, y_pred, tag):
    '''
    Serie des scores classification
    '''
    return pd.Series([tag, accuracy_score(y_test, y_pred), 
                    precision_score(y_test, y_pred, average='weighted'), 
                    f1_score(y_test, y_pred, average='weighted')], 
                    index=['tag', 'accuracy','precision','f1'])

def get_auc(x_test, y_test, model):
    '''
    Obtenir les paramètre pour déssigner la courbe ROC.
    '''
    probs = model.predict_proba(x_test)
    fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
    
def show_auc(x_test, y_test, model, show=True):
    '''
    Dessigner la courbe ROC.
    '''
    from sklearn.metrics import roc_curve, auc
    probs = model.predict_proba(x_test)
    fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig = px.line(x=fpr, y=tpr, title='Modèle RandomForest (auc = %0.2f)' % roc_auc)\
        .add_scatter(x=[0, 1], y=[0, 1], line_dash='dash', name='Aléatoire (auc = 0.5)', mode='lines')
    if not show:
        return fig
    fig.show()

def show_class_matrix(y_test, y_pred, title, show=True):
    fig1 = px.imshow(pd.crosstab(y_test, y_pred, normalize=False), text_auto=True, width=400, height=400, color_continuous_scale='OrRd', labels={'x' : 'Test', 'y' : 'Prediction'}, title=title)
    fig2 = px.imshow(pd.crosstab(y_test, y_pred, normalize=True), text_auto='.0%', width=400, height=400, color_continuous_scale='OrRd', labels={'x' : 'Test', 'y' : 'Prediction'}, title=title)
    if show:
        fig1.show()
        fig2.show()
    return fig1

''' 
Regression linéaires.
'''
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def regress(x, y, regressor=BayesianRidge(), show=True):
    x = x.copy()
    y = y.copy()
    # Divisez les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Définissez les colonnes à encoder avec OneHotEncoder
    columns_to_encode = ['region', 'departement', 'commune']
    # Créez un transformateur pour appliquer OneHotEncoder aux colonnes spécifiées
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), x.select_dtypes(include=['number']).columns)],
        remainder='passthrough'  # Les colonnes non spécifiées seront conservées sans modification
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)  
    ])
    # Entraînez le modèle sur l'ensemble d'entraînement encodé
    model.fit(x_train, y_train)
    # Faites des prédictions sur l'ensemble de test encodé
    y_pred = model.predict(x_test)
    # Évaluez les performances du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Affichez les performances du modèle
    if show:
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'R-squared (R2): {r2}')
        # Error rate is the square root of Mean Squared Error (MSE)
        error_rate = np.sqrt(mse)
        # Print the error rate
        print(f'RMSE: {error_rate}')
    return y_test, y_pred, model['regressor']

def probabilistic_regress(x, y, regressor=BayesianRidge()):
    # TODO: A voir s'il y a vraiment besoin de la debugger
    class ProbabilisticEncoder(TransformerMixin, BaseEstimator):
        def fit(self, x, y=None):
            self.probabilities = {}
            for column in x.columns:
                probabilities = x.groupby(column).size() / len(x)
                self.probabilities[column] = probabilities
            return self

        def transform(self, x):
            x_encoded = x.copy()
            for column, probabilities in self.probabilities.items():
                x_encoded[column] = x[column].map(probabilities)
            return x_encoded

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Colonnes à encoder avec le ProbabilisticEncoder
    #columns_to_encode = ['nom_région', 'nom_département', 'LIBGEO']
    columns_to_encode = ['region', 'departement', 'commune']
    # Transformateur pour appliquer le ProbabilisticEncoder aux colonnes spécifiées
    preprocessor = ColumnTransformer(
        transformers=[
            ('encoder', ProbabilisticEncoder(), columns_to_encode),
            ('scaler', StandardScaler(), x.select_dtypes(include=['number']).columns)
        ],
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    # Entraînement du modèle sur l'ensemble d'entraînement encodé
    model.fit(x_train, y_train)
    # Prédictions sur l'ensemble de test encodé
    y_pred = model.predict(x_test)
    # Évaluation des performances du modèle
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    error_rate = np.sqrt(mse)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R2): {r2}')
    print(f'RMSE: {error_rate}')
    return y_test, y_pred, model['regressor']


def show_regression_results(y_test, y_pred, show=True):
    fig = plt.figure()
    # Scatter plot des prédictions par rapport aux vraies valeurs
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Scatter Plot des Prédictions vs Vraies Valeurs")
    # Tracer la ligne diagonale en rouge
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    if not show:
        return fig
    plt.show()
    
def show_regression_features(x, model, show=True):
    # Données
    variables = x.columns.to_list()
    coefficients = model.coef_
    # Triez les coefficients 
    sorted_indices = sorted(range(len(coefficients)), key=lambda k: abs(coefficients[k]), reverse=False)
    variables_sorted = [variables[i] for i in sorted_indices]
    coefficients_sorted = [coefficients[i] for i in sorted_indices]
    # Graph
    fig = plt.figure(figsize=(10, len(variables)//2))
    plt.barh(variables_sorted, coefficients_sorted, color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Variable')
    plt.title('Coefficients des Variables dans le Modèle')
    if not show:
        return fig
    plt.show()

def show_corr_matrix(df_selected):
    # Créer une carte de corrélation
    correlation_matrix = df_selected.corr()

    # Créer la carte de corrélation avec seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True,mask=np.triu(correlation_matrix, k=1))
    plt.title('Correlation Heatmap')
    plt.show()

def regression_report(x, y, regressor, probabilistic=False):
    print('Resultat de regression {0} pour {1} colonnes'.format(regressor.__class__.__name__, len(x.columns)))
    if probabilistic:
        y_test, y_pred, model = probabilistic_regress(x, y, regressor)
        print('ProbabilisticEncoder utilisé.')
    else:
        y_test, y_pred, model = regress(x, y, regressor)
    show_regression_results(y_test, y_pred)
    show_regression_features(x, model)

def show_decision_tree(X_train, y_train):
    from sklearn.tree import plot_tree    
    #model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model = RandomForestClassifier(random_state=7, max_depth=3)

    model.fit(X_train, y_train)
    #fix, ax = plt.subplots(figsize = (20, 20))
    plot_tree(model, 
            feature_names = X_train.columns.to_list(),
            rounded = True,
            filled = True,
            
            )
    plt.show()

def show_map_class(x, x_dep, show=True):
    from exploration import GEO, VILLESDEFRANCE
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    groupes_salaires = ['Tous',
                        'Niveau de vie faible',
                        'Salaire intermédiaire',
                        'Salaire élevé',
                        'Niveau de la richesse']
    # chargement des données géographiques
    cities = pd.read_csv(VILLESDEFRANCE) #c'est mieux que Google)
    # merge
    x = pd.merge(left = x, right = cities[['insee_code', 'latitude', 'longitude']], left_on='codgeo', right_on='insee_code', how='left')    
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
    x[x.latitude.isna()] = x[x.latitude.isna()].apply(restore_data, axis=1)
    x = x.dropna()
    ## on charge le GeoJSON du data.gouv.fr pour les département
    import json
    with open('input/a-dep2021.json') as f:
        deps = json.load(f)
    fig = go.Figure()
        
    def add_map(x_, x_dep_):
        fig.add_trace(go.Scattermapbox(lon = x_.longitude, 
                                    lat = x_.latitude, 
                                    text = x_.commune + 
                                        ' : ' + x_.salaire.astype(str),
                                    ids = x_.commune.astype(str),
                                    #visible = False,
                                    marker = {
                                        'size' : np.sqrt(x_.population/100),
                                        #'size' : np.sqrt(df_communes[df_communes.eti_ge > eti_ge_margin].eti_ge * 50), #, a_min=0, a_max=150),
                                        'color' : x_.salaire,
                                        'showscale' : True,
                                        'colorscale' : [[0,'blue'], [1,'green']],
                                        'colorbar' : dict(len=0.5, 
                                                        xpad=100, 
                                                        bordercolor='black',
                                                        borderwidth=1, 
                                                        title='Salaires en communes',
                                                        title_side='right')
                                    },
                                    name='Par communes',
                                    # width=1000,
                                    # height=1000,
                                    ))
        fig.add_trace(go.Choroplethmapbox(geojson=deps,
                                        featureidkey='properties.dep',
                                        locations=x_dep_.codgeo_dep.str.rjust(2, '0'), #attention! toujours '01' etc pour les dept 
                                        z = x_dep_.salaire,
                                        visible = False,
                                        zmin = x_dep_.salaire.min(),
                                        zmax = x_dep_.salaire.max(),
                                        zmid = x_dep_.salaire.median(),
                                        text = x_dep_.departement + ' : ' + x_dep_.salaire.astype(str),
                                        ids = x_dep.codgeo_dep.astype(str),
                                        colorscale = [[0,'red'],[0.25,'orange'],[1,'yellow']],
                                        name = 'Par departements',
                                        showscale = True,
                                        colorbar=dict(len=0.5,
                                                    bordercolor='black',
                                                    title='Salaires en départements',
                                                    title_side='right',
                                                    ),
                                        showlegend = True,
                                        # width=1000,
                                        # height=1000,

                            ))
    add_map(x, x_dep)    
    for i in range(4):
        add_map(x[x.cat==i], x_dep[x_dep.cat==i])
    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.update_layout(mapbox_style="carto-positron",
                mapbox = dict(center=dict(lat=46.60, lon=1.98),            
                            zoom=5
                            ))
    fig.update_layout(
        width = 800,
        height = 800,
        title = 'Distribution géographique des catégories de salaire en France',
        )

    steps = []
    for i in range(5):
        step = dict(
            # method="update",
            # args=[{"visible": [False] * 8},
            #     {"title": "Distribution géographique des salaire de groupe: " + groupes_salaires[i]}],  # layout attribute
            method = 'restyle',  
            args = ['visible', [False] * len(fig.data)],
            label = groupes_salaires[i],
        )
        # step["args"][0]['visible'][i*2] = True  # Toggle i'th trace to "visible"
        # step["args"][0]['visible'][i*2 + 1] = True  # Toggle i'th trace to "visible"
        step["args"][1][i*2] = True  # Toggle i'th trace to "visible"
        step["args"][1][i*2 + 1] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        currentvalue={"prefix": "Class: "},
        steps=steps,
        active=0
    )]
    fig.update_layout(
        sliders=sliders
    )
    if not show:
        return fig
    fig.show()
    
