import streamlit as st

st.set_page_config(page_title="Conclusion", page_icon="🗟")

'# _French Industry_: preuves d\'inégalités en France'
'## Conclusion'


'''
Le jeu de données « French Industry » a permis d'identifier des inégalités sociales et économiques à différentes échelles. La distribution des salaires par genre montre une différence de rémunération importante entre les femmes et les hommes. La catégorie des salaires les plus bas est presque exclusivement composée de femmes, tandis que la catégorie des salaires les plus élevés est absolument dominée par les hommes.

Les modèles de régression ont confirmé que le genre est un facteur important dans la prédiction du niveau de salaire. Il est l'un des premiers facteurs qui déterminent le calcul de la rémunération. Cependant, d'autres facteurs jouent également un rôle, tels que les salaires des employés et des travailleurs, ainsi que les revenus des cadres et particulièrement les cadres masculins. Ces modèles étaient facilement sujet à sur-apprentissage à cause de variables explicatives bien souvent extrêmement corrélées. Nous parvenons tout de même à obtenir des résultats très satisfaisants en limitant grandement le nombre de variables explicatives. 

La classification a cependant mis en évidence d'autres facteurs qui, selon les modèles, sont cruciaux pour déterminer le niveau de salaire moyen dans une ville. Parmi eux, on peut noter la présence de grandes entreprises, ainsi que le pourcentage de certaines catégories sociales déterminées par l'Insee, telles que les enfants âgés vivant avec leurs parents, les jeunes seuls avec des enfants et les couples de plus de 50 ans. Cette modélisation est limitée par le fait que les données utilisées sont des salaires moyens. Les revenus extrêmement élevés de certains foyers peuvent donc fausser les estimations. Une alternative serait d'utiliser les revenus médians, qui sont moins sensibles aux valeurs extrêmes.

La cartographie permet de voir la correspondance des localisation des communes avec des salaires moyens les plus élevés et des clusters les plus importants de grandes entreprises. Ces visualisations sont très importantes car les régions étaient globalement très mal représentées dans les modèles de régression et absentes des modèles de classifications.

Il est aussi important d'aborder une limite très importante de notre projet. Nous rappelons que ce projet a une visée pédagogique avant tout et que nous avons tout fait avec les moyens mis à notre disposition pour tirer des insights significatifs et pour faciliter la compréhension de ce jeu de données. Nos résultats mettent en avant de nombreuses inégalités entre homme et femme ou encore entre classe d'âge. Cependant, comme le montre une étude de l'INSEE réalisée en 2021 une grande part de ces inégalités sont expliquées par des emplois non équivalents. 

En effet, [en 2021 l'INSEE met en avant](https://www.insee.fr/fr/statistiques/6960132) un constat effrayant : « le revenu salarial des femmes est inférieur de 24,4 % à celui des hommes ». Cependant l'INSEE dans cette même étude avance un tout autre résultat : « à poste comparable et temps de travail équivalent, l'écart de salaires entre les femmes et hommes atteint 4,3 %. » Cela mets en avant le fait qu'en moyenne les hommes occupent des postes plus rémunérateurs (cela peut être une autre forme d'inégalité mais en dehors de notre étude) et que pour obtenir des résultats plus pertinent dans notre projet il aurait fallu que nous ayons à notre disposition des données sur les types d'emplois occupés. 

'''
