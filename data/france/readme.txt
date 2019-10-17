# Cette oeuvre est mise à disposition sous licence Paternité - Partage à l'Identique 3.0 non transposé.
# Pour voir une copie de cette licence, visitez http://creativecommons.org/licenses/by-sa/3.0/
# ou écrivez à Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.

#--------------------
# Date : 24 mai 2012
# Auteurs : Arthur CHARPENTIER et Ewen GALLIC
# Sources : INSEE, API Google Maps v3 et GeoHack (coordonnées GPS), Propores calculs (estimation de population à partir des données INSEE)
#--------------------

La base comporte 36316 observations.
Les villes pour lesquelles il existe une fusion ou une fusion mentionnée par l'INSEE ont été regroupées.

# Variables :
reg : code region INSEE (character)
dep : code departement INSEE (character, corse 201 et 202 au lieu de 2A et 2B)
com : code commune INSEE (character)
article : article du nom de la commune (character)
com_nom : nom de la commune (character)
long : longitude (numeric)
lat : latitude (numeric)
pop_i : estimation de la population à la date i (ramenée à 1 si <=0), i=1975,...,2010 (numeric)


# Méthode d'estimation de la population :
À l'aide de la fonction bs() du packages splines de R avec un degré de liberté égal à 3
La valeur prédite est ramenée à 1 lorsqu'elle est inférieure ou égale à 1.