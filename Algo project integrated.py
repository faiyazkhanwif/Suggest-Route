#To generate the graphs please open the ipynb file or to see the graph outputs, please see the pdf file.
import numpy as np
import matplotlib.pyplot as plot

plot.rcdefaults()

import gmplot
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="PyCharm")

# (Question 1) : Get and Mark the Location


print("The cities that Ben Sherman will go from Kuala Lumpur are : ")
location = geolocator.geocode("Kuala Lumpur")
print(location.address, location.latitude, location.longitude)

location2 = geolocator.geocode("Jakarta")
print(location2.address, location2.latitude, location2.longitude)

location3 = geolocator.geocode("Bangkok")
print(location3.address, location3.latitude, location3.longitude)

location4 = geolocator.geocode("Taipei")
print(location4.address, location4.latitude, location4.longitude)

location5 = geolocator.geocode("Hong Kong")
print(location5.address, location5.latitude, location5.longitude)

location6 = geolocator.geocode("Tokyo")
print(location6.address, location6.latitude, location6.longitude)

location7 = geolocator.geocode("Beijing")
print(location7.address, location7.latitude, location7.longitude)

location8 = geolocator.geocode("Seoul")
print(location8.address, location8.latitude, location8.longitude)

#mark location : Kuala Lumpur, Jakarta, Bangkok, Taipei, Hong Kong, Tokyo, Beijing, Seoul
lat = [3.1516964, -6.1753942, 13.7542529, 25.0375198, 22.2793278, 35.6828387, 39.906217, 37.5666791]
long = [101.6943271, 106.827183, 100.493087, 121.5636796, 114.1628131, 139.7594549, 116.3912757, 126.9782914]
# plot location : Jakarta
#lat2 = [-6.1753942]
#long2 = [106.827183]
gmapOne = gmplot.GoogleMapPlotter(3.1516964, 101.6943271, 5)
gmapOne.scatter(lat, long, 'red', size = 15000, marker=False)

#gmapTwo = gmplot.GoogleMapPlotter(-6.1753942,106.827183, 15)
#gmapTwo.scatter(lat2, long2, 'blue', size = 50, marker=False)


gmapOne.draw("map.html")
#gmapTwo.draw("map.html")




#Kuala Lumpur, Malaysia 3.1516964 101.6942371
#Daerah Khusus Ibukota Jakarta, Indonesia -6.1753942 106.827183
#กรุงเทพมหานคร, เขตพระนคร, กรุงเทพมหานคร, 10200, ประเทศไทย 13.7542529 100.493087
#臺北市, 信義區, 臺北市, 11008, Taiwan 25.0375198 121.5636796
#香港島 Hong Kong Island, 香港 Hong Kong, China 中国 22.2793278 114.1628131
#東京都, 日本 (Japan) 35.6828387 139.7594549
#北京市, 东城区, 北京市, 100010, China 中国 39.906217 116.3912757
#서울, 대한민국 37.5666791 126.9782914


# Problem 1 (Question 2) : Getting distances between each of the destinations

from geopy.distance import geodesic
KualaLumpur_MAS = (3.1516964, 101.6942371)
Jakarta_INA = (-6.1753942, 106.827183)
Bangkok_THA = (13.7542529, 100.493087)
Taipei_TPE = (25.0375198, 121.5636796)
HongKong_HKG = (22.2793278, 114.1628131)
Tokyo_JPN = (35.6828387, 139.7594549)
Beijing_CHN = (39.906217, 116.3912757)
Seoul_KOR = (37.5666791, 126.9782914)

print("\nBen plans to travel from Kuala Lumpur to other cities")
print("\nThe distance between Kuala Lumpur and Jakarta is : ", geodesic(KualaLumpur_MAS, Jakarta_INA).km)
print("\nThe distance between Kuala Lumpur and Bangkok is : ", geodesic(KualaLumpur_MAS, Bangkok_THA).km)
print("\nThe distance between Kuala Lumpur and Taipei is : ", geodesic(KualaLumpur_MAS, Taipei_TPE).km)
print("\nThe distance between Kuala Lumpur and Hong Kong is : ", geodesic(KualaLumpur_MAS, HongKong_HKG).km)
print("\nThe distance between Kuala Lumpur and Tokyo is : ", geodesic(KualaLumpur_MAS, Tokyo_JPN).km)
print("\nThe distance between Kuala Lumpur and Beijing is : ", geodesic(KualaLumpur_MAS, Beijing_CHN).km)
print("\nThe distance between Kuala Lumpur and Seoul is : ", geodesic(KualaLumpur_MAS, Seoul_KOR).km)

print("***********************************************************")
print("\nBen plans to travel from Jakarta to other cities")
print("\nThe distance between Jakarta and Kuala Lumpur is : ", geodesic(Jakarta_INA, KualaLumpur_MAS).km)
print("\nThe distance between Jakarta and Bangkok is : ", geodesic(Jakarta_INA, Bangkok_THA).km)
print("\nThe distance between Jakarta and Taipei is : ", geodesic(Jakarta_INA, Taipei_TPE).km)
print("\nThe distance between Jakarta and Hong Kong is : ", geodesic(Jakarta_INA, HongKong_HKG).km)
print("\nThe distance between Jakarta and Tokyo is : ", geodesic(Jakarta_INA, Tokyo_JPN).km)
print("\nThe distance between Jakarta and Beijing is : ", geodesic(Jakarta_INA, Beijing_CHN).km)
print("\nThe distance between Jakarta and Seoul is : ", geodesic(Jakarta_INA, Seoul_KOR).km)

print("***********************************************************")
print("\nBen plans to travel from Bangkok to other cities")
print("\nThe distance between Bangkok and Kuala Lumpur is : ", geodesic(Bangkok_THA, KualaLumpur_MAS).km)
print("\nThe distance between Bangkok and Jakarta is : ", geodesic(Bangkok_THA, Jakarta_INA).km)
print("\nThe distance between Bangkok and Taipei is : ", geodesic(Bangkok_THA, Taipei_TPE).km)
print("\nThe distance between Bangkok and Hong Kong is : ", geodesic(Bangkok_THA, HongKong_HKG).km)
print("\nThe distance between Bangkok and Tokyo is : ", geodesic(Bangkok_THA, Tokyo_JPN).km)
print("\nThe distance between Bangkok and Beijing is : ", geodesic(Bangkok_THA, Beijing_CHN).km)
print("\nThe distance between Bangkok and Seoul is : ", geodesic(Bangkok_THA, Seoul_KOR).km)

print("***********************************************************")
print("\nBen plans to travel from Taipei to other cities")
print("\nThe distance between Taipei and Kuala Lumpur is : ", geodesic(Taipei_TPE, KualaLumpur_MAS).km)
print("\nThe distance between Taipei and Jakarta is : ", geodesic(Taipei_TPE, Jakarta_INA).km)
print("\nThe distance between Taipei and Bangkok is : ", geodesic(Taipei_TPE, Bangkok_THA).km)
print("\nThe distance between Taipei and Hong Kong is : ", geodesic(Taipei_TPE, HongKong_HKG).km)
print("\nThe distance between Taipei and Tokyo is : ", geodesic(Taipei_TPE, Tokyo_JPN).km)
print("\nThe distance between Taipei and Beijing is : ", geodesic(Taipei_TPE, Beijing_CHN).km)
print("\nThe distance between Taipei and Seoul is : ", geodesic(Taipei_TPE, Seoul_KOR).km)

print("***********************************************************")
print("\nBen plans to travel from Hong Kong to other cities")
print("\nThe distance between Hong Kong and Kuala Lumpur is : ", geodesic(HongKong_HKG, KualaLumpur_MAS).km)
print("\nThe distance between Hong Kong and Jakarta is : ", geodesic(HongKong_HKG, Jakarta_INA).km)
print("\nThe distance between Hong Kong and Bangkok is : ", geodesic(HongKong_HKG, Bangkok_THA).km)
print("\nThe distance between Hong Kong and Taipei is : ", geodesic(HongKong_HKG, Taipei_TPE).km)
print("\nThe distance between Hong Kong and Tokyo is : ", geodesic(HongKong_HKG, Tokyo_JPN).km)
print("\nThe distance between Hong Kong and Beijing is : ", geodesic(HongKong_HKG, Beijing_CHN).km)
print("\nThe distance between Hong Kong and Seoul is : ", geodesic(HongKong_HKG, Seoul_KOR).km)

print("***********************************************************")
print("\nBen plans to travel from Tokyo to other cities")
print("\nThe distance between Tokyo and Kuala Lumpur is : ", geodesic(Tokyo_JPN, KualaLumpur_MAS).km)
print("\nThe distance between Tokyo and Jakarta is : ", geodesic(Tokyo_JPN, Jakarta_INA).km)
print("\nThe distance between Tokyo and Bangkok is : ", geodesic(Tokyo_JPN, Bangkok_THA).km)
print("\nThe distance between Tokyo and Taipei is : ", geodesic(Tokyo_JPN, Taipei_TPE).km)
print("\nThe distance between Tokyo and Hong Kong is : ", geodesic(Tokyo_JPN, HongKong_HKG).km)
print("\nThe distance between Tokyo and Beijing is : ", geodesic(Tokyo_JPN, Beijing_CHN).km)
print("\nThe distance between Tokyo and Seoul is : ", geodesic(Tokyo_JPN, Seoul_KOR).km)

print("***********************************************************")
print("\nBen plans to travel from Beijing to other cities")
print("\nThe distance between Beijing and Kuala Lumpur is : ", geodesic(Beijing_CHN, KualaLumpur_MAS).km)
print("\nThe distance between Beijing and Jakarta is : ", geodesic(Beijing_CHN, Jakarta_INA).km)
print("\nThe distance between Beijing and Bangkok is : ", geodesic(Beijing_CHN, Bangkok_THA).km)
print("\nThe distance between Beijing and Taipei is : ", geodesic(Beijing_CHN, Taipei_TPE).km)
print("\nThe distance between Beijing and Hong Kong is : ", geodesic(Beijing_CHN, HongKong_HKG).km)
print("\nThe distance between Beijing and Tokyo is : ", geodesic(Beijing_CHN, Tokyo_JPN).km)
print("\nThe distance between Beijing and Seoul is : ", geodesic(Beijing_CHN, Seoul_KOR).km)

print("***********************************************************")
print("\nBen plans to travel from Seoul to other cities")
print("\nThe distance between Seoul and Kuala Lumpur is : ", geodesic(Seoul_KOR, KualaLumpur_MAS).km)
print("\nThe distance between Seoul and Jakarta is : ", geodesic(Seoul_KOR, Jakarta_INA).km)
print("\nThe distance between Seoul and Bangkok is : ", geodesic(Seoul_KOR, Bangkok_THA).km)
print("\nThe distance between Seoul and Taipei is : ", geodesic(Seoul_KOR, Taipei_TPE).km)
print("\nThe distance between Seoul and Hong Kong is : ", geodesic(Seoul_KOR, HongKong_HKG).km)
print("\nThe distance between Seoul and Tokyo is : ", geodesic(Seoul_KOR, Tokyo_JPN).km)
print("\nThe distance between Seoul and Beijing is : ", geodesic(Seoul_KOR, Beijing_CHN).km)


routes = []

#TSP based on brute force
def find_paths(node, cities, path, distance):
    # Add way point
    path.append(node)

    # Calculate path length from current to last node
    if len(path) > 1:
        distance += cities[path[-2]][node]

    # If path contains all cities and is not a dead end,
    # add path from last to first city and return.
    if (len(cities) == len(path)) and (path[0] in cities[path[-1]]):
        global routes
        path.append(path[0])
        distance += cities[path[-2]][path[0]]
        routes.append([distance, path])
        return

    # Fork paths for all possible cities not yet used
    for city in cities:
        if (city not in path) and (node in cities[city]):
            find_paths(city, dict(cities), list(path), distance)


if __name__ == '__main__':
    cities = {
                'KL': {'KL': 0, 'JK': 1178.6718596734863, 'BK': 1180.0700698191881, 'TAI': 3224.7779905401576, 'HK': 2508.4362660368365,'BEI': 4332.215068712132, 'TOK': 5318.677216353379, 'SEO': 4601.871799273804},
                'JK': {'KL': 1178.6718596734863, 'JK': 0, 'BK': 2312.509200540841, 'TAI': 3804.2538748562656, 'HK': 3247.6059266254892,'BEI': 5195.7631594758395, 'TOK': 5773.098750995338, 'SEO': 5274.932678827154},
                'BK': {'KL': 1180.0700698191881, 'JK': 2312.509200540841, 'BK': 0, 'TAI': 2535.8957177665748, 'HK': 1726.1824898477596, 'BEI': 3288.6020977412145, 'TOK': 4610.121572373538, 'SEO': 3719.354478303037},
                'TAI': {'KL': 3224.7779905401576, 'JK': 3804.2538748562656, 'BK': 2535.8957177665748, 'TAI': 0, 'HK': 814.2973250648987, 'BEI': 1718.2312737282145, 'TOK':  2104.313098157768, 'SEO': 1480.973652900838},
                'HK': {'KL': 2508.4362660368365, 'JK': 3247.6059266254892, 'BK': 1726.1824898477596, 'TAI': 814.2973250648987, 'HK': 0, 'BEI': 1965.7256268618175, 'TOK': 2889.715055526659, 'SEO': 2093.6715546233295},
                'BEI': {'KL': 4332.215068712132, 'JK': 5195.7631594758395, 'BK': 3288.6020977412145, 'TAI': 1718.2312737282145, 'HK': 1965.7256268618175, 'BEI': 0, 'TOK': 2104.3499748042273, 'SEO': 955.7689217292321},
                'TOK': {'KL': 5318.677216353379, 'JK': 5773.098750995338, 'BK': 4610.121572373538, 'TAI': 2104.313098157768, 'HK': 2889.715055526659, 'BEI': 2104.3499748042273, 'TOK': 0, 'SEO': 1161.2277477992284},
                'SEO': {'KL': 4601.871799273804, 'JK': 5274.932678827154, 'BK': 3719.354478303037, 'TAI': 1480.973652900838, 'HK': 2093.6715546233295, 'BEI': 955.7689217292321, 'TOK': 1161.2277477992284, 'SEO': 0}
             }

    find_paths('KL', cities, [], 0)
    print("\n")
    routes.sort()
    if len(routes) != 0:
        print("Shortest distance: %s" % round(routes[0][0],3)+" km")
        print("Shortest route based on distance: %s" % routes[0][1])
    else:
        print("FAIL!")

gmap = gmplot.GoogleMapPlotter(3.1516964, 101.6942371, 13)

kl = (3.1516964, 101.6942371)
j = (-6.1753942, 106.827183)
b = (13.7538929, 100.8160803)
t = (25.0375198, 121.5636796)
h = (22.2793278, 114.1628131)
be = (39.906217, 116.3912757)
to = (35.6828387, 139.7594549)
s = (37.5666791, 126.9782914)

lat_list = [3.1516964, 13.7538929, 39.906217, 37.5666791, 35.6828387, 25.0375198, 22.2793278, -6.1753942]
lon_list = [101.6942371, 100.816080, 116.3912757, 126.9782914, 139.7594549, 121.5636796, 114.1628131, 106.827183]
gmap.heatmap(lat_list, lon_list)
gmap.plot(lat_list, lon_list, "cornflowerblue", edge_width=2.5)
gmap.scatter(lat_list, lon_list, size=40, marker=True)
gmap.draw("map2.html")

#Problem 2
# Stopwords' file access
stopwords = open("stopwords.txt", 'r')
stopwords = stopwords.read().splitlines()

# The rabin karp algorithm is made to find stopwords from articles
# An algorithm based on rabin karp is made to find stopwords from articles
def stopwordsearch(stopword, articles):
    sword = len(stopword)
    art = len(articles)
    count = 0

    # Search for stopwords in articles
    for i in range(0, art - sword + 1):
        found = True
        for j in range(0, sword):
            if stopword[j] != articles[i + j]:
                found = False
                break
        if found:
            count += 1

    # Printing out the stopwords' appearance.
    if count > 0:
        print(stopword, ':', count, 'times.')
    else:
        None


# Filtering stopwords from articles to do the count.
def stopwordfilter(filepath):
    # File handling
    article = open(filepath, encoding="utf8")
    article = article.read().splitlines()

    # Search
    for x in article:
        for i in stopwords:
            stopwordsearch(i, x)


def wordcountgraph(filepath):
    # File handling
    file = open(filepath, encoding="utf8")
    file = file.read()
    number_of_characters = len(file)

    # Printing the number of words
    print('Total number of words in the article:', number_of_characters)
    allwords = file.split()

    # Filtering out the stopwords from 'allwords' to plot the graph
    words = [word for word in allwords if word.lower() not in stopwords]
    print('Number of words that is going to be used for finding out the economic sentiment:', len(words))

    # Creating an empty dictionary to store word count
    dic = {}
    wordcount = []
    for wordcount in words:
        dic[wordcount] = dic.get(wordcount, 0) + 1
    print()

    # For words (x axis)
    worda = []
    # For count(y axis)
    countb = []

    # Storing in  list a and countb
    for key, value in dic.items():
        worda.append(key)
        countb.append(value)

    # Length of x axis based on number of words
    x = np.arange(len(worda))

    # Setting bars of words based on count
    plot.bar(x, countb)

    # Counts on y axis
    plot.yticks(fontsize=10)

    # Adding values to x axis
    plot.xticks(x, worda)

    # Rotation of values
    plot.xticks(rotation=90)

    # Labels
    plot.xlabel('Words', fontsize=30)
    plot.ylabel('Count', fontsize=30)

    # Size
    plot.rcParams['figure.figsize'] = (40, 20)


# Bangkok
stopwordfilter("Bangkok.txt")
wordcountgraph('Bangkok.txt')
plot.show() #to see the graph
#plot.close() #after saving

# Beijing
stopwordfilter("Beijing.txt")
wordcountgraph('Beijing.txt')
#plot.show() #to see the graph
#plot.close() #after saving

# Hongkong
stopwordfilter("Hongkong.txt")
wordcountgraph('Hongkong.txt')
#plot.show() #to see the graph
#plot.close() #after saving

# Jakarta
stopwordfilter("Jakarta.txt")
wordcountgraph('Jakarta.txt')
#plot.show() #to see the graph
#plot.close() #after saving

# Seoul
stopwordfilter("Seoul.txt")
wordcountgraph('Seoul.txt')
#plot.show() #to see the graph
#plot.close() #after saving

# Taipei
stopwordfilter("Taipei.txt")
wordcountgraph('Taipei.txt')
#plot.show() #to see the graph
#plot.close() #after saving

# Tokyo
stopwordfilter("Tokyo.txt")
wordcountgraph('Tokyo.txt')
#plot.show() #to see the graph
#plot.close() #after saving

#File handling for positive.txt and negative.txt
positivewords = open("positive.txt", 'r')
positivewords = positivewords.read().splitlines()
negativewords = open("negative.txt", 'r')
negativewords = negativewords.read().splitlines()


def positive_negative_comparison(filepath):
    # file handling
    article = open(filepath, encoding="utf8")
    article = article.read()
    number_of_characters = len(article)

    # Initializing count variables for negative, positive and neutral words.
    poscount = 0
    negcount = 0
    neucount = 0

    # Search for positive and negative words in the articles.
    for word in article.split():
        if word in positivewords:
            poscount = poscount + 1
        elif word in negativewords:
            negcount = negcount + 1

    # Calculating neutral count based on the given condition.
    neucount = number_of_characters - (poscount + negcount)

    # Results of positive, negative and neutral count.
    print("Positive words: ", poscount)
    print("Negative words: ", negcount)
    print("Neutral words: ", neucount)

    # Checking the positive-negative status for the article and coming to a conclusion based on the result.
    if poscount > negcount:
        print("The article is giving positive sentiment.")
        print("This country has a positive economic and financial situation.")
    elif poscount < negcount:
        print("The article is giving negative sentiment.")
        print("This country has a negative economic and financial situation.")
    else:
        print("The article is giving neutral sentiment.")
        print("This country has a neutral economic and financial situation.")

    # Assigning names of the participants of the histogram graph.
    determiners = ("Positive Words", "Negative Words")

    # Length of x axis based on the length of the determiners.
    x = np.arange(len(determiners))

    # Assigning counts as determiners' values.
    det_values = [poscount, negcount]

    # Setting bars based on the values.
    barlist = plot.bar(x, det_values, align='center', alpha=0.5)

    # Decorations
    barlist[0].set_color('green')
    barlist[1].set_color('red')

    # Adding values to x axis
    plot.xticks(x, determiners)

    # Decorations
    plot.yticks(fontsize=10)
    plot.xticks(fontsize=10)

    # Label
    plot.ylabel("Count", fontsize=20)

    # Title
    plot.title("Comparison of positive and negative words.", fontsize=20)

    # Size
    plot.rcParams['figure.figsize'] = (6, 3)

#Bangkok
positive_negative_comparison('Bangkok.txt')
#plot.show() #to see the graph
#plot.close() #after saving

#Beijing
positive_negative_comparison('Beijing.txt')
#plot.show() #to see the graph
#plot.close() #after saving

#Hongkong
positive_negative_comparison('Hongkong.txt')
#plot.show() #to see the graph
#plot.close() #after saving

#Jakarta
positive_negative_comparison('Jakarta.txt')
#plot.show() #to see the graph
#plot.close() #after saving

#Seoul
positive_negative_comparison('Seoul.txt')
#plot.show() #to see the graph
#plot.close() #after saving

#Taipei
positive_negative_comparison('Taipei.txt')
#plot.show() #to see the graph
#plot.close() #after saving

#Tokyo
positive_negative_comparison('Tokyo.txt')
#plot.show() #to see the graph
#plot.close() #after saving




#Problem 3
import itertools #Itertools for possible routes
# calculating final sentiment percentage for finding out probability of a possible route
def calc_sent(filepath):
    # file handling
    file = open(filepath, encoding="utf8")
    file = file.read()
    number_of_characters = len(file)
    allwords = file.split()

    # Filtering out the stopwords again to calculate the probabilty of the route
    words = [word for word in allwords if word.lower() not in stopwords]

    # Number of words responsible the economic sentiment
    n_of_rwords = len(words)

    # positive and negative count
    negativeCount = 0
    positiveCount = 0
    calcCount = 0
    for part in file.split():
        if part in positivewords:
            positiveCount += 1
        elif part in negativewords:
            negativeCount += 1

    # calculating final sentiment percentage
    calcCount = positiveCount - negativeCount
    per_sent_city = (calcCount / n_of_rwords) * 100
    return per_sent_city


# storing final sentiment percentage in variables
sent_jk = calc_sent("Jakarta.txt")
sent_bei = calc_sent("Beijing.txt")
sent_bk = calc_sent("Bangkok.txt")
sent_hk = calc_sent("Hongkong.txt")
sent_seo = calc_sent("Seoul.txt")
sent_tp = calc_sent("Taipei.txt")
sent_tok = calc_sent("Tokyo.txt")


# get all possible path
def possiblePath(citiesList):
    allPath = list(itertools.permutations(citiesList))
    return allPath


path = list(possiblePath(cities))
print("All possible routes for Ben: \n", path, "\n")

# Probability of a possible route
prob = (sent_jk + sent_bei + sent_bk + sent_hk + sent_seo + sent_tp + sent_tok)
print('The probability of a possible route: ', round(prob, 3), '%')

# storing locations in list
location = []
location.append([13.7538929, 100.8160803])  # Bangkok
location.append([39.906217, 116.3912757])  # Beijing
location.append([22.2793278, 114.1628131])  # Hongkong
location.append([-6.1753942, 106.827183])  # Jakarta
location.append([37.5666791, 126.9782914])  # Seoul
location.append([25.0375198, 121.5636796])  # Taiwan
location.append([35.6828387, 139.7594549])  # Tokyo
location.append([3.140853, 101.693207])  # Kuala-Lumpur


# calculating distance to check condition
def distance(x, y, x1, y1):
    firstcity = (x, y)
    secondcity = (x1, y1)
    value = geodesic(firstcity, secondcity).km
    return value


# listing cities to use as keys
cities = ['Bangkok', 'Beijing', 'Hongkong', 'Jakarta', 'Seoul', 'Taipei', 'Tokyo']

# storing sentiment results in list
sentimentlist = []
sentimentlist.append(sent_bk)
sentimentlist.append(sent_bei)
sentimentlist.append(sent_hk)
sentimentlist.append(sent_jk)
sentimentlist.append(sent_seo)
sentimentlist.append(sent_tp)
sentimentlist.append(sent_tok)

# minimum sentiment to be the best choice
minrange_for_best = max(sentimentlist)

# the coordinate of each cities
coordinate = {}
for i in range(len(cities)):
    coordinate[cities[i]] = location[i]

# storing distance for final calculation
for name in coordinate:
    for i in cities:
        dist = distance(coordinate[name][0], coordinate[name][1], coordinate[i][0], coordinate[i][1])
        # print(i,dist)

# get the sentiment analysis of each city
analysis = {}
for i in range(len(cities)):
    analysis[cities[i]] = sentimentlist[i]
print('Economic sentiment for each city: ', analysis)


# check sentiment analysis for first city to check if it is best or not
def check(analysis):
    minimum = minrange_for_best
    for city in analysis:
        if (analysis[city] >= minimum):
            minimum = analysis[city]
            best = city
    return best


best = check(analysis)


# print(best)

# check sentiment analysis for next city
def checkNext(analysis, cities):
    minimum = minrange_for_best
    for city in analysis:
        if (analysis[city] >= minimum and city in cities):
            minimum = analysis[city]
            best = city
    return best


# Verifying condition for 1st city
sentiment = minrange_for_best


def condition(coordinate, dist, city, nearest, sentiment, route, analysis):
    # Initializing given conditions
    minDiff = 2
    pathLength = 0.4 * dist
    bestCity = None

    # Iteration to verify with the conditions
    for name in coordinate:
        if (name != city and name != nearest and name not in route):
            value = distance(coordinate[name][0], coordinate[name][1], coordinate[city][0], coordinate[city][1])
            if (value < pathLength and abs(sentiment - analysis[name]) >= minDiff):
                pathLength = value
                minDiff = abs(sentiment - analysis)
                bestCity = name
    return bestCity


# Verifying condition for next cities
def nextCity(city, coordinate, analysis, route):
    # using a bigger value to store the next calculated distance as minimum
    minimum = 10000
    for name in coordinate:
        if (name != city and name not in route):
            dist = distance(coordinate[name][0], coordinate[name][1], coordinate[city][0], coordinate[city][1])
            if (dist < minimum):
                minimum = dist
                nearest = name
    # print(nearest)
    if nearest != checkNext(analysis, route):
        best = condition(coordinate, minimum, city, nearest, analysis[nearest], route, analysis)
        if (best == None):
            best = nearest
    else:
        best = nearest
    return best


# get the recommended path
def recommendPath(cities, analysis, coordinate):
    route = []
    route.append(check(analysis))
    while (len(route) != len(cities)):
        route.append(nextCity(route[-1], coordinate, analysis, route))
    return route


bestpath = recommendPath(cities, analysis, coordinate)
temp = bestpath.copy()
temp.reverse()
worstpath = temp.copy()
print("The most recommended path for Ben to take based on distance and sentiment:", bestpath)
print("The least recommended path for Ben to take based on distance and sentiment:", worstpath)


