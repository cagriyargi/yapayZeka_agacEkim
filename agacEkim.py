from contextlib import nullcontext
import matplotlib
import numpy as np
import pandas as py
import seaborn as sns
from pandas import read_csv
import matplotlib.pyplot as pyplot
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = read_csv("D:\\dersler\\oruntu\\oruntufinal\\18110131042\\kod\\data.csv")
data = dataset.drop(columns = ['SehirAdi'])

array = data.values
X1 = array[:,0:1] # Ağaç Yoğunluğu
X2 = array[:,1:2] # Nüfus Yoğunluğu
X3 = array[:,2:3] # Hava Kirliliği
Y = array[:,3:4]  # Sonuc

X1_train, X1_validation, Y1_train, Y1_validation = train_test_split(X1, Y, test_size=0.30) # trainkodları
X2_train, X2_validation, Y2_train, Y2_validation = train_test_split(X2, Y, test_size=0.30) # trainkodları
X3_train, X3_validation, Y3_train, Y3_validation = train_test_split(X3, Y, test_size=0.30) # trainkodları


#print(data)
print("###########################")
print("veri seti incelemesi")
print(data.info())         # Veri Setini İnceler
print("###########################")
print("veri seti istatistikler")
print(data.describe())     # Verinin İstatistiklerini Verir
print("###########################")
print("veri seti isNull?")
print(data.isnull().sum()) # Sütunlardaki Boş Değerlerin Sayısını Verir
print("###########################")
print("veri seti kor degerleri")
print(data.corr()) # Korelasyon Değerlerini Verir
print("###########################")
print("veri seti histogram")
print(data.hist()) # Histogram

######################################################################
######################################################################
### SCATTER GRAFİK ###################################################

data.plot(x="AgacYogunlugu", y="Sonuc", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
data.plot(x="NufusYogunlugu", y="Sonuc", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
data.plot(x="HavaKirliligi", y="Sonuc", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

###############################################################################################################
### KOMŞULUK ALGORİTMASI ######################################################################################
###############################################################################################################

# Modeli tanımladık
komsuModeli1 = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
komsuModeli2 = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
komsuModeli3 = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)

# Modeli eğittik
komsuModeli1.fit(X1_train, Y1_train)
komsuModeli2.fit(X2_train, Y2_train)
komsuModeli3.fit(X3_train, Y3_train)

# Sonucu aldık
Y_predictKomsu1 = komsuModeli1.predict(X1_validation)
Y_predictKomsu2 = komsuModeli2.predict(X2_validation)
Y_predictKomsu3 = komsuModeli3.predict(X3_validation)

# Hata Matrisinin Oluşturulması
komsuModeliMatrix1 = confusion_matrix(Y1_validation, Y_predictKomsu1)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(komsuModeliMatrix1, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('KNN Algoritması - Hata Matrisi(Ağaç Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

komsuModeliMatrix2 = confusion_matrix(Y2_validation, Y_predictKomsu2)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(komsuModeliMatrix2, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('KNN Algoritması - Hata Matrisi(Nüfus Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

komsuModeliMatrix3 = confusion_matrix(Y3_validation, Y_predictKomsu3)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(komsuModeliMatrix3, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('KNN Algoritması - Hata Matrisi(Hava Kirliliği)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

komsuSkoru1 = komsuModeli1.score(X1_validation, Y1_validation)
komsuSkoru2 = komsuModeli2.score(X2_validation, Y2_validation)
komsuSkoru3 = komsuModeli3.score(X3_validation, Y3_validation)
print("Komşuluk Algoritması Skoru(Ağaç Yoğunluğu): ", komsuSkoru1)
print("Komşuluk Algoritması Skoru(Nüfus Yoğunluğu): ", komsuSkoru2)
print("Komşuluk Algoritması Skoru(Hava Kirliliği): ", komsuSkoru3)

###############################################################################################################
### KARAR AĞACI ###############################################################################################
###############################################################################################################

# Modeli tanımladık
kararAgaci1 = DecisionTreeClassifier(random_state = 9)
kararAgaci2 = DecisionTreeClassifier(random_state = 9)
kararAgaci3 = DecisionTreeClassifier(random_state = 9)

# Modeli eğittik
kararAgaci1.fit(X1_train, Y1_train)
kararAgaci2.fit(X2_train, Y2_train)
kararAgaci3.fit(X3_train, Y3_train)

# Sonucu aldık
Y_predictAgac1 = kararAgaci1.predict(X1_validation)
Y_predictAgac2 = kararAgaci2.predict(X2_validation)
Y_predictAgac3 = kararAgaci3.predict(X3_validation)

# Hata Matrisinin Oluşturulması
agacMatrix1 = confusion_matrix(Y1_validation, Y_predictAgac1)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(agacMatrix1, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Karar Ağacı Algoritması - Hata Matrisi (Ağaç Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

agacMatrix2 = confusion_matrix(Y1_validation, Y_predictAgac2)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(agacMatrix2, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Karar Ağacı Algoritması - Hata Matrisi (Nüfus Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

agacMatrix3 = confusion_matrix(Y1_validation, Y_predictAgac3)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(agacMatrix3, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Karar Ağacı Algoritması - Hata Matrisi (Hava Kirliliği)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

kararAgaciSkor1 = kararAgaci1.score(X1_validation, Y1_validation)
kararAgaciSkor2 = kararAgaci2.score(X2_validation, Y2_validation)
kararAgaciSkor3 = kararAgaci3.score(X3_validation, Y3_validation)
print("Karar Ağacı Skoru(Ağaç Yoğunluğu): ", kararAgaciSkor1)
print("Karar Ağacı Skoru(Nüfus Yoğunluğu): ", kararAgaciSkor2)
print("Karar Ağacı Skoru(Hava Kirliliği): ", kararAgaciSkor3)

###############################################################################################################
### RASGELE ORMAN ALGORİTMASI #################################################################################
###############################################################################################################

# Modeli tanımladık
rOrman1 = RandomForestClassifier(n_estimators = 100, random_state = 9, n_jobs = -1)
rOrman2 = RandomForestClassifier(n_estimators = 100, random_state = 9, n_jobs = -1)
rOrman3 = RandomForestClassifier(n_estimators = 100, random_state = 9, n_jobs = -1)

# Modeli eğittik
rOrman1.fit(X1_train, Y1_train)
rOrman2.fit(X2_train, Y2_train)
rOrman3.fit(X3_train, Y3_train)

# Sonucu aldık
Y_predictOrman1 = rOrman1.predict(X1_validation)
Y_predictOrman2 = rOrman2.predict(X2_validation)
Y_predictOrman3 = rOrman3.predict(X3_validation)

# Hata Matrisinin Oluşturulması
ormanMatrix1 = confusion_matrix(Y1_validation, Y_predictOrman1)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(ormanMatrix1, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Rasgele Orman Algoritması - Hata Matrisi (Ağaç Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

ormanMatrix2 = confusion_matrix(Y2_validation, Y_predictOrman2)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(ormanMatrix2, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Rasgele Orman Algoritması - Hata Matrisi (Nüfus Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

ormanMatrix3 = confusion_matrix(Y3_validation, Y_predictOrman3)
f, ax = pyplot.subplots(figsize = (5, 5))
sns.heatmap(ormanMatrix3, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Rasgele Orman Algoritması - Hata Matrisi (Hava Kirliliği)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()


ormanSkor1 = rOrman1.score(X1_validation, Y1_validation)
ormanSkor2 = rOrman1.score(X2_validation, Y2_validation)
ormanSkor3 = rOrman1.score(X3_validation, Y3_validation)
print("Rasgele Orman Algoritması Skoru(Ağaç Yoğunluğu): ", ormanSkor1)
print("Rasgele Orman Algoritması Skoru(Nüfus Yoğunluğu): ", ormanSkor2)
print("Rasgele Orman Algoritması Skoru(Hava Kirliliği): ", ormanSkor3)

###############################################################################################################
### N. BAYES ALGORİTMASI ######################################################################################
###############################################################################################################

# Modeli tanımladık
nBayes1 = GaussianNB()
nBayes2 = GaussianNB()
nBayes3 = GaussianNB()

# Modeli eğittik
nBayes1.fit(X1_train, Y1_train)
nBayes2.fit(X2_train, Y1_train)
nBayes3.fit(X3_train, Y1_train)

# Sonucu aldık
Y_predictBayes1 = nBayes1.predict(X1_validation)
Y_predictBayes2 = nBayes2.predict(X2_validation)
Y_predictBayes3 = nBayes3.predict(X3_validation)

# Hata Matrisinin Oluşturulması
bayesMatrix1 = confusion_matrix(Y1_validation, Y_predictBayes1)
f, ax = pyplot.subplots(figsize=(5,5))
sns.heatmap(bayesMatrix1, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Naive Bayes Algoritması - Hata Matrisi (Ağaç Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

bayesMatrix2 = confusion_matrix(Y2_validation, Y_predictBayes2)
f, ax = pyplot.subplots(figsize=(5,5))
sns.heatmap(bayesMatrix2, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Naive Bayes Algoritması - Hata Matrisi (Nüfus Yoğunluğu)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

bayesMatrix3 = confusion_matrix(Y3_validation, Y_predictBayes3)
f, ax = pyplot.subplots(figsize=(5,5))
sns.heatmap(bayesMatrix3, annot = True, linewidth = 0.9, linecolor = 'red', fmt = 'g', ax = ax, cmap = 'rocket_r')
pyplot.title('Naive Bayes Algoritması - Hata Matrisi (Hava Kirliliği)')
pyplot.xlabel('Y Tahmin')
pyplot.ylabel('Y Test')
pyplot.show()

bayesSkor1 = nBayes1.score(X1_validation, Y1_validation)
bayesSkor2 = nBayes2.score(X2_validation, Y2_validation)
bayesSkor3 = nBayes3.score(X3_validation, Y3_validation)
print("N. Bayes Algoritması Skoru(Ağaç Yoğunluğu): ", bayesSkor1)
print("N. Bayes Algoritması Skoru(Nüfus Yoğunluğu): ", bayesSkor2)
print("N. Bayes Algoritması Skoru(Hava Kirliliği): ", bayesSkor3)

###############################################################################################################
### SKOR TABLOSU ##############################################################################################
###############################################################################################################

list_models1 = ["Komşuluk Algoritması", "Karar Ağacı", "Rasgele Orman Algoritması", "N. Bayes Algoritması"]
list_models2 = ["Komşuluk Algoritması", "Karar Ağacı", "Rasgele Orman Algoritması", "N. Bayes Algoritması"]
list_models3 = ["Komşuluk Algoritması", "Karar Ağacı", "Rasgele Orman Algoritması", "N. Bayes Algoritması"]
list_scores1 = [komsuSkoru1, kararAgaciSkor1, ormanSkor1, bayesSkor1]
list_scores2 = [komsuSkoru2, kararAgaciSkor2, ormanSkor2, bayesSkor2]
list_scores3 = [komsuSkoru3, kararAgaciSkor3, ormanSkor3, bayesSkor3]

pyplot.figure(figsize = (12, 4))
pyplot.bar(list_models1, list_scores1, width = 0.2, color = ['red', 'blue', 'brown', 'purple', 'orange'])
pyplot.title('Algoritma - Skor Oranı (Ağaç Yoğunluğu)')
pyplot.xlabel('Algoritmalar')
pyplot.ylabel('Skorlar')
pyplot.show()

pyplot.figure(figsize = (12, 4))
pyplot.bar(list_models2, list_scores2, width = 0.2, color = ['red', 'blue', 'brown', 'purple', 'orange'])
pyplot.title('Algoritma - Skor Oranı (Nüfus Yoğunluğu)')
pyplot.xlabel('Algoritmalar')
pyplot.ylabel('Skorlar')
pyplot.show()

pyplot.figure(figsize = (12, 4))
pyplot.bar(list_models3, list_scores3, width = 0.2, color = ['red', 'blue', 'brown', 'purple', 'orange'])
pyplot.title('Algoritma - Skor Oranı (Hava Kirliliği)')
pyplot.xlabel('Algoritmalar')
pyplot.ylabel('Skorlar')
pyplot.show()
