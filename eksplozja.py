"""Implementacja przykładu uczenia maszynowego;
    Symulacja eksplozji;
    Wykrywanie miejsca eksplozji"""

import numpy as np
import matplotlib.pyplot as plt
import random as r
import math
import pandas as pd

def lokalizacja_sensorow(n_sensorow, promien):
    """Wyznaczenie sensorów o współrzędnych
        położonych na okręgu o promieniu = 3 """
    tab = []
    k = 360 / n_sensorow #dzieki temu sensory beda rowno rozlozone na okregu
    for x in range(0, n_sensorow):
        tab1 = []
        tab1.append(promien * np.cos(k*x*(math.pi/180)))   #dodawanie do listy wspolrzednych na okregu
        tab1.append(promien * np.sin(k*x*(math.pi/180)))
        tab.append(tab1)
    return tab

def frange(start, stop, krok):
    """metoda range dla liczb typu float"""
    #range dziala tylko dla liczb calkowitych
    while start < stop:
        yield start
        start += krok

def lokalizacja_eksplozji(n, krok):
    """Wyznaczenie losowych współrzędnych dla eksplozji wewnatrz okręgu"""
    wspolrzedne_eksplozji = []
    tab = []
    wspol_x = []
    wspol_y = []
    for x in frange(-n, n+1, krok):  #wykorzstanie funkcji frange dla liczb typu float
        wspol_x.append(x)
        for y in frange(-n, n+1, krok):
            if math.sqrt(pow(x,2) + pow(y,2)) < n :  #Z twierdzenia Pitagorasa, wyznaczenie wspołrzednych wewnątrz okręgu o promieniu n
                tab.append(x)
                tab.append(y)
                wspolrzedne_eksplozji.append(tab)
                tab =[]
    wspol_y = wspol_x
    df = pd.DataFrame(index=wspol_y, columns=wspol_x) #Stworzenie obiektu DataFrame z kolumnami jako wspolrzedne x, wierszami jako wspolrzedne y
    miejsce = r.choice(wspolrzedne_eksplozji)         #losowe wybranie wspolrzednych eksplozji z listy wszystkich wspolrzednych dla danego kroku w obrebie danego okregu
    return miejsce, wspolrzedne_eksplozji, df

def sila_odbierana_przez_sensor(tab, e, ro):
    """ Obliczenie
    a) odległości d2 pomiędzy sensorem a eksplozją
    b) siły v odbieranej przez sensor
    c) zastosowanie rozkładu Gaussa"""
    v = []
    d2_tab = []
    for n in range(0, len(tab)):
        x = tab[n][0]
        y = tab[n][1]
        d2 = (pow(x - e[0], 2) + pow(y - e[1], 2)) #oblicznie odległości
        d2_tab.append(d2)
        v.append(1/(d2 + 0.1))   #oblicznie siły
    prawdo = dystrybucja(d2_tab, v, ro)   #wywolanie funkcji obliczajacej rozklad prawopodobienstwa
    return v, d2_tab, prawdo


def dystrybucja(d2_tab, v, ro):
    """ Zastosowanie modelu probabilistycznego (rozkładu Gaussa), biorącego pod uwagę niepewnośc """
    tablica = []
    for x in range(0, len(v)):
        prawdo = (1 / (math.sqrt(2 * math.pi * ro)) *  np.exp(-1 / (2 * ro) * pow(v[x] - (1 / (d2_tab[x] + 0.1)), 2))) #zaimplementowanie wzoru na rozklad gaussa
        tablica.append(prawdo)
    return tablica


def tablica_prawdopodobienstwa_dla_wspol_wewnatrz_kola(wspol, sensory, df, v, ro):
    """Obliczanie rozkładu prawdopodobienstwa dla roznych wspolrzednych
       Zapisywanie wartosci prawdopoobientwa do obiektu DataFrame"""
    p_v_e  = 1
    tablica = []
    for punkt in wspol:
        for x, sensor in enumerate(sensory):
            d2 = pow(sensor[0] - punkt[0], 2) + pow(sensor[1] - punkt[1], 2)
            prawdo = ((1 / (math.sqrt(2 * math.pi * ro))) *  np.exp((-1 / (2 * ro)) * pow(v[x] - (1 / d2 + 0.1), 2))) #zaimplementowanie wzoru na rozklad gaussa
            p_v_e = p_v_e * prawdo
        tablica.append(p_v_e)
        p_v_e = 1
    for x in range(0, len(tablica))  : #Pętla mająca na celu wypelnienie obiektu DataFrame wartosciamu prawdopodobienstwa dla danych wspolrzenych
        df[wspol[x][0]][wspol[x][1]] = tablica[x]
    df = df.fillna(0)                   #zamienienie wartosci NaN w tabeli na 0.0
    #df.to_csv('prawdo.csv')
    return df

def rysuj_wykres(tab, e, wspol):
    """ Wykres poglądowy symulacji.
        Na czerwono zaznaczone są położenia sensorów
        Na zielono zaznaczone są współrzędne wewnątrz okręgu, możliwe miejsca eksplozji
        Na niebiesko - miejsce zaistniałej eksplozji"""
    #wykres tworzony pomocniczo przy pisaniu kodu, jest to opcja wizualizacji sensorow, wspolrzednych i miejsca eksplozji na wykresie
    #Jako że był tworzony pomocniczo, nie jest wywolywany
    os_x = []
    os_y = []
    x_wspol =[]
    y_wspol = []
    for x in range(0, len(tab)): #zapisanie wspolrzednych sensorow x i y do oddzielnych list
        os_x.append(tab[x][0])
        os_y.append(tab[x][1])
    for x in range(0, len(wspol)): #zapisanie wspolrzenych mozliwego miesjca wybuchu, x i y do oddzielnych list
        x_wspol.append(wspol[x][0])
        y_wspol.append(wspol[x][1])
    p = plt.plot(x_wspol, y_wspol, 'go', os_x, os_y, 'ro', e[0], e[1], 'bo') #budowanie wykresu
    plt.show()
    return 0


def rysuj_wykres2(df, miejsce):
    """ Pokazanie na wykresie rozkładu prawdopodobienstwa na zajscie eksplozji dla współrzędnych"""
    wspol_x = df.index
    wspol_x = np.array(wspol_x)
    wspol_y = np.array(wspol_x)
    plt.plot(miejsce[0], miejsce[1], 'ro') #drukowanie miejsca zaistnialej eksplozji jako czerwonej kropki na wykresie
    plt.contourf(wspol_x, wspol_y, df, 20) #graficzne przedstawienie obliczonego rozkladu prawdopodobienstwa na zajscie eksplozji
    plt.show()
    return 0

def wyznaczenie_miejsca_eksplozji(df, krok):
    """Funkcja zwraca wspolrzedne dla najwyzszej wartosci prawdopoobienstwa"""
    m = df.max()
    tab = df.index
    tab1 = []
    wspolrzedne = [0,0]
    maks =0
    for x in frange(tab[0], tab[len(tab)-1], krok): #wykorzystanie funkcji frange jako range dla liczb float
        if df[x].max() > maks:
            maks = df[x].max()
            wspolrzedne[0] = x                #zapisywanie wspolrzednych dla najwyzszej wartosci prawdopodobientwa w obiekcie dataframe
            wspolrzedne[1] = df[x].idxmax()
    return wspolrzedne


#inicjowanie zmiennych
n_sensorow = 200
promien_kola = 5
krok = 0.5
ro = 0.1 #siła zaburzenia

#wywoływanie funkcji
tab = lokalizacja_sensorow(n_sensorow, promien_kola)
miejsce, wspolrzedne, df = lokalizacja_eksplozji(promien_kola, krok)
v, d2_tab, prawdo = sila_odbierana_przez_sensor(tab, miejsce, ro)
#rysuj_wykres(tab, miejsce, wspolrzedne)
df = tablica_prawdopodobienstwa_dla_wspol_wewnatrz_kola(wspolrzedne, tab, df, v, ro)
rysuj_wykres2(df, miejsce)
print("Eksplozja wydarzyła się w: ", miejsce)
print("Eksplozja została przewidziana w: ", wyznaczenie_miejsca_eksplozji(df, krok))

