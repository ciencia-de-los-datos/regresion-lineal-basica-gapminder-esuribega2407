"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    #df = ____

    df = pd.read_csv(
        "gm_2008_region.csv",
        sep=',',  # separador de campos
        thousands=None,  # separador de miles para números
        decimal='.')

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    #y = ____[____].____
    #X = ____[____].____
    y = df["life"]
    X = df["fertility"]

    # Imprima las dimensiones de `y`
    #print(____.____)
    print(y.shape)

    # Imprima las dimensiones de `X`
    # print(____.____)
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    #y_reshaped = y.reshape(____, ____)
    y  = np.array(y)
    y_reshaped = y.reshape(len(y), 1)

    # Trasforme `X` a un array de numpy usando reshape
    #X_reshaped = X.reshape(____, ____)
    X = np.array(X)
    X_reshaped = X.reshape(len(X), 1)

    # Imprima las nuevas dimensiones de `y`
    #print(____.____)
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    #print(____.____)
    print(X_reshaped.shape)

def pregunta_02():
    """
    En este punto se realiza la impresión de algunas estadísticas básicas
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    #df = ____
    df = pd.read_csv(
        "gm_2008_region.csv",
        sep=',',  # separador de campos
        thousands=None,  # separador de miles para números
        decimal='.')

    # Imprima las dimensiones del DataFrame
    #print(____.____)
    print(df.shape)

    # Imprima la correlación entre las columnas `life` y `fertility` con 4 decimales.
    #print(____)
    #corr = round(np.corrcoef(df['life'],df['fertility'])[0][1],4)
    print(round(np.corrcoef(df['life'],df['fertility'])[0][1],4))

    # Imprima la media de la columna `life` con 4 decimales.
    #print(____)
    print(round(np.mean(df['life']),4))

    # Imprima el tipo de dato de la columna `fertility`.
    #print(____)
    print(type(df['fertility']))

    # Imprima la correlación entre las columnas `GDP` y `life` con 4 decimales.
    #print(____)
    print(round(np.corrcoef(df['GDP'], df['life'])[0][1], 4))


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el código presentado a continuación.
    """

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    #df = ____

    df = pd.read_csv(
        "gm_2008_region.csv",
        sep=',',  # separador de campos
        thousands=None,  # separador de miles para números
        decimal='.')

    # Asigne a la variable los valores de la columna `fertility`
    #X_fertility = ____
    X_fertility = np.array(df["fertility"]).reshape(len(df["fertility"]),1)

    # Asigne a la variable los valores de la columna `life`
    #y_life = ____
    y_life =  np.array(df["life"]).reshape(len(df["life"]),1)

    # Importe LinearRegression
    #from ____ import ____
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresión lineal
    #reg = ____
    reg = LinearRegression()


    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    #prediction_space = ____(
    #    ____,
    #    ____,
    #).reshape(____, _____)

    prediction_space = np.linspace(
        X_fertility.min(),
        X_fertility.max(),
    ).reshape(-1, 1)

    # Entrene el modelo usando X_fertility y y_life
    #reg.fit(____, ____)
    reg.fit(X_fertility, y_life)

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    #print(____.score(____, ____).round(____))
    print(reg.score(X_fertility, y_life).round(4))




def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from ____ import ____

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = ____

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = ____

    # Asigne a la variable los valores de la columna `life`
    y_life = ____

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) = ____(
        ____,
        ____,
        test_size=____,
        random_state=____,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = ____

    # Entrene el clasificador usando X_train y y_train
    ____.fit(____, ____)

    # Pronostique y_test usando X_test
    y_pred = ____

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
    rmse = np.sqrt(____(____, ____))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
