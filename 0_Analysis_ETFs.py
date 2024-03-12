
import numpy as np
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

import streamlit as st


# model = load_model('C:/Python/Stock/Stock Predictions Model.keras')
###############################################
# DEFINICIONES
###############################################

# Algoritmo para detectar tendencias basado en EMAs (Average Exponential Moving).
# EMAsignal es una lista que contiene señales de tendencia calculadas para cada fila en el DataFrame df.
# Backcandles define el número de puntos que va a revisar aguas arriba y abajo del punto analizado para ver si se trata de:
#
# 0: No hay señal de tendencia definida.
# 1: Señal de tendencia bajista.
# 2: Señal de tendencia alcista.
# 3: Señal de tendencia neutral o sin definir (ni alcista ni bajista).

def trend_detection(df, backcandles):
    df = df[df['volume'] != 0]
    df.reset_index(drop=True, inplace=True)

    df['EMA'] = ta.ema(df.close, length=50)
    df.tail()

    df = df[0:]
    df.reset_index(drop=True, inplace=True)

    EMAsignal = [0] * len(df)
    for row in range(backcandles, len(df)):
        upt = 1
        dnt = 1
        for i in range(row - backcandles, row + 1):
            if max(df.open[i], df.close[i]) >= df.EMA[i]:
                dnt = 0
            if min(df.open[i], df.close[i]) <= df.EMA[i]:
                upt = 0
        if upt == 1 and dnt == 1:
            EMAsignal[row] = 3
        elif upt == 1:
            EMAsignal[row] = 2
        elif dnt == 1:
            EMAsignal[row] = 1

    df['EMASignal'] = EMAsignal
    return df


# Esta función isPivot es una función que detecta si una vela es un punto pivote o fractal en un conjunto de datos.
# Recibe dos argumentos:
# candle: El índice de la vela que se está considerando como un posible punto pivote.
# window: El número de velas antes y después de la vela candle que se deben considerar al evaluar si es un punto pivote.
# Devuelve un valor entero que indica el tipo de punto pivote detectado: 0, 1, 2, 3

def isPivot(candle, window):
    if candle - window < 0 or candle + window >= len(df):
        return 0

    pivothigh = 1
    pivotlow = 2
    for i in range(candle - window, candle + window + 1):
        if df.iloc[candle].low > df.iloc[i].low:
            pivotlow = 0
        if df.iloc[candle].high < df.iloc[i].high:
            pivothigh = 0
    if (pivothigh and pivotlow):
        return 3
    elif pivothigh:
        return pivothigh
    elif pivotlow:
        return pivotlow
    else:
        return 0

 # Genera un punto elevado desplazado 1e-3 del pivot low y el pivot high

def pointpos(x):
    if x['isPivot']==2:
        return x['low']-1e-3
    elif x['isPivot']==1:
        return x['high']+1e-3
    else:
        return np.nan

################################################
st.header('Stock Market Analysis & Predictor')
################################################

num_symbols = st.number_input('Number of Stock Symbols', min_value=1, step=1, value=5)

# dic_symbols = {'VUAA.F': 'S&P500 (Vanguard)',
#                'VWCE.F': 'FTSE-All World (Vanguard)',
#                'XAIX.F': 'AI & Big Data (Xtrackers)',
#                'SEC0.DE': 'MSCI Global Semiconductors (iShares)',
#                'PRAW.DE': 'Prime Global (Amundi)'}


default_symbols = [
    ('VUAA.F', 'S&P500 (Vanguard)'),
    ('SEC0.DE', 'MSCI Global Semiconductors (iShares)'),
    ('XAIX.F', 'AI & Big Data (Xtrackers)'),
    ('PRAW.DE', 'Prime Global (Amundi)'),
    ('VWCE.F', 'FTSE-All World (Vanguard)')
]

symbols = []
for i in range(num_symbols):
    symbol_name = st.text_input(f'Stock Symbol and Nickname {i+1}',
                                value=default_symbols[i][0] + ': ' + default_symbols[i][1],
                                key=f'symbol_{i}')
    symbols.append(symbol_name)

st.subheader('Selected Stock Symbols')
dic_symbols = {}
for symbol in symbols:
    if symbol:
        s, n = symbol.split(':')
        dic_symbols[s.strip()] = n.strip()

st.write(dic_symbols)

################################################
st.subheader('Precio histórico VS MA60')
################################################

fig, axs = plt.subplots(len(dic_symbols), 1, figsize=(12, 25), dpi=300)
df_stock = {}

for idx, (symbol, name) in enumerate(dic_symbols.items()):
    data = yf.Ticker(symbol)
    data = data.history(period="max")
    data.columns = data.columns.str.lower()

    # Media móvil de 60 días
    ma_60_days = data.close.rolling(60).mean()

    # Guardamos los datos que querremos analizar más adelante
    df_stock[symbol] = data
    
    # Representamos gráficamente las curvas
    data.plot(y='close', color='b', ax=axs[idx], label='close')
    ma_60_days.plot(ax=axs[idx], label='ma_60_days', color='r')
    mean_close = data['close'].mean()  # Calcula el valor medio de 'close' para esta acción
    axs[idx].axhline(mean_close, color='olive', linestyle='--', label='Media_close')
    # Reemplaza el símbolo con el nombre correspondiente del diccionario
    axs[idx].set_title(name)  # Título de la subfigura
    axs[idx].legend()  # Agrega la leyenda al subgráfico

plt.tight_layout()  # Ajusta automáticamente el diseño de los subgráficos para que quepan correctamente
st.pyplot(fig)

########################################
st.subheader('Último día registrado')
for sym, name in dic_symbols.items():
    st.write(df_stock[sym].iloc[-1].name, sym, name, df_stock[sym]['close'].iloc[-1])

################################################
st.subheader('Detector de puntos de inflexibilidad')
################################################

for sym, name in dic_symbols.items():
    data = df_stock[sym]
    # Detección de tendencias
    df = trend_detection(data, 10)
    # Corrobora el tipo de tendencia
    ventana = 5
    df['isPivot'] = df.apply(lambda x: isPivot(x.name, ventana), axis=1)
    # Grafica los puntos en un diagrama
    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

    fig2 = go.Figure(data=[go.Candlestick(x=df.index,
                                          open=df['open'],
                                          high=df['high'],
                                          low=df['low'],
                                          close=df['close'])])

    fig2.add_scatter(x=df.index, y=df['pointpos'], mode="markers",
                     marker=dict(size=5, color="MediumPurple"),
                     name="Pivote")

    fig2.update_layout(xaxis_rangeslider_visible=False,
                       xaxis_title="Días",
                       yaxis_title="Precio de la acción en €",
                       title=f'Evolución del precio € de {name}',
                       font=dict(
                           family="Courier New, monospace",
                           size=14,
                           color="RebeccaPurple")
                       )

    st.plotly_chart(fig2, use_container_width=True)

########################################
# Cálculo de rentabilidad
########################################

df_rent_anual = pd.DataFrame()
df_rent_mensual = pd.DataFrame()

df_rent_mensual_prom = pd.DataFrame()
# Define un diccionario que mapea los nombres de los meses a sus números correspondientes
month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
              'Nov': 11, 'Dec': 12}

for sym, name in dic_symbols.items():
    df = df_stock[sym].copy()

    df['date'] = df.index.strftime('%Y-%m-%d')
    df['month'] = df.index.strftime("%b")
    df['year'] = df.index.year
    df['day'] = df.index.day

    df['ts'] = df['month'] + '_' + df['year'].astype(str)

    # Calculo rentabilidades anuales históricas
    for y in df['year'].unique():
        df_i = df[df['year'] == y].copy()

        df_rent_i = pd.DataFrame({
            'ETF': name,
            'year': y,
            'rent_anual': (df_i.iloc[-1]['close'] - df_i.iloc[0]['close']) / df_i.iloc[0]['close'] * 100
        }, index=[0])

        df_rent_anual = pd.concat([df_rent_anual, df_rent_i], ignore_index=True)

    # Cálculo rentabilidades mensuales históricas
    for ts in df['ts'].unique():
        df_i = df[df['ts'] == ts].copy()

        df_rent_i = pd.DataFrame({
            'ETF': name,
            'year': ts[-4:],
            'month': ts[:3],
            'ts': ts,
            'rent_mensual': (df_i.iloc[-1]['close'] - df_i.iloc[0]['close']) / df_i.iloc[0]['close'] * 100
        }, index=[0])

        df_rent_mensual = pd.concat([df_rent_mensual, df_rent_i], ignore_index=True)

    df_rent_mensual_prom_i = df_rent_mensual[df_rent_mensual['ETF'] == name][['ETF', 'month', 'rent_mensual']].groupby(
        ['ETF', 'month']).mean().round(2).reset_index()
    df_rent_mensual_prom_i['month_number'] = df_rent_mensual_prom_i['month'].map(month_dict)
    df_rent_mensual_prom_i = df_rent_mensual_prom_i.sort_values(by='month_number')
    df_rent_mensual_prom_i = df_rent_mensual_prom_i.drop(columns='month_number')

    df_rent_mensual_prom = pd.concat([df_rent_mensual_prom, df_rent_mensual_prom_i], ignore_index=True)

########################################
# Graficos comparativos rentabilidades ETFs
########################################

st.subheader('Rentabilidad anual (%)')
########

# Redondear el valor de 'rent_anual' al segundo decimal
df_rent_anual['rent_anual'] = df_rent_anual['rent_anual'].round(2)

# Configurar el estilo y el contexto del gráfico con Seaborn
sns.set_style("whitegrid")
sns.set_context("notebook")

# Crear el gráfico de barras utilizando Seaborn
fig = plt.figure(figsize=(20, 10), dpi=300)
ax = sns.barplot(data=df_rent_anual, x='year', y='rent_anual', hue='ETF', palette='colorblind')

# Rotar las etiquetas del eje x para mayor legibilidad
plt.xticks(rotation=0)

# Agregar etiquetas a cada valor
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom',
                fontsize=12, color='black', xytext=(0, 12), textcoords='offset points')

# Establecer título y leyenda del gráfico
plt.title('Rentabilidades anuales por ETF')
plt.legend(title='ETF', loc='upper left')

# Mostrar el gráfico
plt.tight_layout()
# plt.show()
st.pyplot(fig)

##########################################
st.subheader('Rentabilidad mensual (%)')
##########################################

# Redondear el valor de 'rent_mensual' al segundo decimal
df_rent_mensual['rent_mensual'] = df_rent_mensual['rent_mensual'].round(2)

# Obtener los años únicos en el DataFrame
years = sorted(df_rent_mensual['year'].unique())

# Configurar el estilo y el contexto del gráfico con Seaborn
sns.set_style("whitegrid")
sns.set_context("notebook")

# Crear una figura y ejes de subgráficos para cada año
fig, axs = plt.subplots(len(years), 1, figsize=(12, 25), dpi=300, sharex=False)

# Iterar sobre cada año y graficar los datos correspondientes
for i, year in enumerate(years):
    # Filtrar el DataFrame para el año actual
    df_year = df_rent_mensual[df_rent_mensual['year'] == year]

    # Crear el gráfico de barras para el año actual
    sns.barplot(data=df_year, x='ts', y='rent_mensual', hue='ETF', palette='colorblind', ax=axs[i])

    # Rotar las etiquetas del eje x para mayor legibilidad
    axs[i].tick_params(axis='x', rotation=90)

    # Agregar etiquetas a cada valor
    for p in axs[i].patches:
        axs[i].annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom',
                        fontsize=8, color='black', xytext=(0, 5), textcoords='offset points')

    # Establecer título y leyenda del gráfico
    axs[i].set_title(f'Rentabilidades mensuales por ETF - Año {year}')
    axs[i].legend(title='ETF', loc='upper left')

# Ajustar diseño de la figura
plt.tight_layout()
st.pyplot(fig)
# -------------------------------------------------------------------
# Gráfico de las rentabilidades mensuales promedio
st.subheader('Average Monthly Profitability (%)')
fig = plt.figure(figsize=(20, 10), dpi=300)
sns.barplot(data=df_rent_mensual_prom, x='month', y='rent_mensual', hue='ETF', palette='colorblind')

# Agregar etiquetas a cada valor
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom',
                       fontsize=12, color='black', xytext=(0, 12), textcoords='offset points')

# Establecer título y leyenda del gráfico
plt.title('Rentabilidades mensuales promediadas por ETF')
plt.legend(title='ETF', loc='upper left')
# Ajustar el diseño
plt.tight_layout()
st.pyplot(fig)

########################################
# Cálculo volatilidades máximas y mínimas ETFs
########################################

df_vol_anual = pd.DataFrame()
df_vol_mensual = pd.DataFrame()
df_vol_semanal = pd.DataFrame()

top_vol_mensual = pd.DataFrame()
top_vol_semanal = pd.DataFrame()

for sym, name in dic_symbols.items():
    df = df_stock[sym].copy()

    df['date'] = df.index.strftime('%Y-%m-%d')
    df['month'] = df.index.strftime("%b")
    df['year'] = df.index.year
    df['day'] = df.index.day
    df['week'] = df.index.isocalendar().week

    df['ts'] = df['month'] + '_' + df['year'].astype(str)
    df['ts_week'] = df['ts'] + '_' + df['week'].astype(str)

    df['ETF'] = name

    # Calculo volatilidad ANUAL históricas
    df_vol_min = df[['ETF', 'year', 'low']].groupby(['ETF', 'year']).min().reset_index()
    df_vol_max = df[['ETF', 'year', 'high']].groupby(['ETF', 'year']).max().reset_index()

    df_vol_anual_i = df_vol_min.merge(df_vol_max, how='inner', on=['ETF', 'year'])

    df_vol_anual_i['price_day_ref'] = [df[df['year'] == y].iloc[0]['close'].round(2) for y in
                                       df_vol_anual_i['year'].unique()]

    # Cálculo porcentaje volatilidad respecto al primer día de la referencia
    df_vol_anual_i['€_min'] = (df_vol_anual_i['low'] - df_vol_anual_i['price_day_ref'])
    df_vol_anual_i['€_max'] = (df_vol_anual_i['high'] - df_vol_anual_i['price_day_ref'])

    df_vol_anual_i['%_min'] = (df_vol_anual_i['low'] - df_vol_anual_i['price_day_ref']) / df_vol_anual_i[
        'price_day_ref'] * 100
    df_vol_anual_i['%_max'] = (df_vol_anual_i['high'] - df_vol_anual_i['price_day_ref']) / df_vol_anual_i[
        'price_day_ref'] * 100

    # Almacenamos todas las volatilidades de todos los ETFs
    df_vol_anual = pd.concat([df_vol_anual, df_vol_anual_i], ignore_index=True)

    st.write(
        f'El rango de volatilidad ANUAL histórico del {name} es '
        f'{round(df_vol_anual_i["%_min"].min(), 2)} - {round(df_vol_anual_i["%_max"].max(), 2)} %')

    # Calculo volatilidad MENSUAL históricas
    df_vol_min = df[['ETF', 'ts', 'low']].groupby(['ETF', 'ts']).min().reset_index()
    df_vol_max = df[['ETF', 'ts', 'high']].groupby(['ETF', 'ts']).max().reset_index()

    df_vol_mensual_i = df_vol_min.merge(df_vol_max, how='inner', on=['ETF', 'ts'])

    df_vol_mensual_i['price_day_ref'] = [df[df['ts'] == y].iloc[0]['close'].round(2) for y in
                                         df_vol_mensual_i['ts'].unique()]

    # Cálculo porcentaje volatilidad respecto al primer día de la referencia
    df_vol_mensual_i['€_min'] = (df_vol_mensual_i['low'] - df_vol_mensual_i['price_day_ref'])
    df_vol_mensual_i['€_max'] = (df_vol_mensual_i['high'] - df_vol_mensual_i['price_day_ref'])

    df_vol_mensual_i['%_min'] = (df_vol_mensual_i['low'] - df_vol_mensual_i['price_day_ref']) / df_vol_mensual_i[
        'price_day_ref'] * 100
    df_vol_mensual_i['%_max'] = (df_vol_mensual_i['high'] - df_vol_mensual_i['price_day_ref']) / df_vol_mensual_i[
        'price_day_ref'] * 100

    # Almacenamos todas las volatilidades de todos los ETFs
    df_vol_mensual = pd.concat([df_vol_mensual, df_vol_mensual_i], ignore_index=True)

    st.write(
        f'El rango de volatilidad MENSUAL histórico del {name} es '
        f'{round(df_vol_mensual_i["%_min"].min(), 2)} - {round(df_vol_mensual_i["%_max"].max(), 2)} %')
    st.write(df_vol_mensual[df_vol_mensual['ETF'] == name])

    top_min_indices = df_vol_mensual_i.nsmallest(20, '€_min')[['ETF', 'ts', 'price_day_ref', '€_min', '%_min']]
    top_max_indices = df_vol_mensual_i.nlargest(20, '€_max')[['ETF', 'ts', 'price_day_ref', '€_max', '%_max']]

    top_min_indices['type'] = '€_min'
    top_max_indices['type'] = '€_max'
    top_min_indices.rename(columns={'€_min': 'Volatilidad (€)', '%_min': 'Volatilidad (%)'}, inplace=True)
    top_max_indices.rename(columns={'€_max': 'Volatilidad (€)', '%_max': 'Volatilidad (%)'}, inplace=True)

    top_vol_mensual_i = pd.concat([top_min_indices, top_max_indices], ignore_index=True)
    top_vol_mensual = pd.concat([top_vol_mensual, top_vol_mensual_i], ignore_index=True)

    # Calculo volatilidad SEMANAL históricas
    df_vol_min = df[['ETF', 'ts_week', 'low']].groupby(['ETF', 'ts_week']).min().reset_index()
    df_vol_max = df[['ETF', 'ts_week', 'high']].groupby(['ETF', 'ts_week']).max().reset_index()

    df_vol_semanal_i = df_vol_min.merge(df_vol_max, how='inner', on=['ETF', 'ts_week'])
    df_vol_semanal_i['price_day_ref'] = [df[df['ts_week'] == y].iloc[0]['close'].round(2) for y in
                                         df_vol_semanal_i['ts_week'].unique()]

    # Cálculo porcentaje volatilidad respecto al primer día de la referencia
    df_vol_semanal_i['€_min'] = (df_vol_semanal_i['low'] - df_vol_semanal_i['price_day_ref'])
    df_vol_semanal_i['€_max'] = (df_vol_semanal_i['high'] - df_vol_semanal_i['price_day_ref'])

    df_vol_semanal_i['%_min'] = (df_vol_semanal_i['low'] - df_vol_semanal_i['price_day_ref']) / df_vol_semanal_i[
        'price_day_ref'] * 100
    df_vol_semanal_i['%_max'] = (df_vol_semanal_i['high'] - df_vol_semanal_i['price_day_ref']) / df_vol_semanal_i[
        'price_day_ref'] * 100

    # Almacenamos todas las volatilidades de todos los ETFs
    df_vol_semanal = pd.concat([df_vol_semanal, df_vol_semanal_i], ignore_index=True)

    st.write(
        f'El rango de volatilidad SEMANAL histórico del {name} es '
        f'{round(df_vol_semanal_i["%_min"].min(), 2)} - {round(df_vol_semanal_i["%_max"].max(), 2)} %')
    st.write(df_vol_semanal[df_vol_semanal['ETF'] == name])

    top_min_indices = df_vol_semanal_i.nsmallest(20, '€_min')[['ETF', 'ts_week', 'price_day_ref', '€_min', '%_min']]
    top_max_indices = df_vol_semanal_i.nlargest(20, '€_max')[['ETF', 'ts_week', 'price_day_ref', '€_max', '%_max']]

    top_min_indices['type'] = '€_min'
    top_max_indices['type'] = '€_max'
    top_min_indices.rename(columns={'€_min': 'Volatilidad (€)', '%_min': 'Volatilidad (%)'}, inplace=True)
    top_max_indices.rename(columns={'€_max': 'Volatilidad (€)', '%_max': 'Volatilidad (%)'}, inplace=True)

    top_vol_semanal_i = pd.concat([top_min_indices, top_max_indices], ignore_index=True)
    top_vol_semanal = pd.concat([top_vol_semanal, top_vol_semanal_i], ignore_index=True)

########################################
# Gráficos volatilidades máximas y mínimas ETFs
########################################
################################################
st.subheader('Volatilidades mensuales')
################################################
# Configurar el estilo y el contexto del gráfico con Seaborn
sns.set_style("whitegrid")
sns.set_context("notebook")

# Crear una figura y ejes de subgráficos para cada ETF
fig, axs = plt.subplots(len(dic_symbols), 1, figsize=(12, 25), dpi=300, sharex=False)

for idx, (sym, name) in enumerate(dic_symbols.items()):

    df_etf_vol = df_vol_mensual[df_vol_mensual['ETF'] == name].copy()
    df_etf = top_vol_mensual[top_vol_mensual['ETF'] == name].copy()

    df_etf['Volatilidad (€)'] = df_etf['Volatilidad (€)'].round(2)

    # Asignar los valores repetidos a la columna 'x'
    valores_x = np.arange(1, 21)
    df_etf['x'] = np.tile(valores_x, len(df_etf) // len(valores_x) + 1)[:len(df_etf)]

    # Crear el gráfico de barras para el tipo actual
    sns.barplot(data=df_etf, x='x', y='Volatilidad (€)', hue='type', palette='colorblind', ax=axs[idx])

    # Calcular el cuantil
    quantile_max = df_etf_vol['€_max'].quantile(0.65)
    axs[idx].axhline(quantile_max, color='g', linestyle='--', label=f'Quantile 65: {quantile_max:.2f}')

    quantile_min = df_etf_vol['€_min'].quantile(0.35)
    axs[idx].axhline(quantile_min, color='r', linestyle='--', label=f'Quantile 35: {quantile_min:.2f}')

    st.write(f'{name}: \n - Cuantil €_max {quantile_max:.2f} \n  - Cuantil €_min {quantile_min:.2f}')
    st.write(f'Aviso precio COMPRA de acciones: {df_stock[sym]["close"].iloc[-1] + quantile_min:.2f} €')
    st.write(f'Aviso precio VENTA de acciones: {df_stock[sym]["close"].iloc[-1] + quantile_max:.2f} €')

    axs[idx].tick_params(axis='x')

    # Agregar etiquetas a cada valor
    for p in axs[idx].patches:
        axs[idx].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                          va='bottom', fontsize=8, color='black', xytext=(0, 5), textcoords='offset points')

    # Establecer título y leyenda del gráfico
    axs[idx].set_title(f'TOP 20 Volatilidades mensuales por ETF - {name}')
    axs[idx].legend(title='ETF', loc='upper left')

# Ajustar diseño de la figura
plt.tight_layout()
st.pyplot(fig)
# ------------------------------------------------------------------------------------------------------------
st.subheader('Volatilidades semanales')

# Configurar el estilo y el contexto del gráfico con Seaborn
sns.set_style("whitegrid")
sns.set_context("notebook")

# Crear una figura y ejes de subgráficos para cada ETF
fig, axs = plt.subplots(len(dic_symbols), 1, figsize=(12, 25), dpi=300, sharex=False)

for idx, (sym, name) in enumerate(dic_symbols.items()):

    df_etf_vol = df_vol_semanal[df_vol_semanal['ETF'] == name].copy()
    df_etf = top_vol_semanal[top_vol_semanal['ETF'] == name].copy()

    df_etf['Volatilidad (€)'] = df_etf['Volatilidad (€)'].round(2)

    # Asignar los valores repetidos a la columna 'x'
    valores_x = np.arange(1, 21)
    df_etf['x'] = np.tile(valores_x, len(df_etf) // len(valores_x) + 1)[:len(df_etf)]

    # Crear el gráfico de barras para el tipo actual
    sns.barplot(data=df_etf, x='x', y='Volatilidad (€)', hue='type', palette='colorblind', ax=axs[idx])

    # Calcular el cuantil 15
    quantile_max = df_etf_vol['€_max'].quantile(0.85)
    axs[idx].axhline(quantile_max, color='g', linestyle='--', label=f'Quantile 85: {quantile_max:.2f}')

    quantile_min = df_etf_vol['€_min'].quantile(0.15)
    axs[idx].axhline(quantile_min, color='r', linestyle='--', label=f'Quantile 15: {quantile_min:.2f}')

    st.write(f'{name}: \n - Cuantil €_max {quantile_max:.2f} \n  - Cuantil €_min {quantile_min:.2f}')
    st.write(f'Aviso precio COMPRA de acciones: {df_stock[sym]["close"].iloc[-1] + quantile_min:.2f} €')
    st.write(f'Aviso precio VENTA de acciones: {df_stock[sym]["close"].iloc[-1] + quantile_max:.2f} €')

    axs[idx].tick_params(axis='x')

    # Agregar etiquetas a cada valor
    for p in axs[idx].patches:
        axs[idx].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                          va='bottom', fontsize=8, color='black', xytext=(0, 5), textcoords='offset points')

    # Establecer título y leyenda del gráfico
    axs[idx].set_title(f'TOP 20 Volatilidades semanales por ETF - {name}')
    axs[idx].legend(title='ETF', loc='upper left')

# Ajustar diseño de la figura
plt.tight_layout()
st.pyplot(fig)
