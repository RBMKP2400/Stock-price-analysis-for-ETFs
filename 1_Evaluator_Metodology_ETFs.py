

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

import streamlit as st
# import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


################################################
st.header('Stock Market Analysis & Predictor')
################################################

num_symbols = st.number_input('Number of Stock Symbols', min_value=1, step=1, value=5)

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
st.subheader('Historical price VS MA60')
################################################
df_stock = {}

for idx, (symbol, name) in enumerate(dic_symbols.items()):
    data = yf.Ticker(symbol)
    data = data.history(period="max")
    data.columns = data.columns.str.lower()

    # Guardamos los datos que querremos analizar más adelante
    df_stock[symbol] = data

########################################
st.subheader('Last day registered')
for sym, name in dic_symbols.items():
    st.write(df_stock[sym].iloc[-1].name, sym, name, df_stock[sym]['close'].iloc[-1])

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

    top_min_indices = df_vol_semanal_i.nsmallest(20, '€_min')[['ETF', 'ts_week', 'price_day_ref', '€_min', '%_min']]
    top_max_indices = df_vol_semanal_i.nlargest(20, '€_max')[['ETF', 'ts_week', 'price_day_ref', '€_max', '%_max']]

    top_min_indices['type'] = '€_min'
    top_max_indices['type'] = '€_max'
    top_min_indices.rename(columns={'€_min': 'Volatilidad (€)', '%_min': 'Volatilidad (%)'}, inplace=True)
    top_max_indices.rename(columns={'€_max': 'Volatilidad (€)', '%_max': 'Volatilidad (%)'}, inplace=True)

    top_vol_semanal_i = pd.concat([top_min_indices, top_max_indices], ignore_index=True)
    top_vol_semanal = pd.concat([top_vol_semanal, top_vol_semanal_i], ignore_index=True)


################################
# Evaluación de las metodologías
################################
# Definir función para calcular los datos y generar el gráfico
@st.cache_data
def calculos_metodologia(sym, name, df_stock, capital_inicial, q_max1, q_min1, q_max2, q_min2,
                         capital_inverti1, capital_vendi1, capital_inverti2, capital_vendi2,
                         capital_inverti3, capital_inverti4, capital_inverti5):
    # Obtener los datos de este ETF
    df = df_stock[sym].copy()

    # Procesar los datos
    df['date'] = df.index.strftime('%Y-%m-%d')
    df['month'] = df.index.strftime("%b")
    df['year'] = df.index.year
    df['day'] = df.index.day
    df['week'] = df.index.isocalendar().week

    df['ts'] = df['month'] + '_' + df['year'].astype(str)
    df['ts_week'] = df['ts'] + '_' + df['week'].astype(str)

    df['invest_day'] = 0
    # Iteramos sobre cada valor único en la columna 'ts'
    for ts_value in df['ts'].unique():
        # Filtramos el DataFrame para el valor único de 'ts'
        df_subset = df[df['ts'] == ts_value]
        # Buscamos el día 2 o su consecutivo y marcamos como 1
        try:
            mask_3 = df_subset[(df_subset['day'] >= 2)].index[0]
            df.loc[mask_3, 'invest_day'] = 1
        except:
            pass
        try:
            mask_9 = df_subset[(df_subset['day'] >= 9)].index[0]
            df.loc[mask_9, 'invest_day'] = 2
        except:
            pass
        try:
            mask_16 = df_subset[(df_subset['day'] >= 16)].index[0]
            df.loc[mask_16, 'invest_day'] = 3
        except:
            pass
        try:
            mask_23 = df_subset[(df_subset['day'] >= 23)].index[0]
            df.loc[mask_23, 'invest_day'] = 4
        except:
            pass

    # Seleccionamos solo las columnas necesarias
    df = df[['date', 'year', 'ts', 'ts_week', 'invest_day', 'close']]

    # Monthly Quartile Metodology (1)
    df['diff'] = df['close'].diff().fillna(0)
    df['diff_accumulated_1'] = df.groupby('ts')['diff'].cumsum()

    quantile_max = round(df['diff_accumulated_1'].quantile(q_max1), 2)
    quantile_min = round(df['diff_accumulated_1'].quantile(q_min1), 2)

    # Calcular la cantidad de capital a invertir según la metodología
    df['trade_1'] = np.where(df['diff_accumulated_1'] <= quantile_min, capital_inverti1, 0)
    df['trade_1'] += np.where((df['diff_accumulated_1'] >= quantile_max),
                              -capital_vendi1, 0)

    # Si la suma agrupada excede el límite, solo comprar hasta que se cumpla el límite el resto de trade_1 positivos agrupados mensualmente deben de ser 0
    df['trade_accumulated_1'] = df.groupby('ts')['trade_1'].cumsum()
    exceeded_mask = (df['trade_accumulated_1'].gt(capital_inverti3) | df['trade_accumulated_1'].lt(-capital_inverti3))
    df.loc[exceeded_mask, 'trade_1'] = 0
    #df['trade_accumulated_1'] = df.groupby('ts')['trade_1'].cumsum()

    # Calcular el comercio de acciones
    df['stock_trade_1'] = df['trade_1'] / df['close']
    df['stock_trade_1'].iloc[0] = capital_inicial / df['close'].iloc[0]

    # Calcular el stock total y el valor monetario
    df['stock_1'] = df['stock_trade_1'].cumsum()
    df['price_money_1'] = df['stock_1'] * df['close']

    df.drop(['trade_accumulated_1'], axis=1, inplace=True)
    
    # Weekly Quartile Metodology
    df['diff_accumulated_2'] = df.groupby('ts_week')['diff'].cumsum()

    quantile_max = round(df['diff_accumulated_2'].quantile(q_max2), 2)
    quantile_min = round(df['diff_accumulated_2'].quantile(q_min2), 2)

    # Calcular la cantidad de capital a invertir según la metodología
    df['trade_2'] = np.where(df['diff_accumulated_2'] <= quantile_min, capital_inverti2, 0)
    df['trade_2'] += np.where((df['diff_accumulated_2'] >= quantile_max),
                              -capital_vendi2, 0)

    # Si la suma agrupada excede el límite, solo comprar hasta que se cumpla el límite el resto de trade_1 positivos agrupados mensualmente deben de ser 0
    df['trade_accumulated_2'] = df.groupby('ts')['trade_2'].cumsum()
    exceeded_mask = (df['trade_accumulated_2'].gt(capital_inverti3) | df['trade_accumulated_2'].lt(-capital_inverti3))
    df.loc[exceeded_mask, 'trade_2'] = 0
    # df['trade_accumulated_2'] = df.groupby('ts')['trade_2'].cumsum()

    # Calcular el comercio de acciones
    df['stock_trade_2'] = df['trade_2'] / df['close']
    df['stock_trade_2'].iloc[0] = capital_inicial / df['close'].iloc[0]

    # Calcular el stock total y el valor monetario
    df['stock_2'] = df['stock_trade_2'].cumsum()
    df['price_money_2'] = df['stock_2'] * df['close']

    # Monthly trading
    df['trade_3'] = df.apply(lambda row: capital_inverti3 if row['invest_day'] == 1 else 0, axis=1)
    df['stock_trade_3'] = df['trade_3'] / df['close']
    df['stock_trade_3'].iloc[0] = capital_inicial / df['close'].iloc[0]
    df['stock_3'] = df['stock_trade_3'].cumsum()
    df['price_money_3'] = df['stock_3'] * df['close']

    # Bimonthly trading
    df['trade_4'] = df.apply(lambda row: capital_inverti4 if (row['invest_day'] == 1 or row['invest_day'] == 3) else 0, axis=1)
    df['stock_trade_4'] = df['trade_4'] / df['close']
    df['stock_trade_4'].iloc[0] = capital_inicial / df['close'].iloc[0]
    df['stock_4'] = df['stock_trade_4'].cumsum()
    df['price_money_4'] = df['stock_4'] * df['close']

    # Weekly trading
    df['trade_5'] = df.apply(lambda row: capital_inverti5 if (row['invest_day'] != 0) else 0, axis=1)
    df['stock_trade_5'] = df['trade_5'] / df['close']
    df['stock_trade_5'].iloc[0] = capital_inicial / df['close'].iloc[0]
    df['stock_5'] = df['stock_trade_5'].cumsum()
    df['price_money_5'] = df['stock_5'] * df['close']

    # Generar el gráfico

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    # Agregar la línea para la columna 'price_money' en el primer subgráfico
    fig.add_trace(go.Scatter(x=df.index, y=df['price_money_1'], mode='lines', name='MET-1: Quantiles mensual'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['price_money_2'], mode='lines', name='MET-2: Quantiles semanal'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['price_money_3'], mode='lines', name='MET-3: Inversion mensual (Trade Republic)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['price_money_4'], mode='lines', name='MET-4: Inversion bimensual (Trade Republic)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['price_money_5'], mode='lines', name='MET-5: Inversion semanal (Trade Republic)'), row=1, col=1)
    # Agregar el diagrama de barras para la columna 'trade' en el segundo subgráfico
    fig.add_trace(go.Bar(x=df.index, y=df['trade_1'], name='Trade (MET-1)',
                         marker_color=df['trade_1'].apply(lambda x: 'green' if x >= 0 else 'red')), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['trade_2'], name='Trade (MET-2)',
                         marker_color=df['trade_2'].apply(lambda x: 'blue' if x >= 0 else 'yellow')), row=2, col=1)

    # Actualizar diseño del gráfico
    fig.update_layout(
        xaxis_title="Días",
        yaxis_title="Ganancias en €",
        title=f'Evolución de las Ganancias en € de {name}',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"
        )
    )
    fig.update_yaxes(title_text="Trade", row=2, col=1)

    return fig, df


def rentabilidades(df, metodologies, capital_inicial, tax):
    df_rent_anual = pd.DataFrame()
    df_rent_mensual = pd.DataFrame()
    df_rent_mensual_prom = pd.DataFrame()
    # Define un diccionario que mapea los nombres de los meses a sus números correspondientes
    month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                  'Nov': 11, 'Dec': 12}

    for idx, met in enumerate(metodologies):
        # Calculo rentabilidades anuales históricas
        if idx >= 2:
            tax = 0

        unique_year = df['year'].unique()
        for y in unique_year:
            df_i = df[df['year'] == y].copy()

            df_rent_i = pd.DataFrame({
                'Metodología': met,
                'year': y,
                'rent_anual': (df_i.iloc[-1][f'price_money_{idx+1}']
                               - (df[df['year'] <= y][f'trade_{idx + 1}'].sum() + capital_inicial
                                  + df[(df['year'] <= y) & (df[f'trade_{idx + 1}'] != 0)][f'trade_{idx + 1}'].count() * tax))
                              /
                              (df[df['year'] <= y][f'trade_{idx + 1}'].sum() + capital_inicial
                               + df[(df['year'] <= y) & (df[f'trade_{idx + 1}'] != 0)][f'trade_{idx + 1}'].count() * tax) * 100
            }, index=[0])

            df_rent_anual = pd.concat([df_rent_anual, df_rent_i], ignore_index=True)

        # Cálculo rentabilidades mensuales históricas
        unique_ts = df['ts'].unique()
        for pos, ts in enumerate(unique_ts):
            df_i = df[df['ts'] == ts].copy()

            df_rent_i = pd.DataFrame({
                'Metodología': met,
                'year': ts[-4:],
                'month': ts[:3],
                'ts': ts,
                'rent_mensual': (df_i.iloc[-1][f'price_money_{idx+1}']
                                 - (df[df['ts'].isin(unique_ts[:pos+1])][f'trade_{idx+1}'].sum() + capital_inicial
                                    + df[(df['ts'].isin(unique_ts[:pos+1])) & (df[f'trade_{idx+1}'] != 0)][f'trade_{idx+1}'].count()*tax))
                                /
                                (df[df['ts'].isin(unique_ts[:pos + 1])][f'trade_{idx + 1}'].sum() + capital_inicial
                                 + df[(df['ts'].isin(unique_ts[:pos + 1])) & (df[f'trade_{idx + 1}'] != 0)][f'trade_{idx + 1}'].count() * tax) * 100
            }, index=[0])

            df_rent_mensual = pd.concat([df_rent_mensual, df_rent_i], ignore_index=True)

        df_rent_mensual_prom_i = df_rent_mensual[df_rent_mensual['Metodología'] == met][
            ['Metodología', 'month', 'rent_mensual']].groupby(
            ['Metodología', 'month']).mean().round(2).reset_index()
        df_rent_mensual_prom_i['month_number'] = df_rent_mensual_prom_i['month'].map(month_dict)
        df_rent_mensual_prom_i = df_rent_mensual_prom_i.sort_values(by='month_number')
        df_rent_mensual_prom_i = df_rent_mensual_prom_i.drop(columns='month_number')

        df_rent_mensual_prom = pd.concat([df_rent_mensual_prom, df_rent_mensual_prom_i], ignore_index=True)

    # Generar el gráfico
    # Gráfico de rentabilidades anuales
    fig_rent_anual = px.line(df_rent_anual, x='year', y='rent_anual', color='Metodología',
                             title='Rentabilidades Anuales')
    fig_rent_anual.update_layout(xaxis_title='Año', yaxis_title='Rentabilidad (%)')

    # Gráfico de rentabilidades mensuales
    fig_rent_mensual = px.line(df_rent_mensual, x='ts', y='rent_mensual', color='Metodología',
                               title='Rentabilidades Mensuales')
    fig_rent_mensual.update_layout(xaxis_title='Fecha', yaxis_title='Rentabilidad (%)')

    # Gráfico de rentabilidades mensuales promedio por mes
    fig_rent_mensual_prom = px.line(df_rent_mensual_prom, x='month', y='rent_mensual', color='Metodología',
                                   title='Rentabilidades Mensuales Promedio por Mes')
    fig_rent_mensual_prom.update_layout(xaxis_title='Mes', yaxis_title='Rentabilidad Promedio (%)')

    return fig_rent_anual, fig_rent_mensual, fig_rent_mensual_prom



# Código

valores_predeterminados = {
    'VUAA.F': {'tax': 1, 'capital_inicial': 1250, 'q_max1': 0.94, 'q_min1': 0.6, 'q_max2': 0.94, 'q_min2': 0.6,
               'capital_inverti1': 320, 'capital_vendi1': 0, 'capital_inverti2': 80, 'capital_vendi2': 0,
               'capital_inverti3': 320, 'capital_inverti4': 160, 'capital_inverti5': 80},
    'SEC0.DE': {'tax': 1, 'capital_inicial': 1250, 'q_max1': 0.92, 'q_min1': 0.6, 'q_max2': 0.92, 'q_min2': 0.6,
               'capital_inverti1': 320, 'capital_vendi1': 0, 'capital_inverti2': 80, 'capital_vendi2': 0,
               'capital_inverti3': 320, 'capital_inverti4': 160, 'capital_inverti5': 80},
    'XAIX.F': {'tax': 1, 'capital_inicial': 1250, 'q_max1': 0.93, 'q_min1': 0.6, 'q_max2': 0.93, 'q_min2': 0.6,
               'capital_inverti1': 320, 'capital_vendi1': 0, 'capital_inverti2': 80, 'capital_vendi2': 0,
               'capital_inverti3': 320, 'capital_inverti4': 160, 'capital_inverti5': 80},
    'PRAW.DE': {'tax': 1, 'capital_inicial': 1250, 'q_max1': 0.95, 'q_min1': 0.6, 'q_max2': 0.95, 'q_min2': 0.6,
               'capital_inverti1': 320, 'capital_vendi1': 0, 'capital_inverti2': 80, 'capital_vendi2': 0,
               'capital_inverti3': 320, 'capital_inverti4': 160, 'capital_inverti5': 80},
    'VWCE.F': {'tax': 1, 'capital_inicial': 1250, 'q_max1': 0.95, 'q_min1': 0.6, 'q_max2': 0.95, 'q_min2': 0.6,
               'capital_inverti1': 320, 'capital_vendi1': 0, 'capital_inverti2': 80, 'capital_vendi2': 0,
               'capital_inverti3': 320, 'capital_inverti4': 160, 'capital_inverti5': 80},
    }

# Valores predeterminados por defecto en caso de no existir la clave para un ETF específico
valores_por_defecto = {'tax': 1, 'capital_inicial': 1250, 'q_max1': 0.95, 'q_min1': 0.08, 'q_max2': 0.95, 'q_min2': 0.08,
                       'capital_inverti1': 350, 'capital_vendi1': 120, 'capital_inverti2': 175, 'capital_vendi2': 120,
                       'capital_inverti3': 350, 'capital_inverti4': 175, 'capital_inverti5': 87.5}

for sym, name in dic_symbols.items():
    st.subheader(f'Metodología quantiles para {name}')

    # Dividir el espacio en dos columnas
    col1, col2, col3 = st.columns(3)

    # Definir los controles de entrada para este gráfico específico en la primera columna
    with col1:
        # Obtener los valores predeterminados del diccionario o usar los valores predeterminados por defecto
        default_values = valores_predeterminados.get(sym, valores_por_defecto)
        tax = st.number_input(f'Taxes for trading', value=default_values.get('tax'), key=f'tax_{sym}')
        capital_inicial = st.number_input(f'Capital inicial', value=default_values.get('capital_inicial'),
                                          key=f'capital_inicial_{sym}')
        capital_vendi1 = st.number_input(f'Capital a vender (MET-1)',
                                         value=default_values.get('capital_vendi1'), key=f'capital_vendi1_{sym}')
        capital_vendi2 = st.number_input(f'Capital a vender (MET-2)',
                                         value=default_values.get('capital_vendi2'), key=f'capital_vendi2_{sym}')

    # Definir los controles de entrada para este gráfico específico en la segunda columna
    with col2:
        q_max1 = st.number_input(f'Quantil volatilidad máxima mensual (MET-1)',
                                 value=default_values.get('q_max1'), key=f'q_max1_{sym}')
        q_min1 = st.number_input(f'Quantil volatilidad mínima mensual (MET-1)',
                                 value=default_values.get('q_min1'), key=f'q_min1_{sym}')
        q_max2 = st.number_input(f'Quantil volatilidad máxima semanal (MET-2)',
                                 value=default_values.get('q_max2'), key=f'q_max2_{sym}')
        q_min2 = st.number_input(f'Quantil volatilidad mínima semanal (MET-2)',
                                 value=default_values.get('q_min2'), key=f'q_min2_{sym}')

    with col3:
        capital_inverti1 = st.number_input(f'Capital mensual a invertir (MET-1)',
                                           value=default_values.get('capital_inverti1'), key=f'capital_inverti1_{sym}')
        capital_inverti2 = st.number_input(f'Capital semanal a invertir (MET-2)',
                                           value=default_values.get('capital_inverti2'), key=f'capital_inverti2_{sym}')
        capital_inverti3 = st.number_input(f'Capital mensual a invertir (MET-3)',
                                           value=default_values.get('capital_inverti3'), key=f'capital_inverti3_{sym}')
        capital_inverti4 = st.number_input(f'Capital bimensual a invertir (MET-4)',
                                           value=default_values.get('capital_inverti4'), key=f'capital_inverti4_{sym}')
        capital_inverti5 = st.number_input(f'Capital semanal a invertir (MET-5)',
                                           value=default_values.get('capital_inverti5'), key=f'capital_inverti5_{sym}')

    # Generar el gráfico utilizando las variables específicas de este gráfico
    fig, df = calculos_metodologia(sym, name, df_stock, capital_inicial, q_max1, q_min1, q_max2, q_min2,
                                     capital_inverti1, capital_vendi1, capital_inverti2, capital_vendi2,
                                     capital_inverti3, capital_inverti4, capital_inverti5)

    metodologies = ['MET-1: Quantiles mensual',
                    'MET-2: Quantiles semanal',
                    'MET-3: Inversion mensual (Trade Republic)',
                    'MET-4: Inversion bimensual (Trade Republic)',
                    'MET-5: Inversion semanal (Trade Republic)']

    fig_rent_anual, fig_rent_mensual, fig_rent_mensual_prom = rentabilidades(df, metodologies, capital_inicial, tax)
    # Indices de evaluación
    df_resumen = pd.DataFrame()

    df_resumen['Metodología'] = metodologies
    df_resumen['Precio quantile_max'] = [round(df['diff_accumulated_1'].quantile(q_max1), 2),
                                         round(df['diff_accumulated_2'].quantile(q_max2), 2),
                                         None, None, None]
    df_resumen['Precio quantile_min'] = [round(df['diff_accumulated_1'].quantile(q_min1), 2),
                                         round(df['diff_accumulated_2'].quantile(q_min2), 2),
                                         None, None, None]
    df_resumen['Flujo de dinero'] = [df["trade_1"].sum(), df["trade_2"].sum(),
                                     df["trade_3"].sum(), df["trade_4"].sum(), df["trade_5"].sum()]
    df_resumen['Flujo de dinero máximo mensual'] = [int(df[['ts', 'trade_1']].groupby('ts').sum().max().values),
                                                    int(df[['ts', 'trade_2']].groupby('ts').sum().max().values),
                                                    int(df[['ts', 'trade_3']].groupby('ts').sum().max().values),
                                                    int(df[['ts', 'trade_4']].groupby('ts').sum().max().values),
                                                    int(df[['ts', 'trade_5']].groupby('ts').sum().max().values)]
    df_resumen['Flujo de dinero máximo semanal'] = [int(df[['ts_week', 'trade_1']].groupby('ts_week').sum().max().values),
                                                    int(df[['ts_week', 'trade_2']].groupby('ts_week').sum().max().values),
                                                    int(df[['ts_week', 'trade_3']].groupby('ts_week').sum().max().values),
                                                    int(df[['ts_week', 'trade_4']].groupby('ts_week').sum().max().values),
                                                    int(df[['ts_week', 'trade_5']].groupby('ts_week').sum().max().values)]
    df_resumen['Número de transacciones'] = [df[df['trade_1'] != 0]['trade_1'].count(),
                                             df[df['trade_2'] != 0]['trade_2'].count(),
                                             df[df['trade_3'] != 0]['trade_3'].count(),
                                             df[df['trade_4'] != 0]['trade_4'].count(),
                                             df[df['trade_5'] != 0]['trade_5'].count()]
    df_resumen['Inversión total'] = [df["trade_1"].sum() + capital_inicial,
                                     df["trade_2"].sum() + capital_inicial,
                                     df["trade_3"].sum() + capital_inicial,
                                     df["trade_4"].sum() + capital_inicial,
                                     df["trade_5"].sum() + capital_inicial]
    df_resumen['Beneficio a día actual'] = [round(df["price_money_1"].iloc[-1] - df_resumen['Inversión total'].iloc[0] -
                                                  df_resumen['Número de transacciones'].iloc[0] * tax, 2),
                                            round(df["price_money_1"].iloc[-1] - df_resumen['Inversión total'].iloc[1] -
                                                  df_resumen['Número de transacciones'].iloc[1] * tax, 2),
                                            round(df["price_money_3"].iloc[-1] - df_resumen['Inversión total'].iloc[2] -
                                                  df_resumen['Número de transacciones'].iloc[2], 2),
                                            round(df["price_money_4"].iloc[-1] - df_resumen['Inversión total'].iloc[3] -
                                                  df_resumen['Número de transacciones'].iloc[3], 2),
                                            round(df["price_money_5"].iloc[-1] - df_resumen['Inversión total'].iloc[4] -
                                                  df_resumen['Número de transacciones'].iloc[4], 2)]

    df_resumen['Rentabilidad'] = [
        round(df_resumen["Beneficio a día actual"].iloc[0] / df_resumen["Inversión total"].iloc[0] * 100, 2) if
        df_resumen["Inversión total"].iloc[0] != 0 else None,
        round(df_resumen["Beneficio a día actual"].iloc[1] / df_resumen["Inversión total"].iloc[1] * 100, 2) if
        df_resumen["Inversión total"].iloc[1] != 0 else None,
        round(df_resumen["Beneficio a día actual"].iloc[2] / df_resumen["Inversión total"].iloc[2] * 100, 2) if
        df_resumen["Inversión total"].iloc[2] != 0 else None,
        round(df_resumen["Beneficio a día actual"].iloc[3] / df_resumen["Inversión total"].iloc[3] * 100, 2) if
        df_resumen["Inversión total"].iloc[3] != 0 else None,
        round(df_resumen["Beneficio a día actual"].iloc[4] / df_resumen["Inversión total"].iloc[4] * 100, 2) if
        df_resumen["Inversión total"].iloc[4] != 0 else None]

    st.write(df_resumen.T)

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_rent_anual, use_container_width=True)
    st.plotly_chart(fig_rent_mensual, use_container_width=True)
    st.plotly_chart(fig_rent_mensual_prom, use_container_width=True)


