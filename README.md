# Stock-price-analysis-for-ETFs
Este proyecto se centra en la metodología de análisis y evaluación de precios de activos financieros, como ETFs (Exchange-Traded Funds), utilizando datos históricos del mercado. Se ha desarrollado una aplicación utilizando la biblioteca Streamlit para visualizar y ajustar parámetros específicos de diferentes metodologías de inversión. Estos parámetros incluyen impuestos, capital inicial, límites de volatilidad, entre otros.

Esto facilita la elección de la metodología de inversión más rentable a largo plazo basada en el precio histórico del ETF.

El proyecto plantea dos herramientas:
- "0_Analysis_ETFs.py". Se proyectan los siguientes 3 puntos:

  1. Análisis de Mercado de Valores:
     - El código permite al usuario seleccionar que acciones o ETF desea visualizar, mediante su simbolo en yahoofinanzas.
     - Proporciona un análisis histórico de precios de acciones, mostrando el precio de cierre y la media móvil de 60 días.
     - Visualiza gráficamente la evolución de los precios de cierre junto con la media móvil de 60 días para los símbolos seleccionados.
     - 

  2. Rentabilidad de las Acciones:
     - Calcula la rentabilidad anual y mensual de las acciones seleccionadas.
     - Genera gráficos comparativos de la rentabilidad anual y mensual de las acciones.
     - Muestra la rentabilidad promedio mensual de cada acción en un gráfico separado.

  3. Volatilidad de los Precios de las Acciones:
     - Calcula la volatilidad anual, mensual y semanal de los precios de las acciones seleccionadas.
     - Presenta gráficos de las 20 mayores volatilidades mensuales y semanales para cada acción.
     - Ofrece una metodología basada en cuartiles para determinar acciones de compra y venta, considerando la volatilidad de los precios.

- "1_Strategy_Evaluation_ETFs"