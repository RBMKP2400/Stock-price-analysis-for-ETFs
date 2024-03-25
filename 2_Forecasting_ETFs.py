########################################
# Create and evaluate the model
########################################
st.subheader(f'Evaluate the model')
for sym, name in dic_symbols.items():

    df = df_stock[sym].copy()
    df.reset_index(inplace=True)

    data_train = pd.DataFrame(df.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(df.Close[int(len(data) * 0.80): len(data)])

    # Comprobar si ya existe el modelo
    if os.path.exists(f'/models/model_{sym}.keras'):
        # Cargar modelo previamente entrenado
        model = load_model(f'/models/model_{sym}.keras')
    else:
        # Transformamos los valores en un rango de 0 a 1, es decir calculamos los pesos
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_train_scale = scaler.fit_transform(data_train)

        # Generamos las agrupaciones de días que va a utilizar para calcular el día venidero. Se decide tomar una agrupación de 100 en 100.
        # Cuanto mayor sea el número de días a considerar, mejor será la predicción
        x = []
        y = []

        for i in range(100, data_train_scale.shape[0]):
            x.append(data_train_scale[i - 100:i])
            y.append(data_train_scale[i, 0])

        x, y = np.array(x), np.array(y)

        # Creacción de la red neuronal de 4 capas LSTM utilizando un modelo secuencial.
        # capa LSTM (Long Short-Term Memory) es una arquitectura de red neuronal recurrente (RNN) diseñada para manejar secuencias de datos, como series temporales
        # units = numero de neuronas de la capa
        # activation = indica la metodología para calcular la Red Neuronal (RNN)
        # return_sequencies para mostrar el seguimiento de la creacción de la capa
        # input_shape = definir el tamaño del valor de entrada (coge los valores de 100 en 100)

        # La capa Dropout se utiliza comúnmente para regularizar los modelos de redes neuronales y prevenir el sobreajuste (overfitting).
        # El sobreajuste ocurre cuando el modelo se ajusta demasiado a los datos de entrenamiento y tiene dificultades para generalizar a datos nuevos
        # La función de Dropout "apaga" aleatoriamente un porcentaje de unidades (neuronas) en una capa durante el entrenamiento.
        # Esto significa que durante cada paso de entrenamiento, algunas neuronas no contribuirán a la propagación hacia adelante
        # ni a la propagación hacia atrás. Esto ayuda a prevenir la coadaptación de las neuronas, lo que a su vez puede mejorar la
        # capacidad del modelo para generalizar a datos nuevos.

        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True,
                       input_shape=((x.shape[1], 1))))
        model.add(Dropout(0.1))

        model.add(LSTM(units=80, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=100, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))

        model.add(LSTM(units=120, activation='relu'))
        model.add(Dropout(0.4))

        model.add(Dense(units=1))  # Calculamos un unico valor estimado (salida)

        # Definimos el optimizador del modelo y cómo se van a calcular el error del mismo (error cuadrático medio)
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Ejecutamos el modelo
        # epochs: especifica el número de épocas o iteraciones completas sobre el conjunto de datos de entrenamiento. Se entrenarán 50 veces todos los datos.
        # batch_size: Especifica el tamaño del lote utilizado para el entrenamiento. El tamaño del lote determina cuántos ejemplos se utilizan antes de que se actualicen los pesos del modelo.
        # verbose: Este parámetro controla la verbosidad del proceso de entrenamiento. Puede tomar diferentes valores, como 0, 1 o 2. En tu caso, lo has configurado en 1, lo que significa que verás una barra de progreso que muestra el progreso del entrenamiento para cada época.
        model.fit(x, y, epochs=50, batch_size=90, verbose=1)
        model.summary()
        model.save(f'/models/model_{sym}.keras')


    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
    x = []
    y = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])
    x, y = np.array(x), np.array(y)
    y_predict = model.predict(x)
    scale = 1 / scaler.scale_
    y_predict = y_predict * scale
    y = y * scale

    plt.figure(figsize=(10, 8))
    plt.plot(y_predict, 'r', label='Predicted Price')
    plt.plot(y, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()




st.subheader(f'ETFs Forecasting')
for sym, name in dic_symbols.items():
    df = df_stock[sym].copy()

    model = load_model(f'/models/model_{sym}.keras')

    # Obtener los últimos 90 días de datos y seleccionar solo la columna 'Close'
    ultimos_90_dias = df.iloc[-100:]['Close']

    # Escalar los datos de los últimos 90 días
    ultimos_90_dias_scaled = scaler.transform(ultimos_90_dias.values.reshape(-1, 1))

    # Convertir los datos escalados de nuevo a un DataFrame
    ultimos_90_dias_scaled_df = pd.DataFrame(ultimos_90_dias_scaled, index=ultimos_90_dias.index, columns=['Close'])

    # Generar datos de entrada para la predicción de los siguientes 30 días
    x_prediccion_futura = [ultimos_90_dias_scaled_df[-100:]]
    x_prediccion_futura = np.array(x_prediccion_futura)

    # Realizar la predicción con el modelo entrenado
    y_prediccion_futura = model.predict(x_prediccion_futura)

    # Escalar los datos predichos de vuelta al rango original
    y_prediccion_futura = y_prediccion_futura * scaler.scale_

    # Crear un índice de fechas para los próximos 30 días
    fechas_prediccion_futura = pd.date_range(start=ultimos_90_dias.index[-1], periods=100, freq='D')

    # Graficar los últimos 90 días y la predicción de los próximos 30 días
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-100:], df['Close'].iloc[-100:], label='Últimos 90 días')
    plt.plot(fechas_prediccion_futura[:len(y_prediccion_futura)], y_prediccion_futura, 'r--',
             label='Predicción de los próximos 30 días')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre')
    plt.title('Últimos 90 días con Predicción de los Próximos 30 días')
    plt.legend()
    plt.show()
