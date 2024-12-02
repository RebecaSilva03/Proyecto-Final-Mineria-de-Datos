import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def ejecutar_consulta(query):
    engine = create_engine('mysql+pymysql://rebeca:21712rbk.@localhost:3306/mineria_final')
    return pd.read_sql(query, engine)

def clustering_avanzado():
    query = """
    SELECT c.cliente_id, COUNT(v.venta_id) AS total_compras, SUM(v.ven_total) AS total_gastado
    FROM clientes c
    LEFT JOIN venta v ON c.cliente_id = v.cliente_id
    GROUP BY c.cliente_id
    """
    df = ejecutar_consulta(query)
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['total_compras', 'total_gastado']])

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters

    plt.figure(figsize=(15, 9))
    scatter = plt.scatter(
        df['total_compras'], df['total_gastado'],
        c=df['cluster'], cmap='viridis', s=100, alpha=0.7, edgecolor='k'
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
        kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
        c='red', s=200, marker='X', label='Centroides'
    )
    plt.title('Clustering con KMeans', fontsize=16)
    plt.xlabel('Total Compras', fontsize=14)
    plt.ylabel('Total Gastado', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()

def series_temporales_avanzadas():
    query = """
    SELECT DATE(v.ven_fecha) AS fecha, SUM(v.ven_total) AS total_ventas
    FROM venta v
    GROUP BY fecha
    ORDER BY fecha;
    """
    df = ejecutar_consulta(query)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.set_index('fecha', inplace=True)

    decomposition = seasonal_decompose(df['total_ventas'], model='additive', period=30)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], color='blue', title='Serie Observada')
    decomposition.trend.plot(ax=axes[1], color='orange', title='Tendencia')
    decomposition.seasonal.plot(ax=axes[2], color='green', title='Estacionalidad')
    decomposition.resid.plot(ax=axes[3], color='red', title='Residuos')
    
    for ax in axes:
        ax.grid(True)

    plt.suptitle('Descomposición de Series Temporales', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar para que no se superponga el título
    plt.show() 

def tiempos_recogida_con_knn():
    query = """
    SELECT 
        c.cliente_id,
        e.producto_id,
        e.ent_fecha AS fecha_entrada,
        v.ven_fecha AS fecha_recogida,
        TIMESTAMPDIFF(DAY, e.ent_fecha, v.ven_fecha) AS dias_para_recogida
    FROM 
        entradas e
    JOIN 
        detalle_pedido dp ON e.producto_id = dp.producto_id
    JOIN 
        venta v ON dp.producto_id = v.producto_id
    JOIN 
        clientes c ON e.cliente_id = c.cliente_id
    WHERE 
        v.ven_fecha > e.ent_fecha;
    """
    df = ejecutar_consulta(query)
    
    df.dropna(inplace=True)

    X = df[['cliente_id', 'producto_id']]
    y = df['dias_para_recogida']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.7)
    plt.title('Predicción de Tiempos de Recogida con KNN', fontsize=16)
    plt.xlabel('Tiempo Real de Recogida (días)', fontsize=14)
    plt.ylabel('Tiempo Predicho de Recogida (días)', fontsize=14)
    plt.tight_layout()
    plt.show()

def mostrar_graficas():
    plt.ion()
    series_temporales_avanzadas()
    tiempos_recogida_con_knn()
    clustering_avanzado()
    plt.ioff()
    plt.show(block=True)

mostrar_graficas()

