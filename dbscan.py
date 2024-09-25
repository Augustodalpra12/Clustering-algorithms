import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import homogeneity_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler

#============================== TRATAMENTO DE DADOS ================================

# Carregar os dados
dados = pd.read_csv('data/database.csv')

# Substituir NaN por 0
dados.fillna(0, inplace=True)

# Definir o DataFrame X com as colunas desejadas
x = dados.iloc[:, 9:59]
y = dados.iloc[:, 3]

# Identificar colunas que contêm porcentagens e removê-las
colunas_com_porcentagem = []

for col in x.columns:
    # Verificar se a coluna contém algum valor com '%'
    if x[col].astype(str).str.contains('%').any():
        colunas_com_porcentagem.append(col)

# Remover colunas que contêm porcentagens do DataFrame x
x = x.drop(columns=colunas_com_porcentagem)

# Normalizar os dados
scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)

# Transformar o array normalizado de volta para um DataFrame
x_normalized = pd.DataFrame(x_normalized, columns=x.columns)

x = x_normalized

#============================== TREINAMENTO COM DBSCAN ================================

# DBSCAN
# Definir os parâmetros para o GridSearch
param_grid = {
    'eps': [0.3, 0.5, 0.7],  # Valores diferentes de eps
    'min_samples': [5, 10, 15]  # Valores diferentes de min_samples
}

# Inicializar a variável para armazenar o melhor score e os parâmetros correspondentes
best_score = -1
best_params = {}

# Iterar por cada combinação de parâmetros
for params in ParameterGrid(param_grid):
    # Criar um modelo DBSCAN com os parâmetros atuais
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    
    # Ajustar o modelo aos dados
    dbscan.fit(x)
    
    # Filtrar os pontos que não foram considerados como ruído (-1 é o label de ruído no DBSCAN)
    if len(set(dbscan.labels_)) > 1:  # Verifica se há mais de 1 cluster
        score = silhouette_score(x, dbscan.labels_)
    
        # Imprimir os parâmetros e o score atual
        print(f"Parâmetros: {params}, Score de Silhueta: {score}")
        
        # Verificar se o score atual é melhor que o melhor score já encontrado
        if score > best_score:
            best_score = score
            best_params = params

# Exibir os melhores parâmetros e o melhor score encontrado
print(f"\nMelhores Parâmetros: {best_params}")
print(f"Melhor Score de Silhueta: {best_score}")

# Treinar o modelo DBSCAN com os melhores parâmetros encontrados
melhor_eps = best_params['eps']
melhor_min_samples = best_params['min_samples']

dbscan = DBSCAN(eps=melhor_eps, min_samples=melhor_min_samples)
dbscan.fit(x)

#============================== ANALISE ARQUIVO ================================

x['Cluster'] = dbscan.labels_

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

colunas_numericas = x.select_dtypes(include=[float, int]).columns

caracteristicas_cluster = x.groupby('Cluster')[colunas_numericas].mean()

caracteristicas_clusterX = caracteristicas_cluster.round(2)
caracteristicas_cluster.to_csv('caracteristicas_cluster_dbscan.csv', index=True)

print("\nCaracterísticas médias por cluster salvas em 'caracteristicas_cluster.csv'")

#============================== GRAFICO ================================

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

df_visualizacao = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])
df_visualizacao['Cluster'] = dbscan.labels_

# Ajustar o mapa de clusters para o DBSCAN (inclui o rótulo -1, que indica ruído)
mapa_clusters_posicoes = {cluster: f"Cluster {cluster}" for cluster in range(-1, len(set(dbscan.labels_)))}
x['Posicao'] = x['Cluster'].map(mapa_clusters_posicoes)

df_visualizacao['Posicao'] = x['Posicao']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_visualizacao, x='PC1', y='PC2', hue='Posicao', palette='viridis', s=100)
plt.title(f'Visualização dos Clusters por Posição (DBSCAN, eps={melhor_eps}, min_samples={melhor_min_samples})')
plt.xlabel('PC1')  
plt.ylabel('PC2')  
plt.legend(title='Posição')
plt.show()

#============================== GRAFICO ================================

# ============================= AVALIAÇÃO ================================

# Calcular a inércia e o score de silhueta usando DBSCAN
if len(set(dbscan.labels_)) > 1:
    score = silhouette_score(x[colunas_numericas], dbscan.labels_)
    print(f"Score de Silhueta final: {score}")

# Calcular a homogeneidade, Rand index e completude
homogeneity = homogeneity_score(y, dbscan.labels_)
print(f"Homogeneidade do modelo: {homogeneity}")

rand_index = adjusted_rand_score(y, dbscan.labels_)
print(f"Índice Rand Ajustado do modelo: {rand_index}")

completeness = completeness_score(y, dbscan.labels_)
print(f"Completude do modelo: {completeness}")

# Calcular a entropia
entropia_total = 0
n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)

# Calcular a entropia para cada cluster
for cluster in set(dbscan.labels_):
    if cluster != -1:  # Ignorar o ruído (-1)
        counts = np.bincount(y_encoded[dbscan.labels_ == cluster])
        probs = counts / counts.sum() if counts.sum() > 0 else 0  # Proporções
        cluster_entropy = entropy(probs, base=2)
        entropia_total += (counts.sum() / len(y_encoded)) * cluster_entropy

print(f"Entropia total do modelo: {entropia_total}")
