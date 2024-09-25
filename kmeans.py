import pandas as pd
from sklearn.cluster import KMeans
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

#============================== TRATAMENTO DE DADOS ================================

#============================== TREINAMENTO ================================

# K-means
# Definir os parâmetros para o GridSearch
param_grid = {
    'n_clusters': [4],  # Definir 4 clusters para representar cada posição
    'max_iter': [100, 200, 300, 400, 500]
}

# Inicializar a variável para armazenar o melhor score e os parâmetros correspondentes
best_score = -1
best_params = {}

# Iterar por cada combinação de parâmetros
for params in ParameterGrid(param_grid):
    # Criar um modelo KMeans com os parâmetros atuais
    kmeans = KMeans(n_clusters=params['n_clusters'], max_iter=params['max_iter'], random_state=42)
    
    # Ajustar o modelo aos dados
    kmeans.fit(x)
    
    # Calcular o score de silhueta
    score = silhouette_score(x, kmeans.labels_)
    
    # Imprimir os parâmetros e o score atual
    print(f"Parâmetros: {params}, Score de Silhueta: {score}")
    
    # Verificar se o score atual é melhor que o melhor score já encontrado
    if score > best_score:
        best_score = score
        best_params = params

# Exibir os melhores parâmetros e o melhor score encontrado
print(f"\nMelhores Parâmetros: {best_params}")
print(f"Melhor Score de Silhueta: {best_score}")

# Treinar o modelo K-means com os melhores parâmetros encontrados
melhor_n_clusters = best_params['n_clusters']
melhor_max_iter = best_params['max_iter']

kmeans = KMeans(n_clusters=melhor_n_clusters, max_iter=melhor_max_iter, random_state=42)
kmeans.fit(x)

#============================== TREINAMENTO ================================

#============================== ANALISE ARQUIVO ================================

x['Cluster'] = kmeans.labels_

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

colunas_numericas = x.select_dtypes(include=[float, int]).columns


caracteristicas_cluster = x.groupby('Cluster')[colunas_numericas].mean()

caracteristicas_cluster = caracteristicas_cluster.round(2)
caracteristicas_cluster.to_csv('caracteristicas_cluster.csv', index=True)

print("\nCaracterísticas médias por cluster salvas em 'caracteristicas_cluster.csv'")

#============================== ANALISE ARQUIVO ================================

#============================== GRAFICO ================================


pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

df_visualizacao = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])
df_visualizacao['Cluster'] = kmeans.labels_

mapa_clusters_posicoes = {0: 'Goleiro', 1: 'Defensor', 2: 'Meio-Campo', 3: 'Atacante'}
x['Posicao'] = x['Cluster'].map(mapa_clusters_posicoes)

df_visualizacao['Posicao'] = x['Posicao']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_visualizacao, x='PC1', y='PC2', hue='Posicao', palette='viridis', s=100)
plt.title(f'Visualização dos Clusters por Posição (n_clusters={melhor_n_clusters}, max_iter={melhor_max_iter})')
plt.xlabel('PC1')  
plt.ylabel('PC2')  
plt.legend(title='Posição')
plt.show()

#============================== GRAFICO ================================

# ============================= AVALIAÇÃO ================================

# Calcular e imprimir a inércia
inertia = kmeans.inertia_
print(f"Inércia do modelo: {inertia}")

# Calcular e imprimir o score de silhueta usando apenas colunas numéricas
score = silhouette_score(x[colunas_numericas], kmeans.labels_)  # Usar apenas colunas numéricas
print(f"Score de Silhueta final: {score}")

# Calcular a distância média entre os centroides dos clusters
centroids = kmeans.cluster_centers_
distances = pairwise_distances(centroids)
np.fill_diagonal(distances, np.nan)  # Ignorar a diagonal

# Calcular a média das distâncias entre os centroides
mean_distance = np.nanmean(distances)
print(f"Distância média entre centroides: {mean_distance}")

# Calcular a homogeneidade
homogeneity = homogeneity_score(y, kmeans.labels_)
print(f"Homogeneidade do modelo: {homogeneity}")

rand_index = adjusted_rand_score(y, kmeans.labels_)
print(f"Índice Rand Ajustado do modelo: {rand_index}")

completeness = completeness_score(y, kmeans.labels_)
print(f"Completude do modelo: {completeness}")

# Calcular a entropia
entropia_total = 0
n_clusters = kmeans.n_clusters

# Calcular a entropia para cada cluster
for cluster in range(n_clusters):
    # Contar o número de elementos de cada classe no cluster
    counts = np.bincount(y_encoded[kmeans.labels_ == cluster])
    probs = counts / counts.sum() if counts.sum() > 0 else 0  # Proporções
    # Calcular a entropia do cluster
    cluster_entropy = entropy(probs, base=2)
    # Calcular a entropia total ponderada pelo número de amostras no cluster
    entropia_total += (counts.sum() / len(y_encoded)) * cluster_entropy

print(f"Entropia total do modelo: {entropia_total}")


# ============================= AVALIAÇÃO ================================