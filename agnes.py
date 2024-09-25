import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import homogeneity_score, adjusted_rand_score, completeness_score
from scipy.stats import entropy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import ParameterGrid

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

#============================== TREINAMENTO ================================

# Parâmetros para o AGNES
param_grid = {
    'linkage': ['ward', 'complete', 'average', 'single'],
    'distance_threshold': [10, 15, 20, 25]  # Exemplo de diferentes thresholds
}

# Inicializar a variável para armazenar o melhor score e os parâmetros correspondentes
best_score = -1
best_params = {}

# Iterar por cada combinação de parâmetros
for params in ParameterGrid(param_grid):
    # Criar o modelo AgglomerativeClustering sem definir 'n_clusters', mas usando 'distance_threshold'
    agnes = AgglomerativeClustering(linkage=params['linkage'], distance_threshold=params['distance_threshold'], n_clusters=None)
    
    # Ajustar o modelo aos dados
    agnes.fit(x)
    
    # Verificar quantos clusters foram formados
    num_clusters = len(np.unique(agnes.labels_))
    
    # Se houver menos de 2 clusters, ignorar o cálculo do silhouette score
    if num_clusters < 2:
        print(f"Parâmetros: {params}, Número de clusters: {num_clusters}, Não é possível calcular o score de silhueta.")
        continue
    
    # Calcular o score de silhueta
    score = silhouette_score(x, agnes.labels_)
    
    # Imprimir os parâmetros e o score atual
    print(f"Parâmetros: {params}, Score de Silhueta: {score}, Número de clusters: {num_clusters}")
    
    # Verificar se o score atual é melhor que o melhor score já encontrado
    if score > best_score:
        best_score = score
        best_params = params  # Armazenar os melhores parâmetros

# Exibir os melhores parâmetros e o melhor score encontrado
print(f"\nMelhores Parâmetros: {best_params}")
print(f"Melhor Score de Silhueta: {best_score}")

# Armazenar o melhor linkage e o melhor distance_threshold
melhor_linkage = best_params['linkage']
melhor_distance_threshold = best_params['distance_threshold']

print(f"Melhor linkage: {melhor_linkage}")
print(f"Melhor distance_threshold: {melhor_distance_threshold}")

# Treinar o modelo AgglomerativeClustering com os melhores parâmetros
agnes = AgglomerativeClustering(linkage=melhor_linkage, distance_threshold=melhor_distance_threshold, n_clusters=None)
agnes.fit(x)

# Adicionar as labels de cluster ao dataframe
x['Cluster'] = agnes.labels_

# Contar o número de clusters gerados
num_clusters = len(np.unique(agnes.labels_))
print(f"Número de clusters formados: {num_clusters}")
#============================== TREINAMENTO ================================

#============================== ANALISE ARQUIVO ================================



label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

colunas_numericas = x.select_dtypes(include=[float, int]).columns

caracteristicas_cluster = x.groupby('Cluster')[colunas_numericas].mean()

caracteristicas_cluster = caracteristicas_cluster.round(2)
caracteristicas_cluster.to_csv('caracteristicas_cluster_agnes.csv', index=True)

print("\nCaracterísticas médias por cluster salvas em 'caracteristicas_cluster.csv'")

#============================== ANALISE ARQUIVO ================================

#============================== GRAFICO ================================

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

df_visualizacao = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])
df_visualizacao['Cluster'] = agnes.labels_

mapa_clusters_posicoes = {0: 'Goleiro', 1: 'Linha'}
x['Posicao'] = x['Cluster'].map(mapa_clusters_posicoes)

df_visualizacao['Posicao'] = x['Posicao']

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_visualizacao, x='PC1', y='PC2', hue='Posicao', palette='viridis', s=100)
plt.title(f'Visualização dos Clusters por Posição (n_clusters={num_clusters}, linkage={melhor_linkage})')
plt.xlabel('PC1')  
plt.ylabel('PC2')  
plt.legend(title='Posição')
plt.show()

#============================== GRAFICO ================================

# ============================= AVALIAÇÃO ================================

# Como o AgglomerativeClustering não possui um método `inertia_`, alguns métodos de avaliação serão alterados

# Calcular e imprimir o score de silhueta
#score = silhouette_score(x[colunas_numericas], agnes.labels_)  # Usar apenas colunas numéricas
#print(f"Score de Silhueta final: {score}")

# Calcular a homogeneidade
homogeneity = homogeneity_score(y, agnes.labels_)
print(f"Homogeneidade do modelo: {homogeneity}")

rand_index = adjusted_rand_score(y, agnes.labels_)
print(f"Índice Rand Ajustado do modelo: {rand_index}")

completeness = completeness_score(y, agnes.labels_)
print(f"Completude do modelo: {completeness}")

# Calcular a entropia
entropia_total = 0
n_clusters = num_clusters

for cluster in range(n_clusters):
    counts = np.bincount(y_encoded[agnes.labels_ == cluster])
    probs = counts / counts.sum() if counts.sum() > 0 else 0
    cluster_entropy = entropy(probs, base=2)
    entropia_total += (counts.sum() / len(y_encoded)) * cluster_entropy

print(f"Entropia total do modelo: {entropia_total}")

# ============================= AVALIAÇÃO ================================
