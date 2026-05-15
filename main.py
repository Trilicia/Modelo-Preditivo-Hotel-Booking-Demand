import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import seaborn as sns  
import kagglehub
import os


# Download latest version
path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")

print("Path to dataset files:", path)

print(os.listdir(path))

csv_path = os.path.join(path, "hotel_bookings.csv")
df = pd.read_csv(csv_path, encoding='latin-1')

# DADOS DATAFRAME
print(df.head())
print(df.info())
print(df.describe())

df.columns = (
    df.columns
    .str.lower()
    .str.strip()
    .str.replace(' ', '_')
)
print(df.columns)


#NULOS
print(df.isnull().sum())
df['children'] = df['children'].fillna(0)
df['country'] = df['country'].fillna('Unknown')
df['agent'] = df['agent'].fillna(0)
# Remover coluna company, pois tem muitos valores nulos
df = df.drop(columns=['company'])

#DUPLICADOS
print(df.duplicated().sum())
df = df.drop_duplicates()

df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

colunas_categoricas = [
    'hotel',
    'arrival_date_month',
    'meal',
    'country',
    'market_segment',
    'distribution_channel',
    'reserved_room_type',
    'assigned_room_type',
    'deposit_type',
    'customer_type',
    ]

for coluna in colunas_categoricas:
    df[coluna] = df[coluna].astype('category')
print(df.dtypes)  


#OUTLIERS
Q1 = df['adr'].quantile(0.25)
Q3 = df['adr'].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[
    (df['adr'] < limite_inferior) |
    (df['adr'] > limite_superior)
]

print(outliers.shape)

#VISUALIZAR OUTLIERS,
# Foram identificados valores extremos na variável ADR,
# podendo representar erros de registro ou casos atípicos.

plt.boxplot(df['adr'])
plt.title('Boxplot ADR')
plt.show()
df = df[
    (df['adr'] >= limite_inferior) &
    (df['adr'] <= limite_superior)
]

#REMOVER COLUNA reservation_status e reservation_status_date,
#vão atrapalhar o modelo pois já informam o cancelamento
df = df.drop(columns=['reservation_status', 'reservation_status_date'])

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']
X = pd.get_dummies(X, drop_first=True)
print(X.shape)
print(y.shape)


#ANALISE EXPLORATÓRIA
print(df['is_canceled'].value_counts())

print(df['is_canceled'].value_counts(normalize=True) * 100)
# Aproximadamente 37% das reservas foram canceladas.

sns.countplot(data=df, x='is_canceled')

plt.title('Distribuição de Cancelamentos')
plt.xlabel('Cancelamento')
plt.ylabel('Quantidade')

plt.show() 

print(
    df.groupby('is_canceled')['lead_time']
    .mean()
)
## Clientes que realizaram reservas com maior antecedência
# apresentaram maior tendência ao cancelamento.

sns.boxplot(
    data=df,
    x='is_canceled',
    y='lead_time'
)

plt.title('Lead Time vs Cancelamento')

plt.show()

print(
    df.groupby('hotel')['is_canceled']
    .mean()
    .sort_values(ascending=False)
)
#diferença no perfil do cliente, conforme hotel analisado, 
sns.barplot(
    data=df,
    x='hotel',
    y='is_canceled'
)

plt.title('Taxa de Cancelamento por Hotel')

plt.show()

# Reservas com ADR mais elevado apresentaram maior dispersão
# e tendência ao cancelamento
sns.boxplot(
    data=df,
    x='is_canceled',
    y='adr'
)

plt.title('ADR vs Cancelamento')

plt.show()

print(
    df['country']
    .value_counts()
    .head(10)
)

#maior parte das reservas concentrada em poucos países
top_paises = df['country'].value_counts().head(10)

sns.barplot(
    x=top_paises.index,
    y=top_paises.values
)

plt.xticks(rotation=45)

plt.title('Top 10 Países')

plt.show()


print(
    df.groupby('customer_type')['is_canceled']
    .mean()
)
#Clientes do tipo "Transient" maior taxa de cancelamento.
sns.barplot(
    data=df,
    x='customer_type',
    y='is_canceled'
)

plt.title('Cancelamento por Tipo de Cliente')
plt.xticks(rotation=20)
plt.show()

correlacao = df.corr(numeric_only=True)

print(correlacao)

plt.figure(figsize=(12,8))
sns.heatmap(
    correlacao,
    annot=True,
    cmap='coolwarm'
)

plt.title('Matriz de Correlação')

plt.show()

print(
    df.groupby('is_repeated_guest')['is_canceled']
    .mean()
)
# Clientes recorrentes apresentaram menor taxa de cancelamento,
# indicando maior fidelização.

print(
    df.groupby('deposit_type')['is_canceled']
    .mean()
)
# Reservas com depósito tendem a reduzir cancelamentos,
# indicando maior comprometimento do cliente.


#alguns meses apresentaram maior número de reservas, 
# o que pode indicar sazonalidade.
sns.countplot(
    data=df,
    x='arrival_date_month'
)

plt.xticks(rotation=45)

plt.title('Reservas por Mês')

plt.show()

teste