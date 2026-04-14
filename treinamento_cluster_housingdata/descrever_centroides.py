import pickle
import pandas as pd


def carregar_modelo(nome_modelo):
    '''
    carregar o modelo de clusters
    '''
    modelo = pickle.load(open(nome_modelo, 'rb'))

    return modelo

def carregar_normalizador(nome_normalizador):
    '''
    carregar o normalizador dos dados
    '''
    normalizador = pickle.load(open(nome_normalizador, 'rb'))

    return normalizador

def desnormalizar_centroides(centroides, normalizador):
    '''
    desnormaliza o dataframe dos centroides para melhor analise em escala real dos dados
    '''
    centroides_desnorm = normalizador.inverse_transform(centroides)
    #print(centroides_desnorm)

    return centroides_desnorm


def comparar_valores(valor_cluster, media, margem=0.10):
    '''
    compara cada valor de uma linha de cluster com a media geral de cada coluna dos dados originais, respeitando uma margem de 10% para evitar classificar pequenas variações como altas ou baixas
    '''
    if valor_cluster > media * (1 + margem):
        return 'alto'

    elif valor_cluster < media * (1 - margem):
        return 'baixo'

    else:
        return 'medio'


def interpretar_cluster(linha, media_geral, indice):
    '''
    interpreta cada linha dos clusters, de acordo com a comparação entre cada coluna do dataset e com a média geral de ++
    '''
    descricoes = mensagens_colunas()
    print(f'\n>> Cluster: {indice}')

    for coluna in linha.index:

        # ignora colunas sem descrição
        if coluna not in descricoes:
            continue
        
        valor_cluster = linha[coluna]
        media_coluna = media_geral[coluna]

        if coluna == 'CHAS':
            if valor_cluster >= 0.5:
                resultado = 'alto'
            elif valor_cluster <= 0.1:
                resultado = 'baixo'
            else:
                resultado = 'medio'
        else:
            resultado = comparar_valores(valor_cluster, media_coluna)


        frase = descricoes[coluna][resultado]

        print(f'- {coluna}: {valor_cluster:.2f} -> {frase}')
    

def mensagens_colunas():
    '''
    CRIM - maior = pior criminalidade
    ZN - maior = bairros residencias amplos
    INDUS - maior = maior presença de industrias
    CHAS - >= 0.5 -> próximos ao rio Charles
    NOX - maior = maior concentração de óxidos nítricos
    RM - maior = casas maiores
    AGE - maior - ímoveis mais antigos
    DIS - maior - mais distantes dos centros de emprego
    RAD - maior - melhor acesso rodoviário
    TAX - maior = imposto alto
    PTRATIO - maior = mais alunos por professor
    B - maior = maior proporção de população negra na região
    LSTAT - maior = população com menor status socioeconômico
    MEDV - maior = casas mais caras
    '''
    descricoes_cluster = {
        'CRIM': {
            'alto': 'alta criminalidade',
            'baixo': 'baixa criminalidade',
            'medio': 'criminalidade mediana'
        },

        'ZN': {
            'alto': 'maior presença de grandes lotes residenciais',
            'baixo': 'menor presença de grandes lotes residenciais',
            'medio': 'presença mediana de grandes lotes residenciais'
        },

        'INDUS': {
            'alto': 'maior presença industrial',
            'baixo': 'menor presença industrial',
            'medio': 'presença industrial mediana'
        },

        'CHAS': {
            'alto': 'maior presença de imóveis próximos ao rio Charles',
            'baixo': 'menor presença de imóveis próximos ao rio Charles',
            'medio': 'presença moderada de imóveis próximos ao rio Charles'
        },

        'NOX': {
            'alto': 'maior nível de poluição do ar com óxidos nítricos',
            'baixo': 'menor nível de poluição do ar com óxidos nítricos',
            'medio': 'nível mediano de poluição do ar com óxidos nítricos'
        },

        'RM': {
            'alto': 'casas maiores',
            'baixo': 'casas menores',
            'medio': 'casas de tamanho médio'
        },

        'AGE': {
            'alto': 'imóveis mais antigos',
            'baixo': 'imóveis mais novos',
            'medio': 'imóveis com idade mediana'
        },

        'DIS': {
            'alto': 'mais distante dos centros de emprego',
            'baixo': 'mais próximo dos centros de emprego',
            'medio': 'distância mediana dos centros de emprego'
        },

        'RAD': {
            'alto': 'melhor acesso a rodovias principais',
            'baixo': 'menor acesso a rodovias principais',
            'medio': 'acesso mediano a rodovias principais'
        },

        'TAX': {
            'alto': 'impostos mais altos',
            'baixo': 'impostos mais baixos',
            'medio': 'impostos medianos'
        },

        'PTRATIO': {
            'alto': 'maior número de alunos por professor',
            'baixo': 'melhor proporção aluno-professor',
            'medio': 'proporção aluno-professor mediana'
        },

        'B': {
            'alto': 'maior proporção de população negra na região',
            'baixo': 'menor proporção de população negra na região',
            'medio': 'proporção mediana de população negra na região '
        },
        'LSTAT': {
            'alto': 'maior vulnerabilidade socioeconômica',
            'baixo': 'melhor condição socioeconômica',
            'medio': 'condição socioeconômica mediana'
        },

        'MEDV': {
            'alto': 'imóveis mais valorizados',
            'baixo': 'imóveis menos valorizados',
            'medio': 'valor mediano dos imóveis'
        }
    }

    return descricoes_cluster
    
def descrever_clusters(dados_norm, modelo, normalizador, media_geral):
    '''
    descrever de fato o que cada cluster representa e que informações conseguiu tirar dos dados
    '''
    # converter os centroides em dataframe
    centroides = pd.DataFrame(modelo.cluster_centers_, columns=dados_norm.columns)
    centroides_reais = desnormalizar_centroides(centroides, normalizador)

    df_centroides_reais = pd.DataFrame(centroides_reais, columns=dados_norm.columns)
    #print(modelo.cluster_centers_)
    #print(centroides)
    #print(df_centroides_reais)

    for i, linha in df_centroides_reais.iterrows():
        interpretar_cluster(linha, media_geral, i)
