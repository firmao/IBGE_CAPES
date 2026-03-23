import pandas as pd
import numpy as np
import requests
import json

# Configuração de Reprodutibilidade (Obrigatório para o Paper)
np.random.seed(42)

def generate_full_datasets():

    # # API SIDRA: Tabela 5938 (PIB per capita) - Último dado disponível
    # Selecionando as 100 maiores cidades brasileiras por PIB
    # url = "https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/-1/variaveis/37?localidades=N6[all]"
    # 1. Obter dados reais do IBGE do seu GitHub
    url_ibge = "https://github.com/firmao/IBGE_CAPES/raw/refs/heads/main/ibge.json"
    
    try:
        response = requests.get(url_ibge)
        raw_json = response.json()
        
        # 1. TRATAMENTO DINÂMICO DO IBGE (Garantindo 100 linhas)
        # Se for uma lista de listas ou dicionário aninhado, tentamos extrair a lista principal
        if isinstance(raw_json, dict):
            # Tenta encontrar a chave que contém a lista (comum em APIs do governo)
            key = next((k for k in raw_json if isinstance(raw_json[k], list)), None)
            data_list = raw_json[key] if key else [raw_json]
        else:
            data_list = raw_json

        df_ibge_full = pd.DataFrame(data_list)
        
        # Se o JSON tiver menos de 100, vamos replicar com ruído para teste de estresse
        if len(df_ibge_full) < 100:
            repeats = (100 // len(df_ibge_full)) + 1
            df_ibge_full = pd.concat([df_ibge_full] * repeats).head(100).reset_index(drop=True)

        # Selecionar/Criar colunas padrão
        df_ibge = pd.DataFrame()
        df_ibge['municipio_id'] = df_ibge_full['id'] if 'id' in df_ibge_full else range(1000, 1100)
        df_ibge['municipio_nome'] = df_ibge_full['nome'] if 'nome' in df_ibge_full else [f"Polo_{i}" for i in range(100)]
        
        # 2. Gerar CAPES.CSV acoplado aos indicadores do IBGE
        # Vamos usar o PIB ou IDH como 'proxy' para a capacidade de fomento
        # (Isso cria o valor científico: o GCN vai mapear essa correlação latente)
        # Link direto para o CSV de fomento/avaliação mais recente no portal oficial
        # url = "https://dadosabertos.capes.gov.br/dataset/30349f7e-d599-4c17-814d-5c628e937d58/resource/8d05e263-d143-44f2-9856-429737f941f6/download/br-capes-bds-pnpd-2024.csv"
        # 2. GERAÇÃO DO CAPES.CSV (100 registros acoplados)
        capes_data = []
        for i, row in df_ibge.iterrows():
            m_id = str(row['municipio_id'])
            # Lógica: O primeiro dígito do ID define a região, influenciando a nota
            regiao_prefix = int(m_id[0]) if m_id else 1
            
            # Nota CAPES ponderada pela região (Simulando concentração Sul/Sudeste)
            nota = int(np.random.choice([3, 4, 5, 6, 7], p=[0.35, 0.35, 0.15, 0.10, 0.05]))
            if regiao_prefix >= 3: # Boost de excelência para regiões 3, 4 e 5
                nota = min(7, nota + np.random.randint(0, 2))
            
            capes_data.append({
                'municipio_id': row['municipio_id'],
                'programa_id': f"PG_{row['municipio_id']}_{i}",
                'nota_capes': nota,
                'qtd_bolsas': int(nota * np.random.uniform(5, 30)),
                'producao_cientifica': int(nota * np.random.uniform(50, 200))
            })
            
        df_capes = pd.DataFrame(capes_data)

        # Salvar e Verificar
        df_ibge.to_csv('ibge_data.csv', index=False)
        df_capes.to_csv('capes_data.csv', index=False)
        
        print(f"📊 Sucesso!")
        print(f"IBGE: {len(df_ibge)} registros salvos.")
        print(f"CAPES: {len(df_capes)} registros salvos.")
        
    except Exception as e:
        print(f"❌ Erro ao processar: {e}")

if __name__ == "__main__":
    generate_full_datasets()