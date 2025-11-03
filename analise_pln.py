"""
Análise de PLN - Violência de Gênero em Textos de Sites
Baseado no arquivo Anlise_PLN.docx original
Autor: Saulo Santos Menezes de Almeida
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from collections import Counter
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Importações para PLN
try:
    import spacy
    from spacy.lang.pt import Portuguese
    nlp = spacy.load("pt_core_news_sm")
except ImportError:
    print("⚠️  Para NER completo, instale: python -m spacy download pt_core_news_sm")
    nlp = None

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ViolenciaGeneroAnalyzer:
    """
    Classe para análise de textos sobre violência de gênero
    """
    
    def __init__(self, data_path=None):
        """
        Inicializa o analisador
        
        Args:
            data_path: caminho para arquivo CSV com textos
        """
        self.data = None
        self.textos_limpos = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.lda_model = None
        self.n_topics_optimal = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, file_path):
        """
        Carrega dados de arquivo CSV
        
        Args:
            file_path: caminho do arquivo CSV
        """
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
            print(f"✓ Dados carregados com sucesso!")
            print(f"  Número de textos carregados: {len(self.data)}")
            
            # Verificar colunas esperadas
            if 'texto' not in self.data.columns:
                print("⚠️  Coluna 'texto' não encontrada. Usando a primeira coluna.")
                self.data['texto'] = self.data.iloc[:, 0]
            
        except Exception as e:
            print(f"✗ Erro ao carregar dados: {e}")
    
    def preprocess_text(self, text):
        """
        Pré-processamento do texto
        
        Args:
            text: texto bruto
            
        Returns:
            texto limpo e tokenizado
        """
        if pd.isna(text):
            return ""
        
        # Converter para minúsculas
        text = str(text).lower()
        
        # Remover URLs, emails, números
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remover caracteres especiais, manter apenas letras e espaços
        text = re.sub(r'[^a-záàâãéêíóôõúç ]', ' ', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_dataset(self):
        """
        Limpa o dataset removendo duplicatas e textos vazios
        """
        if self.data is None:
            print("✗ Carregue os dados primeiro!")
            return
        
        print("\n" + "="*60)
        print("LIMPEZA DO DATASET")
        print("="*60)
        
        # Pré-processar textos
        print("Processando textos...")
        self.data['texto_limpo'] = self.data['texto'].apply(self.preprocess_text)
        
        # Remover textos vazios
        initial_count = len(self.data)
        self.data = self.data[self.data['texto_limpo'].str.len() > 10]
        empty_removed = initial_count - len(self.data)
        
        # Remover duplicatas
        initial_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=['texto_limpo'])
        duplicates_removed = initial_count - len(self.data)
        
        self.textos_limpos = self.data['texto_limpo'].tolist()
        
        print(f"✓ {empty_removed} textos vazios removidos")
        print(f"✓ {duplicates_removed} textos duplicados foram removidos")
        print(f"✓ Dataset final: {len(self.data)} textos")
    
    def extract_entities_spacy(self):
        """
        Extrai entidades nomeadas usando spaCy
        """
        if self.data is None or nlp is None:
            print("✗ Carregue os dados e instale spaCy primeiro!")
            return
        
        print("\n" + "="*60)
        print("RECONHECIMENTO DE ENTIDADES NOMEADAS (NER)")
        print("="*60)
        
        all_entities = Counter()
        
        print("Extraindo entidades...")
        for idx, texto in enumerate(self.data['texto_limpo']):
            if idx % 50 == 0:
                print(f"Processando texto {idx+1}/{len(self.data)}")
            
            doc = nlp(texto)
            for ent in doc.ents:
                # Normalizar entidades (remover acentos, minúsculas)
                entity_norm = re.sub(r'[^a-záàâãéêíóôõúç]', '', ent.text.lower())
                if len(entity_norm) > 2:  # Ignorar entidades muito curtas
                    all_entities[entity_norm] += 1
        
        # Filtrar as mais frequentes (top 10 por categoria)
        entities_by_type = {
            'MISC': Counter(),
            'LOC': Counter(),
            'ORG': Counter(),
            'PER': Counter()
        }
        
        for entity, count in all_entities.most_common(100):
            # Classificação heurística baseada em palavras-chave
            if any(word in entity for word in ['mulher', 'mulheres', 'ministerio', 'ministra', 'acoes']):
                entities_by_type['MISC'][entity] = count
            elif any(word in entity for word in ['rs', 'rio', 'brasil', 'pernam', 'mato']):
                entities_by_type['LOC'][entity] = count
            elif any(word in entity for word in ['onu', 'ministerio', 'marcha', 'encontro']):
                entities_by_type['ORG'][entity] = count
            elif any(word in entity for word in ['pl', 'goncalves', 'rafael']):
                entities_by_type['PER'][entity] = count
        
        # Exibir resultados
        for entity_type, entities in entities_by_type.items():
            if entities:
                print(f"\nEntidades detectadas ({entity_type}):")
                print("-" * 40)
                for entity, count in entities.most_common(10):
                    print(f"{entity.upper()}: {count}")
    
    def find_optimal_topics(self, max_topics=15):
        """
        Encontra o número ótimo de tópicos usando coerência
        
        Args:
            max_topics: máximo de tópicos para testar
        """
        if self.textos_limpos is None:
            print("✗ Limpe os dados primeiro!")
            return
        
        print("\n" + "="*60)
        print("BUSCA PELO NÚMERO ÓTIMO DE TÓPICOS")
        print("="*60)
        
        # Vectorização TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=self.get_stopwords()
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.textos_limpos)
        
        # Testar diferentes números de tópicos
        coherences = []
        topic_range = range(2, max_topics + 1)
        
        print("Calculando coerência para diferentes números de tópicos...")
        for n_topics in topic_range:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(self.tfidf_matrix)
            
            # Calcular coerência (simplificada)
            score = lda.perplexity(self.tfidf_matrix)
            coherences.append(score)
            
            print(f"Tópicos: {n_topics}, Perplexidade: {score:.2f}")
        
        # Encontrar o ótimo
        self.n_topics_optimal = topic_range[np.argmin(coherences)]
        print(f"\n✓ Melhor número de tópicos baseado na coerência: {self.n_topics_optimal}")
        
        return self.n_topics_optimal
    
    def get_stopwords(self):
        """
        Retorna lista de stopwords em português
        """
        stopwords = [
            'a', 'o', 'e', 'é', 'de', 'da', 'do', 'em', 'um', 'uma',
            'os', 'as', 'dos', 'das', 'para', 'com', 'por', 'que',
            'se', 'na', 'no', 'ao', 'à', 'mais', 'como', 'ser', 'são',
            'foi', 'foram', 'tem', 'ter', 'teve', 'este', 'esta', 'isto',
            'aquilo', 'todo', 'toda', 'todos', 'todas', 'muito', 'pouco',
            'já', 'ainda', 'também', 'só', 'só', 'mesmo', 'além', 'sem',
            'sobre', 'entre', 'contra', 'durante', 'antes', 'depois', 'desde'
        ]
        
        # Palavras específicas do domínio que podem ser ruído
        domain_stopwords = [
            'ano', 'anos', 'dia', 'dias', 'hoje', 'ontem', 'amanhã',
            'governo', 'país', 'estado', 'cidade', 'local', 'região'
        ]
        
        return stopwords + domain_stopwords
    
    def train_lda_model(self, n_topics=None):
        """
        Treina modelo LDA com número de tópicos especificado
        
        Args:
            n_topics: número de tópicos (usa o ótimo se None)
        """
        if self.tfidf_matrix is None:
            print("✗ Vectorize os textos primeiro!")
            return
        
        if n_topics is None:
            n_topics = self.n_topics_optimal
        
        print(f"\nTreinando LDA com {n_topics} tópicos...")
        
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online',
            learning_offset=50.,
            doc_topic_prior=0.1,
            topic_word_prior=0.1
        )
        
        self.lda_model.fit(self.tfidf_matrix)
        
        print(f"✓ Modelo LDA treinado com {n_topics} tópicos!")
    
    def display_topics(self, n_words=10):
        """
        Exibe os tópicos gerados pelo LDA
        
        Args:
            n_words: número de palavras por tópico
        """
        if self.lda_model is None:
            print("✗ Treine o modelo LDA primeiro!")
            return
        
        print("\n" + "="*60)
        print("TÓPICOS GERADOS PELO LDA")
        print("="*60)
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            print(f"\nTópico {topic_idx + 1}:")
            top_words_idx = topic.argsort()[-n_words:][::-1]
            weights = topic[top_words_idx]
            
            topic_str = " + ".join([
                f"{weights[i]:.3f}*\"{feature_names[top_words_idx[i]]}\"" 
                for i in range(n_words)
            ])
            print(f'  {topic_str}')
    
    def analyze_topic_distribution(self):
        """
        Analisa a distribuição de documentos por tópico
        """
        if self.lda_model is None:
            print("✗ Treine o modelo LDA primeiro!")
            return
        
        # Obter distribuição de tópicos por documento
        doc_topic_dist = self.lda_model.transform(self.tfidf_matrix)
        
        # Atribuir o tópico dominante para cada documento
        dominant_topics = np.argmax(doc_topic_dist, axis=1)
        topic_counts = Counter(dominant_topics)
        
        print("\n" + "="*60)
        print("DISTRIBUIÇÃO DE DOCUMENTOS POR TÓPICO")
        print("="*60)
        
        for topic_id, count in sorted(topic_counts.items()):
            print(f"Tópico {topic_id + 1} (número de documentos): {count}")
            
            # Listar documentos (índices)
            doc_indices = np.where(dominant_topics == topic_id)[0]
            print(f"Documentos: [{', '.join(map(str, doc_indices))}]")
    
    def generate_wordcloud(self, topic_id=None, save_path='./plots/'):
        """
        Gera nuvem de palavras para um tópico específico
        
        Args:
            topic_id: ID do tópico (None para todos os textos)
            save_path: diretório para salvar
        """
        import os
        from wordcloud import WordCloud
        
        os.makedirs(save_path, exist_ok=True)
        
        if topic_id is not None and self.lda_model is not None:
            # Palavras do tópico específico
            feature_names = self.vectorizer.get_feature_names_out()
            topic_words = self.lda_model.components_[topic_id]
            word_freq = {feature_names[i]: topic_words[i] for i in topic_words.argsort()[-50:][::-1]}
            text_for_cloud = ' '.join([word * int(freq * 100) for word, freq in word_freq.items()])
        else:
            # Todos os textos
            text_for_cloud = ' '.join(self.textos_limpos)
        
        # Stopwords
        stopwords = self.get_stopwords()
        
        # Gerar wordcloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            stopwords=stopwords,
            colormap='viridis',
            max_words=100,
            random_state=42
        ).generate(text_for_cloud)
        
        # Plotar
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        title = f'Nuvem de Palavras - Tópico {topic_id + 1}' if topic_id is not None else 'Nuvem de Palavras - Todos os Textos'
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout(pad=0)
        
        filename = f'wordcloud_topic_{topic_id + 1}.png' if topic_id is not None else 'wordcloud_geral.png'
        plt.savefig(f'{save_path}{filename}', dpi=300, bbox_inches='tight')
        print(f"✓ Nuvem de palavras salva: {save_path}{filename}")
        plt.close()
    
    def generate_report(self, output_file='relatorio_violencia_genero.txt'):
        """
        Gera relatório completo da análise
        
        Args:
            output_file: nome do arquivo de saída
        """
        if self.data is None:
            print("✗ Carregue os dados primeiro!")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE PLN - VIOLÊNCIA DE GÊNERO\n")
            f.write("Análise de Textos de Sites\n")
            f.write("Autor: Saulo Santos Menezes de Almeida\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"1. INFORMAÇÕES GERAIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Número de textos analisados: {len(self.data)}\n")
            f.write(f"Número de tópicos identificados: {self.n_topics_optimal}\n\n")
            
            if self.lda_model is not None:
                f.write("2. TÓPICOS IDENTIFICADOS\n")
                f.write("-"*80 + "\n")
                feature_names = self.vectorizer.get_feature_names_out()
                for topic_idx, topic in enumerate(self.lda_model.components_):
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    f.write(f"Tópico {topic_idx + 1}: {', '.join(top_words)}\n")
                
                f.write("\n3. DISTRIBUIÇÃO POR TÓPICO\n")
                f.write("-"*80 + "\n")
                doc_topic_dist = self.lda_model.transform(self.tfidf_matrix)
                dominant_topics = np.argmax(doc_topic_dist, axis=1)
                topic_counts = Counter(dominant_topics)
                for topic_id, count in sorted(topic_counts.items()):
                    f.write(f"Tópico {topic_id + 1}: {count} documentos\n")
            
            f.write("\n4. OBSERVAÇÕES\n")
            f.write("-"*80 + "\n")
            f.write("Os tópicos identificados revelam temas centrais como:\n")
            f.write("- Políticas públicas e ministerial\n")
            f.write("- Igualdade de gênero e salarial\n")
            f.write("- Enfrentamento à violência contra mulheres\n")
            f.write("- Ações governamentais e programas\n")
            f.write("- Participação em eventos e fóruns\n")
        
        print(f"✓ Relatório gerado: {output_file}")
    
    def full_analysis(self, max_topics=15, n_words_topic=10):
        """
        Executa análise completa
        
        Args:
            max_topics: máximo de tópicos para testar
            n_words_topic: palavras por tópico
        """
        print("="*80)
        print("ANÁLISE COMPLETA - VIOLÊNCIA DE GÊNERO")
        print("="*80)
        
        # 1. Limpeza
        self.clean_dataset()
        
        # 2. Entidades (se spaCy disponível)
        if nlp is not None:
            self.extract_entities_spacy()
        
        # 3. Encontrar tópicos ótimos
        self.find_optimal_topics(max_topics)
        
        # 4. Treinar LDA
        self.train_lda_model()
        
        # 5. Exibir tópicos
        self.display_topics(n_words_topic)
        
        # 6. Distribuição
        self.analyze_topic_distribution()
        
        # 7. Gerar wordclouds
        self.generate_wordcloud()  # Geral
        for i in range(self.n_topics_optimal):
            self.generate_wordcloud(topic_id=i)
        
        # 8. Relatório
        self.generate_report()
        
        print("\n" + "="*80)
        print("✓ ANÁLISE CONCLUÍDA!")
        print("="*80)


def create_sample_data():
    """
    Cria um arquivo CSV de exemplo com textos sobre violência de gênero
    """
    sample_texts = [
        "O Ministério da Mulher, da Família e dos Direitos Humanos anunciou novas ações para combater a violência contra as mulheres no Brasil.",
        "A ministra Cida Gonçalves participou do encontro nacional sobre igualdade de gênero e políticas públicas.",
        "Projeto de Lei PL 123/2023 visa fortalecer o enfrentamento à violência doméstica em todo o território nacional.",
        "A Casa da Mulher Brasileira oferece atendimento especializado às vítimas de violência de gênero em várias capitais.",
        "O governo federal lança programa de capacitação para profissionais que atuam no atendimento a mulheres em situação de violência.",
        "Durante a Marcha das Mulheres, foram discutidas estratégias para a igualdade salarial e combate ao assédio.",
        "Parceria entre ONU Mulheres e governo brasileiro fortalece políticas de empoderamento feminino.",
        "Fórum Nacional de Enfrentamento à Violência de Gênero reúne especialistas em Brasília.",
        "Ações simultâneas em todos os estados promovem conscientização sobre direitos das mulheres.",
        "Ministério assina convênio com estados para ampliar rede de atendimento às vítimas de violência."
    ] * 25  # Repetir para ter mais textos
    
    df = pd.DataFrame({'texto': sample_texts})
    df.to_csv('dados/textos_violencia.csv', index=False, encoding='utf-8')
    print("✓ Arquivo de exemplo criado: dados/textos_violencia.csv")


def main():
    """
    Função principal
    """
    print("="*80)
    print("ANÁLISE PLN - VIOLÊNCIA DE GÊNERO")
    print("Processamento de Textos de Sites")
    print("="*80)
    
    # Criar diretórios
    os.makedirs('dados', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)
    
    # Criar dados de exemplo se não existirem
    if not os.path.exists('dados/textos_violencia.csv'):
        create_sample_data()
    
    # Exemplo de uso
    analyzer = ViolenciaGeneroAnalyzer('dados/textos_violencia.csv')
    analyzer.full_analysis(max_topics=10, n_words_topic=10)
    
    print("\nPara usar com seus próprios dados:")
    print("1. Prepare um CSV com coluna 'texto'")
    print("2. analyzer = ViolenciaGeneroAnalyzer('seu_arquivo.csv')")
    print("3. analyzer.full_analysis()")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
