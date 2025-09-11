import nltk
import re
from rank_bm25 import BM25Okapi
import pickle
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import json
import logging  
import yaml
# from retriever import Retriever

logger=logging.getLogger("myapp")

# 下载必要的NLTK数据（首次运行时需要）
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')

class engishbm25indexer:
    def __init__(self, config,use_stemming=True, use_lemmatization=False, custom_stop_words=None):
        """
        初始化英文文档索引器
        
        Args:
            use_stemming: 是否使用词干提取
            use_lemmatization: 是否使用词形还原
            custom_stop_words: 自定义停用词列表
        """
        self.config=config
        if os.path.exists(config["bm25_index_name"]):
            self.load_index(config["bm25_index_name"])
                        # 初始化词干提取器和词形还原器
            self.stemmer = PorterStemmer() if use_stemming else None
            self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
            with open(config["doc_embedding_idlistname"],"r")as f:#这个存的是doc_id和向量下标的对应关系
                self.doc_embeddings_docid=json.load(f)#list
            logger.info(self.get_document_stats())
        else:
            self.documents = []
            self.processed_docs = []
            self.bm25 = None
            # 英文停用词
            self.stop_words = set(stopwords.words('english'))
            
            # 添加自定义停用词
            if custom_stop_words:
                self.stop_words.update(custom_stop_words)
            
            # 添加一些学术论文常见的停用词
            academic_stop_words = {
                 'et', 'al', 
            }
            self.stop_words.update(academic_stop_words)
            
            # 初始化词干提取器和词形还原器
            self.stemmer = PorterStemmer() if use_stemming else None
            self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
            
            self.use_stemming = use_stemming
            self.use_lemmatization = use_lemmatization

            with open(config["doc_embedding_idlistname"],"r")as f:#这个存的是doc_id和向量下标的对应关系
                self.doc_embeddings_docid=json.load(f)#list
            documents = []
            for item in self.doc_embeddings_docid:
                documents.append(item[0])
            self.add_documents(documents)
            self.save_index(config["bm25_index_name"])
    
    def preprocess_text(self, text):
        """
        英文文本预处理：清理、分词、去停用词、词干化/词形还原
        """
        # 转换为小写
        text = text.lower()
        
        # 去除URL、邮箱等
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 去除数字（可选，根据需要调整）
        # text = re.sub(r'\d+', '', text)
        
        # 去除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 分词
        tokens = word_tokenize(text)
        
        # 过滤处理
        processed_tokens = []
        for token in tokens:
            # 去除长度过短或过长的词
            if len(token) < 2 or len(token) > 20:
                continue
            
            # 去除停用词
            if token in self.stop_words:
                continue
            
            # 去除纯数字
            if token.isdigit():
                continue
            
            # 词干化
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            
            # 词形还原
            if self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def add_documents(self, documents):
        """
        添加文档到索引
        documents: 文档列表，每个元素是字符串
        """
        print(f"正在处理 {len(documents)} 个文档...")
        
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                print(f"已处理 {i} 个文档")
            
            self.documents.append(doc)
            processed_doc = self.preprocess_text(doc)
            self.processed_docs.append(processed_doc)
        
        print("正在构建BM25索引...")
        self.bm25 = BM25Okapi(self.processed_docs)
        print("索引构建完成！")
    
    def search(self, query, top_k=10):
        """
        搜索文档
        """
        if not self.bm25:
            return []
        
        # 预处理查询
        processed_query = self.preprocess_text(query)
        print(processed_query)
        if not processed_query:
            return []
        
        # 获取分数
        scores = self.bm25.get_scores(processed_query)
        
        # 获取top-k结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)#[:top_k]#在刚开始运行的时候，生成第一批评测样本的时候，这个reverse是true
        
        # results = []
        # for idx in top_indices:
        #     results.append({
        #         'document': self.documents[idx],
        #         'score': scores[idx],
        #         'index': idx,
        #         'preview': self.documents[idx][:200] + '...' if len(self.documents[idx]) > 200 else self.documents[idx],
        #         'docid':self.doc_embeddings_docid[idx][1][0]
        #     })
        
        return top_indices
    
    def get_document_stats(self):
        """
        获取文档库统计信息
        """
        if not self.processed_docs:
            return {}
        
        total_terms = sum(len(doc) for doc in self.processed_docs)
        avg_doc_length = total_terms / len(self.processed_docs)
        
        # 词汇表大小
        vocabulary = set()
        for doc in self.processed_docs:
            vocabulary.update(doc)
        
        return {
            'total_documents': len(self.documents),
            'total_terms': total_terms,
            'average_document_length': avg_doc_length,
            'vocabulary_size': len(vocabulary),
            'unique_terms': len(vocabulary)
        }
    
    def save_index(self, filepath):
        """
        保存索引到文件
        """
        index_data = {
            'documents': self.documents,
            'processed_docs': self.processed_docs,
            'bm25': self.bm25,
            'stop_words': self.stop_words,
            'use_stemming': self.use_stemming,
            'use_lemmatization': self.use_lemmatization
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"bm25索引已保存到 {filepath}")
    
    def load_index(self, filepath):
        """
        从文件加载索引
        """
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data['documents']
        self.processed_docs = index_data['processed_docs']
        self.bm25 = index_data['bm25']
        self.stop_words = index_data['stop_words']
        self.use_stemming = index_data.get('use_stemming', True)
        self.use_lemmatization = index_data.get('use_lemmatization', False)
        print(f"索引已从 {filepath} 加载")

# 文档加载工具函数
def load_documents_from_directory(directory, file_extensions=['.txt', '.md']):
    """
    从目录加载文档
    """
    documents = []
    filenames = []
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in file_extensions):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():  # 只添加非空文档
                        documents.append(content)
                        filenames.append(filename)
            except Exception as e:
                print(f"无法读取文件 {filename}: {e}")
    
    return documents, filenames

def load_documents_from_file_list(file_paths):
    """
    从文件路径列表加载文档
    """
    documents = []
    filenames = []
    
    for filepath in file_paths:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if content.strip():
                    documents.append(content)
                    filenames.append(os.path.basename(filepath))
        except Exception as e:
            print(f"无法读取文件 {filepath}: {e}")
    
    return documents, filenames

# 使用示例
def main():
    # 初始化索引器
    indexer = engishbm25indexer(
        use_stemming=True,  # 使用词干化
        use_lemmatization=False,  # 不使用词形还原（二选一即可）
        custom_stop_words=['doi', 'arxiv', 'pdf', 'htm', 'html']  # 添加自定义停用词
    )
    
    # 示例英文文档
    documents = [
        """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        in contrast to the natural intelligence displayed by humans and animals. 
        Leading AI textbooks define the field as the study of "intelligent agents": 
        any device that perceives its environment and takes actions that maximize 
        its chance of successfully achieving its goals.
        """,
        """
        Machine learning (ML) is a type of artificial intelligence (AI) that allows 
        software applications to become more accurate at predicting outcomes without 
        being explicitly programmed to do so. Machine learning algorithms use historical 
        data as input to predict new output values.
        """,
        """
        Deep learning is part of a broader family of machine learning methods based on 
        artificial neural networks with representation learning. Learning can be 
        supervised, semi-supervised or unsupervised. Deep learning architectures such as 
        deep neural networks, deep belief networks, recurrent neural networks and 
        convolutional neural networks have been applied to fields including computer vision.
        """,
        """
        Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and 
        human language, in particular how to program computers to process and analyze 
        large amounts of natural language data.
        """
    ]
    
    # 或者从目录加载文档
    # documents, filenames = load_documents_from_directory('path/to/your/papers')
    
    # 构建索引
    indexer.add_documents(documents)
    
    # 显示统计信息
    stats = indexer.get_document_stats()
    print("\n文档库统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 搜索示例
    queries = [
        "machine learning algorithms",
        "neural networks deep learning",
        "natural language processing",
        "artificial intelligence applications"
    ]
    
    for query in queries:
        print(f"\n搜索查询：{query}")
        results = indexer.search(query, top_k=3)
        
        print("搜索结果：")
        for i, result in enumerate(results, 1):
            print(f"{i}. 分数: {result['score']:.4f}")
            print(f"   预览: {result['preview']}")
            print()
    
    # 保存索引
    indexer.save_index('english_papers_bm25_index.pkl')



if __name__ == "__main__":
    # Load configuration
    with open('/data4/students/zhangguangyin/chatNum/config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize Retriever
    retriever = engishbm25indexer(config)
    with open(config["doc_embedding_idlistname"],"r")as f:#这个存的是doc_id和向量下标的对应关系
            doc_embeddings_docid=json.load(f)#list
    # Example query
    user_query = "metric SSIM Video Prediction task dataset Moving MNIST (Moving MNIST)"
    retrieved_docs = retriever.search(user_query)[:10]

    for i in retrieved_docs:
        print(doc_embeddings_docid[i][0])

