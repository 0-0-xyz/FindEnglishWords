"""
================================================================================
单词相似检索
功能：
   -自定义混淆词相似匹配 77-80
   -中文释义相似匹配
   -传统机器学习模型（逻辑回归、随机森林、GBDT）删除SVM 227-353
   -深度学习嵌入（LSTM/CNN 字形向量、BERT 中文语义向量） 359-504
   -无监督聚类（KMeans、层次聚类）508-521
   -相似度加权融合、模型评估与可视化 860-891
   -批量+实时交互 893-1096 主程序
================================================================================
"""
import heapq
import re
from collections import defaultdict

import numpy as np#数学计算
import pandas as pd#处理表格数据
import matplotlib.pyplot as plt#画图
import seaborn as sns
#Sklearn包含了机器学习的逻辑回归、随机森林等经典算法。
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch#深度学习的核心
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel #深度学习的框架
from sentence_transformers import SentenceTransformer
import Levenshtein #用来衡量两个单词拼写差多少。
import nltk
import jieba#中文分词
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm  # 进度条库
import os
#并行 ↓
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import threading
import queue


# 设置 Hugging Face 镜像源
#代码中使用了SentenceTransformer和BertTokenizer需要用到Hugging Face (HF)资源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print("已设置 HF_ENDPOINT 镜像源: https://hf-mirror.com") # 仅用于确认

# 下载 nltk 数据（首次运行）
#nltk.download('punkt', quiet=True)
#nltk.download('stopwords', quiet=True)
#nltk.download('wordnet', quiet=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
#忽略警告，使得结果页面更清晰

#全局配置
DATA_FILE = "English_words.csv"  #输入数据
OUTPUT_FILE = "word_similarity_output.csv"#输出
BATCH_SIZE = 32#批处理大小
EMBEDDING_DIM = 128#词嵌入维度，向量维度（把单词转成128个数字）
LSTM_HIDDEN = 64 # LSTM隐藏层大小，每一层有 64 个“记忆单元”
LSTM_LAYERS = 2 #网络有2层
CNN_FILTERS = 100 #CNN过滤器数量
USE_CUDA = torch.cuda.is_available() # 是否有显卡
DEVICE = torch.device("cuda" if USE_CUDA else "cpu") # 用显卡还是CPU

#自定义字形相似规则参数
MAX_LEN_DIFF = 3 #两个单词长度差超过3个字母就不考虑了
ALLOWED_SWAP = True  #允许相邻字母交换
CHAR_COINCIDENCE_THRESHOLD = 0.7 #字母重合度超过70%才算相似


# 数据预处理
class DataPreprocessor:
    """数据清洗、英文词干提取、中文分词"""

    def __init__(self):
        print("  初始化预处理器（加载 NLTK 资源）...")
        self.stemmer = PorterStemmer() # 词干提取器，得词根
        self.lemmatizer = WordNetLemmatizer() # 词形还原器，变原型
        self.stopwords = set(stopwords.words('english')) # 英文停用词
        self.ch_stopwords = set(['的', '了', '和', '与', '或', '一个', '一种', '这个', '那个']) # 中文停用词

    # 得word_clean 清洗后的英文单词
    def clean_word(self, word):
        """英文单词标准化"""
        word = str(word).strip().lower() # 转小写，去掉首尾空格
        word = re.sub(r'[^a-z]', '', word)  # 仅保留字母
        return word

    # 得meaning_clean 清洗后的中文释义
    def clean_meaning(self, meaning):
        """中文释义清洗"""
        meaning = str(meaning).strip()#去空白字符
        meaning = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，,。？?！!；;]', '', meaning)
        return meaning

    # 得meaning_tokens 	中文分词结果
    def tokenize_chinese(self, text):
        """中文分词并去除停用词"""
        words = jieba.lcut(text) #jieba分词
        words = [w for w in words if w not in self.ch_stopwords and len(w.strip()) > 0]#去停用词
        return words

    # 得word_stem 词干提取结果
    def stem_english(self, word):#词干提取
        return self.stemmer.stem(word)

    # 得word_lemma 词形还原结果
    def lemmatize_english(self, word):#词形还原
        return self.lemmatizer.lemmatize(word)

    def preprocess_dataframe(self, df):
        """对 dataframe 进行全量预处理"""
        print("  清洗英文单词...")
        df['word_clean'] = df['word'].apply(self.clean_word)
        print("  清洗中文释义...")
        df['meaning_clean'] = df['meaning'].apply(self.clean_meaning)
        print("  提取词干...")
        df['word_stem'] = df['word_clean'].apply(self.stem_english)
        print("  词形还原...")
        df['word_lemma'] = df['word_clean'].apply(self.lemmatize_english)
        print("  中文分词...")
        tqdm.pandas(desc="  分词进度")
        df['meaning_tokens'] = df['meaning_clean'].progress_apply(self.tokenize_chinese)
        return df

#特征工程（字形相似特征）
class ShapeFeatureExtractor:
    """构造两个单词之间的字符级特征向量"""

    @staticmethod
    def char_coincidence_ratio(w1, w2):#两个单词用了多少相同的字母（不管顺序）
        """字符重合占比（基于集合）"""
        set1, set2 = set(w1), set(w2)# 转换成集合（去重），集合 set 特点：同一个字母只保留 1 次
        if not set1 or not set2:#如果其中一个单词是空字符串，直接返回相似度 0，避免分母为 0 崩溃。
            return 0.0
        return len(set1 & set2) / len(set1 | set2) # 交集（都有字母） ÷ 并集（两者全部字母）

    @staticmethod
    def sequence_similarity(w1, w2):#最长公共子序列的长度比例（考虑顺序）
        """基于最长公共子序列的长度比例"""
        common = sum(
            c1 == c2
            for c1, c2 in zip(w1, w2)
        )
        return common / max(
            len(w1),
            len(w2)
        )

    @staticmethod
    def ngram_similarity(w1, w2, n=2):# 把单词切成连续的n个字母的小片段
        """N-Gram 余弦相似度
        和char_coincidence_ratio类似，char……处理字母，ngram……处理单词片段
        """

        def get_ngrams(s, n):#切分单词
            return [s[i:i + n] for i in range(len(s) - n + 1)]

        grams1 = set(get_ngrams(w1, n))
        grams2 = set(get_ngrams(w2, n))
        #把切好的片段列表转集合，重复片段只保留一份。
        if not grams1 or not grams2:#单词长度小于 n，直接返回相似度 0，防止分母为 0 报错。
            return 0.0
        inter = len(grams1 & grams2)#找交集
        return inter / (len(grams1) + len(grams2) - inter)

    @staticmethod
    def root_match(w1_stem, w2_stem):
        """词根是否相同（词干匹配）"""
        return 1 if w1_stem == w2_stem else 0

    @staticmethod
    def length_diff(w1, w2):#长度相似度
        """长度差（归一化）"""
        diff = abs(len(w1) - len(w2))#长度差
        max_len = max(len(w1), len(w2))#最大长度
        return diff / max_len if max_len > 0 else 0

    @staticmethod
    def edit_distance_norm(w1, w2):
        """归一化编辑距离"""
        dist = Levenshtein.distance(w1, w2)#距离越小越相似
        #编辑距离（Levenshtein距离）是把一个单词变成另一个，需要的最少操作次数（增、删、改）
        max_len = max(len(w1), len(w2))
        return dist / max_len if max_len > 0 else 1.0

    @staticmethod
    def position_shift(w1, w2):#相同字母在两个单词中的位置差了多少
        """衡量字母位置偏移程度（基于索引差的平均）"""
        # 简单实现：遍历较短单词，找在长单词中的位置偏移
        if len(w1) == 0 or len(w2) == 0:
            return 1.0
        shifts = []
        for i, ch in enumerate(w1):
            if ch in w2:
                j = w2.find(ch)
                shifts.append(abs(i - j))
        if not shifts:
            return 1.0
        return np.mean(shifts) / max(len(w1), len(w2))#平均偏移/单词最大长度

    @staticmethod
    def extract_features(word1, word2, stem1, stem2):#提取全部7个特征
        """提取全部特征，返回向量（长度为7）"""
        f1 = ShapeFeatureExtractor.char_coincidence_ratio(word1, word2)
        f2 = ShapeFeatureExtractor.sequence_similarity(word1, word2)
        f3 = ShapeFeatureExtractor.ngram_similarity(word1, word2, 2)
        f4 = ShapeFeatureExtractor.root_match(stem1, stem2)
        f5 = 1 - ShapeFeatureExtractor.length_diff(word1, word2)  # 转为相似度
        f6 = 1 - ShapeFeatureExtractor.edit_distance_norm(word1, word2)
        f7 = 1 - ShapeFeatureExtractor.position_shift(word1, word2)
        return np.array([f1, f2, f3, f4, f5, f6, f7])

#传统机器学习模型
class ShapeMLModel:
    """训练和预测字形相似性"""

    def __init__(self):
        self.models = {
            'LR': LogisticRegression(max_iter=1000),
            #在大数据集下，SVM 训练复杂度是 O(N^3)，极其缓慢。
            # 'SVM': SVC(probability=True),

            #限制随机森林和 GBDT 的最大深度和样本分裂数，防止大数据下树无限生长导致训练慢、过拟合
            'RF': RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42),
            'GBDT': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42 )
        }
        self.best_model =None
        self.scaler = StandardScaler()

    #自动生成训练集
    def generate_training_data(self, df, feature_extractor, num_neg_samples=1):
        """基于规则生成正负样本（模拟人工标注）
           num_neg_samples: 负样本数量与正样本数量之比，1：1"""
        X, y = [], [] # X是特征，y是标签（1=相似，0=不相似）
        words = df['word_clean'].tolist() # 所有单词列表，pandas序列转换成普通Python列表
        stems = df['word_stem'].tolist()# 所有词干列表
        n = len(words)

        # ---------- 正样本 ----------
        print("  生成正样本（遍历单词对）...")
        for i in tqdm(range(n), desc="  正样本生成进度"):
            for j in range(i + 1, min(i + 50, n)):  #只取当前单词后面最多 50 个单词配对，限制配对数，防止爆炸
                w1, w2 = words[i], words[j]
                s1, s2 = stems[i], stems[j]
                if abs(len(w1) - len(w2)) <= MAX_LEN_DIFF:#两个单词长度差不能超过阈值 MAX_LEN_DIFF，长度差太大直接跳过
                    sim = feature_extractor.extract_features(w1, w2, s1, s2)#提取全套相似度特征 sim
                    if sim[1] > 0.6 and sim[5] > 0.5: # 序列相似度>0.6 且 编辑距离相似度>0.5->正样本
                        X.append(sim)
                        y.append(1)

        positive_count = len(y)
        print(f"  正样本数量: {positive_count}")

        # ---------- 负样本 ----------
        import random
        target_negative = int(positive_count * num_neg_samples)#计算负样本数
        print(f"  目标负样本数量: {target_negative}")

        if target_negative <= 0:
            target_negative = positive_count  #

        pbar = tqdm(total=target_negative, desc="  负样本生成进度")
        attempts = 0
        max_attempts = target_negative * 10  # 防止无限循环
        while len(y) - positive_count < target_negative and attempts < max_attempts:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)#随机抽两个不同单词
            if i == j:# 跳过同一个单词自己和自己配对
                continue
            w1, w2 = words[i], words[j]
            s1, s2 = stems[i], stems[j]
            # 判定为不相似的条件（与正样本条件相反）长度差大，或者拼写差异极大
            if abs(len(w1) - len(w2)) > MAX_LEN_DIFF or Levenshtein.ratio(w1, w2) < 0.3:
                # 避免重复添加相同对
                X.append(feature_extractor.extract_features(w1, w2, s1, s2))
                y.append(0)
                pbar.update(1)
            attempts += 1

        pbar.close()
        negative_count = len(y) - positive_count
        print(f"  实际负样本数量: {negative_count}")

        #处理异常情况：如果没生成负样本，用更宽松的条件：
        # 编辑距离>一半长度（改了一半以上字母）即两个单词一半以上字母都不一样，强制当作负样本。
        if negative_count == 0:
            while len(y) - positive_count < target_negative:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                w1, w2 = words[i], words[j]
                s1, s2 = stems[i], stems[j]
                # 强制要求编辑距离较大
                if Levenshtein.distance(w1, w2) > max(len(w1), len(w2)) // 2:
                    X.append(feature_extractor.extract_features(w1, w2, s1, s2))
                    y.append(0)

        X = np.array(X)
        y = np.array(y)
        return X, y

    def train(self, X, y):
        """训练所有模型并选择最佳"""
        print("  标准化特征...")
        X_scaled = self.scaler.fit_transform(X) # 标准化
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        best_score = 0
        print("  训练各个模型...")
        for name, model in self.models.items():
            print(f"    训练 {name} ...")
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            acc = accuracy_score(y_val, pred)
            print(f"      {name} 验证准确率: {acc:.4f}")
            if acc > best_score:
                best_score = acc
                self.best_model = model
        print(f"  [ML] 最佳模型验证准确率: {best_score:.4f}")
        return self.best_model

    def predict_prob(self, features):
        """返回相似概率（单个样本一组单词）"""
        if self.best_model is None:
            return 0.5# 没训练时默认50%
        feat = self.scaler.transform(features.reshape(1, -1))
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(feat)[0, 1]# 返回"相似"的概率
        else:
            return self.best_model.decision_function(feat)

    def predict_batch(self, features_list):
        """批量预测"""
        features_array = np.array(features_list)
        feat_scaled = self.scaler.transform(features_array)
        if hasattr(self.best_model, "predict_proba"):
            return self.best_model.predict_proba(feat_scaled)[:, 1]
        else:
            return self.best_model.decision_function(feat_scaled)


#深度学习模块
#前7个特征是人为设计，局限性表达能力上限极低，不能单独表示单个单词，泛化差，无法捕捉细粒度局部字形
# 深度学习的思路：不设计特征，让神经网络自动学习特征
#单词->固定长度向量
class CharLSTMEncoder(nn.Module):#提取特征的方式按顺序记忆
    """LSTM 编码器，按字母顺序读取单词，用 LSTM 学习字母顺序特征，将单词映射为固定长度向量"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, char_max_len=20):
        # embed_dim：单个字母向量维度 固定 32
        # hidden_dim：LSTM 每层记忆单元数（LSTM_HIDDEN=64）
        # num_layers：LSTM 堆叠层数（ 固定 2 对应LSTM_LAYERS=2）
        # char_max_len=20：单词最多只取前 20 个字母
        super().__init__()
        # 1.字母嵌入层：字母数字→字母向量32
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 2.双向LSTM，读取字母序列
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 3.全连接层，把拼接后的双向隐状态压缩成目标维度
        # hidden_dim * 2 就是正向 + 反向两层记忆拼接。
        self.fc = nn.Linear(hidden_dim * 2, embed_dim)  # 输出 embed_dim 维向量
        # 单词最大字母长度，超过截断、不足补0
        self.char_max_len = char_max_len

    def forward(self, x):
        # 步骤1：字母数字编码 → 字母向量序列
        emb = self.embedding(x)# x形状 [batch, max_len] 32，20
        # 步骤2：送入双向LSTM，输出全部时间步结果 + 每层最后记忆状态
        out, (h_n, c_n) = self.lstm(emb)
        #h_n：每一层 LSTM、正向 / 反向，最后一个字母的记忆隐状态
        # 其余没用到
        # 步骤3：取出双向LSTM最后一层的正向、反向隐状态
        h_n = h_n[-2:, :, :]#切片 [-2:] 取最后两行 最顶层的正向、反向隐状态
        # 步骤4：调换维度，把正向、反向向量拼在一起
        h_n = h_n.permute(1, 0, 2).reshape(h_n.size(1), -1)
        # 步骤5：全连接压缩，输出单词整体向量
        vec = self.fc(h_n)
        return vec

"""将单词转换为固定长度（128）的向量（数值表示）"""
#提取局部关键特征 数字→向量的数学运算
#嵌入层:每个数字→32维向量  CNN卷积: 提取单词片段特征  全连接: 压缩成128维
class CharCNNEncoder(nn.Module):
    """CNN 字符级编码器"""

    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes=[2, 3, 4], char_max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # 嵌入层
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embed_dim)
        self.char_max_len = char_max_len

    def forward(self, x):
        emb = self.embedding(x)# 步骤1: 嵌入[batch, seq_len, embed_dim]
        emb = emb.permute(0, 2, 1)# 步骤2: 转置[batch, embed_dim, seq_len]
        conv_outs = []# 步骤3: 卷积 + 池化
        for conv in self.convs:
            conv_out = torch.relu(conv(emb))# [batch, num_filters, seq_len-k+1]
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)# [batch, num_filters]
            conv_outs.append(pooled)
        concat = torch.cat(conv_outs, dim=1)# 步骤4: 拼接 # [batch, num_filters * len(kernel_sizes)]
        vec = self.fc(concat)# 步骤5: 全连接层[batch, embed_dim]
        return vec


#统一封装单词→向量  字母转数字+填充 LSTM/CNN
class WordEncoder:

#单词->128维数字向量
    def __init__(self, encoder_type='lstm', embed_dim=EMBEDDING_DIM):
        self.encoder_type = encoder_type # 'lstm' 或 'cnn'
        #LSTM：捕捉字母的顺序关系；CNN：捕捉字母的局部组合
        self.embed_dim = embed_dim # 输出向量维度（128）
        self.char_to_idx = {} # 字母→编号映射
        self.idx_to_char = {} # 编号→字母映射
        self.vocab_size = None # 词汇表大小
        self.model = None # 具体的模型（LSTM或CNN）
        self.char_max_len = 20 # 单词最大长度
        self._build_vocab()  # 构建字母表

    def _build_vocab(self):#把字母映射成数字
        chars = "abcdefghijklmnopqrstuvwxyz"
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
        self.char_to_idx['<PAD>'] = 0  # 填充符
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx) # 27（26字母+填充符）

    def word_to_tensor(self, word):#单词转张量
        indices = [self.char_to_idx.get(ch, 0) for ch in word[:self.char_max_len]]#取前self.char_max_len = 20个长度
        if len(indices) < self.char_max_len:
            indices += [0] * (self.char_max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def build_model(self):#实例化深度学习编码器
        if self.encoder_type == 'lstm':
            self.model = CharLSTMEncoder(self.vocab_size, 32, self.embed_dim, 2, self.char_max_len)
            #输入词汇表大小，隐藏层维度，输出向量维度，LSTM层数，序列长度
        else:
            self.model = CharCNNEncoder(self.vocab_size, 32, CNN_FILTERS, [2, 3, 4], self.char_max_len)
            #使用不同大小的卷积核 [2, 3, 4] 捕捉不同长度的字符组合特征
        self.model.to(DEVICE)
        return self.model

    def train_unsupervised(self, word_list, epochs=10):#无监督训练占位函数
        self.build_model()
        print("[DL] 字形编码器使用随机初始化，没有真正训练。")
        return self.model

    def encode_words(self, words):# 批量单词转向量
        if self.model is None:
            self.build_model()
        self.model.eval() # 评估模式（不训练）
        tensors = [self.word_to_tensor(w).to(DEVICE) for w in words]
        batch = torch.stack(tensors) # 堆叠成批次
        with torch.no_grad():  # 不计算梯度
            vectors = self.model(batch)
        return vectors.cpu().numpy() # 转成 numpy 数组

#将中文句子的语义转换为向量表示 Sentence-BERT
class SemanticEncoder:
    """中文语义编码器"""

    def __init__(self):
        # 获取当前代码文件所在文件夹
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接本地模型文件夹路径 local_model
        local_model_path = os.path.join(script_dir, "local_model")
        print(f" 从本地路径加载模型: {local_model_path}")
        #加载本地 Sentence-BERT 中文预训练模型
        self.model = SentenceTransformer(local_model_path)
        #模型迁移到GPU/CPU
        self.model.to(DEVICE)
        print(" 语义模型加载完成")

    def encode_meanings(self, meanings):
        """输入中文句子列表，返回向量，显示进度条"""
        return self.model.encode(meanings, convert_to_numpy=True, batch_size=BATCH_SIZE, show_progress_bar=True)

    def similarity(self, vec1, vec2, metric='cosine'):#相似度度量
        if metric == 'cosine':#余弦相似度 两个向量的夹角余弦值
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity([vec1], [vec2])[0, 0]
        elif metric == 'euclidean':#欧氏距离 空间中的直线距离
            return 1 / (1 + np.linalg.norm(vec1 - vec2))
        elif metric == 'manhattan':#曼哈顿距离 各维度差绝对值之和
            return 1 / (1 + np.sum(np.abs(vec1 - vec2)))
        else:
            return 0.0

# 聚类分析KMeans+Hierarchical
#输入单词的向量表示->自动将相似的单词分成若干组,自动发现单词类别
# 得 semantic_cluster KMeans 聚类标签
class WordCluster:
    def __init__(self, n_clusters=10):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

    def fit_shape_features(self, shape_feature_matrix):
        #字形特征聚类
        self.kmeans.fit(shape_feature_matrix)
        return self.kmeans.labels_

    def fit_hierarchical(self, vectors):
        #层次聚类
        return self.hierarchical.fit_predict(vectors)

#相似度融合及检索,根据字形相似度和语义相似度来查找相似的单词。
#输入一个单词或句子 → 在数据库中查找最相似的单词（支持字形匹配和语义匹配两种方式）
# 得shape_similar_words（字形相似词及分数）、semantic_similar_words(语义相似词及分数)
class SimilaritySearcher:
    def __init__(self, ml_model, shape_encoder, sem_encoder, feature_extractor, words_df):
        self.ml_model = ml_model
        self.shape_encoder = shape_encoder
        self.sem_encoder = sem_encoder
        self.feature_extractor = feature_extractor
        self.df = words_df
        #全局缓存词库列表，避免反复查表
        self.word_list = words_df['word_clean'].tolist()
        self.stem_list = words_df['word_stem'].tolist()
        self.meaning_list = words_df['meaning_clean'].tolist()
        self.sem_vectors = None
        #长度索引 全部扫描->扫描长度相近的单词
        self.length_index = defaultdict(list)
        for idx, word in enumerate(self.word_list):
            self.length_index[len(word)].append(idx)

        self.shape_cache = {}#缓存
        #并行 ↓
        self.max_workers = 12  # 默认线程数
        self.use_parallel = False  # 是否启用并行模式

    #并行配置接口
    def set_parallel_config(self, max_workers=None, enable=True):
        """配置并行搜索参数

        Args:
            max_workers: 最大工作线程数（默认：CPU核心数）
            enable: 是否启用并行模式
        """
        import multiprocessing
        if max_workers is None:
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max_workers
        self.use_parallel = enable
        print(f"[并行] 配置完成: {'启用' if enable else '禁用'}, 线程数={self.max_workers}")

    def _search_single_shape(self, word_stem_pair):
        """单个字形搜索任务（供线程池调用）"""
        word, stem = word_stem_pair
        try:
            results = self.search_shape_similar(word, stem, top_k=5)
            return (word, results)
        except Exception as e:
            print(f"[并行] 搜索 '{word}' 时出错: {e}")
            return (word, [])

    def _search_single_semantic(self, meaning):
        """单个语义搜索任务（供线程池调用）"""
        try:
            results = self.search_semantic_similar(meaning, top_k=5)
            return (meaning, results)
        except Exception as e:
            print(f"[并行] 搜索 '{meaning}' 时出错: {e}")
            return (meaning, [])

    def _search_both(self, query_data):
        """同时搜索字形和语义（供线程池调用）"""
        word, stem, meaning = query_data
        results = {
            'word': word,
            'meaning': meaning,
            'shape_results': [],
            'semantic_results': []
        }

        # 字形搜索
        if word:
            try:
                results['shape_results'] = self.search_shape_similar(word, stem, top_k=5)
            except Exception as e:
                print(f"[并行] 字形搜索 '{word}' 失败: {e}")

        # 语义搜索
        if meaning:
            try:
                results['semantic_results'] = self.search_semantic_similar(meaning, top_k=5)
            except Exception as e:
                print(f"[并行] 语义搜索 '{meaning}' 失败: {e}")

        return results

    #批量查形近词
    def batch_search_shape_parallel(self, query_list, top_k=5, show_progress=True):
        """批量并行搜索字形相似词

        Args:
            query_list: 查询列表，格式为 [(word1, stem1), (word2, stem2), ...]
            top_k: 返回结果数量
            show_progress: 是否显示进度条

        Returns:
            字典列表，格式为 [{'word': word, 'results': [...]}, ...]
        """
        if not self.use_parallel:
            print("[并行] 并行模式未启用，使用单线程模式")
            results = []
            for word, stem in tqdm(query_list, desc="  顺序搜索", disable=not show_progress):
                res = self.search_shape_similar(word, stem, top_k=top_k)
                results.append({'word': word, 'results': res})
            return results

        print(f"[并行] 使用 {self.max_workers} 个线程并行搜索 {len(query_list)} 个单词...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(self._search_single_shape, (word, stem))
                       for word, stem in query_list]

            # 收集结果
            results = []
            if show_progress:
                for future in tqdm(futures, desc="  并行搜索进度"):
                    word, res = future.result()
                    results.append({'word': word, 'results': res})
            else:
                for future in futures:
                    word, res = future.result()
                    results.append({'word': word, 'results': res})

        return results

    #批量查近义词
    def batch_search_semantic_parallel(self, meaning_list, top_k=5, show_progress=True):
        """批量并行搜索语义相似词

        Args:
            meaning_list: 中文释义列表
            top_k: 返回结果数量
            show_progress: 是否显示进度条

        Returns:
            字典列表
        """
        if not self.use_parallel:
            print("[并行] 并行模式未启用，使用单线程模式")
            results = []
            for meaning in tqdm(meaning_list, desc="  顺序搜索", disable=not show_progress):
                res = self.search_semantic_similar(meaning, top_k=top_k)
                results.append({'meaning': meaning, 'results': res})
            return results

        print(f"[并行] 使用 {self.max_workers} 个线程并行搜索 {len(meaning_list)} 个释义...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._search_single_semantic, meaning)
                       for meaning in meaning_list]

            results = []
            if show_progress:
                for future in tqdm(futures, desc="  并行搜索进度"):
                    meaning, res = future.result()
                    results.append({'meaning': meaning, 'results': res})
            else:
                for future in futures:
                    meaning, res = future.result()
                    results.append({'meaning': meaning, 'results': res})

        return results

    #批量同时查形近 + 语义近义词
    def batch_search_both_parallel(self, query_list, top_k=5, show_progress=True):
        """批量并行同时搜索字形和语义

        Args:
            query_list: 查询列表，格式为 [(word1, stem1, meaning1), (word2, stem2, meaning2), ...]
            top_k: 返回结果数量
            show_progress: 是否显示进度条

        Returns:
            包含字形和语义搜索结果的列表
        """
        if not self.use_parallel:
            print("[并行] 并行模式未启用，使用单线程模式")
            results = []
            for word, stem, meaning in tqdm(query_list, desc="  顺序搜索", disable=not show_progress):
                result = {
                    'word': word,
                    'meaning': meaning,
                    'shape_results': self.search_shape_similar(word, stem, top_k=top_k) if word else [],
                    'semantic_results': self.search_semantic_similar(meaning, top_k=top_k) if meaning else []
                }
                results.append(result)
            return results

        print(f"[并行] 使用 {self.max_workers} 个线程并行处理 {len(query_list)} 个查询...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._search_both, (word, stem, meaning))
                       for word, stem, meaning in query_list]

            results = []
            if show_progress:
                for future in tqdm(futures, desc="  并行搜索进度"):
                    results.append(future.result())
            else:
                for future in futures:
                    results.append(future.result())

        return results

    #并行 ↑

    #语义向量预计算
    def precompute_semantic_vectors(self):#计算所有单词的中文释义向量
        print("[语义] 开始预计算所有单词的中文向量...")
        self.sem_vectors = self.sem_encoder.encode_meanings(self.meaning_list)
        print(f"[语义] 已计算 {len(self.sem_vectors)} 个向量")

    #字形相似度打分
    def get_shape_similarity(self, query_word, query_stem, candidate_word, candidate_stem):
        cache_key = (query_word,candidate_word)

        if cache_key in self.shape_cache:
            return self.shape_cache[cache_key]

        #长度过滤：如果长度差超过硬性限制，直接返回 0
        if abs(len(query_word) - len(candidate_word)) > MAX_LEN_DIFF:
            return 0.0

        #编辑距离过滤：快速计算基础编辑距离相似度（至少修改多少字符才能变成另一个词）
        max_len = max(len(query_word), len(candidate_word))
        edit_sim = 1 - Levenshtein.distance(query_word, candidate_word) / max_len if max_len > 0 else 1.0

        #拼写重合度小于 0.25，直接拦截
        if edit_sim < 0.25:
            return 0.0

        #机器学习预测
        features = self.feature_extractor.extract_features(query_word, candidate_word, query_stem, candidate_stem)
        ml_prob = self.ml_model.predict_prob(features)

        #模型给高分，实际基础编辑距离或序列相似度极低，进行惩罚修正（防止不合理）
        rule_score = (features[1] + features[5]) / 2  # 序列相似度与编辑距离相似度的均值
        if ml_prob > 0.5 and rule_score < 0.4:
            ml_prob = ml_prob * 0.2

        #模型6规则4
        final_score = 0.4 * rule_score + 0.6 * ml_prob
        self.shape_cache[cache_key] = final_score
        return final_score

    #比较两个中文句子的语义相似度
    def get_semantic_similarity(self, query_meaning, candidate_meaning, metric='cosine'):
        q_vec = self.sem_encoder.encode_meanings([query_meaning])[0]
        c_vec = self.sem_encoder.encode_meanings([candidate_meaning])[0]
        return self.sem_encoder.similarity(q_vec, c_vec, metric)

    #易混词搜索,单单词形近词检索
    def search_shape_similar(self, query_word, query_stem, top_k=10):
        scores = []
        # 长度索引过滤
        candidate_indices = []

        min_len = len(query_word) - MAX_LEN_DIFF
        max_len = len(query_word) + MAX_LEN_DIFF

        for l in range(min_len, max_len + 1):
            candidate_indices.extend(
                self.length_index.get(l, [])
            )

        for idx in candidate_indices:

            w = self.word_list[idx]
            s = self.stem_list[idx]

            if w == query_word:
                continue
        # 字母重合预过滤
            char_ratio = (ShapeFeatureExtractor.char_coincidence_ratio(query_word,w))

            if char_ratio < 0.30:
                continue

            score = self.get_shape_similarity(
                query_word,
                query_stem,
                w,
                s
            )

            if score > 0.1:
                scores.append(
                    (idx, score)
                )
        # TopK
        top_scores = heapq.nlargest(
            top_k,
            scores,
            key=lambda x: x[1]
        )

        results = []

        for idx, score in top_scores:
            results.append(
                {
                    'word':
                        self.word_list[idx],

                    'meaning':
                        self.meaning_list[idx],

                    'score':
                        score
                }
            )

        return results
    #近义词搜索
    def search_semantic_similar(self, query_meaning, top_k=10, metric='cosine'):
        #预计算所有语义向量
        if self.sem_vectors is None:
            self.precompute_semantic_vectors()
        #编码查询句子
        q_vec = self.sem_encoder.encode_meanings([query_meaning])[0]
        from sklearn.metrics.pairwise import cosine_similarity
        #计算与所有句子的余弦相似度
        scores = cosine_similarity([q_vec], self.sem_vectors)[0]
        #返回最相似的top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if self.meaning_list[idx] == query_meaning and len(results) > 0:
                continue
            results.append({
                'word': self.word_list[idx],
                'meaning': self.meaning_list[idx],
                'score': scores[idx]
            })
        return results

#模型评估与可视化
class Evaluator:
    @staticmethod
    #评估
    def evaluate_shape_model(ml_model, feature_extractor, df, test_pairs, true_labels):
        #提取特征
        X_test = []
        for (w1, w2, s1, s2) in test_pairs:
            X_test.append(feature_extractor.extract_features(w1, w2, s1, s2))
        #模型预测
        pred_probs=ml_model.predict_batch(X_test)#相似度概率
        pred_labels = (pred_probs>0.5).astype(int)#阈值0.5
        #计算评估指标
        acc = accuracy_score(true_labels, pred_labels)
        prec, rec, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
        print("[评估] 字形相似模型：")
        print(f"  准确率: {acc:.4f}, 精确率: {prec:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}")
        return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

    @staticmethod
    #可视化 将高维向量降维到2D
    def visualize_embeddings(vectors, labels, title="Embedding Visualization"):
        print("  正在降维 (PCA) ...")
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        #用不同颜色表示不同类别，使用10种不同的颜色，透明度
        #sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='tab10')
        plt.title(title)
        plt.colorbar()
        plt.show()

#主程序
def main():
    print("单词相似检索系统启动（ML+DL）")

    # 1. 加载数据
    print("\n[1/8] 加载数据...")
    if not os.path.exists(DATA_FILE):
        print(f"错误：数据文件 {DATA_FILE} 不存在！请准备 csv 文件，包含 'word','meaning' 两列。")
        return
    encodings = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']
    for enc in encodings:
        try:
            df_raw = pd.read_csv(DATA_FILE, encoding=enc)
            print(f"成功使用编码: {enc}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("无法识别文件编码，请检查 CSV 文件格式")

    # 限制数据量（用于快速测试）
    """
    MAX_SAMPLES = 500
    if len(df_raw) > MAX_SAMPLES:
        print(f"  数据量较大，仅取前 {MAX_SAMPLES} 条")
        df_raw = df_raw.head(MAX_SAMPLES)
    """
    print(f"  原始数据行数: {len(df_raw)}")
    preprocessor = DataPreprocessor()
    print("  开始预处理...")
    df = preprocessor.preprocess_dataframe(df_raw)
    print(f"  数据预处理完成，共 {len(df)} 条有效单词")

    # 2. 初始化各模块
    print("\n[2/8] 初始化特征提取器...")
    feat_extractor = ShapeFeatureExtractor()
    ml_model = ShapeMLModel()

    # 生成训练数据并训练传统模型
    print("\n[3/8] 传统机器学习模型（字形相似）...")
    print("  生成字形相似训练数据（可能需要几分钟）...")
    X_train, y_train = ml_model.generate_training_data(df, feat_extractor, num_neg_samples=2)
    print(f"  训练样本数: {len(X_train)} (正例: {sum(y_train)}, 负例: {len(y_train) - sum(y_train)})")
    print("  开始训练模型...")
    ml_model.train(X_train, y_train)

    # 深度学习编码器
    print("\n[4/8] 深度学习字形编码器...")
    shape_encoder = WordEncoder(encoder_type='lstm')
    shape_encoder.build_model()
    print("  字形编码器就绪 (随机初始化)")

    print("\n[5/8] 中文语义编码器...")
    sem_encoder = SemanticEncoder()

    # 相似度检索器
    print("\n[6/8] 初始化相似度检索器...")
    searcher = SimilaritySearcher(ml_model, shape_encoder, sem_encoder, feat_extractor, df)
    searcher.precompute_semantic_vectors()

    #并行 ↓
    use_parallel = input("\n是否启用并行搜索？(yes/no，默认yes): ").strip().lower()

    if use_parallel in ['yes', 'y', '']:
        workers = input(f"线程数 (1-{os.cpu_count() * 2}，默认{os.cpu_count()}): ").strip()
        if workers and workers.isdigit():
            max_workers = int(workers)
        else:
            max_workers = os.cpu_count()

        searcher.set_parallel_config(max_workers=max_workers, enable=True)
    else:
        searcher.set_parallel_config(enable=False)
        print("[INFO] 使用单线程模式")
    #并行 ↑

    # 聚类演示
    print("\n[7/8] 无监督聚类（语义）...")
    cluster_model = WordCluster(n_clusters=8)
    sem_vecs = searcher.sem_vectors
    print("  正在执行 KMeans 聚类...")
    cluster_labels = cluster_model.fit_shape_features(sem_vecs)
    df['semantic_cluster'] = cluster_labels
    print("  聚类完成，已将语义相近的单词归入同一簇")

    # 3. 交互分支
    print("\n[8/8] 就绪，等待用户指令")
    choice = input("\n请问是否需要生成【全数据集批量整理后的完整相似词列表文件】？请输入 yes / no : ").strip().lower()

    if choice == 'yes':#批量
        print("\n[批量模式] 正在为所有单词计算相似词...")
    #并行↓
        if searcher.use_parallel:
            print(f"[批量模式] 使用并行搜索 ({searcher.max_workers} 线程)")

            # 准备查询数据
            query_list = []
            for _, row in df.iterrows():
                query_list.append((
                    row['word_clean'],
                    row['word_stem'],
                    row['meaning_clean']
                ))

            # 并行搜索
            results = searcher.batch_search_both_parallel(query_list, top_k=5, show_progress=True)

            # 整理结果
            shape_sim_list = []
            sem_sim_list = []
            for result in results:
                shape_str = "; ".join([f"{r['word']}({r['score']:.3f})：{r['meaning']})" for r in result['shape_results']])
                sem_str = "; ".join([f"{r['word']}:{r['meaning']}({r['score']:.3f})" for r in result['semantic_results']])
                shape_sim_list.append(shape_str)
                sem_sim_list.append(sem_str)

            df['shape_similar_words'] = shape_sim_list
            df['semantic_similar_words'] = sem_sim_list
            #并行↑
        else:#不并行
            shape_sim_list = []
            sem_sim_list = []

            for i, row in tqdm(df.iterrows(), total=len(df), desc="计算相似词进度"):
                q_word = row['word_clean']
                q_stem = row['word_stem']
                q_meaning = row['meaning_clean']

                # 获取 Top5 相似词
                shape_res = searcher.search_shape_similar(q_word, q_stem, top_k=5)
                sem_res = searcher.search_semantic_similar(q_meaning, top_k=5)

                #直接从返回的字典里取 score
                shape_str = "; ".join([f"{r['word']}({r['score']:.3f})：{r['meaning']}" for r in shape_res])
                sem_str = "; ".join([f"{r['word']}:{r['meaning']}({r['score']:.3f})" for r in sem_res])

                shape_sim_list.append(shape_str)
                sem_sim_list.append(sem_str)

            df['shape_similar_words'] = shape_sim_list
            df['semantic_similar_words'] = sem_sim_list
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"[完成] 批量结果已保存至 {OUTPUT_FILE}")

        #可视化语义向量
        print("\n[可视化] 展示语义向量的 PCA 降维分布...")
        Evaluator.visualize_embeddings(sem_vecs, cluster_labels, title="Semantic Clusters of Words")
    else:#实时
        print("\n[实时模式] 请输入单词信息进行相似检索")
        while True:#一直询问
            word = input("\n请输入英文单词: ").strip().lower()
            meaning = input("请输入对应中文释义: ").strip()
            if not word and not meaning:
                print("输入无效，退出实时模式")
                break
            #新增↓
            if word and not meaning:#英文
                word_clean = preprocessor.clean_word(word)
                print(f"正在检索 '{word_clean}' 的字形相似词...")

                word_stem = preprocessor.stem_english(word_clean)
                shape_sim = searcher.search_shape_similar(word_clean, word_stem, top_k=5)

                print("\n【英文字形相似词】")
                if shape_sim:
                    for idx, res in enumerate(shape_sim, 1):
                        print(f"  {idx}. {res['word']} ({res['score']:.3f})—— {res['meaning']}")
                else:
                    print("  未找到字形相似的单词")
            elif meaning and not word:#中文
                meaning_clean = preprocessor.clean_meaning(meaning)
                print(f"正在检索语义相似词...")

                sem_sim = searcher.search_semantic_similar(meaning_clean, top_k=5)

                print("\n【中文语义相似词】")
                if sem_sim:
                    for idx, res in enumerate(sem_sim, 1):
                        print(f"  {idx}. {res['word']} ({res['score']:.3f})—— {res['meaning']}")
                else:
                    print("  未找到语义相似的单词")
            #新增↑
            elif word and meaning:#中英文
                print("处理中……")
                word_clean = preprocessor.clean_word(word)
                word_stem = preprocessor.stem_english(word_clean)
                meaning_clean = preprocessor.clean_meaning(meaning)

                print(f"\n正在检索 '{word_clean}'字形相似词 ...")
                # 字形相似
                shape_sim = searcher.search_shape_similar(word_clean, word_stem, top_k=5)
                print("\n【英文字形相似词】")
                for idx, res in enumerate(shape_sim, 1):
                    print(f"  {idx}. {res['word']} ({res['score']:.3f}) —— {res['meaning']}")
                # 语义相似
                print(f"\n正在检索 '{meaning_clean}'字形相似词 ...")
                sem_sim = searcher.search_semantic_similar(meaning_clean, top_k=5)
                print("\n【中文语义相似词】")
                for idx, res in enumerate(sem_sim, 1):
                    print(f"  {idx}. {res['word']} ({res['score']:.3f}) —— {res['meaning']}")

            cont = input("\n继续检索其他单词？(y/n): ").strip().lower()
            if cont != 'y':
                break


if __name__ == "__main__":
    main()
