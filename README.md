# 单词相似检索系统 📚

基于机器学习和深度学习的智能单词相似度检索工具，支持字形相似度和中文语义相似度双重匹配。

## ✨ 主要功能

- **自定义混淆词相似匹配**：基于字形特征的智能匹配
- **中文释义相似匹配**：使用 BERT 模型进行语义理解
- **传统机器学习模型**：逻辑回归、随机森林、GBDT 集成
- **深度学习嵌入**：LSTM/CNN 字形向量 + BERT 中文语义向量
- **无监督聚类**：KMeans、层次聚类自动发现单词类别
- **相似度加权融合**：多维度综合评分
- **批量+实时交互**：支持两种使用模式
- **并行加速**：多线程批量处理

## 🚀 快速开始

### 环境要求

```bash
Python 3.7+

安装依赖
bash
pip install -r requirements.txt
核心依赖包：

torch - 深度学习框架

transformers - BERT 模型

sentence-transformers - 语义向量编码

scikit-learn - 机器学习算法

pandas - 数据处理

jieba - 中文分词

Levenshtein - 编辑距离计算

数据准备
准备 CSV 文件 English_words.csv，包含两列：

列名	说明	示例
word	英文单词	apple
meaning	中文释义	苹果；苹果公司
运行程序
bash
python word_similarity.py
📖 使用说明
1. 批量模式
批量生成所有单词的相似词列表并保存到文件：

bash
# 程序启动后输入
请问是否需要生成【全数据集批量整理后的完整相似词列表文件】？请输入 yes / no : yes
输出文件：word_similarity_output.csv

2. 实时交互模式
场景一：输入英文查字形相似词
text
请输入英文单词: apple
请输入对应中文释义: 
输出示例：

text
【英文字形相似词】
  1. apply (0.723) —— 申请；应用
  2. ample (0.651) —— 充足的
  3. appeal (0.598) —— 呼吁；上诉
场景二：输入中文查语义相似词
text
请输入英文单词: 
请输入对应中文释义: 高兴
输出示例：

text
【中文语义相似词】
  1. happy (0.856) —— 快乐的；幸福的
  2. joyful (0.792) —— 喜悦的
  3. delighted (0.734) —— 高兴的
场景三：同时输入中英文双向检索
text
请输入英文单词: happy
请输入对应中文释义: 快乐
输出字形和语义两个维度的相似词。

3. 并行加速配置
程序支持多线程并行处理，大幅提升批量检索速度：

text
是否启用并行搜索？(yes/no，默认yes): yes
线程数 (1-8，默认4): 4
🏗️ 系统架构
text
数据预处理 → 特征提取 → 模型训练 → 相似度计算 → 结果输出
    ↓           ↓           ↓           ↓           ↓
  清洗分词   字形特征    ML/DL模型   加权融合    批量/实时
核心模块说明
模块	功能	关键技术
DataPreprocessor	数据清洗、分词、词干提取	jieba, NLTK
ShapeFeatureExtractor	字形特征提取	编辑距离、N-Gram
ShapeMLModel	传统机器学习模型	LR, RF, GBDT
WordEncoder	字形向量编码	LSTM/CNN
SemanticEncoder	语义向量编码	BERT
SimilaritySearcher	相似度检索与融合	加权评分
📊 相似度算法
字形相似度特征（7维）
字符重合度

最长公共子序列相似度

N-Gram 相似度（2-gram）

词根匹配

长度相似度

编辑距离相似度

位置偏移度

最终评分公式
text
final_score = 0.4 × rule_score + 0.6 × ml_prob
rule_score：序列相似度与编辑距离的均值

ml_prob：机器学习模型预测的概率

⚙️ 配置参数
在代码开头的全局配置区域可调整：

python
DATA_FILE = "English_words.csv"      # 输入数据文件
OUTPUT_FILE = "word_similarity_output.csv"  # 输出文件
MAX_SAMPLES = 500                    # 最大处理数量（测试用）
MAX_LEN_DIFF = 3                     # 单词长度最大差异
BATCH_SIZE = 32                      # 批处理大小
EMBEDDING_DIM = 128                  # 词嵌入维度
📁 项目结构
text
word-similarity/
├── word_similarity.py      # 主程序
├── English_words.csv       # 输入数据
├── word_similarity_output.csv  # 输出结果
├── local_model/            # 本地 BERT 模型
├── requirements.txt        # 依赖包列表
└── README.md              # 说明文档
🔧 常见问题
Q1: 提示找不到 English_words.csv
A: 确保 CSV 文件放在程序同目录下，且包含 word 和 meaning 两列。

Q2: BERT 模型下载慢
A: 程序已配置国内镜像源：

python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
Q3: 内存不足
A: 减少 MAX_SAMPLES 或 BATCH_SIZE 的值。

Q4: 并行搜索不生效
A: 检查是否在启动时选择了 yes，Python 版本需 3.7+。

📝 输出文件说明
word_similarity_output.csv 包含：

列名	说明
word_clean	清洗后的英文单词
meaning_clean	清洗后的中文释义
shape_similar_words	字形相似词及分数
semantic_similar_words	语义相似词及分数
semantic_cluster	语义聚类标签
🎯 应用场景
📖 英语学习助手：查找易混淆词、近义词

🔍 词典检索增强：智能纠错和联想

🧠 NLP 预处理：构建词义消歧数据集

📊 文本分析：自动发现单词聚类

📄 许可证
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

🤝 贡献
欢迎提交 Issue 和 Pull Request！

📧 联系方式
如有问题，请通过 Issue 反馈
