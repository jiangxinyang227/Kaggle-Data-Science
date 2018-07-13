"""
英文文档预处理和中文文档预处理的区别
1、不需要进行编码解码，在中文中若存在Unicode编码，需要转为utf-8
2、不许要做分词处理，英文中单词间自带空格
3、处理拼写错误问题，如Hell World
4、词干提取，例如countries和country我们希望作为一个单词处理
"""

"""
具体处理流程如下
1、数据收集。中英文处理类似，一种是使用现成的语料库作为我们的语料库；另一种是自己去爬取数据来作为我们的语料库

2、除去数据中非文本部分，例如特殊字符等。中英文处理类似，可以用re或者beautifulSoup（主要是html标签）

3、英文文本的拼写错误，安装pyenchant库来处理
"""

from enchant.checker import SpellChecker

checker = SpellChecker("en_US")
checker.set_text("Many peope likke to watch in the name of people")
for err in checker:
    print("Error", err.word)

"""
4、词干提取和词性还原，词性还原比词干提取更温柔些，例如对于imaging，词干提取可能得到imag，而词性还原更容易得到image
"""

import nltk
nltk.download()
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("countries")

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
wnl.lemmatize("countries")

"""
5、将大小写统一转为小写

6、引入停用词，去除文本中的一些停用次

7、特征处理，一般都是向量化，TF-IDF，标准化三步。对于大文本，又没有什么好的分布式处理的条件，可以用hash_trick进行降维处理

8、建立分析模型

注意：有时候在处理英文时也需要做分词，主要是针对一些地名，比如New York
"""


