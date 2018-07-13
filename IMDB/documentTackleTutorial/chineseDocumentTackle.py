import jieba
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize():
    wordList = ["沙瑞金", "易学习", "王大路", "京州"]
    for word in wordList:
        jieba.suggest_freq(word, True)

    with open("./data/nlp_test0.txt") as f:
        document = f.read()

        document_cut = jieba.cut(document)
        result = " ".join(document_cut)

        with open("./data/nlp_test1.txt", 'w') as f2:
            f2.write(result)

    with open("./data/nlp_test2.txt") as f3:
        document = f3.read()
        document_cut = jieba.cut(document)
        result = " ".join(document_cut)

        with open("./data/nlp_test3.txt", "w") as f4:
            f4.write(result)


def main():
    with open("./data/stop_words.txt", "rb") as fr:
        stop_words = fr.read()
        stopWordList = stop_words.splitlines()

    with open("./data/nlp_test1.txt", "r") as f:
        words1 = f.read()

    with open("./data/nlp_test3.txt", "r") as f1:
        words2 = f1.read()

    corups = [words1, words2]
    vector = TfidfVectorizer(stop_words=stopWordList)
    tfidf = vector.fit_transform(corups)
    wordList = vector.get_feature_names()
    weightList = tfidf.toarray()

    print(wordList)
    print(weightList)


if __name__ == '__main__':
    main()
