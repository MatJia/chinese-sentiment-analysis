#分词，获取两个dict，第一个dict为vocab，记录中文词语转数字，额外有两个【空白】以及【新增】占位
#vocab中，每个词语对应一个对应数字
#第二个dict为frequency，简单易懂，词语对应出现频率

import pandas as pd
import jieba as jb
import numpy as np

def set_dictionary(Text: list, MinFrq: int) -> (dict, dict): #MinFrq为最低出现频率，少了不要
    vocab, word_frq = {}, {}
    for text in Text:
        words = list(jb.cut(text))
        for word in words:
            word = word.strip()
            if word != "":
                word_frq[word] = word_frq.get(word, 0) + 1
    vocab["Empty"] = 0 #<PAD> for padding
    vocab["Unknown"] = 1 #<UNK> for unknown
    for word,freq in word_frq.items():
        if freq >= MinFrq:
            vocab[word] = len(vocab)
    return vocab, word_frq

def get_padding_len(text_origin: list, predict_padding_line: int) -> int:
    '''这版问题：数据量太大了太慢（重复分词），padding_line每次+2太靠蒙，重写了
    text = text_origin
    cover_line = predict_padding_line
    covered_text = 0
    total_text = len(text)
    test_calculator = 0 ############################################################
    while covered_text < total_text * 0.95:
        test_calculator += 1  #################################################################
        #for test, please annotate when release
        print(f"we've set padding line for {test_calculator} times") #######################
        for i in range(len(text)):
            words = list(jb.cut(text[i]))
            words = [x for x in words if x.strip() != ""]
            len_word = len(words)
            if len_word == 0:
                print("this sentence covered!") #################################################
                continue
            if len_word <= cover_line:
                covered_text += 1
                print(f"find {words} have {len_word} words, under cover_line{cover_line}")###########
                text[i] = ""
        if covered_text < total_text * 0.95:
            cover_line += 2
    return cover_line
    '''
    count = []
    for sentence in text_origin:
        count_words = len([x for x in list(jb.cut(sentence)) if x.strip() != ""])
        count.append(count_words)
    return int(np.percentile(count,95) + 1)

def word_to_num(text: list, tar_len: int, vocab: dict) -> list:
    tar_list = []
    for sentence in text:
        words_list = [x for x in list(jb.cut(sentence)) if x.strip() != ""]
        len_word = len(words_list)
        for i in range(tar_len):
            if i < len_word:
                words_list[i] = vocab.get(words_list[i], vocab["Unknown"])
            else:
                words_list.append(vocab["Empty"])
        tar_list.append(words_list)
    #print(tar_list)
    return tar_list

if __name__ == "__main__":
    csv_source = pd.read_csv('../data/sentiment_data.csv')
    label_list = csv_source["label"].tolist()
    vocab, word_frq = set_dictionary(csv_source["text"].tolist(),1)
    padding_len = get_padding_len(csv_source["text"].tolist(), 7)
    padded_list = word_to_num(csv_source["text"].tolist(), padding_len, vocab)
    train_size = int(len(padded_list) * 0.8)
    train_padded_list = padded_list[:train_size]
    train_label_list = label_list[:train_size]
    predict_padded_list = padded_list[train_size:]
    predict_label_list = label_list[train_size:]