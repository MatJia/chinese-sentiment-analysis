from dataset import load_set

def calculate_label(label_list: list) -> (int, int, int, int):
    result = [0, 0, 0, 0]
    for i in label_list:
        result[int(i)] += 1
    return tuple(result)

def accuracy(label_list: list, predict_label: int) -> float:
    result = 0
    for i in label_list:
        if int(i) == predict_label:
            result += 1
    return result/len(label_list)

def get_major_label(label_list: list) -> int:
    get_count_list = calculate_label(label_list)
    return get_count_list.index(max(get_count_list))

if __name__ == "__main__":
    vocab, word_frq, train_size, train_padded_list, train_label_list, predict_padded_list, predict_label_list = load_set()
    print(calculate_label(train_label_list))
    print(calculate_label(predict_label_list))
    major_label = get_major_label(train_label_list)
    print(major_label)
    print(accuracy(predict_label_list, major_label))