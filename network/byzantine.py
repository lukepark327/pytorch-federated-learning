from enum import Enum


class ByzantineType(Enum):
    BIASED_DATA_SET = 'biased data set'
    CORRUPTED_DATA_SET = 'corrupted data set'
    CORRUPTED_META = 'corrupted meta'

class Byzantine:
    def __init__(self, byzantine_type: ByzantineType):
        self.type = byzantine_type


def corrupt_ten_labeled_data(x_train, y_train, swap_label_1, swap_label_2):
    for i in range(len(x_train)):
        if y_train[i].item() is swap_label_1:
            y_train[i] = swap_label_2
        elif y_train[i].item() is swap_label_2:
            y_train[i] = swap_label_1
