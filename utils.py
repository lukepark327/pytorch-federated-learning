import os


def recursive_mkdir(path_str):
    if path_str == "":
        return

    if path_str[-1] == "/":
        path_str = path_str[:-1]

    if not os.path.exists(path_str):
        upper = os.path.dirname(path_str)

        if not os.path.exists(upper):
            recursive_mkdir(upper)

        os.mkdir(path_str)


if __name__ == "__main__":
    recursive_mkdir("clients/1/model")
