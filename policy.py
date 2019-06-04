def bubble_sort(arr, reverse):
    n = len(arr)
    res = arr[:]
    idx = list(range(n))
 
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n - i - 1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if res[j] > res[j + 1]:
                res[j], res[j + 1] = res[j + 1], res[j]
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
    
    if reverse:
        res.reverse()
        idx.reverse()

    return res, idx


def greedy(reputations, num, reverse=False):
    _, idx = bubble_sort(reputations, reverse)

    return idx[:num]


def probabilistic_greedy(reputations, num):
    pass