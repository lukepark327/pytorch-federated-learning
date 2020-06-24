import torch
from torch.autograd import Variable

import math
from numbers import Number


class Weights():
    def __init__(self,
                 params):

        if isinstance(params, Weights):
            params = params.to_dict()
        elif isinstance(params, dict):
            pass
        else:
            try:
                params = dict(params)
            except:
                raise ValueError("params must be `Weights (dict)` or `generator` which is retured by named_parameters() but {}.".format(type(params)))

        self.params = params

    def to_dict(self):
        return self.params

    """container
    # TBA
    """

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def __getitem__(self, key):
        if type(key) != str:
            raise TypeError("key must be a `str` but {}.".format(type(key)))
        if key not in self.params.keys():
            raise KeyError("key '{}' is not in params.".format(key))
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return self.params.__iter__()

    def __contains__(self, key):
        return self.params.__contains__(key)

    """arithmetic
    # *_ : in-place version
    """

    # -x
    def neg(self):
        res = dict()
        for key, value in self.items():
            res[key] = -1 * value.data
        return Weights(res)

    def neg_(self):
        self.params = self.neg()

    def __neg__(self):
        return self.neg()

    # x + (y: dict or Weights)
    # or
    # x + (y: Number)
    def add(self, other):
        res = dict()

        if isinstance(other, dict) or isinstance(other, Weights):
            for key, value in self.items():
                if key not in other.keys():
                    raise KeyError("'{}' is not in a argument.".format(key))
                res[key] = value.add(other[key].data)
        elif isinstance(other, Number):
            s = Variable(torch.Tensor([other]).double())
            for key, value in self.items():
                res[key] = value.add(s.expand(value.size()))
        else:
            raise TypeError("The argument must be `Weights (dict)` or `Number` but {}.".format(type(other)))

        return Weights(res)

    def add_(self, other):
        self.params = self.add(other)

    def __add__(self, other):
        return self.add(other)

    # x - y
    def sub(self, other):
        # return self.add(-other)
        res = dict()

        if isinstance(other, dict) or isinstance(other, Weights):
            for key, value in self.items():
                if key not in other.keys():
                    raise KeyError("'{}' is not in a argument.".format(key))
                res[key] = value.sub(other[key].data)
        elif isinstance(other, Number):
            s = Variable(torch.Tensor([other]).double())
            for key, value in self.items():
                res[key] = value.sub(s.expand(value.size()))
        else:
            raise TypeError("The argument must be `Weights (dict)` or `Number` but {}.".format(type(other)))

        return Weights(res)

    def sub_(self, other):
        self.params = self.sub(other)

    def __sub__(self, other):
        return self.sub(other)

    # x * (y: dict or Weights): Hadamard product
    # or
    # x * (y: Number): scalar multiplication
    def mul(self, other):
        res = dict()

        if isinstance(other, dict) or isinstance(other, Weights):
            for key, value in self.items():
                if key not in other.keys():
                    raise KeyError("'{}' is not in a argument.".format(key))
                res[key] = value.mul(other[key].data)
        elif isinstance(other, Number):
            s = Variable(torch.Tensor([other]).double())
            for key, value in self.items():
                res[key] = value.mul(s.expand(value.size()))
        else:
            raise TypeError("The argument must be `Weights (dict)` or `Number` but {}.".format(type(other)))

        return Weights(res)

    def mul_(self, other):
        self.params = self.mul(other)

    def __mul__(self, other):
        return self.mul(other)

    # TODO: x @ y
    # def __matmul__(self, other):
    #     # @: inner product
    #     pass
    # mm

    # x / (y: dict or Weights): inverse of Hadamard product
    # or
    # x / (y: Number): inverse of scalar multiplication
    def div(self, other):
        res = dict()

        if isinstance(other, dict) or isinstance(other, Weights):
            for key, value in self.items():
                if key not in other.keys():
                    raise KeyError("'{}' is not in a argument.".format(key))
                res[key] = value.div(other[key].data)
        elif isinstance(other, Number):
            s = Variable(torch.Tensor([other]).double())
            for key, value in self.items():
                res[key] = value.div(s.expand(value.size()))
        else:
            raise TypeError("The argument must be `Weights (dict)` or `Number` but {}.".format(type(other)))

        return Weights(res)

    def div_(self, other):
        self.params = self.div(other)

    def __truediv__(self, other):
        return self.div(other)

    # x // (y: dict or Weights): element-wise floor_divide
    # or
    # x // (y: Number): floor_divide with scalar
    def floor_divide(self, other):
        res = dict()

        if isinstance(other, dict) or isinstance(other, Weights):
            for key, value in self.items():
                if key not in other.keys():
                    raise KeyError("'{}' is not in a argument.".format(key))
                res[key] = value.floor_divide(other[key].data)
        elif isinstance(other, Number):
            s = Variable(torch.Tensor([other]).double())
            for key, value in self.items():
                res[key] = value.floor_divide(s.expand(value.size()))
        else:
            raise TypeError("The argument must be `Weights (dict)` or `Number` but {}.".format(type(other)))

        return Weights(res)

    def floor_divide_(self, other):
        self.params = self.floor_divide(other)

    def __floordiv__(self, other):
        return self.floor_divide(other)

    # x % (y: dict or Weights): element-wise mod operator
    # or
    # x % (y: Number): mod operator with scalar
    def remainder(self, other):
        res = dict()

        if isinstance(other, dict) or isinstance(other, Weights):
            for key, value in self.items():
                if key not in other.keys():
                    raise KeyError("'{}' is not in a argument.".format(key))
                res[key] = value.remainder(other[key].data)
        elif isinstance(other, Number):
            s = Variable(torch.Tensor([other]).double())
            for key, value in self.items():
                res[key] = value.remainder(s.expand(value.size()))
        else:
            raise TypeError("The argument must be `Weights (dict)` or `Number` but {}.".format(type(other)))

        return Weights(res)

    def remainder_(self, other):
        self.params = self.remainder(other)

    def __mod__(self, other):
        return self.remainder(other)

    # divmod()
    def __divmod__(self, other):
        return (self.div(other), self.remainder(other))

    # x ** (y: dict or Weights): element-wise
    # or
    # x ** (y: Number): power of scalar
    def pow(self, other):
        res = dict()

        if isinstance(other, dict) or isinstance(other, Weights):
            for key, value in self.items():
                if key not in other.keys():
                    raise KeyError("'{}' is not in a argument.".format(key))
                res[key] = value.pow(other[key].data)
        elif isinstance(other, Number):
            s = Variable(torch.Tensor([other]).double())
            for key, value in self.items():
                res[key] = value.pow(s.expand(value.size()))
        else:
            raise TypeError("The argument must be `Weights (dict)` or `Number` but {}.".format(type(other)))

        return Weights(res)

    def pow_(self, other):
        self.params = self.pow(other)

    def __pow__(self, other):
        return self.pow(other)

    # round()
    def round(self):
        res = dict()
        for key, value in self.items():
            res[key] = value.round()
        return Weights(res)

    def round_(self):
        self.params = self.round()

    def __round__(self):
        return self.round()

    """cmp
    # *_ : in-place version
    """

    def __lt__(self, other):
        # < other
        pass

    def __le__(self, other):
        # <= other
        pass

    def __gt__(self, other):
        # > other
        pass

    def __ge__(self, other):
        # >= other
        pass

    def __eq__(self, other):
        # == other
        pass

    def __ne__(self, other):
        # != other
        pass

    """type
    # TBA
    """

    def __hash__(self):
        pass  # TODO: SHA256

    """copy
    # TBA
    """

    # copy
    def _copy(self, other):
        if not isinstance(other, Weights):
            raise TypeError("The argument must be `Weight` but {}.".format(type(other)))
        return other.params  # deepcopy

    def copy_(self, other):
        self.params = self._copy(other)

    """tensors
    # TBA
    """

    # zeros
    def _zeros(self):
        res = dict()
        for key, value in self.items():
            res[key] = torch.zeros_like(value)
        return res

    def zeros(self):
        return Weights(self._zeros())

    def zeros_(self):
        self.params = self._zeros()

    # ones
    def _ones(self):
        res = dict()
        for key, value in self.items():
            res[key] = torch.ones_like(value)
        return res

    def ones(self):
        return Weights(self._ones())

    def ones_(self):
        self.params = self._ones()

    # fill and full
    def _pack(self, value):
        res = dict()
        for key, elem in self.items():
            res[key] = torch.empty_like(elem).fill_(value)
        return res

    def fill_(self, value):
        self.params = self._pack(value)

    def full(self, value):
        return Weights(self._pack(value))

    # empty
    def _empty(self):
        res = dict()
        for key, value in self.items():
            res[key] = torch.empty_like(value)
        return res

    def empty(self):
        return Weights(self._empty())

    """random
    # TBA
    """

    def _rand(self):
        res = dict()
        for key, value in self.items():
            res[key] = torch.rand_like(value)
        return res

    def rand_(self):
        self.params = self._rand()

    def rand(self):
        return Weights(self._rand())

    def _randn(self):
        res = dict()
        for key, value in self.items():
            res[key] = torch.randn_like(value)
        return res

    def randn_(self):
        self.params = self._randn()

    def randn(self):
        return Weights(self._randn())

    def randint_(self, high):
        self.params = self._randint(high)

    def _randint(self, high):
        res = dict()
        for key, value in self.items():
            res[key] = torch.randint_like(value, high)
        return res

    def randint(self, high):
        return Weights(self._randint(high))

    """TODO
    # type
    # cat
    # split
    """


"""distance
# TBA
"""


def FilterNorm(weights):
    # Filter-wise Normalization

    theta = Frobenius(weights)

    res = dict()
    for key, value in weights.items():
        d = Frobenius({key: value})
        d += 1e-10  # Ref. https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py#L111
        res[key] = value.div(d).mul(theta)

    if isinstance(weights, Weights):
        return Weights(res)
    else:
        return res


def Frobenius(weights, base_weights=None):
    # Frobenius Norm.

    total = 0.
    for key, value in weights.items():
        if base_weights is not None:
            elem = value.sub(base_weights[key])
        else:
            elem = value.clone().detach()

        elem.mul_(elem)
        total += torch.sum(elem).item()

    return math.sqrt(total)


if __name__ == "__main__":
    from net import DenseNet

    net1 = DenseNet(
        growthRate=12,
        depth=100,
        reduction=0.5,
        bottleneck=True,
        nClasses=10)
    w1 = Weights(net1.named_parameters())

    print(Frobenius(w1))
    print(Frobenius(FilterNorm(w1)))
    print(Frobenius(w1))
    print(Frobenius(w1, base_weights=w1))
