from lark import Lark, Transformer, exceptions
from copy import deepcopy as dc
from functools import reduce
import numpy as np
import math
from util import *
import re

class Entropy:
    factor = 0
    variables = []
    conditions = None

    def __init__(self, variables, conditions=None, factor=1):
        self.factor = factor
        self.variables = sorted(variables)
        self.conditions = conditions
        if conditions:
            self.conditions = sorted(self.conditions)

        check = variables.copy()
        if conditions:
            check += conditions
        if len(check) != len(set(check)):
            raise ValueError('Dualplicate variables found in {}'.format(self))

    def __str__(self):
        s = 'H('
        if close(self.factor, -1):
            s = '-' + s
        elif not close(self.factor, 1):
            s = str(self.factor) + s
        s += ','.join(self.variables)
        if self.conditions:
            s += '|' + ','.join(self.conditions)
        s += ')'
        return s

    def __repr__(self):
        return str(self)

    def canonical(self):
        if not self.conditions:
            return Combination(self)
        else:
            e1 = Entropy(self.variables + self.conditions, None, self.factor)
            e2 = Entropy(self.conditions.copy(), None, self.factor)
            c = Combination(e1)
            c.sub(e2)
            return c

    def copy(self):
        return dc(self)

    def rvs(self):
        check = self.variables.copy()
        if self.conditions:
            check += self.conditions.copy()
        return list(set(check))

    def get_b(self, n, rvs):
        if self.conditions:
            raise ValueError('Cannot get b vector from conditonal entropy. Please turn it to canonical form first.')
        idx = list(map(lambda x: rvs.index(x), self.variables))
        idx = list(map(lambda x: 1 << x, idx))
        idxx = reduce(lambda x, y: x | y, idx, 0) - 1
        k = get_k(n)
        out = np.zeros(k)
        out[idxx] = self.factor
        return out

    def replace_var(self, target, replacement):
        if self.conditions:
            raise ValueError('Cannot replace variables in conditonal entropy. Please turn it to canonical form first.')
        if not set(target).issubset(set(self.variables)):
            return
        self.variables = list(set(self.variables) - set(target))
        self.variables.append(replacement)

class MutualInformation:
    factor = 0
    leftVariables = []
    rightVariables = []
    conditions = None

    def __init__(self, left, right, conditions=None, factor=1):
        self.factor = factor
        self.conditions = conditions
        if conditions:
            self.conditions = sorted(self.conditions)
        self.leftVariables = sorted(left)
        self.rightVariables = sorted(right)

        check = left + right
        if len(check) != len(set(check)):
            raise ValueError('dualplicate found in {}'.format(self))

    def __str__(self):
        s = 'I('
        if close(self.factor, -1):
            s = '-' + s
        elif not close(self.factor, 1):
            s = str(self.factor) + s
        s += ','.join(self.leftVariables)
        s += ';'
        s += ','.join(self.rightVariables)
        if self.conditions:
            s += '|' + ','.join(self.conditions)
        s += ')'
        return s

    def __repr__(self):
        return str(self)

    def canonical(self):
        if not self.conditions:
            el = Entropy(self.leftVariables.copy(), None, self.factor)
            er = Entropy(self.rightVariables.copy(), None, self.factor)
            eall = Entropy(self.leftVariables.copy() + self.rightVariables.copy(), None, self.factor)

            c = Combination([el, er])
            c.sub(eall)
            return c
        else:
            el = Entropy(list(set(self.leftVariables.copy() + self.conditions.copy())), None, self.factor)
            er = Entropy(list(set(self.rightVariables.copy() + self.conditions.copy())), None, self.factor)
            eall = Entropy(list(set(self.leftVariables.copy() + self.rightVariables.copy() + self.conditions.copy())), None, self.factor)
            ec = Entropy(self.conditions.copy(), None, self.factor)

            c = Combination([el, er])
            c.sub(eall)
            c.sub(ec)
            return c

    def copy(self):
        return dc(self)

    def rvs(self):
        check = self.leftVariables.copy()
        check += self.rightVariables.copy()
        if self.conditions:
            check += self.conditions.copy()
        return list(set(check))

    def get_b(self, n, rvs):
        raise ValueError('Cannot get b from mutual information. Please convert it to canonical form before doing so')

    def replace_var(self, target, replacement):
        raise ValueError('Cannot replace variables in mutual information. Please convert it to canonical form before doing so')

class Combination:
    measures = []

    def __init__(self, measures):
        if type(measures) == list:
            self.measures = measures.copy()
        elif type(measures) == Combination:
            self.measures = measures.measures.copy()
        else:
            self.measures = [measures.copy()]
    
    @staticmethod
    def from_b(b, n, rvs, cadmm):
        idx = np.where(b != 0)[0]
        val = b[idx]

        rv_idx = list(map(lambda x: cadmm.get_elements(x + 1), idx))

        def f(idx):
            ln = list(map(lambda x: int(math.log(x, 2)), idx))
            return list(map(lambda x: rvs[x], ln))

        rvlst = list(map(lambda x: f(x), rv_idx))
        elst = list(map(lambda t: Entropy(t[1], None, val[t[0]]), enumerate(rvlst)))
        return Combination(elst)

    def add(self, measure):
        self.measures.append(measure.copy())

    def sub(self, measure):
        measure.factor *= -1
        self.measures.append(measure.copy())

    def canonical(self):
        cs = list(map(lambda x: x.canonical(), self.measures))
        ms = list(map(lambda x: x.measures.copy(), cs))
        mms = reduce(lambda x, y: x + y, ms, [])
        return Combination(mms)

    def scale(self, f):
        def s(m, f):
            mm = m.copy()
            mm.factor = mm.factor * f
            return mm
        m = list(map(lambda x: s(x, f), self.measures))
        return Combination(m)

    def __mul__(self, other):
        if type(other) not in (float, int):
            raise TypeError('You can only muliply a combination with a float or int')
        return self.scale(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def combine(self, c):
        cc = Combination(self.measures + c.measures)
        return cc

    def __str__(self):
        if len(self.measures) == 0:
            return '0'
        res = ' + '.join(map(lambda x: str(x), self.measures))
        return re.sub(' \+ \-', ' - ', res)

    def __repr__(self):
        return str(self)

    def copy(self):
        return dc(self)

    def rvs(self):
        return list(set(reduce(lambda x, y: x + y.rvs(), self.measures, [])))

    def get_b(self, n, rvs):
        return reduce(lambda x, y: x + y.get_b(n, rvs), self.measures, 0)

    def replace_var(self, target, replacement):
        for m in self.measures:
            m.replace_var(target, replacement)

class Relation:
    lhs = None
    rhs = None
    relation = None
    extra_rvs = []

    def __init__(self, lhs, rhs, relation):
        self.lhs = lhs
        self.rhs = rhs
        if type(lhs) != float:
            self.lhs = lhs.copy()
        if type(rhs) != float:
            self.rhs = rhs.copy()

        if relation == 'ge' or relation == '>=':
            self.relation = '>='
        elif relation == 'le' or relation == '<=':
            self.relation = '<='
        else:
            self.relation = '='

        if (type(lhs) == float and type(rhs) == float):
            raise ValueError('At least one side of the input should be an information expression')
        if (type(lhs) == float and lhs != 0) or (type(rhs) == float and rhs != 0):
            raise ValueError('The number on one side has to be exactly 0')

    def __str__(self):
        return str(self.lhs) + ' ' + self.relation + ' ' + str(self.rhs)

    def __repr__(self):
        return str(self)

    def canonical(self):
        l = self.lhs
        r = self.rhs
        if type(self.lhs) != float:
            l = self.lhs.copy()
            l = l.canonical()
        if type(self.rhs) != float:
            r = self.rhs.copy()
            r = r.canonical()
        return Relation(l, r, self.relation)

    def push_left(self):
        l = self.lhs
        r = self.rhs
        if type(l) != float:
            l = self.lhs.copy()
        if type(r) != float:
            r = self.rhs.copy()

        if self.relation == '>=' or self.relation == '=':
            if r == 0:
                return Relation(l, r, self.relation)
            r = r * -1
            if type(l) == float:
                l = r
            else:
                l = l.combine(r)
            r = float(0)
            return Relation(l, r, self.relation)
        else:
            return Relation(l * -1, r * -1, '>=').push_left()

    def copy(self):
        return dc(self)

    def add_rvs(self, ervs):
        self.extra_rvs += ervs
        self.extra_rvs = list(set(self.extra_rvs))

    def rvs(self):
        check = []
        if type(self.lhs) != float:
            check += self.lhs.rvs()
        if type(self.rhs) != float:
            check += self.rhs.rvs()
        check += self.extra_rvs
        return sorted(list(set(check)))

    def nrv(self):
        return len(self.rvs())

    def get_b(self, n=None, rvs=None):
        if not n:
            n = self.nrv()
        if not rvs:
            rvs = self.rvs()
        if self.rhs != 0:
            raise ValueError('Please use push left before getting b')
        return self.lhs.get_b(n, rvs)

    def b(self, n=None, rvs=None):
        return self.canonical().push_left().get_b(n, rvs)

    def replace_var(self, target, replacement):
        if self.rhs != 0:
            raise ValueError('Please use push left before replacing variables')
        self.lhs.replace_var(target, replacement)

parser = Lark(r"""
        int: /[0-9]/+
        num: SIGNED_NUMBER | "-"
        rv: /[A-Za-z]/ [int]
        rvlist: rv ("," rv)*
        pair: rvlist ";" rvlist

        measure: entropy | mutual_information
        entropy: [num] "H(" rvlist ["|" rvlist] ")"
        mutual_information: [num] "I(" pair ["|" rvlist] ")"

        macro1: rvlist "->" rvlist "->" rvlist ("->" rvlist)*
        macro2: rvlist ":" rvlist
        macro3: rvlist "." rvlist ("." rvlist)*

        macro: macro1 | macro2 | macro3

        combination: sum | diff | measure
        sum: combination "+" measure
        diff: combination "-" measure

        relation: ge | le | eq | macro
        ge: (combination | num) ">=" (combination | num)
        le: (combination | num) "<=" (combination | num)
        eq: (combination | num) "=" (combination | num)

        %import common.ESCAPED_STRING
        %import common.SIGNED_NUMBER
        %import common.WS
        %ignore WS

        """, start='relation')

class T(Transformer):
    def int(self, n):
        return n[0]
    def num(self, n):
        if len(n) == 0:
            return float(-1)
        return float(n[0])
    def rv(self, v):
        return ''.join(v)
    def rvlist(self, items):
        return items
    def pair(self, items):
        return (items[0], items[1])
    def entropy(self, items):
        e = None
        if type(items[0]) == float:
            f = items[0]
            if len(items) == 3:
                e = Entropy(items[1], items[2], f)
            else:
                e = Entropy(items[1], None, f)
        else:
            if len(items) == 2:
                e = Entropy(items[0], items[1])
            else:
                e = Entropy(items[0], None)
        return e
    def mutual_information(self, items):
        m = None
        if type(items[0]) == float:
            f = items[0]
            if len(items) == 3:
                m = MutualInformation(items[1][0], items[1][1], items[2], f)
            else:
                m = MutualInformation(items[1][0], items[1][1], None, f)
        else:
            if len(items) == 2:
                m = MutualInformation(items[0][0], items[0][1], items[1])
            else:
                m = MutualInformation(items[0][0], items[0][1])
        return m
    def measure(self, items):
        return items[0]
    def combination(self, items):
        return items[0]
    def sum(self, items):
        return Combination(items)
    def diff(self, items):
        c = Combination(items[0])
        c.sub(items[1])
        return c
    def macro1(self, items):
        def expand(lst):
            return [(lst[:i],lst[i],lst[i+1]) for i in range(1,len(lst)-1)]
        tuples = expand(items)
        def flatten(lst):
            return [i for sub in lst for i in sub]
        i_lst = map(lambda t: MutualInformation(flatten(t[0]), t[2], t[1]), tuples)
        c_lst = map(lambda i: Combination([i]), i_lst)
        r_lst = list(map(lambda c: Relation(c, 0.0, 'eq'), c_lst))
        return r_lst
    def macro2(self, items):
        h = Entropy(items[0], items[1])
        c = Combination([h])
        r = Relation(c, 0.0, 'eq')
        return [r]
    def macro3(self, items):
        def flatten(lst):
            return [i for sub in lst for i in sub]
        c1 = Combination([Entropy(flatten(items))])
        c2 = Combination([Entropy(i) for i in items])
        r = Relation(c1, c2, 'eq')
        return [r]
    def macro(self, items):
        return items[0]
    def relation(self, items):
        if type(items[0]) == list:
            return items[0]
        re = items[0].data
        ch = items[0].children
        r = Relation(ch[0], ch[1], re)
        return r

def parse(text):
    try:
        tree = parser.parse(text)
        return T().transform(tree)
    except exceptions.ParseError:
        raise ValueError('Invalid input.')

