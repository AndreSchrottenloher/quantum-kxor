
"""
This file implements the merging trees of the paper 
'Improved Quantum Algorithms for the k-XOR Problem' (https://eprint.iacr.org/2021/407)

It allows to obtain the optimization results of the paper WITHOUT quantum walks.

During the optimization, it considers all possible binary trees, which is why the
time increases significantly with k (it starts to be slow from k = 8).

The code relies on the SCIP Optimization suite (https://www.scipopt.org)
and its python interface pyscipopt.

"""

from decimal import Decimal
from fractions import Fraction
import random
from functools import lru_cache
from pyscipopt import Model, quicksum
from operator import itemgetter
import math
import copy


FLAGS = {
    "classical-seq" : "",
    "quantum-seq" : "",
    "classical-ram": "",
    "quantum-ram" : ""
}

# choose whether we minimize the time or the time-memory product
MINIMIZER_FLAG = {
    "time": "the time complexity only",
    "product": "the time-memory product"
}



#===================================================
# Tree representation

#Merging trees can be exported to text or tikz code. Here is how to interpret the
#text output, with the single-solution 7-xor algorithm of the paper (time
#and memory 0.2857)

#(0.2857143, 0.2857143, 
# Repeated 0.429  (3/7)  times:
#(L0_0|7|qsample|s=0|z=0.571  (4/7) |st=0.071  (1/14) )
#     (L1_0|4|qsample|s=0.143  (1/7) |z=0.286  (2/7) |st=0)
#          (L2_0|2|qsample|s=0.143  (1/7) |z=0.143  (1/7) |st=0)
#               (L3_0|1|qsample|s=0.143  (1/7) |z=0|st=0)
#               (L3_1|1|cstored|s=0.143  (1/7) |z=0|st=0)
#          Subtree Outside the loops:
#          (L2_1|2|cstored|s=0.143  (1/7) |z=0.143  (1/7) |st=1)
#               (L3_2|1|csample|s=0.143  (1/7) |z=0|st=0)
#               Subtree Outside the loops:
#               (L3_3|1|cstored|s=0.143  (1/7) |z=0|st=0)
#     Repeated 0.286  (2/7)  times:
#     (L1_1|3|cstored|s=0.143  (1/7) |z=0.286  (2/7) |st=0)
#          (L2_2|1|csample|s=0.143  (1/7) |z=0|st=0)
#          Subtree Outside the loops:
#          (L2_3|2|cstored|s=0.286  (2/7) |z=0|st=1)
#               (L3_6|1|csample|s=0.143  (1/7) |z=0|st=0)
#               Subtree Outside the loops:
#               (L3_7|1|cstored|s=0.143  (1/7) |z=0|st=0))

#Each node specifies whether it's sampled or stored, and if it's a quantum
#or classical list. The size, prefix size (z), and sample time are given.
#The repetition loops concern the right children of the main subbranch. The
#repetition variables are given (they do not take Grover search into account,
#contrary to the sample times). Children of these repeated subtrees inherit
#from the repetition variable. Sometimes, a subtree is "outside the loops",
#so it doesn't need to be repeated. Its total time complexity is computed
#and taken separately.

#The corresponding algorithm follows:
#Outside the loops, build the full space for L3_1 in time 0.143  (1/7)  
#Outside the loops, build the full space for L3_3 in time 0.143  (1/7)  
#Outside the loops, build the full space for L2_1 in time 0.286  (2/7)  
#Outside the loops, build the full space for L2_3 in time 0.286  (2/7)  
#For 0.286  (2/7)  repetitions of L1_1 do
# |Create a list L2_3 of 0.286  (2/7)  2-XORs on 0 bits with outside-loop data (no time)
# |Build the list L1_1 of 0.143  (1/7)  3-XORs on 0.286  (2/7)  bits in time 0.143  (1/7)  by repeating the following:
# |    Sample an element with 0 zeroes
# |    Match with L2_3 and (try to) obtain a 3-XOR on 0.286  (2/7)  bits
# |Create a list L2_1 of 0.286  (2/7)  2-XORs on 0 bits with outside-loop data (no time)
# |For 0.143  (1/7)  repetitions of L3_1 do
# | |Create a list L3_1 of 0 1-XORs on 0 bits with outside-loop data (no time)
# | |Sample an element with 0 zeroes
# | |Match with L3_1 and (try to) obtain a 2-XOR on 0 bits
# | |Match with L2_1 and (try to) obtain a 4-XOR on 0.286  (2/7)  bits
#
# It's a little different, but equivalent, from the algorithm given in the paper.


    
def strappr(nb):
    """
    Prints a float approximated to some decimals and also to a fraction.
    """
    if type(nb) is not float:
        return ""
    d = Decimal(nb)
    if nb == 0:
        return "0"
    if nb == int(nb):
        return str(int(nb))
    return  str(round(nb, 3)) + "  (" + str(Fraction(d).limit_denominator(10000)) + ") "


def myround(n):
    # if a small rounding is required
    return round(n,3)


HEADER = """
%====================================================
% tikz picture of a merging tree
% /!\ lateX document preamble should contain the following (at least):
%\\documentclass{article}
%\\usepackage{tikz}
%\\usetikzlibrary{trees}
%\\usetikzlibrary{arrows}

\\begin{tikzpicture}[grow=up,level distance=30mm, nodes={draw,rectangle,rounded corners=.25cm,->}, 
    level 1/.style={sibling distance=70mm}, level 2/.style={sibling distance=35mm}, 
    level 3/.style={sibling distance=17mm}]
"""


class Node:

    def __init__(self, flag="sampled", left=None, right=None, k=0):
        if flag not in ["sampled", "stored"]:
            raise ValueError("invalid flag:" + flag)
        self.l = left
        self.r = right
        if left is not None and not left.is_sampled:
            raise ValueError("invalid tree shape")
        if right is not None and right.is_sampled:
            raise ValueError("invalid tree shape")
        self.flag = flag
        self.name = ""
        self.is_leaf = (self.l is None and self.r is None)
        self.is_sampled = (self.flag == "sampled")
        self.k = k
        
        self.s = None
        self.z = None
        self.st = None
        self.isqs = None
        self.repeats = 0
        self.is_root = False
        self.outside_loops = 0
        

    def __str__(self):
        node_type = "qsample"
        if (str(self.flag) == "sampled" and self.isqs == 0):
            node_type = "csample"
        elif (str(self.flag) == "stored" and self.isqs == 0):
            node_type = "cstored"
        elif (str(self.flag) == "stored" and self.isqs == 1):
            node_type = "qstored"
            
        if self.repeats is None or self.repeats == 0.:
            res = ""
        else:
            if self.flag == "stored" and self.outside_loops == 1:
                res = "Subtree outside the loops:\n"
            else:
                res = "Repeated " + strappr(self.repeats) + " times:\n"
        res += ("(%s|%i|%s|s=%s|z=%s|st=%s)" 
                % (self.name, self.k, node_type, strappr(self.s), strappr(self.z), strappr(self.st) ) )
        if self.l is not None:
            l = str(self.l).split("\n")
            for s in l:
                res += "\n" + "     " + s
        if self.r is not None:
            l = str(self.r).split("\n")
            for s in l:
                res += "\n" + "     " + s
        return res

    def _labels_to_tex(self):
        node_type = "qsample"
        if (str(self.flag) == "sampled" and self.isqs == 0):
            node_type = "csample"
        elif (str(self.flag) == "stored" and self.isqs == 0):
            node_type = "cstored"
        elif (str(self.flag) == "stored" and self.isqs == 1):
            node_type = "qstored"
        
        tmp = ""
        if (str(self.flag) == "sampled" and self.isqs == 0):
            tmp = "[dashed]"
        elif  (str(self.flag) == "sampled" and self.isqs == 1):
            tmp = "[dashed, very thick]"
        
        
        repeatsline = ("""\\\\ r = %s """ % strappr(self.repeats)) if self.repeats != 0. else ""
        outsideline = ("""\\\\ outside loops""" if self.outside_loops != 0 else "")
        res = ("""node%s {\\begin{tabular}{c} k = %i \\\\ s = %s \\\\ z = %s \\\\ st = %s %s %s \end{tabular}}""" 
                % (tmp, self.k, strappr(self.s), strappr(self.z), strappr(self.st), repeatsline, outsideline ) )
        
        return res


    def to_tex(self):
        res = ""
        if self.is_root:
            res += HEADER
            res += """\\%s\n""" % self._labels_to_tex()
        else:
            res += """%s\n""" % self._labels_to_tex()
        if self.r is not None:
            res += "child { %s }\n" % self.r.to_tex()
        if self.l is not None:
            res += "child { %s }" % self.l.to_tex()
        if self.is_root:
            res += """;\n\\end{tikzpicture}"""
        return res


    def __repr__(self):
        return self.__str__()


    def post_order_iter(self):
        if self.l is not None:
            for n in self.l.post_order_iter():
                yield n
        if self.r is not None:
            for n in self.r.post_order_iter():
                yield n
        yield self


    def iter_left_branch(self):
        if self.l is not None:
            for n in self.l.iter_left_branch():
                yield n
        yield self


    def right_nodes_of_left_branch(self):
        if self.r is not None:
            yield self.r
        if self.l is not None:
            for n in self.l.right_nodes_of_left_branch():
                yield n
    
    # left branch including self, leaf first
    def left_branch(self):
        if self.l is not None:
            for n in self.l.left_branch():
                yield n
        yield self
       

    def name_nodes(self, current_path="0", level=0):
        self.name = "L%i_%i" % (level, int(current_path, 2))
        # 0: left, 1: right
        if not self.is_leaf:
            self.l.name_nodes(current_path+"0", level=level+1)
            self.r.name_nodes(current_path+"1", level=level+1)


    def to_algorithm(self):
        res = ""
        for node in self.post_order_iter():
            if node.outside_loops > 0:
                res += ("Outside the loops, build the full space for %s in time %s \n" % 
                                    (node.name , strappr(float(node.k) / float(node.totalk))) )
        return res + self._to_algorithm()
        

    def _to_algorithm(self, indent=""):
        res = ""
        if self.is_leaf:
            if self.flag == "sampled":
                res += indent + ("Sample %s: an element with %s prefix\n" % (self.name, strappr(node.z)))
            else:
                res += indent + ("Build the list %s of %s elements with %s prefix in time %s\n" %
                        (self.name, strappr(self.s), strappr(self.z), strappr(self.time)))
            return res
        
        right_subtrees = self.right_nodes_of_left_branch()
        new_indent = indent

        # first explain how we build the right subtrees
        for node in right_subtrees:
            if node.repeats > 0:
                res += new_indent + "For %s repetitions of %s do\n" % (strappr(node.repeats), node.name)
                new_indent += " |"
            if node.outside_loops > 0:
                res += new_indent + ("Create a list %s of %s %i-XORs on %s bits with outside-loop data (no time)\n"
                     % (node.name, strappr(node.s), node.k, strappr(node.z)))
            else:
                res += node._to_algorithm(new_indent)
        # then how we sample from this node and build the list (if stored)

        if self.flag == "stored":
            res += new_indent + ("Build the list %s of %s %i-XORs on %s bits in time %s by repeating the following:\n" 
                        % (self.name, strappr(self.s), self.k, strappr(self.z), strappr(self.time)) )
            new_indent += "    "
        left_branch = self.left_branch()
        if self.flag == "sampled":
            res += new_indent + ("Sample %s (sample time %s) by repeating the following:\n"
                                % (self.name, strappr(self.st) ))
            new_indent += "  "
        for node in left_branch:
            if node.is_leaf:
                res += new_indent + ("Sample %s: an element with %s prefix\n" 
                    % (node.name, strappr(node.z)))
            else:
                res += new_indent + ("Match with %s and (try to) obtain a %i-XOR on %s bits\n" 
                                % (node.r.name, node.k, strappr(node.z) ))
        return res


@lru_cache(maxsize=None)
def all_binary_trees(k, flag="sampled"):
    """
    
    >>> len(all_binary_trees(3))
    2
    """
    if k == 0:
        return []
    elif k == 1:
        return [Node(flag, k=1)]
    else:
        res = []
        for i in range(1,k):# 1 to k-1 included
            tmp = all_binary_trees(i,flag="sampled")
            ttmp = all_binary_trees(k-i, flag="stored")
            for t in tmp:
                for tt in ttmp:
                    ntmp = Node(flag=flag, left=copy.deepcopy(t), right=copy.deepcopy(tt), k=k)
                    res.append(ntmp)
        return res


@lru_cache(maxsize=None)
def opt_binary_tree(k, flag="sampled"):
    # a binary tree that divides k in half, and puts more weight on the quantum (left) part
    # always if necessary
    if k == 1:
        return Node(flag, k=1)
    else:
        if k % 2 == 0:
            return Node(flag=flag, left=copy.deepcopy(opt_binary_tree(k//2, "sampled")),
                                    right=copy.deepcopy(opt_binary_tree(k//2, "stored")), k=k)
        else:
            return Node(flag=flag, left=copy.deepcopy(opt_binary_tree(k//2 + 1, "sampled")),
                                    right=copy.deepcopy(opt_binary_tree(k//2, "stored")), k=k)



#===============================================



def optimize_kxor_tree(t,d=1,memub=100,verb=True, flag="quantum-ram",
        access=True,multencr=False,minimize="time",export=False):
    """
    Optimizes a given (extended) merging tree.
    
    Parameters:
    - t -> shape of the tree (as in the file kxor_tree.py). It contains the
    information of k.
    - d -> size of the domain, in number of bits, in multiples of the codomain n
    (the number of bits to zero)
    - 0 and k. If d is smaller than 1/k this should yield an error
    - memub -> memory constraint
    - flag -> computation model (classical, quantum, type of memory access)
    - multencr -> True if we are in the multiple-encryption case (this does
    not seem to make any difference for our algorithms)
    - minimize -> whether we minimize the time or time-memory product

    We assume that quantum oracle access is given (if the domain size is small,
    this does not make any difference).

    The following flags are accepted:
    - quantum-seq: quantum computations allowed, but no qRAM (memory accesses
            are sequential). Classical RAM is allowed.
      Also, exponential qubits are allowed, so the results will differ from
      an optimization with classical memory only.
    - quantum-ram: quantum computations allowed, and qRAM (either QRACM / QRAQM)
    - classical-seq: classical computations, and no RAM (memory accesses are
            sequential)
    - classical-ram: classical computations and RAM
    
    /!\ the program has mostly been tested in the many-solution and single-solution
    cases, and not extensively tested with the "sequential" flags. It may not
    work correctly in cases not described in the paper.

    """

    if flag not in FLAGS or minimize not in MINIMIZER_FLAG:
        raise ValueError("Unrecognized flag: " + str(flag))
    if d > 1 or d < 0:
        raise ValueError("Domain size must be 0 <= d <= 1")
    if verb:
        print(t.k)
    if multencr and d != 1/t.k:
        raise ValueError("This cannot be the multiple-encryption problem, d should be set to 1/k")

    #=================================================
    model = Model("k-xor")
    model.hideOutput()
    
    # total time: maximum of individual time complexities of stored
    # lists, and taking into account repetition loops
    total_time = model.addVar(vtype="C", lb=0, ub=1)
    # total memory: maximum of stored list sizes
    total_mem = model.addVar(vtype="C", lb=0, ub=1)
    
    model.addCons( total_mem <= memub )
    
    # decides if the "repeat" loops are done with Grover search, or classically
    qrepeats = 0 if "classical" in flag else 1
    
    t.name_nodes()
    
    t.is_root = True
    # create all the variables attached to each node
    for node in t.post_order_iter():
        # size of the list
        node.s = model.addVar(vtype="C", lb=0, ub=1)
        # number of "zeroes" (prefix size)
        node.z = model.addVar(vtype="C", lb=0, ub=1)
        # decides if the list is a quantum sample
        node.isqs = model.addVar(vtype="B", lb=0, ub=1)
        # sample time of the list
        node.st = model.addVar(vtype="C", lb=0, ub=1)
        # number of repetitions of the list
        node.repeats = model.addVar(vtype="C", lb=0, ub=1)
        # total time complexity to create the list (if stored)
        # this include the child nodes
        node.time = model.addVar(vtype="C", lb=0, ub=1)
        node.totalk = t.k

    # structural constraints
    # on the list sizes, zeroes and wether they are sampled quantumly
    for node in t.post_order_iter():
        if "classical" in flag:
            # if the computation model is classical, no node can be quantumly
            # (with quantum search) sampled
            model.addCons( node.isqs == 0 )
        # in the "multiple-encryption" case (or bicomposite in general),
        # the prefixes must always be multiples of 1/k
        if multencr:
            inttmp = model.addVar(vtype="I", lb=0, ub=t.k)
            model.addCons( (node.z)*t.k == inttmp )

        # maximal size of this list
        model.addCons( node.s <= node.k*d - node.z )

        if not node.is_leaf:
            # the node has a left and a right child. Their prefix sizes are equal
            model.addCons( node.l.z == node.r.z )

            # the list size is obtained from the children list sizes and the new
            # prefix constraint
            model.addCons( node.s ==  node.l.s + node.r.s - node.z + node.r.z )

            # if we sample quantumly from this node, then we must sample
            # quantumly from its left child
            model.addCons( node.l.isqs >= node.isqs )


    # constraints on the sample time and total time
    for node in t.post_order_iter():
        
        if node.is_leaf:
            # if node is a leaf, it is sampled by a simple exhaustive search
            model.addCons( node.st == node.z*(1-0.5*node.isqs) )
        else:

            # now the constraints to compute the sample time: tmp is the cost
            # of a search iterate
            tmp = model.addVar(vtype="C", lb=0, ub=1)
            # a search iterate requires to sample the left list
            model.addCons( tmp >= node.l.st )
            # in the "sequential" case, we also need sequential membership tests to the right list
            if flag == "quantum-seq":
                # but these cost only if we are doing a quantum sample (otherwise
                # we may use classical RAM)
                model.addCons( tmp >= node.isqs*node.r.s )
            if flag == "classical-seq":
                model.addCons( tmp >= node.r.s )
            
            # number of search iterates to sample an element
            iterates = model.addVar(vtype="C", lb=0, ub=1)
            # there are (rare) cases where node.z - node.r.z - node.r.s is smaller than 0
            # the folllowing constraint:
#            model.addCons( iterates >= (node.z - node.r.z - node.r.s) )
            # would suffice, but we want to have an equality, hence the following:
            #=====
            tmpc = model.addVar(vtype="C", lb=-1, ub=1)
            model.addCons( tmpc == (node.z - node.r.z - node.r.s) )
            tmpbool = model.addVar(vtype="B")
            model.addCons( tmpbool* tmpc >= 0 )
            model.addCons( (1-tmpbool)* tmpc <= 0 )
            model.addCons( iterates == tmpbool*tmpc )
            #====
            
            # sample time formula
            model.addCons( node.st == (1-0.5*node.isqs)*iterates + tmp )

        if node.flag == "stored":
            model.addCons( total_mem >= node.s )

            # here a special trick: if the node is a full list, we can take
            # it out from its repetition loop (this is actually a particular
            # example of nested repetition loops. More complex nestings don't help
            # in the single-solution case.)
            node.outside_loops = model.addVar(vtype="B")

            # if the node is "outside the loops", we count its individual time as 0, 
            # and add a global time and memory factor
            # (also, this works only with RAM / qRAM)
            model.addCons( node.time >= (1-node.outside_loops)*(node.st + node.s) )
            if not node.is_leaf:
                model.addCons( node.time >= node.r.time )

            model.addCons( total_time >= node.outside_loops*node.k*d )
            model.addCons( total_mem >= node.outside_loops*node.k*d )
            

    for node in t.post_order_iter():

        if node.is_leaf and not access:
            # if no quantum oracle access
            # we need a list of size s + z, and classical queries to build it
            # also, this works only with qRAM / RAM
            model.addCons( total_mem >= node.s + node.z )
            model.addCons( total_time >= node.s + node.z )
       

    # these nodes / subtrees are the repetition loops
    rnlb = list(t.right_nodes_of_left_branch())
    # start with the node attached at level 1, then 2, etc
    
    for node in rnlb:
        model.addCons( node.repeats <= node.k*d - node.s )
        
    model.addCons( t.repeats == quicksum( [n.repeats for n in rnlb] ))
    model.addCons( t.s == 0 )
    model.addCons( t.z + t.repeats == 1 )
    model.addCons( total_time >= t.st + t.repeats*(1 - 0.5*qrepeats) )

    
    for i in range(len(rnlb)):
        # repeat variables here are the repetition loops of stored lists
        model.addCons( total_time >= 
                quicksum([rnlb[j].repeats for j in range(0,i+1) ])*(1 - 0.5*qrepeats) + rnlb[i].time )


    if minimize == "time":
        # minimizing the time complexity and, for a given time, also
        # taking the minimum memory
        model.setObjective(100*total_time + total_mem,sense="minimize")
    elif minimize == "product":
        # minimizing the time-memory product
        model.setObjective(total_time + total_mem, sense="minimize")
    model.optimize()

    
    for node in t.post_order_iter():
        node.repeats = model.getVal(node.repeats)
        node.s = model.getVal(node.s)
        node.z = model.getVal(node.z)
        node.st = model.getVal(node.st)
        node.isqs = model.getVal(node.isqs)
        node.time = model.getVal(node.time)
        if node.flag == "stored":
            node.outside_loops = model.getVal(node.outside_loops)

    if model.getStatus() != "optimal":
        raise Exception("Could not find an optimal solution")
    return (model.getVal( total_time ), model.getVal( total_mem ), t)



def find_kxor_tree(k,d=1,memub=100,verb=False,flag="quantum-ram",
        access=True,multencr=False,minimize="time",export=False,onlybalanced=False):
    """
    
    >>> find_kxor_tree(4,d=1, flag="quantum-ram", onlybalanced=True)[0]
    0.25
    
    >>> find_kxor_tree(4,d=1, flag="classical-ram", onlybalanced=True)[0]
    0.333333

    >>> find_kxor_tree(4,d=0.25, flag="quantum-ram", onlybalanced=True)[0]
    0.3125
    
    >>> find_kxor_tree(5,d=0.2, flag="quantum-ram", onlybalanced=True)[0]
    0.3
    
    >>> find_kxor_tree(5,d=1/5., flag="classical-ram", onlybalanced=True, multencr=False)[0]
    0.6
    
    """

    trees = [opt_binary_tree(k)] if onlybalanced else all_binary_trees(k)
    results = []
    for t in trees:
        try:
            tmp = optimize_kxor_tree(t,d=d,memub=memub,verb=False, 
                        flag=flag,access=access,multencr=multencr,
                        minimize=minimize,export=export)
            results.append( (round(tmp[0],6), round(tmp[1],6), tmp[2] ))
        except Exception as e:
            print(e)
    
    if not results:
        raise Exception("All optimizations raised an error")

    if minimize == "time":
    # take the best times
        best = sorted(results, key=lambda x: x[0])
        bestvalue = best[0][0]
        best = [t for t in best if abs(t[0]-bestvalue) < 0.00001]
        # take the best memory
        best = sorted(best, key=lambda x: x[1])[0]
    elif minimize == "product":
        # take the best time-memory product
        best = sorted(results, key=lambda x: x[0] + x[1])
        bestvalue = best[0][0] + best[0][1]
        best = [t for t in best if abs(t[0] +t[1] -bestvalue) < 0.00001]
        # take the best time
        best = sorted(best, key=itemgetter(0))[0]

    if verb:
        print(best[2])
        print(best[2].to_algorithm())
    
    return best


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    print("All tests completed\n")
    
    # testing
#    oracle_access = True
#    for k in range(2,10):
#        res = find_kxor_tree(k,d=1./k,verb=False,flag="quantum-ram",access=oracle_access, minimize="time")
#        print(k, res[0],  res[0] + res[1])
    
#    for i in range(1,5):
#        res = find_kxor_tree(8,d=float(i)/8.,verb=False,flag="quantum-ram",access=oracle_access, minimize="product")
#        print(i, res[0] + res[1])
    
#    find_kxor_tree(4,d=0.25, flag="quantum-ram", onlybalanced=True)
    
#    print(find_kxor_tree(7,d=1./7.,verb=True,flag="quantum-ram",access=True, minimize="time"))



