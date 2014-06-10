#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# File Name: KDTree.py
# Author: narutoacm
# Mail: narutoacm@gmail.com
# Website: www.narutoacm.com
# Created Time: 2014年06月06日 星期五 22时06分22秒
#########################################################################

import numpy as np
import heapq as hq
import math

def eculid_dis(vec1, vec2):
    d = vec1-vec2
    return np.dot(d, d)

class KDTree:
    class KDNode:
        def __init__(self, feats, vals):
            self.feats = feats
            self.vals = vals
            self.leaf = (len(feats) <= 1)
            self.splitdim = -1
            self.splitval = 0
            self.splitidx = -1
            self.leftchild = None
            self.rightchild = None

        def split(self):
            ''' split the node, and recursion to child, to build a kd tree '''
            n = len(self.feats)
            if n <= 1:
                self.leaf = True
                self.splitdim = -1
                self.splitval = 0
                self.splitidx = -1
                self.leftchild = None
                self.rightchild = None
                return

            # find the split dimension and the split value on the dimension
            d = len(self.feats[0])
            var_max = np.var(self.feats[:,0])
            self.splitdim = 0
            for i in range(1, d):
                var = np.var(self.feats[:,i])
                if var > var_max:
                    var_max = var
                    self.splitdim = i
            self.splitval = np.median(self.feats[:,self.splitdim])

            # partition the feats array, just like the partition algorithm in
            # quick sort algorithm
            j = -1
            for i in range(n):
                if self.feats[i][self.splitdim] <= self.splitval:
                    j += 1
                    tmp = np.copy(self.feats[j])
                    self.feats[j] = self.feats[i]
                    self.feats[i] = tmp
                    tmp = np.copy(self.vals[j])
                    self.vals[j] = self.vals[i]
                    self.vals[i] = tmp
                    if j == 0 or self.feats[j][self.splitdim] >= \
                            self.feats[self.splitidx][self.splitdim]:
                        self.splitidx = j

            tmp = np.copy(self.feats[self.splitidx])
            self.feats[self.splitidx] = self.feats[j]
            self.feats[j] = tmp
            tmp = np.copy(self.vals[self.splitidx])
            self.vals[self.splitidx] = self.vals[j]
            self.vals[j] = tmp

            # after partition, feats[0:splitidx+1] are leftchild node, the
            # rest are rightchild
            self.splitidx = j

            # if all features at one side, this is a leaf node
            if j == n-1:
                self.leaf = True
                self.leftchild = None
                self.rightchild = None
                return

            # recursion split the children
            self.leftchild = KDTree.KDNode(self.feats[:j+1,:], self.vals[:j+1,:])
            self.rightchild = KDTree.KDNode(self.feats[j+1:,:], self.vals[j+1:,:])
            self.leftchild.split()
            self.rightchild.split()


    def __init__(self):
        self.root = None

    def build(self, feats, vals):
        ''' build the tree '''
        feats = np.array(feats).reshape((len(feats),-1))
        vals = np.array(vals).reshape((len(vals),-1))
        self.root = KDTree.KDNode(feats, vals)
        self.root.split()

    def bbf_knn(self, feat, k, max_nn_chks = 100, dist_func = eculid_dis):
        ''' search k neighbors use bbf algorithm '''
        if self.root == None:
            return
        feat = np.array(feat)
        cnt, t = 0, 0
        res = []
        Q = [] # a prior queue based on the distance with feat
        hq.heapify(Q)
        hq.heappush(Q,(0, cnt, self.root))
        cnt += 1
        while len(Q) > 0 and t < max_nn_chks:
            t += 1
            node = hq.heappop(Q)[2]

            # go to the leaf node, and append the relative node to Q
            while node != None and not node.leaf:
                splitdim = node.splitdim
                splitval = node.splitval
                if feat[splitdim] <= splitval:
                    unexpl = node.rightchild
                    node = node.leftchild
                else:
                    unexpl = node.leftchild
                    node = node.rightchild
                hq.heappush(Q, (math.fabs(splitval - feat[splitdim]), cnt,
                    unexpl))
                cnt += 1

            # update the res table
            for i in range(len(node.feats)):
                dis = dist_func(node.feats[i], feat)
                #print feat,node.feats[i],dis
                if len(res) == 0:
                    res.append((dis, node.feats[i], node.vals[i]))
                else:
                    if dis >= res[len(res)-1][0]:
                        if len(res) < k:
                            res.append((dis, node.feats[i], node.vals[i]))
                    else:
                        if len(res) < k:
                            res.append(res[len(res)-1])
                        j = len(res) - 2
                        while j >= 0 and res[j][0] > dis:
                            res[j+1] = res[j]
                            j -= 1
                        res[j+1] = (dis, node.feats[i], node.vals[i])
        return res

if __name__ == '__main__':
    feats = [[1,1],[0,0],[3,1],[4,3],[-1,3],[2,5],[6,7]]
    vals = [0, 1, 2, 3, 4, 5, 6]
    t = KDTree()
    t.build(feats, vals)
    print t.bbf_knn([1,1], 3)
