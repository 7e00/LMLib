#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# Author: narutoacm - www.narutoacm.com
# Email: narutoacm@gmail.com
# File Name:		CART.py
# Last modified:	2014-07-05 11:26
#########################################################################

import numpy as np
import math
class CART:
    class Node:
        def __init__(self, featidx = -1, splitfunc = None, result = None, depth
                = 0, leafnum = 0, childs = None):
            self.featidx = featidx
            self.splitfunc = splitfunc
            self.result = result
            self.depth = depth
            self.leafnum = leafnum
            self.childs = childs


    @staticmethod
    def defaultSplitFunc(data, interval, featidx, threshold):
        if type(threshold) == int or type(threshold) == float:
            return lambda x: x >= threshold
        else:
            return lambda x: x == threshold

    @staticmethod
    def splitData(data, interval, featidx, splitfunc):
        ''' split data given featidx and split function on featidx '''

        data[interval[0]:interval[1]] = sorted(data[interval[0]:interval[1]], key=lambda x:splitfunc(x[featidx]))
        catego = splitfunc(data[interval[0]][featidx])
        l, r = interval[0]+1, interval[1]
        while l < r:
            mid = int((l+r)/2)
            if splitfunc(data[mid][featidx]) != catego:
                r = mid
            else:
                l = mid+1
        return [(interval[0],l),(l,interval[1])]

    @staticmethod
    def distribution(data, interval):
        ''' calculate the distribution of the labes '''
        cnts = {}
        if interval[0] == interval[1]:
            return cnts
        labelidx = len(data[interval[0]])-1
        for i in range(interval[0],interval[1]):
            cnts.setdefault(data[i][labelidx], 0)
            cnts[data[i][labelidx]] += 1
        return cnts

    @staticmethod
    def giniimpurity(distri):
        ''' calculate the giniimpurity given the distribution '''
        imp = 0.0
        N = sum(distri.values())
        for i in distri:
            p1 = float(distri[i])/N
            for j in distri:
                if j == i:
                    continue
                p2 = float(distri[j])/N
                imp += p1*p2
        return imp

    @staticmethod
    def entropy(distri):
        ''' calculate the entropy given the distribution '''
        ent = 0.0
        log2 = lambda x:math.log(x)/math.log(2)
        N = sum(distri.values())
        for i in distri:
            p = float(distri[i])/N
            if p != 0:
                ent -= p*log2(p)
        return ent

    @staticmethod
    def MSE(distri):
        mean = 0.0
        N = 0
        for key,value in distri.items():
            N += value
            mean += key*value
        if N == 0:
            return 0.0
        mean /= N
        mse = 0.0
        for key,value in distri.items():
            mse += value * ((key-mean)**2)
        mse /= N
        return mse

    @staticmethod
    def build(data, interval, scorefunc, maxtreedepth = -1):
        ''' build a CART tree '''

        print "interval",interval
        if interval[0] == interval[1]:
            return None

        distri = CART.distribution(data, interval)

        if maxtreedepth == 1:
            return CART.Node(result=distri, depth=1, leafnum=1)

        # calculate the score on the data
        score = scorefunc(distri)
        maxgain = 0.0
        bestsplitcriteria = None

        N = interval[1] - interval[0] # samples number
        F = len(data[interval[0]])-1 # feature dimension

        # at each featrue idx, split the data and calculate the 
        # score after split, then choose the best feature idx
        # to use to split the data and recursive build the 
        # children
        for f in range(F):
            allvalue = {}
            for row in data:
                allvalue[row[f]] = 1
            for val in allvalue.keys():
                splitfunc = CART.defaultSplitFunc(data, interval, f, val)
                res = CART.splitData(data, interval, f, splitfunc)
                print "f,val,splitinterval,splitdata",f,val,res,data
                p = float(res[0][1]-res[0][0])/N
                distri1 = CART.distribution(data, res[0])
                distri2 = CART.distribution(data, res[1])
                splitscore = p*scorefunc(distri1) + (1.0-p)*scorefunc(distri2)
                gain = score - splitscore
                if gain > maxgain:
                    maxgain = gain
                    bestsplitcriteria = (f, splitfunc, val)

        if maxgain > 0:
            bestsplitintvals = CART.splitData(data, interval,
                    bestsplitcriteria[0], bestsplitcriteria[1])
            print "bestf,bestval,bestsplitinterval,bestsplitdata",bestsplitcriteria[0],bestsplitcriteria[2],bestsplitintvals,data
            chlds = []
            maxdepth = 0
            totalleaf = 0
            for each in bestsplitintvals:
                chlds.append(CART.build(data, each, scorefunc, maxtreedepth-1))
                maxdepth = max(maxdepth, chlds[len(chlds)-1].depth)
                totalleaf += chlds[len(chlds)-1].leafnum
            return CART.Node(featidx=bestsplitcriteria[0], splitfunc =
                    bestsplitcriteria[1], depth = maxdepth+1, leafnum =
                    totalleaf, childs = chlds)
        else:
            # no gain, then it is a leafnode
            return CART.Node(result=distri, depth=1, leafnum=1)

    @staticmethod
    def prune(rootnode, scorefunc, alpha = 0.1):
        maxdepth = 0
        totalleaf = 0
        for child in rootnode.childs:
            if child.result == None:
                CART.prune(child, scorefunc, alpha)
            maxdepth = max(maxdepth, child.depth)
            totalleaf += child.leafnum
        rootnode.depth = maxdepth+1
        rootnode.leafnum = totalleaf
        
        for child in rootnode.childs:
            if child.result == None:
                break
        else:
            distri = {}
            totalnum = 0
            childsnum = []
            childscore = []
            for child in rootnode.childs:
                chddistri = child.result
                childscore.append(scorefunc(chddistri))
                num = 0
                for key,value in chddistri.items():
                    distri.setdefault(key, 0)
                    distri[key] += value
                    num += value
                childsnum.append(num)
                totalnum += num
            score = scorefunc(distri)
            oldscore = 0.0
            for i in range(len(childsnum)):
                oldscore += float(childsnum[i])/totalnum * childscore[i]
            if score - oldscore < alpha:
                rootnode.result = distri
                rootnode.leafnum = 1
                rootnode.depth = 1
    
    @staticmethod
    def search(rootnode, feat):
        ''' search one the tree '''
        if rootnode.result != None:
            return rootnode.result

        # if has not the feature, search all child node, and added the 
        # results by weights
        if feat[rootnode.featidx] == None:
            distri = {}
            res = []
            resweights = []
            totalnum = 0
            for child in rootnode.childs:
                res.append(CART.search(child, feat))
                num = 0;
                for key,value in res[len(res)-1].items():
                    num += value
                resweights.append(num)
                totalnum += num
            for i in range(len(res)):
                weight = float(resweights[i])/totalnum
                for key,value in res[i].items():
                    distri.setdefault(key, 0.0)
                    distri[key] += value*weight
            return distri
        else:
            childidx = rootnode.splitfunc(feat[rootnode.featidx])
            return CART.search(rootnode.childs[childidx], feat)

    def __init__(self):
        self.root = None
        self.classifier = None

    def learn(self, data, classifier = True, maxtreedepth = -1, alpha = 0.1):
        ''' learn a CART classifier or regressor 
        data is the train set whose last colum is the 
        labels or regression values '''
        self.classifier = classifier
        scorefunc = CART.giniimpurity
        if not classifier:
            scorefunc = CART.MSE
        self.root = CART.build(data, (0,len(data)), scorefunc, maxtreedepth)
        CART.prune(self.root, scorefunc, alpha)

    def predict(self, feat):
        distri = CART.search(self.root, feat)
        totalnum = 0
        maxval = 0
        label = None
        mean = 0.0
        print distri
        for key,value in distri.items():
            if value > maxval:
                maxval = value
                label = key
            mean += key*value
            totalnum += value
        mean /= totalnum
        if self.classifier:
            return (label, float(maxval)/totalnum)
        else:
            return mean

if __name__ == '__main__':
    c = CART()
    data = [[1,"a", 0],[2,"b", 0], [3,"a",1],[3,"b",0],[4,"a",1]]
    c.learn(data)
    print c.predict([0,None])
    print c.predict([5,"a"])
    print c.predict([4,"b"])
    
    data = [[1,"a",0.0],[2,"b",1.0],[0,"b",1.0],[3,"a",3.0],[3,"b",3.0],[4,"a",4.0]]
    c.learn(data, False)
    print c.predict([-1,None])
    print c.predict([5,"a"])
    print c.predict([4,"b"])
