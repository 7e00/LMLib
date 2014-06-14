#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# File Name: NaiveBayesClassifier.py
# Author: narutoacm
# Mail: narutoacm@gmail.com
# Website: www.narutoacm.com
# Created Time: 2014年06月14日 星期六 14时26分36秒
#########################################################################

class NaiveBayesClassifier:
    def __init__(self):
        self.probs = {}

    def learn(self, feats, labels, reg = 1.0):
        ''' learn a naive bayes classifier from data '''

        # probs[label][0] = P(Y=label)
        # probs[label][1][j][val] = P(X(j)=val|Y=label)
        # here X(j) stand for j dimension of X=(X(0),X(1),...X(F-1))
        self.probs = {}

        N,F = len(feats),len(feats[0])

        # used for find all values in each dimension of X
        fdlist = [{} for i in range(F)]

        for i in range(N):
            self.probs.setdefault(labels[i], [0.0,[{} for j in range(F)]])
            self.probs[labels[i]][0] += 1
            conprobs = self.probs[labels[i]][1]
            xi = feats[i]
            for j in range(F):
                conprobs[j].setdefault(xi[j], 0.0)
                conprobs[j][xi[j]] += 1
                fdlist[j][xi[j]] = 1

        # S[i] is the number of different values that X(i) can have
        S = [len(fdlist[i].keys()) for i in range(F)]
        # K is the number of all labels
        K = len(self.probs.keys())
        for label in self.probs.keys():
            cnt = self.probs[label][0]
            conprobs = self.probs[label][1]
            for j in range(F):
                for key in fdlist[j].keys():
                    val = conprobs[j].setdefault(key, 0.0)
                    conprobs[j][key] = (val+reg)/(cnt+S[j]*reg)
            self.probs[label][0] = (cnt+reg)/(N+K*reg)

        return self

    def predict(self, feats):
        N,F = len(feats),len(feats[0])
        res = []
        for x in feats:
            maxplabel = 0.0
            bestlabel = None
            for label, prob in self.probs.items():
                # plabel is P(Y=label)
                plabel = prob[0]

                # naive bayes, assume F features are indepent
                for i in range(F):
                    if x[i] not in prob[1][i]:
                        plabel = 0.0
                        break
                    plabel *= prob[1][i][x[i]]

                if plabel > maxplabel:
                    maxplabel = plabel
                    bestlabel = label
            res.append((bestlabel, maxplabel))
        return res

if __name__ == '__main__':
    feats = \
    [[1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],[2,'M'],[2,'M'],[2,'L'],[2,'L'],[3,'L'],[3,'M'],[3,'M'],[3,'L'],[3,'L']]
    labels = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]
    c = NaiveBayesClassifier()
    print c.learn(feats, labels, 0).predict([[2,'S']])
