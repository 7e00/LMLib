#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# File Name: PerceptronClassifier.py
# Author: narutoacm
# Mail: narutoacm@gmail.com
# Website: www.narutoacm.com
# Created Time: 2014年05月29日 星期四 23时00分50秒
#########################################################################

import numpy as np
class PerceptronClassifier:
    def __init__(self):
        pass
    def learn_raw(self, feats, labels, rate = 0.4, max_iter = 1000):
        ''' raw style learn 
            feats: n*k
            labels: n*1, value 1 represents +case, -1 means -case'''

        # change feats and labels to numpy array
        feats = np.array(feats).reshape((len(feats),-1)) 
        labels = np.array(labels).flatten()

        # add a colum with all 1 to feats
        exfeats = np.hstack((np.array([[1] for i in range(len(feats))]),
            feats))

        # w0 is b
        self.w = np.array([0.0]*exfeats.shape[1])
        itercnt = 0
        while itercnt < max_iter:
            itercnt += 1
            flag = 0 # indicate if there are wrong classified cases
            for i in range(len(exfeats)):
                if labels[i] * np.dot(self.w, exfeats[i,:]) <= 0:
                    flag = 1
                    self.w += labels[i]*rate*exfeats[i,:]
            if flag == 0:
                break
        return self

    def learn_pair(self, feats, labels, rate = 0.4, max_iter = 1000):
        ''' another way to learn perceptron '''
        feats = np.array(feats).reshape((len(feats),-1))
        labels = np.array(labels).flatten()
        exfeats = np.hstack((np.array([[1] for i in range(len(feats))]),
            feats))
        gram = np.dot(exfeats, np.transpose(exfeats))
        alpha = np.array([0 for i in range(len(exfeats))])
        itercnt = 0
        while itercnt < max_iter:
            itercnt += 1
            flag = 0
            for i in range(len(exfeats)):
                val = 0.0
                for j in range(len(exfeats)):
                    val += alpha[j]*labels[j]*gram[j][i]
                if val * labels[i] <= 0:
                    flag = 1
                    alpha[i] += rate
            if flag == 0:
                break
        self.w = np.array([0.0]*exfeats.shape[1])
        for i in range(len(exfeats)):
            self.w += alpha[i]*labels[i]*exfeats[i,:]
        return self

    def learn(self, feats, labels, rate = 0.4, max_iter = 1000, method = "raw"):
        if method == "raw":
            return self.learn_raw(feats, labels, rate, max_iter)
        elif method == "pair":
            return self.learn_pair(feats, labels, rate, max_iter)
        else:
            return self

    def predict(self, feats):
        exfeats = np.hstack((np.array([[1] for i in range(len(feats))]),
            feats))
        labels = np.array([1]*len(feats))
        for i in range(len(feats)):
            if np.dot(self.w, exfeats[i,:]) <= 0:
                labels[i] = -1
        return labels

if __name__ == '__main__':
    c = PerceptronClassifier()
    c.learn([[1,1],[2,1],[3,3],[4,3]], [1,1,-1,-1])
    print c.w
    print c.predict([[1,1],[2,1],[3,3],[4,3]])
