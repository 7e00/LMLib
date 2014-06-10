#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# File Name: KNeighborClassifier.py
# Author: narutoacm
# Mail: narutoacm@gmail.com
# Website: www.narutoacm.com
# Created Time: 2014年05月31日 星期六 23时20分27秒
#########################################################################

import KDTree as kd

class KNeighborClassifier:
    ''' K Neighbor Classifier '''
    def __init__(self):
        self.kdtree = kd.KDTree()

    def learn(self, feats, labels):
        self.kdtree.build(feats, labels)
        return self

    def predict(self, feats, k = 5, max_nn_chks = 100, dist_func =
            kd.eculid_dis):
        ''' give the best guess based on k neighbor class '''
        res = []
        for i in range(len(feats)):
            d = {}
            neigs = self.kdtree.bbf_knn(feats[i], k, max_nn_chks, dist_func)
            for each in neigs:
                if each[2][0] not in d:
                    d[each[2][0]] = 1
                else:
                    d[each[2][0]] += 1
            tmp = []
            for key,value in d.items():
                tmp.append((value*1.0/len(neigs),key))
            tmp.sort()
            res.append(tmp[len(tmp)-1][1])
        return res

if __name__ == '__main__':
    c = KNeighborClassifier()
    c.learn([[1,1],[1,2],[3,4],[4,3]],['+','+','-','-'])
    print c.predict([[1,1],[1,2],[3,4],[4,3],[0,0],[5,5]], 1)
