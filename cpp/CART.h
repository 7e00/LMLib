/*************************************************************************
    # Author: narutoacm - www.narutoacm.com
    # Email: narutoacm@gmail.com
    # File Name: CART.h
    # Last modified: 2014-07-05 21:46
 ************************************************************************/

#ifndef __CART_H__
#define __CART_H__

#include <map>
#include <opencv2/opencv.hpp>

struct Node
{
    int featidx;
    double splitval;
    int depth;
    int leafnum;
    std::map<double, double> result;
    Node *childs[2];
    Node(int fidx, double sv, int dph, int ln, Node *clds[2])
        :featidx(fidx), splitval(sv), depth(dph), leafnum(ln)
    {
        childs[0] = clds[0];
        childs[1] = clds[1];
    }
    Node(std::map<double,double> &res)
        :featidx(-1),splitval(0),result(res),depth(1),leafnum(1)
    {
        childs[0] = childs[1] = 0;
    }
};

int splitData(cv::Mat &data, int s, int e, int featidx, double splitval);
void distribution(const cv::Mat &data, int s, int e, std::map<double, double> &distri);
double giniimpurity(const std::map<double,double> &distri);
double MSE(const std::map<double,double> &distri);
typedef double (*scorefunc)(const std::map<double,double> &);
Node *build(cv::Mat &data, int s, int e, scorefunc sf, int maxtreedepth = -1);
void prune(Node *rootnode, scorefunc sf, double alpha);
void search(Node *rootnode, const cv::Mat &feat, std::map<double,double> &result);
void freetree(Node *rootnode);

class CART
{
public:
    Node *rootnode;
    bool classifier;
    CART()
        :rootnode(0),classifier(false)
    {}
    ~CART()
    {
        if (rootnode)
            freetree(rootnode);
    }

    void learn(cv::Mat &data, bool classifier = true, int maxtreedepth = -1, double alpha = 0.1);
    double predict(cv::Mat &feat, double *prob = 0);
};

#endif //CART_H__

