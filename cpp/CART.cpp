/*************************************************************************
    # Author: narutoacm - www.narutoacm.com
    # Email: narutoacm@gmail.com
    # File Name: CART.cpp
    # Last modified: 2014-07-05 21:59
 ************************************************************************/

#include "CART.h"

using namespace std;
using namespace cv;

int splitData(cv::Mat &data, int s, int e, int featidx, double splitval)
{
    double *ptr = (double *)(data.data) + s*data.cols + featidx;
    int l = s;
    for (int i = s; i < e; ++i)
    {
        if (*ptr < splitval)
        {
            Mat rowl = data.row(l).clone();
            data.row(i).copyTo(data.row(l));
            rowl.copyTo(data.row(i));
            ++l;
        }
        ptr += data.cols;
    }
    return l;
}

void distribution(const cv::Mat &data, int s, int e, std::map<double, double> &distri)
{
    distri.clear();
    if (s == e)
        return;
    int labelidx = data.cols-1;
    double *ptr = (double *)(data.data) + s*data.cols + labelidx;
    for (int i = s; i < e; ++i)
    {
        if (distri.count(*ptr))
            distri[*ptr] += 1;
        else
            distri[*ptr] = 1;
        ptr += data.cols;
    }
}

double giniimpurity(const std::map<double,double> &distri)
{
    double imp = 0;
    map<double,double>::const_iterator iter;
    double N = 0;
    for (iter = distri.begin(); iter != distri.end(); ++iter)
        N += (*iter).second;
    for (iter = distri.begin(); iter != distri.end(); ++iter)
    {
        double p1 = (*iter).second/N;
        for (map<double,double>::const_iterator iter2 = distri.begin();
                iter2 != distri.end(); ++iter2)
        {
            if (iter == iter2)
                continue;
            double p2 = (*iter2).second/N;
            imp += p1*p2;
        }
    }
    return imp;
}

double MSE(const std::map<double,double> &distri)
{
    double mean = 0;
    double N = 0;
    map<double,double>::const_iterator iter;
    for (iter = distri.begin(); iter != distri.end(); ++iter)
    {
        N += iter->second;
        mean += iter->first * iter->second;
    }
    if (N == 0)
        return 0;
    mean /= N;
    double mse = 0;
    for (iter = distri.begin(); iter != distri.end(); ++iter)
    {
        mse += iter->second * (iter->first - mean) * (iter->first - mean);
    }
    mse /= N;
    return mse;
}

Node *build(cv::Mat &data, int s, int e, scorefunc sf, int maxtreedepth)
{
    if (s == e)
        return 0;
    map<double,double> distri;
    distribution(data, s, e, distri);
    
    if (maxtreedepth == 1)
        return new Node(distri);

    double score = sf(distri);
    //cout<<"score "<<score<<endl;
    double maxgain = 0;
    int bestfidx = 0;
    double bestspv = 0;

    int N = (e-s), F = data.cols-1;
    for (int f = 0; f < F; ++f)
    {
        map<double,int> allvalue;
        double *ptr = (double *)(data.data) + s*data.cols + f;
        for (int i = s; i < e; ++i)
        {
            allvalue[*ptr]  = 1;
            ptr += data.cols;
        }
        map<double,int>::iterator iter;
        for (iter = allvalue.begin(); iter != allvalue.end(); ++iter)
        {
            int sp = splitData(data, s, e, f, iter->first);
            //cout<<"split "<<f<<' '<<iter->first<<' '<<sp<<' '<<data<<endl;
            double p = (double)(sp-s)/(double)N;
            map<double,double> distri1;
            map<double,double> distri2;
            distribution(data, s, sp, distri1);
            distribution(data, sp, e, distri2);
            double splitscore = p*sf(distri1) + (1.0-p)*sf(distri2);
            double gain = score - splitscore;
            if (gain > maxgain)
            {
                maxgain = gain;
                bestfidx = f;
                bestspv = iter->first;
            }
        }
    }

    if (maxgain > 0)
    {
        int sp = splitData(data, s, e, bestfidx, bestspv);
        //cout<<"bestsplit "<<bestfidx<<' '<<bestspv<<' '<<sp<<' '<<data<<endl;
        int maxdepth = 0;
        int totalleaf = 0;
        Node *chlds[2];
        chlds[0] = build(data, s, sp, sf, maxtreedepth);
        chlds[1] = build(data, sp, e, sf, maxtreedepth);
        maxdepth = std::max(chlds[0]->depth,chlds[1]->depth);
        totalleaf = chlds[0]->leafnum + chlds[1]->leafnum;
        return new Node(bestfidx,bestspv,maxdepth+1,totalleaf,chlds);
    }
    else
        return new Node(distri);
}

void prune(Node *rootnode, scorefunc sf, double alpha)
{
    if (!rootnode->result.empty())
        return;
    int maxdepth = 0;
    int totalleaf = 0;
    for (int i = 0; i < 2; ++i)
    {
        if (rootnode->childs[i])
        {
            prune(rootnode->childs[i], sf, alpha);
            maxdepth = std::max(maxdepth, rootnode->childs[i]->depth);
            totalleaf += rootnode->childs[i]->leafnum;
        }
    }
    rootnode->depth = maxdepth+1;
    rootnode->leafnum = totalleaf;
    
    if (rootnode->childs[0]->result.empty() && 
            rootnode->childs[1]->result.empty())
    {
        map<double,double> distri;
        double totalnum = 0;
        vector<double> childsnum;
        vector<double> childscore;
        for (int i = 0; i < 2; ++i)
        {
            childscore.push_back(sf(rootnode->childs[i]->result));
            double num = 0;
            map<double,double>::iterator iter;
            for (iter = rootnode->childs[i]->result.begin();
                    iter != rootnode->childs[i]->result.end();
                    ++iter)
            {
                if (distri.count(iter->first))
                    distri[iter->first] += iter->second;
                else
                    distri[iter->first] = iter->second;
                num += iter->second;
            }
            childsnum.push_back(num);
            totalnum += num;
        }
        double score = sf(distri);
        double oldscore = 0;
        for (int i = 0; i < 2; ++i)
        {
            double p = childsnum[i]/totalnum;
            oldscore += p * childscore[i];
        }
        if (score - oldscore < alpha)
        {
            rootnode->result = distri;
            rootnode->leafnum = 1;
            rootnode->depth = 1;
        }
    }
}

void search(Node *rootnode, const cv::Mat &feat, std::map<double,double> &result)
{
    if (!rootnode->result.empty())
    {
        result = rootnode->result;
        return;
    }

    Node *child = rootnode->childs[feat.at<double>(rootnode->featidx) >= rootnode->splitval];
    search(child, feat, result);
}

void freetree(Node *rootnode)
{
    if (rootnode->result.empty())
    {
        freetree(rootnode->childs[0]);
        freetree(rootnode->childs[1]);
    }
    delete rootnode;
}

void write(Node *rootnode, ofstream &f)
{
    f << "(" << endl;
    f << "result: ";
    if (!rootnode->result.empty())
    {
        f << "{" << endl;
        for (map<double,double>::iterator iter = rootnode->result.begin();
                iter != rootnode->result.end();
                ++iter)
        {
            f << iter->first << ' ' << iter->second << endl;
        }
        f << "}" << endl;
        f << ")" << endl;
    }
    else
    {
        f << "{}" << endl;
        f << "featidx: " << rootnode->featidx << endl;
        f << "splitval: " << rootnode->splitval << endl;
        f << "depth: " << rootnode->depth << endl;
        f << "leafnum: " << rootnode->leafnum << endl;
        f << ")" << endl;
        write(rootnode->childs[0], f);
        write(rootnode->childs[1], f);
    }
}

Node *read(ifstream &f)
{
    int featidx;
    double splitval;
    int depth;
    int leafnum;
    map<double,double> result;
    string str;
    f >> str;
    if (str != "(")
        return 0;
    f >> str;
    if (str != "result:")
        return 0;
    f >> str;
    if (str == "{")
    {
        getline(f, str);
        while (getline(f, str))
        {
            if (str == "}")
                break;
            stringstream ss(str);
            double key, value;
            ss >> key >> value;
            result[key] = value;
        }
        f >> str;
        return new Node(result);
    }
    else if (str == "{}")
    {
        f >> str >> featidx;
        f >> str >> splitval;
        f >> str >> depth;
        f >> str >> leafnum;
        f >> str;
        Node *chlds[2];
        chlds[0] = read(f);
        chlds[1] = read(f);
        return new Node(featidx, splitval, depth, leafnum, chlds);
    }
    else
        return 0;
}

void CART::learn(cv::Mat &data, bool classifier, int maxtreedepth, double alpha)
{
    if (rootnode)
        freetree(rootnode);

    this->classifier = classifier;
    scorefunc sf = giniimpurity;
    if (!classifier)
        sf = MSE;
    rootnode = build(data, 0, data.rows, sf, maxtreedepth);
    prune(rootnode, sf, alpha);
}
double CART::predict(cv::Mat &feat, double *prob)
{
    map<double,double> distri;
    search(rootnode, feat, distri);
    double totalnum = 0;
    double maxval = 0;
    double label = 0;
    double mean = 0;
    map<double,double>::iterator iter;
    for (iter = distri.begin(); iter != distri.end(); ++iter)
    {
        if (iter->second > maxval)
        {
            maxval = iter->second;
            label = iter->first;
        }
        mean += iter->first * iter->second;
        totalnum += iter->second;
    }
    mean /= totalnum;
    if (classifier)
    {
        if (prob)
            *prob = maxval/totalnum;
        return label;
    }
    else
        return mean;
}
