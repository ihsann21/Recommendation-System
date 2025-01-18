// KDTree.h
#ifndef KD_TREE_H
#define KD_TREE_H

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

struct KDNode {
    std::vector<float> point;
    int index;
    KDNode *left = nullptr, *right = nullptr;
};

class KDTree {
private:
    int K;
    KDNode *root = nullptr;
    std::vector<std::unique_ptr<KDNode>> nodes;

    KDNode* build(std::vector<int> &items, int s, int e, int depth,
                  const std::vector<float> &mat);
    float dist2(const std::vector<float> &a, const std::vector<float> &b) const;
    void search(KDNode* node, const std::vector<float> &q, int depth, int n,
                std::vector<std::pair<float, int>> &best) const;

public:
    KDTree(int dim);
    void buildTree(const std::vector<float> &mat, int count);
    void searchTopN(const std::vector<float> &q, int n,
                    std::vector<std::pair<float, int>> &best) const;
};

#endif // KD_TREE_H