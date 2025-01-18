// KDTree.cpp
#include "kdtTree.h"

KDTree::KDTree(int dim) : K(dim) {}

KDNode* KDTree::build(std::vector<int> &items, int s, int e, int depth,
                       const std::vector<float> &mat) {
    if (s >= e) return nullptr;
    int axis = depth % K;
    int mid = (s + e) / 2;
    std::nth_element(items.begin() + s, items.begin() + mid, items.begin() + e,
                     [&](int a, int b) {
                         float va = mat[a * K + axis];
                         float vb = mat[b * K + axis];
                         return va < vb;
                     });
    int idx = items[mid];
    auto* nd = new KDNode();
    nodes.push_back(std::unique_ptr<KDNode>(nd));
    nd->index = idx;
    nd->point.resize(K);
    for (int d = 0; d < K; d++) {
        nd->point[d] = mat[idx * K + d];
    }
    nd->left = build(items, s, mid, depth + 1, mat);
    nd->right = build(items, mid + 1, e, depth + 1, mat);
    return nd;
}

float KDTree::dist2(const std::vector<float> &a, const std::vector<float> &b) const {
    float s = 0.f;
    for (int i = 0; i < K; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

void KDTree::search(KDNode* node, const std::vector<float> &q, int depth, int n,
                    std::vector<std::pair<float, int>> &best) const {
    if (!node) return;
    float d2 = dist2(q, node->point);
    if ((int)best.size() < n) {
        best.push_back({d2, node->index});
        std::push_heap(best.begin(), best.end(),
                       [](auto &a, auto &b) { return a.first < b.first; });
    } else {
        if (d2 < best.front().first) {
            std::pop_heap(best.begin(), best.end(),
                          [](auto &a, auto &b) { return a.first < b.first; });
            best.pop_back();
            best.push_back({d2, node->index});
            std::push_heap(best.begin(), best.end(),
                           [](auto &a, auto &b) { return a.first < b.first; });
        }
    }
    int axis = depth % K;
    float diff = q[axis] - node->point[axis];
    KDNode* first = diff < 0 ? node->left : node->right;
    KDNode* second = diff < 0 ? node->right : node->left;
    search(first, q, depth + 1, n, best);
    if (diff * diff < best.front().first || (int)best.size() < n) {
        search(second, q, depth + 1, n, best);
    }
}

void KDTree::buildTree(const std::vector<float> &mat, int count) {
    std::vector<int> idx(count);
    for (int i = 0; i < count; i++) idx[i] = i;
    root = build(idx, 0, count, 0, mat);
}

void KDTree::searchTopN(const std::vector<float> &q, int n,
                        std::vector<std::pair<float, int>> &best) const {
    best.clear();
    best.reserve(n);
    search(root, q, 0, n, best);
    std::sort(best.begin(), best.end(),
              [](auto &a, auto &b) { return a.first < b.first; });
}
