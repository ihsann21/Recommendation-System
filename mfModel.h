// MFModel.h
#ifndef MF_MODEL_H
#define MF_MODEL_H

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <atomic>
#include <thread>
#include <algorithm>
#include "rating.h" // Assuming Rating struct/class is defined elsewhere.

class MFModel {
public:
    int U, I, K, iters;
    float lr, reg;
    float globalMean;

    std::vector<float> userFactors, itemFactors;
    std::vector<float> userBias, itemBias;
    std::mt19937 rng;

    MFModel(int u, int i, int k = 1, int it = 1, float learn = 0.04f, float r = 0.01f, float gm = 3.0f);

    float predict(int u, int m) const;
    float quickRMSE(const std::vector<Rating> &d);
    void train(std::vector<Rating> &dat, int numThreads, const std::vector<Rating> &val);
};

#endif // MF_MODEL_H