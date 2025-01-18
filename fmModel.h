// FMModel.h
#ifndef FM_MODEL_H
#define FM_MODEL_H

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <atomic>
#include <thread>
#include <algorithm>
#include "rating.h" // Rating struct/class needs to be defined separately.

class FMModel {
public:
    int U, I, K, iters;
    float lr, reg;
    float globalMean;

    std::vector<float> wUser, wItem;
    std::vector<float> vUser, vItem;
    std::mt19937 rng;

    FMModel(int u, int i, int k = 1, int it = 1, float learn = 0.04f, float r = 0.01f, float gm = 3.0f);

    float predict(int u, int m) const;
    float quickRMSE(const std::vector<Rating> &d);
    void train(std::vector<Rating> &dat, int numThreads, const std::vector<Rating> &val);
};

#endif // FM_MODEL_H
