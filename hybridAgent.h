// HybridAgent.h
#ifndef HYBRID_AGENT_H
#define HYBRID_AGENT_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "mfModel.h" // Assuming MFModel is defined elsewhere.
#include "fmModel.h" // Assuming FMModel is defined elsewhere.
#include "rating.h" // Assuming UserRatings is defined elsewhere.
#include "kdtTree.h" // Assuming KDTree is defined elsewhere.

class HybridAgent {
private:
    const MFModel &mf;
    const FMModel &fm;
    std::vector<UserRatings> userData;
    std::vector<float> itemAvg;
    float globalMean;

public:
    float alpha, beta, gamma, delta;
    KDTree kdt;
    const std::vector<float> &mfItemFactors; // from MF
    int K, I;
    int topK;
    float simThreshold;

    HybridAgent(const MFModel &m, const FMModel &f,
                const std::vector<UserRatings> &uData,
                const std::vector<float> &itAvg,
                float gm,
                const std::vector<float> &mfFactors,
                int factorDim,
                int itemCount,
                int tK = 1,
                float sThresh = 0.05f,
                float A = 0.4f,  // MF pay
                float B = 0.4f,  // FM pay
                float G = 0.15f, // IBCF pay
                float D = 0.05f  // pop pay
               );

    float WeightedCosineSim(int i, int j) const;
    float predict(int u, int i) const;
};

#endif // HYBRID_AGENT_H