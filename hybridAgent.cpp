// HybridAgent.cpp
#include "hybridAgent.h"

HybridAgent::HybridAgent(const MFModel &m, const FMModel &f,
                         const std::vector<UserRatings> &uData,
                         const std::vector<float> &itAvg,
                         float gm,
                         const std::vector<float> &mfFactors,
                         int factorDim,
                         int itemCount,
                         int tK,
                         float sThresh,
                         float A, float B, float G, float D)
    : mf(m), fm(f), userData(uData), itemAvg(itAvg), globalMean(gm),
      alpha(A), beta(B), gamma(G), delta(D),
      kdt(factorDim),
      mfItemFactors(mfFactors),
      K(factorDim),
      I(itemCount),
      topK(tK),
      simThreshold(sThresh) {
    kdt.buildTree(mfItemFactors, I);
}

float HybridAgent::WeightedCosineSim(int i, int j) const {
    if (i == j) return 1.f;
    float wa = itemAvg[i];
    float wb = itemAvg[j];
    if (std::fabs(wa) < 1e-8 || std::fabs(wb) < 1e-8) return 0.f;
    const float *fi = &mfItemFactors[i * K];
    const float *fj = &mfItemFactors[j * K];
    double dot = 0.0, sumA = 0.0, sumB = 0.0;
    for (int f = 0; f < K; f++) {
        double x = fi[f] * wa;
        double y = fj[f] * wb;
        dot += x * y;
        sumA += x * x;
        sumB += y * y;
    }
    if (sumA < 1e-12 || sumB < 1e-12) return 0.f;
    double sim = dot / (std::sqrt(sumA) * std::sqrt(sumB));
    return static_cast<float>(sim);
}

float HybridAgent::predict(int u, int i) const {
    float mfVal = mf.predict(u, i);
    float fmVal = fm.predict(u, i);

    float ibVal = globalMean;
    float num = 0.f, den = 0.f;
    if (i >= 0 && i < I) {
        std::vector<float> q(K);
        for (int f = 0; f < K; f++) {
            q[f] = mfItemFactors[i * K + f];
        }
        std::vector<std::pair<float, int>> nb;
        kdt.searchTopN(q, 2 * topK, nb);
        const auto &uVec = userData[u].ratings;
        for (auto &cand : nb) {
            int nbI = cand.second;
            if (nbI == i) continue;
            float s = WeightedCosineSim(i, nbI);
            if (s > simThreshold) {
                auto it = std::lower_bound(uVec.begin(), uVec.end(), nbI,
                                           [](auto &xx, int val) { return xx.first < val; });
                if (it != uVec.end() && it->first == nbI) {
                    float r = it->second;
                    num += s * (r - itemAvg[nbI]);
                    den += std::fabs(s);
                }
            }
        }
    }
    if (den > 1e-10f) {
        ibVal = globalMean + num / den;
        if (ibVal > 5.f) ibVal = 5.f; else if (ibVal < 1.f) ibVal = 1.f;
    }

    float popVal = (i >= 0 && i < I) ? itemAvg[i] : globalMean;

    float finalPred = alpha * mfVal + beta * fmVal + gamma * ibVal + delta * popVal;
    if (finalPred > 5.f) finalPred = 5.f; else if (finalPred < 1.f) finalPred = 1.f;
    return finalPred;
}
