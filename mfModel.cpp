// MFModel.cpp
#include "mfModel.h"

MFModel::MFModel(int u, int i, int k, int it, float learn, float r, float gm)
    : U(u), I(i), K(k), iters(it), lr(learn), reg(r), globalMean(gm), rng(1234) {
    userFactors.resize(U * K);
    itemFactors.resize(I * K);
    userBias.resize(U, 0.f);
    itemBias.resize(I, 0.f);

    std::normal_distribution<float> dist(0.f, 0.01f);
    for (auto &x : userFactors) x = dist(rng);
    for (auto &x : itemFactors) x = dist(rng);
}

float MFModel::predict(int u, int m) const {
    if (u < 0 || u >= U || m < 0 || m >= I) return globalMean;
    float p = globalMean + userBias[u] + itemBias[m];
    const float *uf = &userFactors[u * K];
    const float *mf = &itemFactors[m * K];
    float dot = 0.f;
    for (int f = 0; f < K; f++) {
        dot += uf[f] * mf[f];
    }
    p += dot;
    if (p > 5.f) p = 5.f; else if (p < 1.f) p = 1.f;
    return p;
}

float MFModel::quickRMSE(const std::vector<Rating> &d) {
    if (d.empty()) return -1.f;
    double s = 0.0;
    for (const auto &r : d) {
        float pr = predict(r.userId, r.itemId);
        double df = static_cast<double>(r.rating) - static_cast<double>(pr);
        s += df * df;
    }
    return static_cast<float>(std::sqrt(s / d.size()));
}

void MFModel::train(std::vector<Rating> &dat, int numThreads, const std::vector<Rating> &val) {
    size_t sz = dat.size();
    std::mt19937 g(12345);
    float curLR = lr;
    float bestVal = std::numeric_limits<float>::infinity();

    for (int ep = 0; ep < iters; ep++) {
        std::shuffle(dat.begin(), dat.end(), g);

        std::atomic<size_t> idx(0);
        std::vector<std::thread> ths;
        ths.reserve(numThreads);

        auto worker = [&]() {
            size_t i;
            while ((i = idx.fetch_add(1)) < sz) {
                int u = dat[i].userId;
                int m = dat[i].itemId;
                float r = dat[i].rating;

                float pred = globalMean + userBias[u] + itemBias[m];
                const float *uf = &userFactors[u * K];
                const float *mf = &itemFactors[m * K];
                float sumF = 0.f;
                for (int f = 0; f < K; f++) {
                    sumF += uf[f] * mf[f];
                }
                pred += sumF;
                float err = r - pred;

                userBias[u] += curLR * (err - reg * userBias[u]);
                itemBias[m] += curLR * (err - reg * itemBias[m]);
                for (int ff = 0; ff < K; ff++) {
                    float oldU = userFactors[u * K + ff];
                    float oldI = itemFactors[m * K + ff];
                    userFactors[u * K + ff] += curLR * (err * oldI - reg * oldU);
                    itemFactors[m * K + ff] += curLR * (err * oldU - reg * oldI);
                }
            }
        };

        for (int t = 0; t < numThreads; t++) {
            ths.emplace_back(worker);
        }
        for (auto &th : ths) th.join();
        ths.clear();

        float vRMSE = quickRMSE(val);
        if (vRMSE > 0.f && vRMSE < bestVal) {
            bestVal = vRMSE;
        } else {
            curLR *= 1.02f; if (curLR > 0.02f) curLR = 0.02f;
            reg *= 0.98f;   if (reg < 0.001f) reg = 0.001f;
        }
        curLR *= 0.9f;
    }
}
