// FMModel.cpp
#include "fmModel.h"

FMModel::FMModel(int u, int i, int k, int it, float learn, float r, float gm)
    : U(u), I(i), K(k), iters(it), lr(learn), reg(r), globalMean(gm), rng(1234) {
    wUser.resize(U, 0.f);
    wItem.resize(I, 0.f);
    vUser.resize(U * K);
    vItem.resize(I * K);

    std::normal_distribution<float> dist(0.f, 0.01f);
    for (auto &xx : vUser) xx = dist(rng);
    for (auto &xx : vItem) xx = dist(rng);
}

float FMModel::predict(int u, int m) const {
    if (u < 0 || u >= U || m < 0 || m >= I) return globalMean;
    float p = globalMean + wUser[u] + wItem[m];
    const float *vu = &vUser[u * K];
    const float *vi = &vItem[m * K];
    float sumF = 0.f;
    for (int f = 0; f < K; f++) {
        sumF += vu[f] * vi[f];
    }
    p += sumF;
    if (p > 5.f) p = 5.f; else if (p < 1.f) p = 1.f;
    return p;
}

float FMModel::quickRMSE(const std::vector<Rating> &d) {
    if (d.empty()) return -1.f;
    double s = 0.0;
    for (const auto &r : d) {
        float pr = predict(r.userId, r.itemId);
        double df = static_cast<double>(r.rating) - static_cast<double>(pr);
        s += df * df;
    }
    return static_cast<float>(std::sqrt(s / d.size()));
}

void FMModel::train(std::vector<Rating> &dat, int numThreads, const std::vector<Rating> &val) {
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

                float p = predict(u, m);
                float err = r - p;

                float wuOld = wUser[u];
                wUser[u] += curLR * (err - reg * wuOld);
                float wiOld = wItem[m];
                wItem[m] += curLR * (err - reg * wiOld);

                float *vu = &vUser[u * K];
                float *vi = &vItem[m * K];
                for (int f = 0; f < K; f++) {
                    float vuf = vu[f];
                    float vif = vi[f];
                    vu[f] += curLR * (err * vif - reg * vuf);
                    vi[f] += curLR * (err * vuf - reg * vif);
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
            curLR *= 1.02f;
            if (curLR > 0.02f) curLR = 0.02f;
            reg *= 0.98f;
            if (reg < 0.001f) reg = 0.001f;
        }
        curLR *= 0.9f;
    }
}
