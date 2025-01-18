#include "evaluate.h"
#include <cmath> // sqrt için

float RMSE(const std::vector<float> &actual, const std::vector<float> &pred) {
    if (actual.size() != pred.size() || actual.empty()) {
        return -1.f;
    }

    double sum = 0.0;
    for (size_t i = 0; i < actual.size(); i++) {
        double d = (double)actual[i] - (double)pred[i];
        sum += d*d;
    }

    return (float)std::sqrt(sum / actual.size());
}
