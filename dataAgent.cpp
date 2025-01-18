#include "dataAgent.h"

void DataAgent::parseBuffer(const std::string &buf, std::vector<Rating> &out, size_t maxLines) {
    size_t pos = 0;
    {
        auto e = buf.find('\n', pos);
        if (e == std::string::npos) return;
        pos = e + 1;
    }
    size_t c = 0;
    while (pos < buf.size()) {
        if (maxLines > 0 && c >= maxLines) break;
        size_t e = buf.find('\n', pos);
        if (e == std::string::npos) e = buf.size();
        std::string line = buf.substr(pos, e - pos);
        pos = e + 1;
        if (line.empty()) continue;

        size_t p1 = line.find(',');
        if (p1 == std::string::npos) continue;
        size_t p2 = line.find(',', p1 + 1);
        if (p2 == std::string::npos) continue;

        int u = std::atoi(line.substr(0, p1).c_str());
        int i = std::atoi(line.substr(p1 + 1, p2 - (p1 + 1)).c_str());
        float r = std::atof(line.substr(p2 + 1).c_str());
        if (r < 1.f || r > 5.f) continue;
        out.push_back({u, i, r});
        c++;
    }
}

void DataAgent::loadTrainingData(const std::string &fname, size_t maxLines) {
    std::string buffer;
    if (!readFileIntoBuffer(fname, buffer)) {
        std::cerr << "Can't load " << fname << "\n";
        std::exit(1);
    }
    std::vector<Rating> raw;
    parseBuffer(buffer, raw, maxLines);
    buffer.clear();
    buffer.shrink_to_fit();

    if (raw.empty()) {
        std::cerr << "No training data!\n";
        std::exit(1);
    }
    int maxU = 0, maxI = 0;
    double sum = 0.0;
    for (auto &r : raw) {
        if (r.userId > maxU) maxU = r.userId;
        if (r.itemId > maxI) maxI = r.itemId;
    }
    userMap.assign(maxU + 1, -1);
    itemMap.assign(maxI + 1, -1);

    int cu = 0, ci = 0;
    std::vector<Rating> all;
    all.reserve(raw.size());
    for (auto &r : raw) {
        if (userMap[r.userId] < 0) userMap[r.userId] = cu++;
        if (itemMap[r.itemId] < 0) itemMap[r.itemId] = ci++;
    }
    numUsers = cu;
    numItems = ci;

    for (auto &r : raw) {
        int uu = userMap[r.userId];
        int ii = itemMap[r.itemId];
        all.push_back({uu, ii, r.rating});
        sum += r.rating;
    }
    raw.clear();
    raw.shrink_to_fit();

    globalMean = (float)(sum / all.size());

    size_t vsize = std::min((size_t)500, all.size());
    for (size_t i = 0; i < vsize; i++) {
        valData.push_back(all[i]);
    }
    for (size_t i = vsize; i < all.size(); i++) {
        trainingData.push_back(all[i]);
    }
}

void DataAgent::loadTestData(const std::string &fname, size_t maxLines) {
    std::string buffer;
    if (!readFileIntoBuffer(fname, buffer)) {
        std::cerr << "Can't load test " << fname << "\n";
        return;
    }
    std::vector<Rating> raw;
    parseBuffer(buffer, raw, maxLines);
    buffer.clear();
    buffer.shrink_to_fit();

    testData.reserve(raw.size());
    for (auto &r : raw) {
        if (r.userId < (int)userMap.size() && userMap[r.userId] >= 0 &&
            r.itemId < (int)itemMap.size() && itemMap[r.itemId] >= 0) {
            int uu = userMap[r.userId];
            int ii = itemMap[r.itemId];
            testData.push_back({uu, ii, r.rating});
        }
    }
    raw.clear();
    raw.shrink_to_fit();
}

bool DataAgent::readFileIntoBuffer(const std::string &fname, std::string &buffer) {
    std::ifstream file(fname);
    if (!file.is_open()) {
        return false;
    }
    buffer.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return true;
}
