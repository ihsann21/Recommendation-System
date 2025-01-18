#ifndef DATA_AGENT_H
#define DATA_AGENT_H

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#include "rating.h" // Rating sınıfı veya struct'ını burada dahil etmelisiniz.

class DataAgent {
public:
    std::vector<Rating> trainingData, testData, valData;

    int numUsers = 0, numItems = 0;
    float globalMean = 3.0f;

    std::vector<int> userMap, itemMap;

    void loadTrainingData(const std::string &fname, size_t maxLines = 0);
    void loadTestData(const std::string &fname, size_t maxLines = 0);

private:
    static void parseBuffer(const std::string &buf, std::vector<Rating> &out, size_t maxLines = 0);
    bool readFileIntoBuffer(const std::string &fname, std::string &buffer);
};

#endif // DATA_AGENT_H
