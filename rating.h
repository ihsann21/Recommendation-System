#ifndef RATING_STRUCT_H
#define RATING_STRUCT_H

#include <vector>
#include <utility>
#include <string>
#include <fstream>
#include <iostream>

struct Rating {
    int userId;
    int itemId;
    float rating;
};

struct UserRatings {
    std::vector<std::pair<int, float>> ratings; 
};

bool readFileIntoBuffer(const std::string &fname, std::string &buffer);

#endif 
