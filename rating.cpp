// RatingStruct.cpp
#include "rating.h"

bool readFileIntoBuffer(const std::string &fname, std::string &buffer) {
    std::ifstream in(fname, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        std::cerr << "Cannot open " << fname << "\n";
        return false;
    }
    std::streamsize sz = in.tellg();
    in.seekg(0, std::ios::beg);
    if (sz <= 0) {
        std::cerr << "File empty?\n";
        return false;
    }
    buffer.resize((size_t)sz);
    if (!in.read(&buffer[0], sz)) {
        std::cerr << "Error reading " << fname << "\n";
        return false;
    }
    return true;
}
