#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <random>
#include <limits>
#include <atomic>
#include <mutex>
#include <numeric>
#include <functional>
#include <memory>
#include "dataAgent.h"
#include "fmModel.h"
#include "mfModel.h"
#include "hybridAgent.h"
#include "evaluate.h"
#include "rating.h"
#include "kdtTree.h"
#


int main(){
    std::ios::sync_with_stdio(false);

    auto start= std::chrono::high_resolution_clock::now();

    DataAgent data;
    data.loadTrainingData("training_data_son.csv");
    data.loadTestData("test_data_son.csv");

    unsigned int numThreads= std::max(1u, std::thread::hardware_concurrency());
    std::cout<<"Users= "<< data.numUsers 
             <<" Items= "<< data.numItems 
             <<" globalMean= "<< data.globalMean <<"\n";

    // itemAvg
    std::vector<float> itemAvg(data.numItems,0.f);
    {
        std::vector<double> sum(data.numItems,0.0);
        std::vector<int> cnt(data.numItems,0);
        for(auto &r: data.trainingData){
            sum[r.itemId]+= r.rating;
            cnt[r.itemId]++;
        }
        for(int i=0; i<data.numItems; i++){
            if(cnt[i]>0){
                itemAvg[i]= (float)(sum[i]/ cnt[i]);
            } else {
                itemAvg[i]= data.globalMean;
            }
        }
    }

    // userDataVec
    std::vector<UserRatings> userDataVec(data.numUsers);
    {
        std::mutex mx;
        size_t sz= data.trainingData.size();
        size_t blk= (sz+ numThreads-1)/ numThreads;
        std::vector<std::thread> ths; ths.reserve(numThreads);

        auto worker=[&](size_t s, size_t e){
            for(size_t x=s; x< e; x++){
                const auto &rr= data.trainingData[x];
                std::lock_guard<std::mutex> lock(mx);
                userDataVec[rr.userId].ratings.push_back({rr.itemId, rr.rating});
            }
        };
        for(unsigned t=0; t<numThreads; t++){
            size_t st= t*blk;
            size_t ed= std::min(st+blk, sz);
            if(st<ed) ths.emplace_back(worker, st, ed);
        }
        for(auto &th: ths) th.join();

        // sort
        ths.clear();
        blk= (data.numUsers+ numThreads-1)/ numThreads;
        auto sworker=[&](size_t s, size_t e){
            for(size_t u=s; u< e; u++){
                std::sort(userDataVec[u].ratings.begin(), userDataVec[u].ratings.end(),
                    [](auto &a, auto &b){return a.first< b.first;});
            }
        };
        for(unsigned t=0; t<numThreads; t++){
            size_t st= t*blk;
            size_t ed= std::min(st+blk, (size_t)data.numUsers);
            if(st<ed) ths.emplace_back(sworker, st, ed);
        }
        for(auto &th: ths) th.join();
    }

    // Train MF
    MFModel mf(
       data.numUsers, data.numItems,
       /*K=*/1, 
       /*iters=*/1, 
       /*lr=*/0.04f,
       /*reg=*/0.01f, 
       /*gm=*/ data.globalMean
    );
    mf.train(data.trainingData, numThreads, data.valData);

    // Train FM
    FMModel fm(
       data.numUsers, data.numItems,
       /*K=*/1,
       /*iters=*/1,
       /*learn=*/0.04f,
       /*r=*/0.01f,
       /*gm=*/ data.globalMean
    );
    fm.train(data.trainingData, numThreads, data.valData);

    // HybridAgent (MF + FM + IBCF + pop)
    // Param: topK=5, simThreshold=0.05, alpha=0.4, beta=0.4, gamma=0.15, delta=0.05
    HybridAgent hybrid(
       mf,
       fm,
       userDataVec,
       itemAvg,
       data.globalMean,
       mf.itemFactors, // from MF
       /*factorDim=*/14,
       data.numItems,
       /*topK=*/1,
       /*sThresh=*/0.05f,
       /*A=*/0.4f, // MF
       /*B=*/0.4f, // FM
       /*G=*/0.15f,// IBCF
       /*D=*/0.05f // pop
    );

    // Test
    std::vector<float> actual, preds;
    actual.reserve(data.testData.size());
    preds.reserve(data.testData.size());

    {
        std::mutex mx;
        size_t sz= data.testData.size();
        size_t blk= (sz+ numThreads-1)/ numThreads;
        std::vector<std::thread> ths; ths.reserve(numThreads);

        auto wkr=[&](size_t s, size_t e){
            std::vector<float> la, lp;
            la.reserve(e-s); lp.reserve(e-s);
            for(size_t x=s; x< e; x++){
                const auto &r= data.testData[x];
                float p= hybrid.predict(r.userId, r.itemId);
                la.push_back(r.rating);
                lp.push_back(p);
            }
            std::lock_guard<std::mutex> lock(mx);
            actual.insert(actual.end(), la.begin(), la.end());
            preds.insert(preds.end(), lp.begin(), lp.end());
        };

        for(unsigned t=0; t<numThreads; t++){
            size_t st= t*blk;
            size_t ed= std::min(st+blk, sz);
            if(st<ed) ths.emplace_back(wkr, st, ed);
        }
        for(auto &th: ths) th.join();
    }

    float finalRMSE = RMSE(actual, preds);

    auto end= std::chrono::high_resolution_clock::now();
    double sec= std::chrono::duration<double>(end - start).count();

   

    // print predictions
    for(size_t i=0; i< preds.size(); i++){
        std::cout<<"TestRow="<< i 
                 <<" Actual="<< actual[i]
                 <<" Pred="<< preds[i] <<"\n";
    }
 std::cout<<"Final RMSE= "<< finalRMSE <<"\n";
    std::cout<<"Time= "<< sec <<" s\n";
    return 0;
}