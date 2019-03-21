// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// test_cpp_parametric_functions.cpp

#include "macros.hpp"

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <climits>
#include <gtest/gtest.h>

#include <nbla/exception.hpp>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/initializer.hpp>

using std::shared_ptr;
using std::make_shared;
using std::vector;
using std::pair;
using namespace nbla;

class InitializerTest : public ::testing::Test {
protected:
    Context ctx_;
    CgVariablePtr var_;

    virtual void SetUp() override {
        ctx_.array_class = "CpuArray";

        var_ = make_shared<CgVariable>(Shape_t({5, 5}), false);

        resetVariable();
    }

    void resetVariable() {
        // init all values by 0
        auto *d = var_->variable()->cast_data_and_get_pointer<float>(ctx_, true);

        REP(i, var_->variable()->size()) {
            d[i] = 0;
        }
    }

    template <typename T>
    T* initVar(Initializer *initializer) {
        initializer->initialize(var_->variable()->data());

        return var_->variable()->cast_data_and_get_pointer<T>(ctx_, false);
    }

    void checkUniformInitializer(const shared_ptr<UniformInitializer> &init, float min, float max) {
        auto *d = initVar<float>(init.get());

        REP(i, var_->variable()->size()) {
            EXPECT_TRUE(d[i] <= max || d[i] >= min);
        }
    }

    void checkConstantInitializer(const shared_ptr<ConstantInitializer> &init, float value) {
        auto *d = initVar<float>(init.get());

        REP(i, var_->variable()->size()) {
            EXPECT_EQ(d[i], value);
        }
    }

    void checkNormalInitializer(const shared_ptr<NormalInitializer> &init, float mu, float sigma) {
        //just check call
        EXPECT_NO_THROW(initVar<float>(init.get()));
    }

    void checkUniformIntInitializer(const shared_ptr<UniformIntInitializer> &init, int min, int max) {
        auto *d = initVar<int>(init.get());

        REP(i, var_->variable()->size()) {
            EXPECT_TRUE(d[i] <= max || d[i] >= min);
        }
    }
};

TEST_F(InitializerTest, UniformInitializer) {
    //default constructor -> init by [-1, 1]
    auto initializer = make_shared<UniformInitializer>();

    checkUniformInitializer(initializer, -1, 1);

    // check all cases
    vector<pair<float, float>> check_cases{
            {2, 3}, {2, 2}, {2, 1}, {2, 0}, {2, -1}, // min > 0
            {0, 2}, {0, 0}, {0, -2}, // min = 0
            {-2, 1}, {-2, 0}, {-2, -1}, {-2, -2}, {-2, -3} // min < 0
    };

    for(auto p: check_cases) {
        float min = p.first;
        float max = p.second;

        if (min > max) {
            EXPECT_THROW(UniformInitializer(min, max), Exception);
        } else {
            initializer = make_shared<UniformInitializer>(min, max);
            checkUniformInitializer(initializer, min, max);
        }
    }
}

TEST_F(InitializerTest, ConstantInitializer) {
    //default constructor -> init by 0
    auto initializer = make_shared<ConstantInitializer>();

    checkConstantInitializer(initializer, 0.0);

    // check all cases
    vector<float> check_cases{-1, 0, 1};

    for(float v: check_cases) {
        initializer = make_shared<ConstantInitializer>(v);
        checkConstantInitializer(initializer, v);
    }
}

TEST_F(InitializerTest, NormalInitializer) {
    //default constructor -> init by 0
    auto initializer = make_shared<NormalInitializer>();

    checkNormalInitializer(initializer, 0.0, 1.0);

    // check all cases
    vector<pair<float, float>> check_cases{
        {1, 1}, {1, 0}, {1, -1}, // mu > 0
        {0, 1}, {0, 0}, {0, -1}, // mu = 0
        {-1, 1}, {-1, 0}, {-1, -1} // mu < 0
        };

    for(auto p: check_cases) {
        float mu = p.first;
        float sigma = p.second;

        if (sigma < 0) {
            EXPECT_THROW(NormalInitializer(mu, sigma), Exception);
        } else {
            initializer = make_shared<NormalInitializer>(mu, sigma);
            checkNormalInitializer(initializer, mu, sigma);
        }
    }
}

TEST_F(InitializerTest, UniformIntInitializer) {
    //default constructor -> init by [0, INT_MAX]
    auto initializer = make_shared<UniformIntInitializer>();

    checkUniformIntInitializer(initializer, 0, INT_MAX);

    // check all cases
    vector<pair<int, int>> check_cases{
            {2, 3}, {2, 2}, {2, 1}, {2, 0}, {2, -1}, // min > 0
            {0, 2}, {0, 0}, {0, -2}, // min = 0
            {-2, 1}, {-2, 0}, {-2, -1}, {-2, -2}, {-2, -3} // min < 0
    };

    for(auto p: check_cases) {
        int min = p.first;
        int max = p.second;

        if (min > max) {
            EXPECT_THROW(UniformIntInitializer(min, max), Exception);
        } else {
            auto initializer = make_shared<UniformIntInitializer>(min, max);
            checkUniformIntInitializer(initializer, min, max);
        }
    }
}


