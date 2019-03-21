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
#include <gtest/gtest.h>

#include <nbla/exception.hpp>
#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/parametric_functions.hpp>

using namespace nbla;
namespace PF = nbla::parametric_functions;


class ParameterDirectoryTest : public ::testing::Test {
protected:
    Context ctx_;

    virtual void SetUp() override {
        ctx_.array_class = "CpuArray";
    }

    template<typename T>
    bool checkValuesSame(VariablePtr var, T v_ref) {
        T *d = var->cast_data_and_get_pointer<T>(ctx_, false);

        for (int i = 0; i < var->size(); ++i) {
            if (d[i] != v_ref) {
                return false;
            }
        }

        return true;
    }
};

template<>
bool ParameterDirectoryTest::checkValuesSame<VariablePtr>(VariablePtr var, VariablePtr v_ref) {
    // check shape (and size)
    if (var->shape() != v_ref->shape()) {
        return false;
    }

    // check all values
    auto *d1 = var->cast_data_and_get_pointer<float>(ctx_, false);
    auto *d2 = v_ref->cast_data_and_get_pointer<float>(ctx_, false);

    for (int i = 0; i < var->size(); ++i) {
        if (d1[i] != d2[i]) {
            return false;
        }
    }
    return true;
}


// ParameterDirectory::get_parameter_or_create(string, std::initializer_list, nbla::Initializer, bool) => CgVariablePtr
TEST_F(ParameterDirectoryTest, get_parameter_or_create) {
    ParameterDirectory params;
    string param_name = "test_parameter";

    // create parameter
    auto init = make_shared<ConstantInitializer>(1);
    auto shape = Shape_t{3, 3};
    CgVariablePtr created1 = params.get_parameter_or_create(param_name, shape, init.get(), false);

    // check shape
    ASSERT_TRUE(created1->variable()->shape() == shape);
    ASSERT_EQ(created1->variable()->size(), 9);

    // check all values
    ASSERT_TRUE(checkValuesSame<float>(created1->variable(), 1));

    // check create parameter which already exists
    init = make_shared<ConstantInitializer>(2); // dummy
    CgVariablePtr created2 = params.get_parameter_or_create(param_name, shape, init.get(), false);

    ASSERT_THROW(params.get_parameter_or_create(param_name, {5, 5}, init.get(), false), Exception);

    // check all values
    ASSERT_TRUE(checkValuesSame<VariablePtr>(created1->variable(), created2->variable()));
}


// ParameterDirectory::get_parameter(string) => CgVariablePtr
TEST_F(ParameterDirectoryTest, get_parameter) {
    ParameterDirectory params;
    string param_name = "test_parameter";

    // before creating
    ASSERT_TRUE(nullptr == params.get_parameter(param_name));

    // create parameter
    auto init = make_shared<ConstantInitializer>(1);
    CgVariablePtr created1 = params.get_parameter_or_create(param_name, {3, 3}, init.get(), false);

    // after creating
    auto ptr1 = params.get_parameter(param_name);

    ASSERT_TRUE(checkValuesSame<VariablePtr>(ptr1->variable(), created1->variable()));

    // check create parameter which already exists
    init = make_shared<ConstantInitializer>(2); // dummy initializer
    CgVariablePtr created2 = params.get_parameter_or_create(param_name, {3, 3}, init.get(), false);

    auto ptr2 = params.get_parameter(param_name);
    ASSERT_TRUE(checkValuesSame<VariablePtr>(ptr1->variable(), ptr2->variable()));
}


// parameterDirectory::get_parameters() => vector<pair<string, VariablePtr>>
TEST_F(ParameterDirectoryTest, get_parameters) {
    ParameterDirectory params;
    pair<string, VariablePtr> registered;
    vector<pair<string, VariablePtr>> all_variables;
    string param_name = "test_parameter";

    // before creating
    all_variables = params.get_parameters();
    ASSERT_EQ(all_variables.size(), 0);

    // create parameter
    auto init = make_shared<ConstantInitializer>(1);
    CgVariablePtr created1 = params.get_parameter_or_create(param_name, {3, 3}, init.get(), false);

    // after creating
    all_variables = params.get_parameters();
    ASSERT_EQ(all_variables.size(), 1);
    registered = all_variables[0];
    ASSERT_EQ("/" + param_name, registered.first); // parameter name check
    ASSERT_TRUE(checkValuesSame(created1->variable(), registered.second));

    // create same named parameter again
    init = make_shared<ConstantInitializer>(2); // dummy initializer
    CgVariablePtr created2 = params.get_parameter_or_create(param_name, {3, 3}, init.get(), false);

    all_variables = params.get_parameters();
    ASSERT_EQ(all_variables.size(), 1);
    registered = all_variables[0];
    ASSERT_EQ("/" + param_name, registered.first); // parameter name check
    ASSERT_TRUE(checkValuesSame(created2->variable(), registered.second));

    // create other parameter
    string new_param_name = "new_parameter";
    CgVariablePtr created3 = params.get_parameter_or_create(new_param_name, {3, 3}, init.get(),
                                                            false); // ConstantInitializer(2)

    all_variables = params.get_parameters();
    ASSERT_EQ(all_variables.size(), 2);

    vector<string> name_list = {param_name, new_param_name};
    vector<CgVariablePtr> param_list = {created1, created3};
    REP(i, 2) {
        registered = all_variables[i];
        ASSERT_EQ("/" + name_list[i], registered.first); // parameter name check
        ASSERT_TRUE(checkValuesSame(param_list[i]->variable(), registered.second));
    }

    // test scope (scope/new_parameter)
    init = make_shared<ConstantInitializer>(3);
    all_variables = params["scope"].get_parameters();
    ASSERT_EQ(all_variables.size(), 0);

    CgVariablePtr created4 = params["scope"].get_parameter_or_create(new_param_name, {4, 4}, init.get(),
                                                                     false); // ConstantInitializer(3)

    all_variables = params.get_parameters();
    ASSERT_EQ(all_variables.size(), 3);

    all_variables = params["scope"].get_parameters();
    ASSERT_EQ(all_variables.size(), 1);
    ASSERT_TRUE(checkValuesSame<VariablePtr>(all_variables[0].second, created4->variable()));
}

// ParameterDirectory::deep_copy() => parameterDirectory
TEST_F(ParameterDirectoryTest, deep_copy) {
    ParameterDirectory params;
    vector<string> name_list = {"param1", "param2"};
    vector<Shape_t> shape_list = {Shape_t{3, 3}, Shape_t{5, 5}};
    vector<shared_ptr<ConstantInitializer>> init_list = {make_shared<ConstantInitializer>(1),
                                                         make_shared<ConstantInitializer>(2)};

    REP(i, name_list.size()) {
        params.get_parameter_or_create(name_list[i], shape_list[i], init_list[i].get(), false);
    }

    // deep copy
    ParameterDirectory params_copy = params.create_deep_copy();

    // check size
    ASSERT_EQ(params.get_parameters().size(), 2);
    ASSERT_EQ(params_copy.get_parameters().size(), 2);

    // check all parameters are same
    REP(i, name_list.size()) {
        ASSERT_TRUE(params_copy.get_parameter(name_list[i])->variable()->shape() == shape_list[i]);
        ASSERT_TRUE(checkValuesSame<VariablePtr>(params.get_parameter(name_list[i])->variable(),
                                                 params_copy.get_parameter(name_list[i])->variable()));
    }

    // insert new parameter to params_copy
    auto init = make_shared<ConstantInitializer>(3);
    params_copy.get_parameter_or_create("only_for_params_copy", {2, 2}, init.get(), false);

    ASSERT_EQ(params.get_parameters().size(), 2);
    ASSERT_EQ(params_copy.get_parameters().size(), 3);

    // change the values of a parameter in params_copy
    string target_name = name_list[0];
    VariablePtr var = params_copy.get_parameter(target_name)->variable();
    auto *d = var->cast_data_and_get_pointer<float>(ctx_, true);

    REP(i, var->size()) {
        d[i] = 0;
    }

    ASSERT_TRUE(checkValuesSame<float>(params_copy.get_parameter(target_name)->variable(), 0));
    ASSERT_TRUE(checkValuesSame<float>(params.get_parameter(target_name)->variable(), 1));
}


