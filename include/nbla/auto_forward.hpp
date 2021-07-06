// Copyright 2019,2020,2021 Sony Corporation.
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

#ifndef __NBLA_AUTO_FORWARD_HPP__
#define __NBLA_AUTO_FORWARD_HPP__
#include <nbla/defs.hpp>
#include <nbla/singleton_manager.hpp>

namespace nbla {
/**
Singleton class for storing global context.
*/
class NBLA_API AutoForward {
  bool current_;

public:
  ~AutoForward();
  /** Get auto forward.
   */
  bool get_auto_forward() const;

  /** Set current context.
   */
  void set_auto_forward(bool autoforward);

private:
  friend SingletonManager;
  // Never called by users.
  AutoForward();
  DISABLE_COPY_AND_ASSIGN(AutoForward);
};
}
#endif
