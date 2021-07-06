// Copyright 2021 Sony Corporation.
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

#ifndef __NBLA_EVENT_HPP__
#define __NBLA_EVENT_HPP__

#include <nbla/common.hpp>
#include <nbla/context.hpp>
#include <nbla/defs.hpp>

#include <memory>

namespace nbla {

using std::shared_ptr;

class NBLA_API Event {
public:
  Event() {}
  virtual ~Event();

  // Return the flag which is true if this event can be delete.
  virtual void wait_event(const Context ctx,
                          const int async_flags = AsyncFlag::NONE) {}
};

/// Shared pointer of Event.
typedef shared_ptr<Event> EventPtr;
}
#endif