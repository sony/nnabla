//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NBLA_STD_HPP__
#define __NBLA_STD_HPP__

#include <deque>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <functional>
#include <memory>
#include <type_traits>

#include <cstdlib>

namespace nbla {

inline void *malloc(size_t size) { return std::malloc(size); }

inline void free(void *p) { return std::free(p); }

using std::deque;
using std::function;
using std::hash;
using std::list;
using std::map;
using std::multimap;
using std::ostringstream;
using std::priority_queue;
using std::queue;
using std::set;
using std::stack;
using std::string;
using std::stringstream;
using std::unordered_map;
using std::unordered_set;
using std::vector;

using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;

using std::enable_shared_from_this;
using std::make_shared;
using std::make_unique;

using std::to_string;

/** Allocate memory and call constructor of a new object.

   This macro enables us to create a object with private constructor.
 */
#define NBLA_NEW_OBJECT(TYPE, ...) (new TYPE(__VA_ARGS__))

/** Call destructor and deallocate memory of a object.

   This macro enables us to delete a object with private destructor.
 */
#define NBLA_DELETE_OBJECT(p)                                                  \
  do {                                                                         \
    delete (p);                                                                \
  } while (false)

/** Allocate memory and call constructor of a new object.
 */
template <typename T, typename... Args> T *new_object(Args &&... args) {
  static_assert(std::is_constructible<T, Args...>::value,
                "Unable to call constructor.");
  return new T(std::forward<Args>(args)...);
}

/** Call destructor and deallocate memory of a object.
 */
template <typename T> void delete_object(T *p) {
  static_assert(std::is_destructible<T>::value, "Unable to call destructor.");
  delete p;
}

/** Allocate memory and call constructor of a new array.
 */
template <typename T> T *new_array(size_t n) {
  static_assert(std::is_default_constructible<T>::value,
                "Unable to call constructor.");
  return new T[n];
}

/** Call destructor and deallocate memory of an array.
 */
template <typename T> void delete_array(T *p) {
  static_assert(std::is_destructible<T>::value, "Unable to call destructor.");
  delete[] p;
}
} // namespace nbla

#endif
