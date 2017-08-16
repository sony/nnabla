#include "protobuf_internal.hpp"

#include <nbla/logger.hpp>
#include <fstream>

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/variable.hpp>
#include <nbla/function.hpp>
{function_includes}


namespace nbla_utils 
{{
    namespace NNP
    {{
        shared_ptr<nbla::CgFunction> _proto_internal::create_cgfunction(const Function& func) 
        {{
            {function_creator}
            return 0;
        }}
    }}
}}

