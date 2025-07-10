
#include "dcn_v3.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dcn_v3_forward", &dcn_v3_forward, "dcn_v3_forward");
  m.def("dcn_v3_backward", &dcn_v3_backward, "dcn_v3_backward");
}
