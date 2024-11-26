#pragma once
#include <torch/extension.h>

#include "bfs.h"
#include "mst.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    /* build trees */
    m.def("bfs_forward", &bfs_forward, "bfs_forward");
    m.def("mst_forward", &mst_forward, "mst_forward");
}
