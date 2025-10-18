// views.hpp - mdspan aliases and helpers
#pragma once

#include "cpp_hf/hf_mdspan.hpp"
#include <cstddef>

namespace hf {

using Ext2 = stdx::extents<std::size_t, stdx::dynamic_extent, stdx::dynamic_extent>;
using Ext4 = stdx::extents<std::size_t, stdx::dynamic_extent, stdx::dynamic_extent,
                                      stdx::dynamic_extent, stdx::dynamic_extent>;

template <class T>
using Grid2 = stdx::mdspan<T, Ext2, stdx::layout_right>;

template <class T>
using Grid4 = stdx::mdspan<T, Ext4, stdx::layout_right>;

// Access helper to avoid depending on operator() availability across lib versions
template <class Mdspan, class... Idx>
inline decltype(auto) mds_get(const Mdspan& m, Idx... idx) {
    return m.data_handle()[ m.mapping()(static_cast<std::size_t>(idx)...) ];
}

} // namespace hf
