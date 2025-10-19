// views.hpp - mdspan aliases and helpers
#pragma once

#include "cpp_hf/mdspan_compat.hpp"
#include <cstddef>

namespace hf {

// Standard mdspan aliases (via compat namespace alias `md`)
using Ext2 = md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent>;
using Ext4 = md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent,
                                      md::dynamic_extent, md::dynamic_extent>;

template <class T>
using Grid2 = md::mdspan<T, Ext2, md::layout_right>;

template <class T>
using Grid4 = md::mdspan<T, Ext4, md::layout_right>;

// Access helper to avoid depending on operator() availability across lib versions
template <class Mdspan, class... Idx>
inline decltype(auto) mds_get(const Mdspan& m, Idx... idx) {
    return m.data_handle()[ m.mapping()(static_cast<std::size_t>(idx)...) ];
}

} // namespace hf
