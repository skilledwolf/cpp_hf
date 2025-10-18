// views.hpp - mdspan aliases and helpers
#pragma once

#include <mdspan>
#include <cstddef>

namespace hf {

using Ext2 = std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>;
using Ext4 = std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent,
                                      std::dynamic_extent, std::dynamic_extent>;

template <class T>
using Grid2 = std::mdspan<T, Ext2, std::layout_right>;

template <class T>
using Grid4 = std::mdspan<T, Ext4, std::layout_right>;

// Access helper to avoid depending on operator() availability across lib versions
template <class Mdspan, class... Idx>
inline decltype(auto) mds_get(const Mdspan& m, Idx... idx) {
    return m.data_handle()[ m.mapping()(static_cast<std::size_t>(idx)...) ];
}

} // namespace hf

