#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>

using rank_t = uint8_t;

template <typename dim_t, rank_t r> class basic_shape
{
  public:
    static constexpr rank_t rank = r;

    constexpr explicit basic_shape(const std::array<dim_t, r> &dims)
        : dims(dims)
    {
    }

    template <typename... D>
    constexpr explicit basic_shape(D... d) : dims({static_cast<dim_t>(d)...})
    {
        static_assert(sizeof...(D) == r, "invalid number of dims");
    }

    template <typename... I> dim_t index(I... args) const
    {
        static_assert(sizeof...(I) == r, "invalid number of indexes");

        // TODO: expand the expression
        const std::array<dim_t, r> offs({static_cast<dim_t>(args)...});
        dim_t off = 0;
        for (rank_t i = 0; i < r; ++i) { off = off * dims[i] + offs[i]; }
        return off;
    }

    dim_t size() const
    {
        return std::accumulate(dims.begin(), dims.end(), 1,
                               std::multiplies<dim_t>());
    }

  private:
    const std::array<dim_t, r> dims;
};

using dim_t = uint32_t;

template <rank_t r> using shape = basic_shape<dim_t, r>;
