//
// Created by ganler on 2/10/20.
//

#include "spinlock.hpp"

void spinlock::lock() noexcept
{
    while (m_af.test_and_set(std::memory_order_acquire))
        ;
}

void spinlock::unlock() noexcept { m_af.clear(std::memory_order_release); }