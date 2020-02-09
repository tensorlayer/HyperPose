//
// Created by ganler on 2/10/20.
//

#pragma once

#include <atomic>
#include <iostream>
#include <thread>

class spinlock
{
  private:
    std::atomic_flag m_af;

  public:
    spinlock() : m_af(ATOMIC_FLAG_INIT) {}
    void lock() noexcept;
    void unlock() noexcept;
};