#ifndef SWAPENDIAN_H
#define SWAPENDIAN_H

#pragma once
#include <cstdint>

/**
 * \brief Swaps the byte order of a 32-bit unsigned integer.
 *
 * This function takes a 32-bit unsigned integer and swaps its byte order
 * (endianness). It is useful for converting between big-endian and little-endian
 * representations.
 *
 * \param value The 32-bit unsigned integer whose byte order is to be swapped.
 * \return The 32-bit unsigned integer with its byte order swapped.
 */
std::uint32_t swap_endian(std::uint32_t value);

#include "swapEndian.tpp"

#endif //SWAPENDIAN_H
