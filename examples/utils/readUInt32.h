#ifndef READUINT32_H
#define READUINT32_H

#include <fstream>
#include "swapEndian.h"

/**
 * \brief Reads a 32-bit unsigned integer from a file stream in big-endian format.
 *
 * This function reads 4 bytes from the given file stream, interprets them as a
 * 32-bit unsigned integer in big-endian format, and returns the value with the
 * correct byte order for the host system.
 *
 * \param stream The input file stream to read from.
 * \return The 32-bit unsigned integer read from the file stream.
 * \throws std::runtime_error if the read operation fails.
*/
std::uint32_t readUInt32(std::ifstream& stream);

#endif // READUINT32_H