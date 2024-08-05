#include "readUInt32.h"

std::uint32_t readUInt32(std::ifstream& stream) {
    std::uint32_t value;
    // Read 4 bytes from the stream into the value
    if (!stream.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        throw std::runtime_error("Failed to read 32-bit unsigned integer from stream");
    }
    // Swap the byte order to match the host system's endianness
    return swap_endian(value);
}