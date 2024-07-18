#ifndef MixedPrecisionFloat16_H
#define MixedPrecisionFloat16_H

#include <cstdint>
#include <iostream>

class float_16 {
public:
    uint16_t value;

    float_16();
    explicit float_16(float f);

    explicit operator float() const;

private:
    static uint16_t float32_to_float16(float f);
    static float float16_to_float32(uint16_t f);
};

float_16 operator+(const float_16& a, const float_16&b);
float_16 operator-(const float_16& a, const float_16&b);
float_16 operator*(const float_16& a, const float_16&b);
float_16 operator/(const float_16& a, const float_16&b);


#endif // MixedPrecisionFloat16_H
