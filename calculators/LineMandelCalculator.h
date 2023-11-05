/**
 * @file LineMandelCalculator.h
 * @author Matěj Konopík <xkonop03@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 4.11.2023
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    int* data;
    float* z_x_temp;
    float* z_y_temp;
    int half_height;
};