/**
 * @file BatchMandelCalculator.h
 * @author Matěj Konopík <xkonop03@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD parallelization over small batches
 * @date 4.11.2023
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
    int* data;
    float* z_x_temp;
    float* z_y_temp;
    int half_height;
    unsigned matrix_base_size;
};

#endif