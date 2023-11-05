/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD parallelization over lines
 * @date DATE
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <cstring>


#include "LineMandelCalculator.h"

using std::cout;
using std::endl;
using std::memcpy;

#define ALIGN_SIZE 64                           // align memory to 64 bytes (for the AVX512 registers: 64B = 512b)
#define SIMD_LEN_INT (512/(sizeof(int)*8))      // number of integers in AVX512 register
#define SIMD_LEN_FLOAT (512/(sizeof(float)*8))  // number of floats in AVX512 register
#define MEM_ALLOC_ERR 1000                      // error code for memory allocation failure


#define DEBUG   // comment this line to disable debug printing
#ifdef DEBUG
#define D_PRINT(x) std::cout << "DEBUG: " << x << std::endl
#else
#define D_PRINT(x)
#endif


LineMandelCalculator::LineMandelCalculator(unsigned matrixBaseSize, unsigned limit) :
        BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator") {
    // allocate main data matrix
    data = (int *) (aligned_alloc(ALIGN_SIZE, height * width * sizeof(int)));
    // allocate helper arrays
    z_x_temp = (float *) (aligned_alloc(ALIGN_SIZE, width * sizeof(float)));
    z_y_temp = (float *) (aligned_alloc(ALIGN_SIZE, width * sizeof(float)));
    // check allocation success
    if (data == nullptr or z_x_temp == nullptr or z_y_temp == nullptr) {
        std::cout << typeid(*this).name() << " : Memory allocation failed. Aborting." << endl;
        exit(MEM_ALLOC_ERR);
    }
    // we use the fact that mandelbrot is symmetrical, therefore we only calculate half and then copy it
    half_height = height / 2;
    D_PRINT(typeid(*this).name() << " : half_height=" << half_height
                                 << " height=" << height
                                 << " width=" << width
                                 << " limit=" << limit
                                 << endl);
}

LineMandelCalculator::~LineMandelCalculator() {
    // free allocated data with checking against the memory validity
    if (data != nullptr) {
        free(data);
    }
    if (z_x_temp != nullptr) {
        free(z_x_temp);
    }
    if (z_y_temp != nullptr) {
        free(z_y_temp);
    }
}


int *LineMandelCalculator::calculateMandelbrot() {

    // prefill default values to the data array, vectorize it
#pragma omp simd aligned(data:ALIGN_SIZE) simdlen(SIMD_LEN_INT) uniform(limit)
    for (auto i = 0; i < height * width; i++) {
        data[i] = limit;
    }

    for (auto y_index = 0; y_index <= half_height; y_index++) {
        // use omd simd to calculate mandelbrot using vectorization
        D_PRINT("Calculating line " << y_index << " of " << half_height);
        auto y_value = float(y_start + y_index * dy);

        // prepare the current values for given line (y_index)
#pragma omp simd aligned(z_y_temp:ALIGN_SIZE, z_x_temp:ALIGN_SIZE) simdlen(SIMD_LEN_FLOAT) uniform(y_value, y_start, y_index, dy)
        for (auto x_index = 0; x_index <= width; x_index++) {
            z_y_temp[x_index] = float(y_start + y_index * dy);
            z_y_temp[x_index] = y_value;
        }

        // calculate mandelbrot for given line (y_index)
        for (auto calc_iter = 0; calc_iter < limit; ++calc_iter) {
            // calculate z_x^2 and z_y^2
#pragma omp simd aligned(z_x_temp:ALIGN_SIZE, z_y_temp:ALIGN_SIZE) simdlen(SIMD_LEN_FLOAT) uniform(y_value)
            for (auto x_index = 0; x_index <= width; x_index++) {
                z_x_temp[x_index] = z_x_temp[x_index] * z_x_temp[x_index];
                z_y_temp[x_index] = z_y_temp[x_index] * z_y_temp[x_index];
            }
            // check if the value is above the limit and the value is default, rewrite with current calc iteration value
#pragma omp simd aligned(z_x_temp:ALIGN_SIZE, z_y_temp:ALIGN_SIZE, data:ALIGN_SIZE) simdlen(SIMD_LEN_FLOAT) uniform(limit)
            for (auto x_index = 0; x_index <= width; x_index++) {
                if (z_x_temp[x_index] + z_y_temp[x_index] > 4.0f && data[y_index * width + x_index] == limit) {
                    D_PRINT("Saving explicit value " << calc_iter << " to data[" << y_index * width + x_index << "]");
                    data[y_index * width + x_index] = calc_iter;
                }
            }
            // calculate z_x and z_y
#pragma omp simd aligned(z_x_temp:ALIGN_SIZE, z_y_temp:ALIGN_SIZE) simdlen(SIMD_LEN_FLOAT) uniform(y_value)
            for (auto x_index = 0; x_index <= width; x_index++) {
                z_x_temp[x_index] = z_x_temp[x_index] - z_y_temp[x_index] + float(x_start + x_index * dx);
                z_y_temp[x_index] = 2.0f * z_x_temp[x_index] * z_y_temp[x_index] + y_value;
            }
        }

        // copy the calculated line to the second half of the matrix
#pragma omp simd aligned(data: ALIGN_SIZE) uniform(y_index, width)
        for (auto x_index = 0; x_index < width; x_index++) {
            memcpy(&data[(height - y_index - 1) * width + x_index], &data[y_index * width + x_index], sizeof(int));
        }
    }
    return data;
}
