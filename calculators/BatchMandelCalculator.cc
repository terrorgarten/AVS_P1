/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <cstdlib>
#include <stdexcept>

#include "BatchMandelCalculator.h"

using std::cout;
using std::cerr;
using std::endl;

#define ALIGN_SIZE 64                           // align memory to 64 bytes (for the AVX512 registers: 64B = 512b)
#define SIMD_LEN_INT (512/(sizeof(int)*8))      // number of integers in AVX512 register
#define SIMD_LEN_FLOAT (512/(sizeof(float)*8))  // number of floats in AVX512 register
#define BATCH_SIZE 64                           // number of cells to calculate in one batch
#define BATCH_MEM_ALLOC_ERR 2000                // error code for memory allocation failure


//#define DEBUG   // uncomment this line to enable debug printing
#ifdef DEBUG
#define D_PRINT(x) std::cout << "BATCH_DEBUG: " << x << std::endl
#else
#define D_PRINT(x)
#endif

BatchMandelCalculator::BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit) :
        BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator") {
    // allocate main data matrix
    data = (int *) (aligned_alloc(ALIGN_SIZE, height * width * sizeof(int)));
    // allocate helper arrays
    z_x_temp = (float *) (aligned_alloc(ALIGN_SIZE, BATCH_SIZE * sizeof(float)));
    z_y_temp = (float *) (aligned_alloc(ALIGN_SIZE, BATCH_SIZE * sizeof(float)));
    // check allocation success
    if (data == nullptr or z_x_temp == nullptr or z_y_temp == nullptr) {
        cerr << typeid(*this).name() << " : Memory allocation failed. Aborting." << endl;
        exit(BATCH_MEM_ALLOC_ERR);
    }
    // we use the fact that mandelbrot is symmetrical, therefore we only calculate half and then copy it
    half_height = height / 2;
    matrix_base_size = matrixBaseSize;
    D_PRINT(typeid(*this).name() << " : half_height=" << half_height
                                 << " height=" << height
                                 << " width=" << width
                                 << " limit=" << limit
                                 << endl);
}

BatchMandelCalculator::~BatchMandelCalculator() {
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


int *BatchMandelCalculator::calculateMandelbrot() {
    D_PRINT(typeid(*this).name() << " : calculateMandelbrot(): " << matrix_base_size / BATCH_SIZE << endl);
    // prefill default values to the data array, vectorize it
    for (auto i = 0; i < height * width; i++) {
        data[i] = limit;
    }

    // iterate over the first half of the lines
    for (auto y_index = 0; y_index <= half_height; y_index++) {
        // calculate the y value for the current line (given by the y_index)
        auto y_value = float(y_start + y_index * dy);
        // iterate over the batches in the current line
        D_PRINT("y_index: " << y_index << " y_value: " << y_value << endl);
        for (auto batch_start_index = 0; batch_start_index < width; batch_start_index += BATCH_SIZE) {
            // fill up the helper arrays with the current batch values
            D_PRINT("batch_start_index: " << batch_start_index << endl);
            for (auto batch_inner_index = 0; batch_inner_index < BATCH_SIZE; batch_inner_index++) {
                z_x_temp[batch_inner_index] = float(x_start + (batch_start_index + batch_inner_index) * dx);
                z_y_temp[batch_inner_index] = y_value;
                D_PRINT("TOTAL INDEX: " << batch_start_index + batch_inner_index << endl);
            }

            // calculate the mandelbrot values for the current batch
            for (auto iteration = 0; iteration < limit; iteration++) {
                // cycle over the helper arrays and calculate
                for (auto batch_inner_index = 0; batch_inner_index < BATCH_SIZE; batch_inner_index++) {
                    auto x_index = batch_start_index + batch_inner_index;
                    if (data[y_index * width + x_index] == limit) { // TODO - move to if on 104 to improve vectorization?

                        auto x_value = float(x_start + x_index * dx);

                        auto z_x = z_x_temp[batch_inner_index];
                        auto z_y = z_y_temp[batch_inner_index];

                        auto z_x2 = z_x * z_x;
                        auto z_y2 = z_y * z_y;

                        if (z_x2 + z_y2 > 4.0f) {
                            data[y_index * width + x_index] = iteration;
                        } else {
                            z_y_temp[batch_inner_index] = 2.0f * z_x * z_y + y_value;
                            z_x_temp[batch_inner_index] = z_x2 - z_y2 + x_value;
                        }
                    }
                }
            }
        }
        // copy the calculated line to the second half of the matrix
        for (auto x_index = 0; x_index < width; x_index++) {
            data[(height - y_index - 1) * width + x_index] = data[y_index * width + x_index];
        }
    }
    return data;
}