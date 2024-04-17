#pragma once
#include "hnswlib.h"

namespace hnswlib {
    /**
     * For a given loop unrolling factor K, distance type dist_t, and data type data_t,
     * calculate the Hamming distance between two vectors.
     * The compiler should automatically do the loop unrolling for us here and vectorize as appropriate.
     */    
    template<typename dist_t, typename data_t = dist_t, int K = 1>
    static dist_t
    Tanimoto(const void *__restrict pVect1, const void *__restrict vect1SetBits_ptr, const void *__restrict pVect2, const void *__restrict vect2SetBits_ptr, const void *__restrict qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t vect1SetBits = *((size_t *) vect1SetBits_ptr);
        size_t vect2SetBits = *((size_t *) vect2SetBits_ptr);

        data_t *a = (data_t *) pVect1;
        data_t *b = (data_t *) pVect2;

        size_t common = 0;

        qty = qty / K;

        for (size_t i = 0; i < qty; i++) {
            for (size_t j = 0; j < K; j++) {
                const size_t index = (i * K) + j;
                const unsigned char _a = a[index];
                const unsigned char _b = b[index];

                // Count the number of set bits in the intersection of both fingerprints using
                // Brian Kernighan's algorithm
                unsigned char and_result = _a & _b;
                while (and_result) {
                    ++common;
                    and_result &= and_result - 1;
                }

            }
        }
        
        if ((vect1SetBits == 0) && (vect2SetBits == 0))
            return 1.0;

        // Calculate Tanimoto distance
        dist_t tanimoto = (dist_t) common / (vect1SetBits + vect2SetBits - common);
        dist_t tanimoto_distance = 1.0 - tanimoto;
        return (tanimoto_distance);
    }
    

    template<typename dist_t, typename data_t = dist_t, int K>
    static dist_t
    TanimotoAtLeast(const void *__restrict pVect1, const void *__restrict vect1SetBits_ptr, const void *__restrict pVect2, const void *__restrict vect2SetBits_ptr, const void *__restrict qty_ptr) {
        size_t k = K;
        size_t remainder = *((size_t *) qty_ptr) - K;

        data_t *a = (data_t *) pVect1;
        data_t *b = (data_t *) pVect2;

        size_t vect1SetBits = *((size_t *) vect1SetBits_ptr);
        size_t vect2SetBits = *((size_t *) vect2SetBits_ptr);

        return Tanimoto<dist_t, data_t, K>(a, &vect1SetBits, b, &vect2SetBits, &k)
             + Tanimoto<dist_t, data_t, 1>(a + K, &vect1SetBits, b + K, &vect2SetBits, &remainder);
    }


    template <typename dist_t, typename data_t = dist_t>
    class TanimotoSpace : public SpaceInterface<dist_t> {
        DISTFUNC<dist_t> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        TanimotoSpace(size_t dim) : dim_(dim), data_size_(dim * sizeof(data_t)) {
            if (dim % 128 == 0)
                fstdistfunc_ = Tanimoto<dist_t, data_t, 128>;
            else if (dim % 64 == 0)
                fstdistfunc_ = Tanimoto<dist_t, data_t, 64>;
            else if (dim % 32 == 0)
                fstdistfunc_ = Tanimoto<dist_t, data_t, 32>;
            else if (dim % 16 == 0)
                fstdistfunc_ = Tanimoto<dist_t, data_t, 16>;
            else if (dim % 8 == 0)
                fstdistfunc_ = Tanimoto<dist_t, data_t, 8>;
            else if (dim % 4 == 0)
                fstdistfunc_ = Tanimoto<dist_t, data_t, 4>;

            else if (dim > 128)
                fstdistfunc_ = TanimotoAtLeast<dist_t, data_t, 128>;            
            else if (dim > 64)
                fstdistfunc_ = TanimotoAtLeast<dist_t, data_t, 64>;
            else if (dim > 32)
                fstdistfunc_ = TanimotoAtLeast<dist_t, data_t, 32>;
            else if (dim > 16)
                fstdistfunc_ = TanimotoAtLeast<dist_t, data_t, 16>;
            else if (dim > 8)
                fstdistfunc_ = TanimotoAtLeast<dist_t, data_t, 8>;
            else if (dim > 4)
                fstdistfunc_ = TanimotoAtLeast<dist_t, data_t, 4>;
            else
                fstdistfunc_ = Tanimoto<dist_t, data_t>;
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<dist_t> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~TanimotoSpace() {}
    };

}
