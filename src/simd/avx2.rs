//! AVX2/AVX-512 Intrinsics for Parallel Order Book Processing
//! 
//! Accelerates microprice calculations and Z-score normalizations by processing
//! 8 to 16 data points simultaneously per CPU cycle.

use std::arch::x86_64::*;

/// Process 4 price levels using AVX for microprice calculation
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn compute_microprice_avx(
    bids: [f64; 4],
    asks: [f64; 4],
    bid_sizes: [f64; 4],
    ask_sizes: [f64; 4],
) -> f64 {
    // Load data into AVX registers
    let bid_vec = _mm256_loadu_pd(bids.as_ptr());
    let ask_vec = _mm256_loadu_pd(asks.as_ptr());
    let bid_size_vec = _mm256_loadu_pd(bid_sizes.as_ptr());
    let ask_size_vec = _mm256_loadu_pd(ask_sizes.as_ptr());

    // Calculate weighted mid: (bid*bid_size + ask*ask_size) / (bid_size + ask_size)
    let bid_weighted = _mm256_mul_pd(bid_vec, bid_size_vec);
    let ask_weighted = _mm256_mul_pd(ask_vec, ask_size_vec);
    let numerator = _mm256_add_pd(bid_weighted, ask_weighted);

    let denominator = _mm256_add_pd(bid_size_vec, ask_size_vec);

    // Divide
    let result = _mm256_div_pd(numerator, denominator);

    // Horizontal sum and average
    let mut sums = [0.0f64; 4];
    _mm256_storeu_pd(sums.as_mut_ptr(), result);

    (sums[0] + sums[1] + sums[2] + sums[3]) / 4.0
}

/// Compute Z-scores for 8 values using AVX2
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn compute_zscores_avx(values: &[f64; 8], mean: f64, std_dev: f64) -> [f64; 8] {
    let val_vec = _mm256_loadu_pd(values.as_ptr());
    let val_vec2 = _mm256_loadu_pd(values.as_ptr().add(4));

    let mean_vec = _mm256_set1_pd(mean);
    let std_vec = _mm256_set1_pd(std_dev);

    // Z = (X - mean) / std
    let diff1 = _mm256_sub_pd(val_vec, mean_vec);
    let diff2 = _mm256_sub_pd(val_vec2, mean_vec);

    let z1 = _mm256_div_pd(diff1, std_vec);
    let z2 = _mm256_div_pd(diff2, std_vec);

    let mut result = [0.0f64; 8];
    _mm256_storeu_pd(result.as_mut_ptr(), z1);
    _mm256_storeu_pd(result.as_mut_ptr().add(4), z2);

    result
}

/// Parallel comparison of order book deltas using AVX2
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn compare_deltas_avx(
    old_prices: [f64; 4],
    new_prices: [f64; 4],
) -> u32 {
    let old_vec = _mm256_loadu_pd(old_prices.as_ptr());
    let new_vec = _mm256_loadu_pd(new_prices.as_ptr());

    // Compare: returns mask where new > old
    let cmp = _mm256_cmp_pd(new_vec, old_vec, _CMP_GT_OQ);

    // Extract mask
    let mask = _mm256_movemask_pd(cmp) as u32;

    mask
}

/// Sum 8 values using AVX2 horizontal add
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn horizontal_sum_avx(values: &[f64; 8]) -> f64 {
    let vec1 = _mm256_loadu_pd(values.as_ptr());
    let vec2 = _mm256_loadu_pd(values.as_ptr().add(4));

    let sum = _mm256_add_pd(vec1, vec2);

    // Horizontal add within each 128-bit lane
    let hi = _mm256_permute2f128_pd(sum, sum, 0x1);
    let sum2 = _mm256_add_pd(sum, hi);

    let shuffled = _mm256_shuffle_pd(sum2, sum2, 0x5);
    let final_sum = _mm256_add_pd(sum2, shuffled);

    let mut result = [0.0f64; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), final_sum);

    result[0]
}

/// Find maximum value in array using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn find_max_avx(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let len = values.len();
    let mut max_val = f64::NEG_INFINITY;

    // Process 4 at a time
    let mut i = 0;
    if len >= 4 {
        let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);

        while i + 4 <= len {
            let vec = _mm256_loadu_pd(values.as_ptr().add(i));
            max_vec = _mm256_max_pd(max_vec, vec);
            i += 4;
        }

        // Extract max from vector
        let mut temps = [0.0f64; 4];
        _mm256_storeu_pd(temps.as_mut_ptr(), max_vec);
        max_val = temps[0].max(temps[1]).max(temps[2]).max(temps[3]);
    }

    // Handle remainder
    while i < len {
        max_val = max_val.max(values[i]);
        i += 1;
    }

    max_val
}

/// Find minimum value in array using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn find_min_avx(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }

    let len = values.len();
    let mut min_val = f64::INFINITY;

    // Process 4 at a time
    let mut i = 0;
    if len >= 4 {
        let mut min_vec = _mm256_set1_pd(f64::INFINITY);

        while i + 4 <= len {
            let vec = _mm256_loadu_pd(values.as_ptr().add(i));
            min_vec = _mm256_min_pd(min_vec, vec);
            i += 4;
        }

        // Extract min from vector
        let mut temps = [0.0f64; 4];
        _mm256_storeu_pd(temps.as_mut_ptr(), min_vec);
        min_val = temps[0].min(temps[1]).min(temps[2]).min(temps[3]);
    }

    // Handle remainder
    while i < len {
        min_val = min_val.min(values[i]);
        i += 1;
    }

    min_val
}

/// AVX-512 implementation for processing 8 doubles at once
#[cfg(target_feature = "avx512f")]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn compute_microprice_avx512(
    bids: [f64; 8],
    asks: [f64; 8],
    bid_sizes: [f64; 8],
    ask_sizes: [f64; 8],
) -> f64 {
    use std::arch::x86_64::*;

    let bid_vec = _mm512_loadu_pd(bids.as_ptr());
    let ask_vec = _mm512_loadu_pd(asks.as_ptr());
    let bid_size_vec = _mm512_loadu_pd(bid_sizes.as_ptr());
    let ask_size_vec = _mm512_loadu_pd(ask_sizes.as_ptr());

    let bid_weighted = _mm512_mul_pd(bid_vec, bid_size_vec);
    let ask_weighted = _mm512_mul_pd(ask_vec, ask_size_vec);
    let numerator = _mm512_add_pd(bid_weighted, ask_weighted);
    let denominator = _mm512_add_pd(bid_size_vec, ask_size_vec);

    let result = _mm512_div_pd(numerator, denominator);

    // Horizontal sum
    let mut sums = [0.0f64; 8];
    _mm512_storeu_pd(sums.as_mut_ptr(), result);

    let total: f64 = sums.iter().sum();
    total / 8.0
}

/// Runtime CPU feature detection and dispatch
pub fn compute_microprice_dispatch(
    bids: &[f64],
    asks: &[f64],
    bid_sizes: &[f64],
    ask_sizes: &[f64],
) -> f64 {
    let len = bids.len().min(asks.len()).min(bid_sizes.len()).min(ask_sizes.len());
    
    if len == 0 {
        return 0.0;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && len >= 8 {
            unsafe {
                let mut b = [0.0; 8];
                let mut a = [0.0; 8];
                let mut bs = [0.0; 8];
                let mut as_ = [0.0; 8];
                
                for i in 0..8.min(len) {
                    b[i] = bids[i];
                    a[i] = asks[i];
                    bs[i] = bid_sizes[i];
                    as_[i] = ask_sizes[i];
                }
                
                return compute_microprice_avx512(b, a, bs, as_);
            }
        }

        if is_x86_feature_detected!("avx2") && len >= 4 {
            unsafe {
                let mut b = [0.0; 4];
                let mut a = [0.0; 4];
                let mut bs = [0.0; 4];
                let mut as_ = [0.0; 4];
                
                for i in 0..4.min(len) {
                    b[i] = bids[i];
                    a[i] = asks[i];
                    bs[i] = bid_sizes[i];
                    as_[i] = ask_sizes[i];
                }
                
                return compute_microprice_avx(b, a, bs, as_);
            }
        }
    }

    // Fallback scalar implementation
    let mut total = 0.0;
    let mut weight = 0.0;
    for i in 0..len {
        let w = bid_sizes[i] + ask_sizes[i];
        total += (bids[i] * bid_sizes[i] + asks[i] * ask_sizes[i]);
        weight += w;
    }
    
    if weight > 0.0 {
        total / weight
    } else {
        0.0
    }
}

/// Safe wrapper for AVX2 operations with runtime detection
pub struct Avx2Accelerator {
    avx2_available: bool,
    avx512_available: bool,
}

impl Avx2Accelerator {
    pub fn new() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            avx2_available: is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            avx2_available: false,

            #[cfg(target_arch = "x86_64")]
            avx512_available: is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            avx512_available: false,
        }
    }

    pub fn has_avx2(&self) -> bool {
        self.avx2_available
    }

    pub fn has_avx512(&self) -> bool {
        self.avx512_available
    }

    /// Safe Z-score computation with fallback
    pub fn compute_zscores(&self, values: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
        if self.avx2_available && values.len() >= 8 {
            // Process in chunks of 8
            let mut result = Vec::with_capacity(values.len());
            
            unsafe {
                let mut chunk = [0.0; 8];
                let mut i = 0;
                
                while i + 8 <= values.len() {
                    chunk.copy_from_slice(&values[i..i+8]);
                    let zscores = compute_zscores_avx(&chunk, mean, std_dev);
                    result.extend_from_slice(&zscores);
                    i += 8;
                }
                
                // Handle remainder
                for &v in values.iter().skip(i) {
                    result.push((v - mean) / std_dev);
                }
            }
            
            result
        } else {
            // Scalar fallback
            values.iter().map(|&v| (v - mean) / std_dev).collect()
        }
    }

    /// Safe max finding with fallback
    pub fn find_max(&self, values: &[f64]) -> f64 {
        if self.avx2_available && values.len() >= 4 {
            unsafe {
                find_max_avx(values)
            }
        } else {
            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    }

    /// Safe min finding with fallback
    pub fn find_min(&self, values: &[f64]) -> f64 {
        if self.avx2_available && values.len() >= 4 {
            unsafe {
                find_min_avx(values)
            }
        } else {
            values.iter().cloned().fold(f64::INFINITY, f64::min)
        }
    }
}

impl Default for Avx2Accelerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerator_detection() {
        let accelerator = Avx2Accelerator::new();
        
        #[cfg(target_arch = "x86_64")]
        {
            println!("AVX2 available: {}", accelerator.has_avx2());
            println!("AVX-512 available: {}", accelerator.has_avx512());
        }
    }

    #[test]
    fn test_zscores_scalar_fallback() {
        let accelerator = Avx2Accelerator::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std_dev = 1.414;

        let zscores = accelerator.compute_zscores(&values, mean, std_dev);
        
        assert_eq!(zscores.len(), 5);
        // Z-score of mean should be ~0
        assert!(zscores[2].abs() < 0.01);
    }

    #[test]
    fn test_find_max() {
        let accelerator = Avx2Accelerator::new();
        let values = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        
        let max = accelerator.find_max(&values);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_find_min() {
        let accelerator = Avx2Accelerator::new();
        let values = vec![5.0, 2.0, 8.0, 1.0, 9.0];
        
        let min = accelerator.find_min(&values);
        assert_eq!(min, 1.0);
    }
}
