//! SSE4.2 Fallback Routines for Cross-Platform SIMD Acceleration
//! 
//! Provides SIMD acceleration for environments lacking AVX support.

use std::arch::x86_64::*;

/// Process 2 price levels using SSE4.2 for microprice calculation
#[target_feature(enable = "sse4.2")]
#[inline]
pub unsafe fn compute_microprice_sse(
    bids: [f64; 2],
    asks: [f64; 2],
    bid_sizes: [f64; 2],
    ask_sizes: [f64; 2],
) -> f64 {
    // Load data into SSE registers (2 doubles per register)
    let bid_vec = _mm_loadu_pd(bids.as_ptr());
    let ask_vec = _mm_loadu_pd(asks.as_ptr());
    let bid_size_vec = _mm_loadu_pd(bid_sizes.as_ptr());
    let ask_size_vec = _mm_loadu_pd(ask_sizes.as_ptr());

    // Calculate weighted mid: (bid*bid_size + ask*ask_size) / (bid_size + ask_size)
    let bid_weighted = _mm_mul_pd(bid_vec, bid_size_vec);
    let ask_weighted = _mm_mul_pd(ask_vec, ask_size_vec);
    let numerator = _mm_add_pd(bid_weighted, ask_weighted);

    let denominator = _mm_add_pd(bid_size_vec, ask_size_vec);

    // Divide
    let result = _mm_div_pd(numerator, denominator);

    // Horizontal sum and average
    let mut sums = [0.0f64; 2];
    _mm_storeu_pd(sums.as_mut_ptr(), result);

    (sums[0] + sums[1]) / 2.0
}

/// Compute Z-scores for 4 values using SSE4.2
#[target_feature(enable = "sse4.2")]
#[inline]
pub unsafe fn compute_zscores_sse(values: &[f64; 4], mean: f64, std_dev: f64) -> [f64; 4] {
    let val_vec1 = _mm_loadu_pd(values.as_ptr());
    let val_vec2 = _mm_loadu_pd(values.as_ptr().add(2));

    let mean_vec = _mm_set1_pd(mean);
    let std_vec = _mm_set1_pd(std_dev);

    // Z = (X - mean) / std
    let diff1 = _mm_sub_pd(val_vec1, mean_vec);
    let diff2 = _mm_sub_pd(val_vec2, mean_vec);

    let z1 = _mm_div_pd(diff1, std_vec);
    let z2 = _mm_div_pd(diff2, std_vec);

    let mut result = [0.0f64; 4];
    _mm_storeu_pd(result.as_mut_ptr(), z1);
    _mm_storeu_pd(result.as_mut_ptr().add(2), z2);

    result
}

/// Parallel comparison of order book deltas using SSE4.2
#[target_feature(enable = "sse4.2")]
#[inline]
pub unsafe fn compare_deltas_sse(
    old_prices: [f64; 2],
    new_prices: [f64; 2],
) -> u32 {
    let old_vec = _mm_loadu_pd(old_prices.as_ptr());
    let new_vec = _mm_loadu_pd(new_prices.as_ptr());

    // Compare: returns mask where new > old
    let cmp = _mm_cmpgt_pd(new_vec, old_vec);

    // Extract mask
    let mask = _mm_movemask_pd(cmp) as u32;

    mask
}

/// Sum 4 values using SSE horizontal add
#[target_feature(enable = "sse4.2")]
#[inline]
pub unsafe fn horizontal_sum_sse(values: &[f64; 4]) -> f64 {
    let vec = _mm_loadu_pd(values.as_ptr());
    let vec2 = _mm_loadu_pd(values.as_ptr().add(2));

    let sum = _mm_add_pd(vec, vec2);

    // Shuffle and add
    let shuffled = _mm_shuffle_pd(sum, sum, 0x1);
    let final_sum = _mm_add_pd(sum, shuffled);

    let mut result = [0.0f64; 2];
    _mm_storeu_pd(result.as_mut_ptr(), final_sum);

    result[0] + result[1]
}

/// Find maximum value in array using SSE4.2
#[target_feature(enable = "sse4.2")]
pub unsafe fn find_max_sse(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let len = values.len();
    let mut max_val = f64::NEG_INFINITY;

    // Process 2 at a time
    let mut i = 0;
    if len >= 2 {
        let mut max_vec = _mm_set1_pd(f64::NEG_INFINITY);

        while i + 2 <= len {
            let vec = _mm_loadu_pd(values.as_ptr().add(i));
            max_vec = _mm_max_pd(max_vec, vec);
            i += 2;
        }

        // Extract max from vector
        let mut temps = [0.0f64; 2];
        _mm_storeu_pd(temps.as_mut_ptr(), max_vec);
        max_val = temps[0].max(temps[1]);
    }

    // Handle remainder
    while i < len {
        max_val = max_val.max(values[i]);
        i += 1;
    }

    max_val
}

/// Find minimum value in array using SSE4.2
#[target_feature(enable = "sse4.2")]
pub unsafe fn find_min_sse(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }

    let len = values.len();
    let mut min_val = f64::INFINITY;

    // Process 2 at a time
    let mut i = 0;
    if len >= 2 {
        let mut min_vec = _mm_set1_pd(f64::INFINITY);

        while i + 2 <= len {
            let vec = _mm_loadu_pd(values.as_ptr().add(i));
            min_vec = _mm_min_pd(min_vec, vec);
            i += 2;
        }

        // Extract min from vector
        let mut temps = [0.0f64; 2];
        _mm_storeu_pd(temps.as_mut_ptr(), min_vec);
        min_val = temps[0].min(temps[1]);
    }

    // Handle remainder
    while i < len {
        min_val = min_val.min(values[i]);
        i += 1;
    }

    min_val
}

/// Dot product of two vectors using SSE4.2
#[target_feature(enable = "sse4.2")]
#[inline]
pub unsafe fn dot_product_sse(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut i = 0;

    // Process 2 at a time
    if len >= 2 {
        let mut sum_vec = _mm_setzero_pd();

        while i + 2 <= len {
            let va = _mm_loadu_pd(a.as_ptr().add(i));
            let vb = _mm_loadu_pd(b.as_ptr().add(i));
            let prod = _mm_mul_pd(va, vb);
            sum_vec = _mm_add_pd(sum_vec, prod);
            i += 2;
        }

        // Horizontal sum
        let mut temps = [0.0f64; 2];
        _mm_storeu_pd(temps.as_mut_ptr(), sum_vec);
        sum = temps[0] + temps[1];
    }

    // Handle remainder
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

/// Safe wrapper for SSE4.2 operations with runtime detection
pub struct SseAccelerator {
    sse_available: bool,
    sse42_available: bool,
}

impl SseAccelerator {
    pub fn new() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            sse_available: is_x86_feature_detected!("sse") && is_x86_feature_detected!("sse2"),
            #[cfg(not(target_arch = "x86_64"))]
            sse_available: false,

            #[cfg(target_arch = "x86_64")]
            sse42_available: is_x86_feature_detected!("sse4.2"),
            #[cfg(not(target_arch = "x86_64"))]
            sse42_available: false,
        }
    }

    pub fn has_sse(&self) -> bool {
        self.sse_available
    }

    pub fn has_sse42(&self) -> bool {
        self.sse42_available
    }

    /// Safe Z-score computation with fallback
    pub fn compute_zscores(&self, values: &[f64], mean: f64, std_dev: f64) -> Vec<f64> {
        if self.sse42_available && values.len() >= 4 {
            // Process in chunks of 4
            let mut result = Vec::with_capacity(values.len());
            
            unsafe {
                let mut chunk = [0.0; 4];
                let mut i = 0;
                
                while i + 4 <= values.len() {
                    chunk.copy_from_slice(&values[i..i+4]);
                    let zscores = compute_zscores_sse(&chunk, mean, std_dev);
                    result.extend_from_slice(&zscores);
                    i += 4;
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
        if self.sse42_available && values.len() >= 2 {
            unsafe {
                find_max_sse(values)
            }
        } else {
            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
    }

    /// Safe min finding with fallback
    pub fn find_min(&self, values: &[f64]) -> f64 {
        if self.sse42_available && values.len() >= 2 {
            unsafe {
                find_min_sse(values)
            }
        } else {
            values.iter().cloned().fold(f64::INFINITY, f64::min)
        }
    }

    /// Safe dot product with fallback
    pub fn dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        if self.sse42_available && a.len() >= 2 && b.len() >= 2 {
            unsafe {
                dot_product_sse(a, b)
            }
        } else {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
        }
    }
}

impl Default for SseAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerator_detection() {
        let accelerator = SseAccelerator::new();
        
        #[cfg(target_arch = "x86_64")]
        {
            println!("SSE available: {}", accelerator.has_sse());
            println!("SSE4.2 available: {}", accelerator.has_sse42());
        }
    }

    #[test]
    fn test_zscores_scalar_fallback() {
        let accelerator = SseAccelerator::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std_dev = 1.414;

        let zscores = accelerator.compute_zscores(&values, mean, std_dev);
        
        assert_eq!(zscores.len(), 5);
        assert!(zscores[2].abs() < 0.01);
    }

    #[test]
    fn test_find_max() {
        let accelerator = SseAccelerator::new();
        let values = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        
        let max = accelerator.find_max(&values);
        assert_eq!(max, 9.0);
    }

    #[test]
    fn test_dot_product() {
        let accelerator = SseAccelerator::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let result = accelerator.dot_product(&a, &b);
        assert!((result - 70.0).abs() < 0.001);
    }
}
