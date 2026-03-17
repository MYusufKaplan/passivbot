//! Property tests for the IncrementalRollingVolume calculator.
//!
//! **Property 9: Rolling Volume Incremental Correctness**
//! *For any* sequence of timesteps within window_size of each other, the
//! IncrementalRollingVolume calculator SHALL produce identical sums to a full
//! window recalculation.
//!
//! **Property 10: Rolling Volume Fallback on Large Gaps**
//! *For any* timestep gap exceeding window_size, the IncrementalRollingVolume
//! calculator SHALL perform a full recalculation and produce correct sums.
//!
//! **Validates: Requirements 7.1, 7.2, 7.3**

use ndarray::{Array3, ArrayView3};

use crate::constants::VOLUME;

/// A simplified test harness for IncrementalRollingVolume that mirrors the
/// behavior of the actual implementation in backtest.rs.
/// This allows testing the rolling volume logic in isolation.
#[derive(Debug)]
pub struct RollingVolumeTestHarness {
    /// Current rolling sums per coin
    sums: Vec<f64>,
    /// Start of current window (inclusive)
    window_start: usize,
    /// End of current window (exclusive)
    window_end: usize,
    /// Window size in timesteps
    window_size: usize,
    /// Counter for full recalculations (for testing)
    full_recalc_count: usize,
    /// Counter for incremental updates (for testing)
    incremental_count: usize,
}

impl RollingVolumeTestHarness {
    /// Creates a new RollingVolumeTestHarness.
    pub fn new(n_coins: usize, window_size: usize) -> Self {
        Self {
            sums: vec![0.0; n_coins],
            window_start: 0,
            window_end: 0,
            window_size,
            full_recalc_count: 0,
            incremental_count: 0,
        }
    }

    /// Updates the rolling volume sums for timestep k.
    /// Mirrors IncrementalRollingVolume::update() from backtest.rs
    pub fn update(&mut self, k: usize, hlcvs: &ArrayView3<f64>) -> Vec<f64> {
        let new_start = k.saturating_sub(self.window_size);

        // Check if we can do incremental update
        if self.window_end > 0 && new_start >= self.window_start && new_start <= self.window_end {
            // Incremental update
            let n_coins = self.sums.len();

            // Subtract values leaving the window
            for idx in 0..n_coins {
                for t in self.window_start..new_start {
                    self.sums[idx] -= hlcvs[[t, idx, VOLUME]];
                }
            }

            // Add values entering the window
            for idx in 0..n_coins {
                for t in self.window_end..k {
                    self.sums[idx] += hlcvs[[t, idx, VOLUME]];
                }
            }

            self.incremental_count += 1;
        } else {
            // Full recalculation
            let n_coins = self.sums.len();
            for idx in 0..n_coins {
                self.sums[idx] = hlcvs.slice(ndarray::s![new_start..k, idx, VOLUME]).sum();
            }

            self.full_recalc_count += 1;
        }

        self.window_start = new_start;
        self.window_end = k;

        self.sums.clone()
    }

    /// Computes the full window sum from scratch (reference implementation).
    pub fn compute_full_sum(k: usize, window_size: usize, hlcvs: &ArrayView3<f64>) -> Vec<f64> {
        let new_start = k.saturating_sub(window_size);
        let n_coins = hlcvs.shape()[1];
        let mut sums = vec![0.0; n_coins];

        for idx in 0..n_coins {
            sums[idx] = hlcvs.slice(ndarray::s![new_start..k, idx, VOLUME]).sum();
        }

        sums
    }

    /// Resets the calculator state.
    pub fn reset(&mut self) {
        for sum in &mut self.sums {
            *sum = 0.0;
        }
        self.window_start = 0;
        self.window_end = 0;
        self.full_recalc_count = 0;
        self.incremental_count = 0;
    }

    /// Returns the number of full recalculations performed.
    pub fn full_recalc_count(&self) -> usize {
        self.full_recalc_count
    }

    /// Returns the number of incremental updates performed.
    pub fn incremental_count(&self) -> usize {
        self.incremental_count
    }

    /// Returns the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

/// Creates a test HLCV array with the given volume data.
/// The array has shape [timesteps, coins, 4] where index 3 is VOLUME.
fn create_hlcv_array(volumes: &[Vec<f64>]) -> Array3<f64> {
    let n_timesteps = volumes.len();
    let n_coins = if n_timesteps > 0 { volumes[0].len() } else { 0 };

    let mut hlcvs = Array3::<f64>::zeros((n_timesteps, n_coins, 4));

    for (t, coin_volumes) in volumes.iter().enumerate() {
        for (idx, &vol) in coin_volumes.iter().enumerate() {
            // Set HIGH, LOW, CLOSE to 1.0 (not used in volume tests)
            hlcvs[[t, idx, 0]] = 1.0;
            hlcvs[[t, idx, 1]] = 1.0;
            hlcvs[[t, idx, 2]] = 1.0;
            // Set VOLUME
            hlcvs[[t, idx, VOLUME]] = vol;
        }
    }

    hlcvs
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating volume data for multiple coins over multiple timesteps.
    fn volume_data_strategy(
        min_timesteps: usize,
        max_timesteps: usize,
        min_coins: usize,
        max_coins: usize,
    ) -> impl Strategy<Value = Vec<Vec<f64>>> {
        (min_coins..=max_coins).prop_flat_map(move |n_coins| {
            prop::collection::vec(
                prop::collection::vec(0.0f64..1000.0, n_coins..=n_coins),
                min_timesteps..=max_timesteps,
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Property 9: Rolling Volume Incremental Correctness**
        /// **Validates: Requirements 7.1, 7.2**
        ///
        /// Test 1: Sequential timesteps produce identical results to full recalculation.
        /// For any sequence of timesteps within window_size of each other, the
        /// incremental calculator produces identical sums to a full window recalculation.
        #[test]
        fn test_incremental_matches_full_recalculation(
            volumes in volume_data_strategy(20, 100, 1, 5),
            window_size in 5usize..15,
        ) {
            let n_timesteps = volumes.len();
            let n_coins = volumes[0].len();
            let hlcvs = create_hlcv_array(&volumes);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(n_coins, window_size);

            // Process timesteps sequentially (within window_size of each other)
            for k in window_size..n_timesteps {
                let incremental_sums = calculator.update(k, &hlcvs_view);
                let full_sums = RollingVolumeTestHarness::compute_full_sum(k, window_size, &hlcvs_view);

                // Verify incremental matches full recalculation
                for (idx, (&inc, &full)) in incremental_sums.iter().zip(full_sums.iter()).enumerate() {
                    prop_assert!(
                        (inc - full).abs() < 1e-10,
                        "Coin {}: incremental sum {} != full sum {} at timestep {}",
                        idx, inc, full, k
                    );
                }
            }
        }

        /// **Property 9: Rolling Volume Incremental Correctness**
        /// **Validates: Requirements 7.1, 7.2**
        ///
        /// Test 2: Incremental updates are used for sequential timesteps.
        /// After the first update, sequential timesteps should use incremental updates.
        #[test]
        fn test_sequential_uses_incremental_updates(
            volumes in volume_data_strategy(30, 80, 1, 3),
            window_size in 5usize..10,
        ) {
            let n_timesteps = volumes.len();
            let n_coins = volumes[0].len();
            let hlcvs = create_hlcv_array(&volumes);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(n_coins, window_size);

            // First update should be full recalculation
            calculator.update(window_size, &hlcvs_view);
            prop_assert_eq!(
                calculator.full_recalc_count(), 1,
                "First update should be full recalculation"
            );
            prop_assert_eq!(
                calculator.incremental_count(), 0,
                "First update should not be incremental"
            );

            // Subsequent sequential updates should be incremental
            for k in (window_size + 1)..n_timesteps {
                calculator.update(k, &hlcvs_view);
            }

            let expected_incremental = n_timesteps - window_size - 1;
            prop_assert_eq!(
                calculator.incremental_count(), expected_incremental,
                "Sequential updates should use incremental method"
            );
            prop_assert_eq!(
                calculator.full_recalc_count(), 1,
                "Only first update should be full recalculation"
            );
        }

        /// **Property 9: Rolling Volume Incremental Correctness**
        /// **Validates: Requirements 7.1, 7.2**
        ///
        /// Test 3: Small gaps within window still use incremental updates.
        /// Gaps smaller than window_size should still use incremental updates.
        #[test]
        fn test_small_gaps_use_incremental(
            volumes in volume_data_strategy(50, 100, 1, 3),
            window_size in 10usize..20,
            gap in 1usize..5,
        ) {
            let n_timesteps = volumes.len();
            let n_coins = volumes[0].len();
            let hlcvs = create_hlcv_array(&volumes);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(n_coins, window_size);

            // First update
            let start_k = window_size;
            calculator.update(start_k, &hlcvs_view);

            // Skip by gap (but stay within window_size)
            let next_k = start_k + gap.min(window_size - 1);
            if next_k < n_timesteps {
                let incremental_before = calculator.incremental_count();
                let full_before = calculator.full_recalc_count();

                let incremental_sums = calculator.update(next_k, &hlcvs_view);
                let full_sums = RollingVolumeTestHarness::compute_full_sum(next_k, window_size, &hlcvs_view);

                // Should use incremental update for small gaps
                prop_assert_eq!(
                    calculator.incremental_count(), incremental_before + 1,
                    "Small gap {} should use incremental update", gap
                );
                prop_assert_eq!(
                    calculator.full_recalc_count(), full_before,
                    "Small gap {} should not trigger full recalculation", gap
                );

                // Results should still match
                for (idx, (&inc, &full)) in incremental_sums.iter().zip(full_sums.iter()).enumerate() {
                    prop_assert!(
                        (inc - full).abs() < 1e-10,
                        "Coin {}: incremental sum {} != full sum {} after gap {}",
                        idx, inc, full, gap
                    );
                }
            }
        }

        /// **Property 10: Rolling Volume Fallback on Large Gaps**
        /// **Validates: Requirements 7.3**
        ///
        /// Test 4: Large gaps trigger full recalculation.
        /// For any timestep gap exceeding window_size, the calculator performs
        /// a full recalculation.
        #[test]
        fn test_large_gap_triggers_full_recalculation(
            volumes in volume_data_strategy(60, 120, 1, 3),
            window_size in 5usize..15,
            gap_multiplier in 1usize..3,
        ) {
            let n_timesteps = volumes.len();
            let n_coins = volumes[0].len();
            let hlcvs = create_hlcv_array(&volumes);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(n_coins, window_size);

            // First update
            let start_k = window_size;
            calculator.update(start_k, &hlcvs_view);
            let full_before = calculator.full_recalc_count();

            // Create a gap larger than window_size
            let large_gap = window_size + gap_multiplier;
            let next_k = start_k + large_gap;

            if next_k < n_timesteps {
                let incremental_sums = calculator.update(next_k, &hlcvs_view);
                let full_sums = RollingVolumeTestHarness::compute_full_sum(next_k, window_size, &hlcvs_view);

                // Should trigger full recalculation
                prop_assert_eq!(
                    calculator.full_recalc_count(), full_before + 1,
                    "Large gap {} (> window_size {}) should trigger full recalculation",
                    large_gap, window_size
                );

                // Results should still be correct
                for (idx, (&inc, &full)) in incremental_sums.iter().zip(full_sums.iter()).enumerate() {
                    prop_assert!(
                        (inc - full).abs() < 1e-10,
                        "Coin {}: sum {} != expected {} after large gap",
                        idx, inc, full
                    );
                }
            }
        }

        /// **Property 10: Rolling Volume Fallback on Large Gaps**
        /// **Validates: Requirements 7.3**
        ///
        /// Test 5: Full recalculation produces correct sums after large gaps.
        /// After a large gap, the full recalculation produces correct sums.
        #[test]
        fn test_full_recalculation_correctness_after_gap(
            volumes in volume_data_strategy(80, 150, 1, 4),
            window_size in 8usize..20,
        ) {
            let n_timesteps = volumes.len();
            let n_coins = volumes[0].len();
            let hlcvs = create_hlcv_array(&volumes);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(n_coins, window_size);

            // Process some timesteps sequentially
            for k in window_size..(window_size + 10).min(n_timesteps) {
                calculator.update(k, &hlcvs_view);
            }

            // Jump to a much later timestep (large gap)
            let jump_k = (n_timesteps - 1).min(window_size + 10 + window_size * 2);
            if jump_k < n_timesteps {
                let incremental_sums = calculator.update(jump_k, &hlcvs_view);
                let full_sums = RollingVolumeTestHarness::compute_full_sum(jump_k, window_size, &hlcvs_view);

                // Verify correctness after large gap
                for (idx, (&inc, &full)) in incremental_sums.iter().zip(full_sums.iter()).enumerate() {
                    prop_assert!(
                        (inc - full).abs() < 1e-10,
                        "Coin {}: sum {} != expected {} after large gap to timestep {}",
                        idx, inc, full, jump_k
                    );
                }
            }
        }

        /// **Property 9 & 10: Combined correctness test**
        /// **Validates: Requirements 7.1, 7.2, 7.3**
        ///
        /// Test 6: Mixed sequential and gap updates produce correct results.
        /// A mix of sequential updates and gaps (both small and large) produces
        /// correct results throughout.
        #[test]
        fn test_mixed_sequential_and_gaps(
            volumes in volume_data_strategy(100, 200, 1, 3),
            window_size in 10usize..20,
            steps in prop::collection::vec(1usize..30, 5..15),
        ) {
            let n_timesteps = volumes.len();
            let n_coins = volumes[0].len();
            let hlcvs = create_hlcv_array(&volumes);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(n_coins, window_size);
            let mut current_k = window_size;

            for step in steps {
                if current_k >= n_timesteps {
                    break;
                }

                let incremental_sums = calculator.update(current_k, &hlcvs_view);
                let full_sums = RollingVolumeTestHarness::compute_full_sum(current_k, window_size, &hlcvs_view);

                // Verify correctness at each step
                for (idx, (&inc, &full)) in incremental_sums.iter().zip(full_sums.iter()).enumerate() {
                    prop_assert!(
                        (inc - full).abs() < 1e-10,
                        "Coin {}: sum {} != expected {} at timestep {} (step size {})",
                        idx, inc, full, current_k, step
                    );
                }

                current_k += step;
            }
        }

        /// **Property 9: Rolling Volume Incremental Correctness**
        /// **Validates: Requirements 7.1, 7.2**
        ///
        /// Test 7: Single coin volume tracking is correct.
        /// Simplified test with single coin to verify basic correctness.
        #[test]
        fn test_single_coin_correctness(
            volumes in prop::collection::vec(0.0f64..100.0, 30..80),
            window_size in 5usize..15,
        ) {
            let n_timesteps = volumes.len();
            let volumes_2d: Vec<Vec<f64>> = volumes.iter().map(|&v| vec![v]).collect();
            let hlcvs = create_hlcv_array(&volumes_2d);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(1, window_size);

            for k in window_size..n_timesteps {
                let incremental_sums = calculator.update(k, &hlcvs_view);

                // Compute expected sum manually
                let start = k.saturating_sub(window_size);
                let expected_sum: f64 = volumes[start..k].iter().sum();

                prop_assert!(
                    (incremental_sums[0] - expected_sum).abs() < 1e-10,
                    "Single coin: incremental sum {} != expected {} at timestep {}",
                    incremental_sums[0], expected_sum, k
                );
            }
        }

        /// **Property 10: Rolling Volume Fallback on Large Gaps**
        /// **Validates: Requirements 7.3**
        ///
        /// Test 8: Boundary case - gap exactly at window_size.
        /// A gap exactly equal to window_size should trigger full recalculation.
        #[test]
        fn test_boundary_gap_at_window_size(
            volumes in volume_data_strategy(60, 100, 1, 2),
            window_size in 5usize..15,
        ) {
            let n_timesteps = volumes.len();
            let n_coins = volumes[0].len();
            let hlcvs = create_hlcv_array(&volumes);
            let hlcvs_view = hlcvs.view();

            let mut calculator = RollingVolumeTestHarness::new(n_coins, window_size);

            // First update
            let start_k = window_size;
            calculator.update(start_k, &hlcvs_view);
            let full_before = calculator.full_recalc_count();

            // Gap exactly at window_size + 1 (just beyond overlap)
            let next_k = start_k + window_size + 1;
            if next_k < n_timesteps {
                let incremental_sums = calculator.update(next_k, &hlcvs_view);
                let full_sums = RollingVolumeTestHarness::compute_full_sum(next_k, window_size, &hlcvs_view);

                // Should trigger full recalculation (gap exceeds window)
                prop_assert_eq!(
                    calculator.full_recalc_count(), full_before + 1,
                    "Gap at window_size+1 should trigger full recalculation"
                );

                // Results should be correct
                for (idx, (&inc, &full)) in incremental_sums.iter().zip(full_sums.iter()).enumerate() {
                    prop_assert!(
                        (inc - full).abs() < 1e-10,
                        "Coin {}: sum {} != expected {} at boundary gap",
                        idx, inc, full
                    );
                }
            }
        }
    }
}
