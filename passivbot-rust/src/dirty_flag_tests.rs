//! Property tests for the dirty flag mechanism used in position key caching.
//!
//! **Property 8: Position Key Dirty Flag Mechanism**
//! *For any* position change (add or remove), the Rust Backtest engine SHALL mark
//! the corresponding key buffer as dirty, and the next access SHALL trigger a
//! re-sort only if the dirty flag is set.
//!
//! **Validates: Requirements 5.3**

use std::collections::HashMap;

/// A simplified test harness that isolates the dirty flag mechanism.
/// This mirrors the behavior of the Backtest struct's position key caching
/// without requiring the full HLCV data initialization.
#[derive(Debug)]
pub struct DirtyFlagTestHarness {
    /// Simulated long positions (coin_idx -> exists)
    positions_long: HashMap<usize, bool>,
    /// Simulated short positions (coin_idx -> exists)
    positions_short: HashMap<usize, bool>,
    /// Pre-allocated buffer for sorted long keys
    long_keys_buffer: Vec<usize>,
    /// Pre-allocated buffer for sorted short keys
    short_keys_buffer: Vec<usize>,
    /// Dirty flag for long keys
    long_keys_dirty: bool,
    /// Dirty flag for short keys
    short_keys_dirty: bool,
    /// Counter for how many times long keys were recomputed
    long_recompute_count: usize,
    /// Counter for how many times short keys were recomputed
    short_recompute_count: usize,
}

impl DirtyFlagTestHarness {
    pub fn new() -> Self {
        Self {
            positions_long: HashMap::new(),
            positions_short: HashMap::new(),
            long_keys_buffer: Vec::new(),
            short_keys_buffer: Vec::new(),
            long_keys_dirty: true, // Start dirty to force initial population
            short_keys_dirty: true,
            long_recompute_count: 0,
            short_recompute_count: 0,
        }
    }

    /// Returns cached sorted long position keys, recomputing only when dirty.
    /// This mirrors Backtest::get_sorted_long_keys()
    pub fn get_sorted_long_keys(&mut self) -> Vec<usize> {
        if self.long_keys_dirty {
            self.long_keys_buffer.clear();
            self.long_keys_buffer.extend(self.positions_long.keys());
            self.long_keys_buffer.sort_unstable();
            self.long_keys_dirty = false;
            self.long_recompute_count += 1;
        }
        self.long_keys_buffer.clone()
    }

    /// Returns cached sorted short position keys, recomputing only when dirty.
    /// This mirrors Backtest::get_sorted_short_keys()
    pub fn get_sorted_short_keys(&mut self) -> Vec<usize> {
        if self.short_keys_dirty {
            self.short_keys_buffer.clear();
            self.short_keys_buffer.extend(self.positions_short.keys());
            self.short_keys_buffer.sort_unstable();
            self.short_keys_dirty = false;
            self.short_recompute_count += 1;
        }
        self.short_keys_buffer.clone()
    }

    /// Marks the long keys buffer as dirty.
    pub fn mark_long_keys_dirty(&mut self) {
        self.long_keys_dirty = true;
    }

    /// Marks the short keys buffer as dirty.
    pub fn mark_short_keys_dirty(&mut self) {
        self.short_keys_dirty = true;
    }

    /// Add a long position (simulates entry fill)
    pub fn add_long_position(&mut self, idx: usize) {
        let is_new = !self.positions_long.contains_key(&idx);
        self.positions_long.insert(idx, true);
        if is_new {
            self.mark_long_keys_dirty();
        }
    }

    /// Remove a long position (simulates close fill)
    pub fn remove_long_position(&mut self, idx: usize) {
        if self.positions_long.remove(&idx).is_some() {
            self.mark_long_keys_dirty();
        }
    }

    /// Add a short position (simulates entry fill)
    pub fn add_short_position(&mut self, idx: usize) {
        let is_new = !self.positions_short.contains_key(&idx);
        self.positions_short.insert(idx, true);
        if is_new {
            self.mark_short_keys_dirty();
        }
    }

    /// Remove a short position (simulates close fill)
    pub fn remove_short_position(&mut self, idx: usize) {
        if self.positions_short.remove(&idx).is_some() {
            self.mark_short_keys_dirty();
        }
    }

    /// Get the current recompute count for long keys
    pub fn long_recompute_count(&self) -> usize {
        self.long_recompute_count
    }

    /// Get the current recompute count for short keys
    pub fn short_recompute_count(&self) -> usize {
        self.short_recompute_count
    }

    /// Check if long keys are currently dirty
    pub fn is_long_dirty(&self) -> bool {
        self.long_keys_dirty
    }

    /// Check if short keys are currently dirty
    pub fn is_short_dirty(&self) -> bool {
        self.short_keys_dirty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating position operations
    #[derive(Debug, Clone)]
    enum PositionOp {
        AddLong(usize),
        RemoveLong(usize),
        AddShort(usize),
        RemoveShort(usize),
        GetLongKeys,
        GetShortKeys,
    }

    fn position_op_strategy() -> impl Strategy<Value = PositionOp> {
        prop_oneof![
            (0usize..100).prop_map(PositionOp::AddLong),
            (0usize..100).prop_map(PositionOp::RemoveLong),
            (0usize..100).prop_map(PositionOp::AddShort),
            (0usize..100).prop_map(PositionOp::RemoveShort),
            Just(PositionOp::GetLongKeys),
            Just(PositionOp::GetShortKeys),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Property 8: Position Key Dirty Flag Mechanism**
        /// **Validates: Requirements 5.3**
        ///
        /// Test 1: After marking dirty, the next get_sorted_*_keys() call triggers a re-sort.
        /// For any position change (add or remove), the dirty flag is set, and the next
        /// access triggers a re-sort.
        #[test]
        fn test_dirty_flag_triggers_resort_on_position_change(
            initial_positions in prop::collection::vec(0usize..50, 0..20),
            position_to_add in 0usize..100,
        ) {
            let mut harness = DirtyFlagTestHarness::new();

            // Add initial positions
            for idx in &initial_positions {
                harness.add_long_position(*idx);
            }

            // First access - should trigger recompute (dirty from adds)
            let _ = harness.get_sorted_long_keys();
            let count_after_first = harness.long_recompute_count();

            // Second access without changes - should NOT trigger recompute
            let _ = harness.get_sorted_long_keys();
            prop_assert_eq!(
                harness.long_recompute_count(),
                count_after_first,
                "Consecutive access without changes should not trigger recompute"
            );

            // Add a new position - should mark dirty
            harness.add_long_position(position_to_add);

            // If position was new, dirty flag should be set
            if !initial_positions.contains(&position_to_add) {
                prop_assert!(
                    harness.is_long_dirty(),
                    "Adding new position should mark keys dirty"
                );

                // Next access should trigger recompute
                let _ = harness.get_sorted_long_keys();
                prop_assert_eq!(
                    harness.long_recompute_count(),
                    count_after_first + 1,
                    "Access after adding position should trigger recompute"
                );
            }
        }

        /// **Property 8: Position Key Dirty Flag Mechanism**
        /// **Validates: Requirements 5.3**
        ///
        /// Test 2: Consecutive get_sorted_*_keys() calls without marking dirty return
        /// the same result without re-sorting.
        #[test]
        fn test_consecutive_access_no_resort(
            positions in prop::collection::vec(0usize..100, 0..30),
        ) {
            let mut harness = DirtyFlagTestHarness::new();

            // Add positions
            for idx in &positions {
                harness.add_long_position(*idx);
            }

            // First access
            let keys1 = harness.get_sorted_long_keys();
            let count1 = harness.long_recompute_count();

            // Multiple consecutive accesses should not trigger recompute
            for _ in 0..5 {
                let keys = harness.get_sorted_long_keys();
                prop_assert_eq!(
                    harness.long_recompute_count(),
                    count1,
                    "Consecutive access should not trigger recompute"
                );
                prop_assert_eq!(
                    keys,
                    keys1.clone(),
                    "Consecutive access should return same keys"
                );
            }
        }

        /// **Property 8: Position Key Dirty Flag Mechanism**
        /// **Validates: Requirements 5.3**
        ///
        /// Test 3: The dirty flag is correctly set when positions are added/removed.
        #[test]
        fn test_dirty_flag_set_on_position_changes(
            ops in prop::collection::vec(position_op_strategy(), 1..50),
        ) {
            let mut harness = DirtyFlagTestHarness::new();
            let mut expected_long_positions: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut expected_short_positions: std::collections::HashSet<usize> = std::collections::HashSet::new();

            for op in ops {
                match op {
                    PositionOp::AddLong(idx) => {
                        let was_new = !expected_long_positions.contains(&idx);
                        harness.add_long_position(idx);
                        expected_long_positions.insert(idx);

                        if was_new {
                            prop_assert!(
                                harness.is_long_dirty(),
                                "Adding new long position {} should mark dirty",
                                idx
                            );
                        }
                    }
                    PositionOp::RemoveLong(idx) => {
                        let existed = expected_long_positions.contains(&idx);
                        harness.remove_long_position(idx);
                        expected_long_positions.remove(&idx);

                        if existed {
                            prop_assert!(
                                harness.is_long_dirty(),
                                "Removing existing long position {} should mark dirty",
                                idx
                            );
                        }
                    }
                    PositionOp::AddShort(idx) => {
                        let was_new = !expected_short_positions.contains(&idx);
                        harness.add_short_position(idx);
                        expected_short_positions.insert(idx);

                        if was_new {
                            prop_assert!(
                                harness.is_short_dirty(),
                                "Adding new short position {} should mark dirty",
                                idx
                            );
                        }
                    }
                    PositionOp::RemoveShort(idx) => {
                        let existed = expected_short_positions.contains(&idx);
                        harness.remove_short_position(idx);
                        expected_short_positions.remove(&idx);

                        if existed {
                            prop_assert!(
                                harness.is_short_dirty(),
                                "Removing existing short position {} should mark dirty",
                                idx
                            );
                        }
                    }
                    PositionOp::GetLongKeys => {
                        let keys = harness.get_sorted_long_keys();
                        // Verify keys match expected positions
                        let mut expected: Vec<usize> = expected_long_positions.iter().cloned().collect();
                        expected.sort_unstable();
                        prop_assert_eq!(
                            keys,
                            expected,
                            "Long keys should match expected positions"
                        );
                        // After access, dirty flag should be cleared
                        prop_assert!(
                            !harness.is_long_dirty(),
                            "Dirty flag should be cleared after access"
                        );
                    }
                    PositionOp::GetShortKeys => {
                        let keys = harness.get_sorted_short_keys();
                        // Verify keys match expected positions
                        let mut expected: Vec<usize> = expected_short_positions.iter().cloned().collect();
                        expected.sort_unstable();
                        prop_assert_eq!(
                            keys,
                            expected,
                            "Short keys should match expected positions"
                        );
                        // After access, dirty flag should be cleared
                        prop_assert!(
                            !harness.is_short_dirty(),
                            "Dirty flag should be cleared after access"
                        );
                    }
                }
            }
        }

        /// **Property 8: Position Key Dirty Flag Mechanism**
        /// **Validates: Requirements 5.3**
        ///
        /// Test 4: Keys are always sorted after access regardless of insertion order.
        #[test]
        fn test_keys_always_sorted(
            positions in prop::collection::vec(0usize..1000, 1..50),
        ) {
            let mut harness = DirtyFlagTestHarness::new();

            // Add positions in arbitrary order
            for idx in &positions {
                harness.add_long_position(*idx);
            }

            let keys = harness.get_sorted_long_keys();

            // Verify keys are sorted
            for i in 1..keys.len() {
                prop_assert!(
                    keys[i - 1] < keys[i],
                    "Keys should be sorted: {} should be < {}",
                    keys[i - 1],
                    keys[i]
                );
            }
        }

        /// **Property 8: Position Key Dirty Flag Mechanism**
        /// **Validates: Requirements 5.3**
        ///
        /// Test 5: Re-adding an existing position does not mark dirty.
        #[test]
        fn test_readd_existing_position_no_dirty(
            positions in prop::collection::vec(0usize..50, 1..20),
        ) {
            let mut harness = DirtyFlagTestHarness::new();

            // Add initial positions
            for idx in &positions {
                harness.add_long_position(*idx);
            }

            // Access to clear dirty flag
            let _ = harness.get_sorted_long_keys();
            prop_assert!(!harness.is_long_dirty(), "Dirty flag should be cleared after access");

            // Re-add existing positions - should NOT mark dirty
            for idx in &positions {
                harness.add_long_position(*idx);
                prop_assert!(
                    !harness.is_long_dirty(),
                    "Re-adding existing position {} should not mark dirty",
                    idx
                );
            }
        }

        /// **Property 8: Position Key Dirty Flag Mechanism**
        /// **Validates: Requirements 5.3**
        ///
        /// Test 6: Removing non-existent position does not mark dirty.
        #[test]
        fn test_remove_nonexistent_no_dirty(
            positions in prop::collection::vec(0usize..50, 0..20),
            nonexistent in 50usize..100,
        ) {
            let mut harness = DirtyFlagTestHarness::new();

            // Add initial positions (all < 50)
            for idx in &positions {
                harness.add_long_position(*idx);
            }

            // Access to clear dirty flag
            let _ = harness.get_sorted_long_keys();
            prop_assert!(!harness.is_long_dirty(), "Dirty flag should be cleared after access");

            // Remove non-existent position (>= 50) - should NOT mark dirty
            harness.remove_long_position(nonexistent);
            prop_assert!(
                !harness.is_long_dirty(),
                "Removing non-existent position {} should not mark dirty",
                nonexistent
            );
        }
    }
}
