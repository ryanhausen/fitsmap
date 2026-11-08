use pyo3::prelude::*;

#[pyfunction]
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}


/// Calculates the total number of tiles for a given range of zoom levels.
///
/// # Examples
/// ```rust
/// use _tiling::get_total_tiles;
///
/// let total = get_total_tiles(0, 2);
/// assert_eq!(total, Some(21));
/// ```
pub fn get_total_tiles(min_zoom: u8, max_zoom: u8) -> Option<u64> {
    // The sum in this expression can't over flow because the maximum value of
    // `max_zoom` is 31 and the sum to that point is still less than u64::MAX.
    (min_zoom..=max_zoom)
        .map(|z| 4u64.checked_pow(z.into()))
        .sum()
}

#[pymodule]
mod _tiling {

    #[pymodule_export]
    use super::add;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_get_total_tiles() {
        let total = get_total_tiles(0, 2);
        assert_eq!(total, Some(21));
    }

    #[test]
    fn test_get_total_tiles_max() {
        let total = get_total_tiles(0, 31);
        assert_eq!(total, Some(6148914691236517205));
    }

    #[test]
    fn test_get_total_tiles_zooms_equal() {
        let total = get_total_tiles(3, 3);
        assert_eq!(total, Some(64));
    }

    #[test]
    fn test_get_total_tiles_min_gt_max() {
        let total = get_total_tiles(3, 2);
        assert_eq!(total, Some(0));
    }

    #[test]
    fn test_get_total_tiles_overflows() {
        let total = get_total_tiles(0, 32);
        assert_eq!(total, None);
    }

    #[test]
    fn test_get_total_tiles_overflows_z_not_zero() {
        let total = get_total_tiles(31, 32);
        assert_eq!(total, None);
    }
}
