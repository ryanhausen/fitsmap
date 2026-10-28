use pyo3::prelude::*;

#[pyfunction]
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub fn get_total_tiles(min_zoom: u8, max_zoom: u8) -> Option<u64> {
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
    fn test_get_total_tiles_overflows() {
        let total = get_total_tiles(0, 32);
        assert_eq!(total, None);
    }
}
