use pyo3::prelude::*;

#[pyfunction]
pub fn add(left: u64, right: u64) -> u64 {
    left + right
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
}
