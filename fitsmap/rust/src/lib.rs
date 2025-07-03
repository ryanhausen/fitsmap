use pyo3::prelude::*;
use image::{ImageBuffer, Rgba, imageops};
use fitsio::FitsFile;
use rayon::prelude::*;
use std::path::Path;
use anyhow::anyhow;
use ndarray::{s, Array2, ArrayD, Ix2};
use pyo3::exceptions::PyValueError;

// Custom error that can be converted to a PyErr
#[derive(Debug)]
enum MyError {
    Anyhow(anyhow::Error),
    Io(std::io::Error),
}

impl From<anyhow::Error> for MyError {
    fn from(err: anyhow::Error) -> MyError {
        MyError::Anyhow(err)
    }
}

impl From<std::io::Error> for MyError {
    fn from(err: std::io::Error) -> MyError {
        MyError::Io(err)
    }
}

impl From<MyError> for PyErr {
    fn from(err: MyError) -> PyErr {
        match err {
            MyError::Anyhow(e) => PyValueError::new_err(e.to_string()),
            MyError::Io(e) => PyValueError::new_err(e.to_string()),
        }
    }
}

#[pyfunction]
fn tile_img_rust(
    file_location: String,
    out_dir: String,
    min_zoom: u32,
    max_zoom: u32,
    tile_size: u32,
) -> Result<(), MyError> {
    let array = get_array_rust(&file_location)?;

    make_dirs_rust(&out_dir, min_zoom, max_zoom)?;

    for zoom in (min_zoom..=max_zoom).rev() {
        let scale = 2_u32.pow(max_zoom - zoom);
        let scaled_tile_size = tile_size * scale;

        let slice_params = slice_idx_generator_rust(array.dim(), zoom, scaled_tile_size);

        slice_params
            .into_par_iter()
            .for_each(|(z, y, x, (slice_y, slice_x))| {
                let tile_data = array.slice(s![slice_y, slice_x]);
                let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(tile_size, tile_size);

                // This is a simplified version of the tiling logic.
                // A more complete implementation would handle different image types and normalization.
                for ((in_y, in_x), pixel) in tile_data.indexed_iter() {
                    if (in_x as u32) < tile_size && (in_y as u32) < tile_size {
                        let val = (*pixel as f32 / 65535.0 * 255.0) as u8;
                        img.put_pixel(in_x as u32, in_y as u32, Rgba([val, val, val, 255]));
                    }
                }

                let resized_img = imageops::resize(&img, tile_size, tile_size, imageops::FilterType::Lanczos3);

                let path_str = build_path_rust(z, y, x, &out_dir);
                let path = Path::new(&path_str);
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).unwrap();
                }
                resized_img.save(path).unwrap();
            });
    }

    Ok(())
}

fn get_array_rust(file_location: &str) -> Result<Array2<f32>, anyhow::Error> {
    let mut fptr = FitsFile::open(file_location)?;
    let hdu = fptr.primary_hdu()?;
    let array: Vec<f32> = hdu.read_image(&mut fptr)?;
    let naxis1 = hdu.read_key::<i64>(&mut fptr, "NAXIS1")?;
    let naxis2 = hdu.read_key::<i64>(&mut fptr, "NAXIS2")?;
    let shape = (naxis2 as usize, naxis1 as usize);
    Ok(Array2::from_shape_vec(shape, array)?)
}

fn make_dirs_rust(out_dir: &str, min_zoom: u32, max_zoom: u32) -> std::io::Result<()> {
    for z in min_zoom..=max_zoom {
        let num_tiles = 2_u32.pow(z);
        for y in 0..num_tiles {
            let dir = Path::new(out_dir).join(z.to_string()).join(y.to_string());
            std::fs::create_dir_all(dir)?;
        }
    }
    Ok(())
}

fn build_path_rust(z: u32, y: u32, x: u32, out_dir: &str) -> String {
    Path::new(out_dir)
        .join(z.to_string())
        .join(y.to_string())
        .join(format!("{}.png", x))
        .to_str()
        .unwrap()
        .to_string()
}

fn slice_idx_generator_rust(
    shape: (usize, usize),
    zoom: u32,
    tile_size: u32,
) -> Vec<(u32, u32, u32, (std::ops::Range<usize>, std::ops::Range<usize>))> {
    let (height, width) = shape;
    let num_tiles_y = (height as f32 / tile_size as f32).ceil() as u32;
    let num_tiles_x = (width as f32 / tile_size as f32).ceil() as u32;
    let mut params = Vec::new();

    for y in 0..num_tiles_y {
        for x in 0..num_tiles_x {
            let y_start = (y * tile_size) as usize;
            let y_end = ((y + 1) * tile_size) as usize;
            let x_start = (x * tile_size) as usize;
            let x_end = ((x + 1) * tile_size) as usize;

            params.push((
                zoom,
                y,
                x,
                (y_start..y_end.min(height), x_start..x_end.min(width)),
            ));
        }
    }
    params
}

#[pymodule]
fn fitsmap_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tile_img_rust, m)?)?;
    Ok(())
}