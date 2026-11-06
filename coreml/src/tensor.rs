//! Typed model inputs and outputs.

use std::ffi::c_void;
use std::ptr::NonNull;

use half::f16;

use crate::DataType;

/// An element type a multi-array holds: `i32`, [`half::f16`] or `f32`.
pub trait Element: Copy + Send + Sync + 'static + sealed::Bytes {
    /// The multi-array data type of `Self`.
    const DATA_TYPE: DataType;
}

impl Element for i32 {
    const DATA_TYPE: DataType = DataType::Int32;
}

impl Element for f16 {
    const DATA_TYPE: DataType = DataType::Float16;
}

impl Element for f32 {
    const DATA_TYPE: DataType = DataType::Float32;
}

pub(crate) mod sealed {
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use half::f16;

    use crate::DataType;

    /// An element's native-endian bytes.
    pub trait Bytes: Sized {
        /// The element whose native-endian bytes start `bytes`, which holds
        /// at least `size_of::<Self>()` of them.
        fn read(bytes: &[u8]) -> Self;
    }

    impl Bytes for i32 {
        fn read(bytes: &[u8]) -> Self {
            Self::from_ne_bytes(bytes[..4].try_into().expect("four bytes"))
        }
    }

    impl Bytes for f16 {
        fn read(bytes: &[u8]) -> Self {
            Self::from_ne_bytes(bytes[..2].try_into().expect("two bytes"))
        }
    }

    impl Bytes for f32 {
        fn read(bytes: &[u8]) -> Self {
            Self::from_ne_bytes(bytes[..4].try_into().expect("four bytes"))
        }
    }

    /// What the model needs of an input to build its multi-array.
    pub trait Fill {
        /// The feature name.
        fn name(&self) -> &str;
        /// The element type.
        fn data_type(&self) -> DataType;
        /// The sequence length before padding.
        fn len(&self) -> usize;
        /// Writes the values, then padding up to `padded` elements.
        ///
        /// # Safety
        ///
        /// `base` points to storage for `padded` elements of
        /// [`data_type`](Self::data_type), `stride` elements apart, writable
        /// for the call; `padded` is at least [`len`](Self::len).
        unsafe fn fill(&self, base: NonNull<c_void>, stride: usize, padded: usize);
    }
}

/// A model input: something the model pads to a bucket and feeds by name.
/// Implemented by [`Sequence`].
pub trait Input: sealed::Fill {}

/// A sequence input: `values` under the feature `name`, followed by `pad`
/// up to the length of the bucket that runs.
#[derive(Debug, Clone, Copy)]
pub struct Sequence<'a, T: Element> {
    /// The feature name the manifest declares.
    pub name: &'a str,
    /// The values before padding.
    pub values: &'a [T],
    /// The padding value.
    pub pad: T,
}

impl<T: Element> sealed::Fill for Sequence<'_, T> {
    fn name(&self) -> &str {
        self.name
    }

    fn data_type(&self) -> DataType {
        T::DATA_TYPE
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    unsafe fn fill(&self, base: NonNull<c_void>, stride: usize, padded: usize) {
        let base = base.as_ptr().cast::<T>();
        // SAFETY: the caller guarantees `padded` elements of `T`, `stride`
        // apart, from `base`; `values.len() <= padded`, so every index below
        // is in bounds and each element is written exactly once.
        unsafe {
            for (i, &value) in self.values.iter().enumerate() {
                base.add(i * stride).write(value);
            }
            for i in self.values.len()..padded {
                base.add(i * stride).write(self.pad);
            }
        }
    }
}

impl<T: Element> Input for Sequence<'_, T> {}

/// A model output, copied out of its multi-array in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    shape: Box<[usize]>,
    data: Vec<T>,
}

impl<T: Element> Tensor<T> {
    /// Gathers the elements of a multi-array of `shape` and `strides`
    /// (in elements) out of its raw storage `bytes`.
    ///
    /// # Panics
    ///
    /// When `shape` and `strides` address an element outside `bytes`: Core
    /// ML described its own array wrongly.
    pub(crate) fn gather(bytes: &[u8], shape: &[usize], strides: &[isize]) -> Self {
        let width = size_of::<T>();
        let count = shape.iter().product::<usize>();
        let mut data = Vec::with_capacity(count);
        let mut index = vec![0usize; shape.len()];
        for _ in 0..count {
            let offset: isize = index
                .iter()
                .zip(strides)
                .map(|(&i, &stride)| i as isize * stride)
                .sum();
            let start =
                usize::try_from(offset).expect("a multi-array offset is non-negative") * width;
            data.push(T::read(&bytes[start..start + width]));
            // Row-major odometer: bump the last axis, carry leftwards.
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        Self {
            shape: shape.into(),
            data,
        }
    }

    /// The shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The elements, row-major.
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// The elements, row-major.
    #[must_use]
    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    fn bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    #[test]
    fn contiguous_storage_gathers_in_order() {
        let storage = bytes(&[1, 2, 3, 4, 5, 6]);
        let tensor = Tensor::<i32>::gather(&storage, &[2, 3], &[3, 1]);
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(tensor.data(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn padded_rows_skip_their_padding() {
        // Two rows of three, each padded to four elements.
        let storage = bytes(&[1, 2, 3, -1, 4, 5, 6, -1]);
        let tensor = Tensor::<i32>::gather(&storage, &[2, 3], &[4, 1]);
        assert_eq!(tensor.data(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn transposed_strides_read_column_major_storage() {
        // A 2×3 array stored column-major.
        let storage = bytes(&[1, 4, 2, 5, 3, 6]);
        let tensor = Tensor::<i32>::gather(&storage, &[2, 3], &[1, 2]);
        assert_eq!(tensor.data(), [1, 2, 3, 4, 5, 6]);
    }
}
