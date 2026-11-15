//! Core ML-owned, typed multidimensional tensors.
use crate::model::core_ml;
use crate::{DataType, Element, Error, TensorData};
use block2::StackBlock;
use objc2::{AllocAnyThread, rc::Retained, runtime::AnyObject};
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureValue, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSInteger, NSNumber, NSString};
use std::{cell::RefCell, collections::BTreeMap, ffi::c_void, marker::PhantomData, ptr::NonNull};

/// Immutable typed tensor retaining Core ML storage. Cloning retains the
/// same array; passing it to another function does not materialize host data.
#[derive(Debug)]
pub struct Tensor<T: Element> {
    pub(crate) array: Retained<MLMultiArray>,
    shape: Box<[usize]>,
    element: PhantomData<T>,
}
impl<T: Element> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            shape: self.shape.clone(),
            element: PhantomData,
        }
    }
}
impl<T: Element> Tensor<T> {
    /// Copy row-major values into a new native tensor of an explicit shape.
    pub fn from_slice(shape: &[usize], values: &[T]) -> Result<Self, Error> {
        let count = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        if shape.is_empty() || shape.contains(&0) || count != Some(values.len()) {
            return Err(Error::TensorShape {
                shape: shape.to_vec(),
                elements: values.len(),
            });
        }
        let dims = NSArray::from_retained_slice(
            &shape
                .iter()
                .map(|&d| NSNumber::new_usize(d))
                .collect::<Vec<_>>(),
        );
        // SAFETY: dimensions are nonzero and their product matches the values.
        let array = unsafe {
            MLMultiArray::initWithShape_dataType_error(
                MLMultiArray::alloc(),
                &dims,
                ml_type(T::DATA_TYPE),
            )
        }
        .map_err(core_ml)?;
        let fill = StackBlock::new(
            |ptr: NonNull<c_void>, _: NSInteger, strides: NonNull<NSArray<NSNumber>>| {
                // SAFETY: Core ML lends these strides for this writable buffer;
                // requesting mutable access can replace the backing storage.
                let strides: Vec<usize> = unsafe { strides.as_ref() }
                    .iter()
                    .map(|n| n.as_usize())
                    .collect();
                let base = ptr.as_ptr().cast::<T>();
                for (offset, &value) in offsets(shape, &strides).zip(values) {
                    // SAFETY: offsets follow the shape and current buffer strides.
                    unsafe { base.add(offset).write(value) };
                }
            },
        );
        // SAFETY: the handler writes only valid elements and retains no pointer.
        unsafe { array.getMutableBytesWithHandler(&fill) };
        Ok(Self {
            array,
            shape: shape.into(),
            element: PhantomData,
        })
    }
    pub(crate) fn from_array(array: Retained<MLMultiArray>, name: &str) -> Result<Self, Error> {
        // SAFETY: array is retained from a Core ML feature value.
        let actual = unsafe { array.dataType() };
        if actual != ml_type(T::DATA_TYPE) {
            return Err(Error::OutputType {
                name: name.into(),
                expected: T::DATA_TYPE,
                actual,
            });
        }
        // SAFETY: shape belongs to the retained array.
        let shape = unsafe { array.shape() }
            .iter()
            .map(|n| n.as_usize())
            .collect();
        Ok(Self {
            array,
            shape,
            element: PhantomData,
        })
    }
    /// The logical shape, independent of storage padding.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Explicitly read the tensor into row-major host storage, respecting strides.
    pub fn read(&self) -> TensorData<T> {
        // SAFETY: the array stays retained through the synchronous handler.
        let strides: Vec<isize> = unsafe { self.array.strides() }
            .iter()
            .map(|n| n.as_isize())
            .collect();
        let bytes = RefCell::new(Vec::new());
        let copy = StackBlock::new(|ptr: NonNull<c_void>, size: NSInteger| {
            // SAFETY: Core ML lends exactly size bytes for this handler.
            let data =
                unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<u8>(), size as usize) };
            bytes.borrow_mut().extend_from_slice(data);
        });
        // SAFETY: copy reads the lent storage without retaining a pointer.
        unsafe { self.array.getBytesWithHandler(&copy) };
        TensorData::gather(&bytes.into_inner(), &self.shape, &strides)
    }
}
fn offsets<'a>(shape: &'a [usize], strides: &'a [usize]) -> impl Iterator<Item = usize> + 'a {
    (0..shape.iter().product()).map(move |mut flat| {
        let mut offset = 0;
        for (&dim, &stride) in shape.iter().zip(strides).rev() {
            offset += (flat % dim) * stride;
            flat /= dim;
        }
        offset
    })
}
pub(crate) fn ml_type(dtype: DataType) -> MLMultiArrayDataType {
    match dtype {
        DataType::Int32 => MLMultiArrayDataType::Int32,
        DataType::Float16 => MLMultiArrayDataType::Float16,
        DataType::Float32 => MLMultiArrayDataType::Float32,
    }
}
/// Named tensor inputs. Different features may have different shapes and types.
#[derive(Default)]
pub struct Inputs {
    values: BTreeMap<String, Retained<MLFeatureValue>>,
}
impl Inputs {
    pub fn new() -> Self {
        Self::default()
    }
    /// Retain a native tensor as an input, without copying its elements.
    pub fn insert<T: Element>(
        &mut self,
        name: impl Into<String>,
        tensor: &Tensor<T>,
    ) -> Result<(), Error> {
        let name = name.into();
        if self.values.contains_key(&name) {
            return Err(Error::DuplicateInput(name));
        }
        // SAFETY: featureValue retains the live immutable array.
        self.values.insert(name, unsafe {
            MLFeatureValue::featureValueWithMultiArray(&tensor.array)
        });
        Ok(())
    }
    pub(crate) fn provider(&self) -> Result<Retained<MLDictionaryFeatureProvider>, Error> {
        let keys: Vec<_> = self.values.keys().map(|k| NSString::from_str(k)).collect();
        let keys: Vec<_> = keys.iter().map(|k| &**k).collect();
        let values: Vec<&AnyObject> = self.values.values().map(|v| v.as_ref()).collect();
        let dictionary = NSDictionary::from_slices(&keys, &values);
        // SAFETY: every key is a string and every value an MLFeatureValue.
        unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dictionary,
            )
        }
        .map_err(core_ml)
    }
}
