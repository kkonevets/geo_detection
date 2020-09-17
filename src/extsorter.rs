use itertools::{Itertools, KMerge};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Result, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

pub trait Sortable: Eq + Ord + Sized + Send + Sync {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()>;
    fn decode<R: Read>(reader: &mut R) -> Option<Self>;
}

pub struct ExternalSorter {
    max_mem: usize,
    tempdir: PathBuf,
    file_count: usize,
}

impl ExternalSorter {
    pub fn new(tempdir: PathBuf, max_mem: usize) -> ExternalSorter {
        assert!(max_mem > 0);
        ExternalSorter {
            max_mem,
            tempdir,
            file_count: 0,
        }
    }

    fn file_name(&self, n: usize) -> PathBuf {
        self.tempdir.join(format!("{}", n))
    }

    fn write_chunk<T: Sortable>(&mut self, buf: &mut Vec<T>) -> Result<()> {
        let file = File::create(self.file_name(self.file_count))?;
        let mut out = BufWriter::new(file);
        for item in buf.iter() {
            item.encode(&mut out)?;
        }
        out.flush()?;
        self.file_count += 1;
        Ok(())
    }

    /// Sort a given iterator, returning a new iterator with items
    pub fn sort_unstable_by_key<T, I, K>(
        &mut self,
        iterator: I,
        f: fn(&T) -> K,
    ) -> Result<KMerge<<ChunkReader<T> as IntoIterator>::IntoIter>>
    where
        T: Sortable,
        I: Iterator<Item = T>,
        K: Ord,
    {
        let max_size = self.max_mem / std::mem::size_of::<T>() as usize;
        let mut buf = Vec::<T>::with_capacity(max_size);
        for item in iterator {
            if buf.len() == max_size {
                buf.par_sort_unstable_by_key(f);
                self.write_chunk(&mut buf)?;
                buf.clear();
            }
            buf.push(item);
        }
        if !buf.is_empty() {
            buf.par_sort_unstable_by_key(f);
            self.write_chunk(&mut buf)?;
        }
        std::mem::drop(buf);
        let mut readers = vec![];
        for n in 0..self.file_count {
            readers.push(ChunkReader::new(self.file_name(n))?);
        }
        Ok(readers.into_iter().kmerge())
    }
}

pub struct ChunkReader<T: Sortable> {
    reader: BufReader<std::fs::File>,
    phantom: PhantomData<T>,
}

impl<T: Sortable> ChunkReader<T> {
    pub fn new<P: AsRef<Path>>(fname: P) -> Result<ChunkReader<T>> {
        Ok(ChunkReader {
            reader: BufReader::with_capacity(2usize.pow(30), File::open(fname)?),
            phantom: PhantomData,
        })
    }
}

impl<T: Sortable> Iterator for ChunkReader<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        T::decode(&mut self.reader)
    }
}
