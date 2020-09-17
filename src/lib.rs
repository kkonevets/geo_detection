#[macro_use]
extern crate diesel;
extern crate dotenv;
extern crate num;
extern crate rayon;
// extern crate num_iter;

pub mod extsorter;
pub mod lmdb_client;
pub mod messages;
pub mod models;
pub mod preprocessing;
pub mod processing;
pub mod schema;

pub type TYPE = u32;
pub type EdgeReader<T> = ListReader<EdgeItem<T>>;
pub type EdjReader<T> = ListReader<EdjItem<T>>;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diesel::pg::PgConnection;
use diesel::prelude::*;
use dotenv::dotenv;
use extsorter::{ExternalSorter, Sortable};
use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Result, Write};
use std::marker::PhantomData;
use std::path::Path;

pub trait LeBytes:
    num::NumCast
    + num::Zero
    + num::One
    + num::Bounded
    + std::fmt::Display
    + std::fmt::Debug
    + std::ops::Rem
    + std::ops::Add
    + std::ops::AddAssign
    + std::ops::Sub
    + std::str::FromStr
    + PartialEq
    + Default
    + Sized
    + Copy
    + Sync
    + Send
    + Clone
{
    fn read_le_bytes<R: ReadBytesExt>(reader: &mut R) -> Result<Self>;
    fn write_le_bytes<W: WriteBytesExt>(self, writer: &mut W) -> Result<()>;
}

impl LeBytes for u8 {
    fn read_le_bytes<R: ReadBytesExt>(reader: &mut R) -> Result<Self> {
        reader.read_u8()
    }
    fn write_le_bytes<W: WriteBytesExt>(self, writer: &mut W) -> Result<()> {
        writer.write_u8(self)
    }
}

impl LeBytes for i32 {
    fn read_le_bytes<R: ReadBytesExt>(reader: &mut R) -> Result<Self> {
        reader.read_i32::<LittleEndian>()
    }
    fn write_le_bytes<W: WriteBytesExt>(self, writer: &mut W) -> Result<()> {
        writer.write_i32::<LittleEndian>(self)
    }
}

impl LeBytes for u32 {
    fn read_le_bytes<R: ReadBytesExt>(reader: &mut R) -> Result<Self> {
        reader.read_u32::<LittleEndian>()
    }
    fn write_le_bytes<W: WriteBytesExt>(self, writer: &mut W) -> Result<()> {
        writer.write_u32::<LittleEndian>(self)
    }
}

impl LeBytes for f32 {
    fn read_le_bytes<R: ReadBytesExt>(reader: &mut R) -> Result<Self> {
        reader.read_f32::<LittleEndian>()
    }
    fn write_le_bytes<W: WriteBytesExt>(self, writer: &mut W) -> Result<()> {
        writer.write_f32::<LittleEndian>(self)
    }
}

impl LeBytes for i64 {
    fn read_le_bytes<R: ReadBytesExt>(reader: &mut R) -> Result<Self> {
        reader.read_i64::<LittleEndian>()
    }
    fn write_le_bytes<W: WriteBytesExt>(self, writer: &mut W) -> Result<()> {
        writer.write_i64::<LittleEndian>(self)
    }
}

impl LeBytes for u64 {
    fn read_le_bytes<R: ReadBytesExt>(reader: &mut R) -> Result<Self> {
        reader.read_u64::<LittleEndian>()
    }
    fn write_le_bytes<W: WriteBytesExt>(self, writer: &mut W) -> Result<()> {
        writer.write_u64::<LittleEndian>(self)
    }
}

// =======================================================================================

pub trait ListItem: Sized {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()>;
    fn decode<R: Read>(reader: &mut R) -> Option<Self>;
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct EdgeItem<T: LeBytes>(pub T, pub T);

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct EdgeValueItem<T: LeBytes, V: LeBytes>(pub T, pub T, pub V);

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct EdjItem<T: LeBytes> {
    pub k: T,
    pub v: Vec<T>,
}

pub struct ListReader<T: ListItem> {
    reader: BufReader<std::fs::File>,
    phantom: PhantomData<T>,
}

impl<T: ListItem> ListReader<T> {
    pub fn new<P: AsRef<Path>>(fname: P) -> Result<ListReader<T>> {
        Ok(ListReader {
            reader: BufReader::new(File::open(fname)?),
            phantom: PhantomData,
        })
    }
    pub fn with_capacity<P: AsRef<Path>>(capacity: usize, fname: P) -> Result<ListReader<T>> {
        Ok(ListReader {
            reader: BufReader::with_capacity(capacity, File::open(fname)?),
            phantom: PhantomData,
        })
    }
}

impl<T: ListItem> Iterator for ListReader<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        T::decode(&mut self.reader)
    }
}

impl<T: LeBytes> EdjItem<T> {
    pub fn new(k: T, v: Vec<T>) -> EdjItem<T> {
        EdjItem { k: k, v: v }
    }
}

impl<T: Ord + LeBytes> Sortable for EdgeItem<T> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        ListItem::encode(self, writer)?;
        Ok(())
    }
    fn decode<R: Read>(reader: &mut R) -> Option<Self> {
        ListItem::decode(reader)
    }
}

impl<T: LeBytes> ListItem for EdgeItem<T> {
    fn encode<W: Write>(&self, mut writer: &mut W) -> Result<()> {
        self.0.write_le_bytes(&mut writer)?;
        self.1.write_le_bytes(&mut writer)?;
        Ok(())
    }

    fn decode<R: Read>(mut reader: &mut R) -> Option<Self> {
        match T::read_le_bytes(&mut reader) {
            Ok(left) => {
                let right = T::read_le_bytes(&mut reader).unwrap();
                Some(EdgeItem::<T>(left, right))
            }
            Err(_) => None,
        }
    }
}

impl<T: Ord + LeBytes, V: Ord + LeBytes> Sortable for EdgeValueItem<T, V> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        ListItem::encode(self, writer)?;
        Ok(())
    }
    fn decode<R: Read>(reader: &mut R) -> Option<Self> {
        ListItem::decode(reader)
    }
}

impl<T: LeBytes, V: LeBytes> ListItem for EdgeValueItem<T, V> {
    fn encode<W: Write>(&self, mut writer: &mut W) -> Result<()> {
        self.0.write_le_bytes(&mut writer)?;
        self.1.write_le_bytes(&mut writer)?;
        self.2.write_le_bytes(&mut writer)?;
        Ok(())
    }

    fn decode<R: Read>(mut reader: &mut R) -> Option<Self> {
        match T::read_le_bytes(&mut reader) {
            Ok(left) => {
                let right = T::read_le_bytes(&mut reader).unwrap();
                let value = V::read_le_bytes(&mut reader).unwrap();
                Some(EdgeValueItem::<T, V>(left, right, value))
            }
            Err(_) => None,
        }
    }
}

impl<T: num::Integer + LeBytes> ListItem for EdjItem<T> {
    fn encode<W: Write>(&self, writer: &mut W) -> Result<()> {
        cast::<usize, T>(self.v.len()).write_le_bytes(writer)?;
        self.k.write_le_bytes(writer)?;
        for j in &self.v {
            j.write_le_bytes(writer)?;
        }
        Ok(())
    }

    fn decode<R: Read>(mut reader: &mut R) -> Option<Self> {
        match T::read_le_bytes(&mut reader) {
            Ok(len) => {
                let ix = T::read_le_bytes(&mut reader).unwrap();
                let len = cast(len);
                let mut rec = Vec::<T>::with_capacity(len);
                for _ in 0..len {
                    let id = T::read_le_bytes(&mut reader).unwrap();
                    if id != ix {
                        &rec.push(id);
                    }
                }
                Some(EdjItem::new(ix, rec))
            }
            Err(_) => None,
        }
    }
}

pub fn recreate_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = Path::new(path.as_ref());
    if path.exists() {
        fs::remove_dir_all(path)?;
    }
    fs::create_dir(path)
}

pub fn pgsql_connection() -> PgConnection {
    dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url).expect(&format!("Error connecting to {}", database_url))
}

#[allow(dead_code)]
fn people2pgsql() {
    // sort -n -k 1,1 -T . people.csv > people2.csv

    // sudo vi /var/lib/pgsql/12/data/postgresql.conf
    // wal_level = minimal # default is replica
    // max_wal_senders = 0 # default is 10
    // sudo systemctl restart postgresql-12

    // psql -U postgres -d geography

    // \timing
    // TRUNCATE TABLE geography;
    // ALTER TABLE geography
    //   DROP CONSTRAINT geography_pkey;
    // COPY geography(extid,profile_proc_location,profile_proc_city,profile_proc_hometown)
    // FROM '/data/1/data/vk/people.csv' DELIMITER ',' CSV HEADER;
}

#[allow(dead_code)]
fn pgsql2people() {
    // psql -U postgres -d geography
    // COPY geography TO '/data/1/data/company/people.csv' DELIMITER ',' CSV HEADER;
}

#[allow(dead_code)]
pub fn get_progress_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {wide_bar} {percent}% (ETA {eta}) {msg}"),
    );
    // pb.set_draw_delta(len / 100);
    pb
}

pub fn read_vec<T: LeBytes, R: ReadBytesExt>(mut reader: R) -> Result<Vec<T>> {
    let mut ret = vec![];
    while let Ok(el) = T::read_le_bytes(&mut reader) {
        ret.push(el);
    }
    Ok(ret)
}

pub fn write_vec<T: LeBytes, W: WriteBytesExt>(v: &[T], mut writer: W) -> Result<()> {
    for val in v {
        val.write_le_bytes(&mut writer)?;
    }
    Ok(())
}

pub fn read_vec_file<T: LeBytes>(fname: &Path) -> Result<Vec<T>> {
    let reader = BufReader::new(File::open(fname)?);
    read_vec(reader)
}

pub fn write_vec_file<T: LeBytes, P: AsRef<Path>>(v: &[T], fname: P) -> Result<()> {
    let mut out = BufWriter::new(File::create(fname)?);
    write_vec(v, &mut out)?;
    out.flush()
}

fn cast<T: num::NumCast + num::Integer, K: num::NumCast + num::Integer>(x: T) -> K {
    num::cast::<T, K>(x).unwrap()
}

pub fn list_update<T: Copy>(list: &mut Vec<Vec<T>>, left: usize, right: T) {
    match list.get_mut(left) {
        Some(row) => (*row).push(right),
        None => panic!("no such index {}", left),
    }
}

#[allow(dead_code)]
pub fn edjlist_init<T, I>(nnodes: usize, reader: I, directed: bool) -> Vec<Vec<T>>
where
    T: num::Integer + LeBytes,
    I: Iterator<Item = (T, T)>,
{
    let mut elist = vec![Vec::<T>::new(); nnodes];
    for (left, right) in reader {
        if left == right {
            continue;
        }
        list_update(&mut elist, cast::<T, usize>(left), right);
        if !directed {
            list_update(&mut elist, cast::<T, usize>(right), left);
        }
    }
    elist
}

pub fn is_sorted_strict<T>(data: &[T]) -> bool
where
    T: Ord,
{
    data.windows(2).all(|w| w[0] < w[1])
}

pub fn file_size<T>(fname: &Path) -> Result<usize> {
    let metadata = fs::metadata(fname)?;
    let size = (metadata.len() as usize) / (std::mem::size_of::<T>());
    Ok(size)
}
