extern crate rayon;

use crate::{read_vec, recreate_dir, write_vec, write_vec_file, EdjReader, LeBytes, TYPE};
use lmdb::{
    Cursor, Database, DatabaseFlags, Environment, EnvironmentFlags, RoTransaction, Transaction,
    WriteFlags,
};
// use rayon::prelude::*;

use std::io::Result;
use std::path::Path;
use thousands::Separable;

pub struct GraphReader<'env, I>
where
    I: Iterator<Item = TYPE>,
{
    db: Database,
    pub txn: &'env RoTransaction<'env>,
    ids: I,
}

impl<'env, I> GraphReader<'env, I>
where
    I: Iterator<Item = TYPE>,
{
    pub fn new(db: Database, txn: &'env RoTransaction, ids: I) -> GraphReader<'env, I> {
        GraphReader {
            db: db,
            txn: txn,
            ids: ids,
        }
    }
    fn get(&self, id: TYPE) -> lmdb::Result<Vec<TYPE>> {
        self.txn
            .get(self.db, &id.to_le_bytes())
            .map(|blob| read_vec(blob).unwrap())
    }
}

impl<'env, I> Iterator for GraphReader<'env, I>
where
    I: Iterator<Item = TYPE>,
{
    type Item = (TYPE, (Vec<TYPE>, Option<Vec<TYPE>>));
    fn next(&mut self) -> Option<(TYPE, (Vec<TYPE>, Option<Vec<TYPE>>))> {
        self.ids.next().map(|ix| {
            (
                ix,
                match self.get(ix) {
                    Ok(v) => (v, None),
                    Err(e) => panic!("{}: could't find {} index", e, ix),
                },
            )
        })
    }
}

pub fn get_environment<P: AsRef<Path>>(
    path: P,
    flags: EnvironmentFlags,
) -> lmdb::Result<Environment> {
    let mut builder = Environment::new();
    builder.set_max_dbs(10).set_map_size(2usize.pow(40)); /* 1Tb */
    builder.set_flags(flags);
    builder.open(path.as_ref())
}

pub fn to_embeded_db() -> Result<()> {
    let path = "../data/vk/graph_db";
    recreate_dir(path)?;

    let flags = EnvironmentFlags::empty();
    let env = get_environment(path, flags).unwrap();

    let mut db_flags = DatabaseFlags::empty();
    db_flags.set(DatabaseFlags::INTEGER_KEY, true);
    let db = env.create_db(Some("social_net"), db_flags).unwrap();
    let mut txn = env.begin_rw_txn().unwrap();
    {
        let mut writer = txn.open_rw_cursor(db).unwrap();

        let mut w_flags = WriteFlags::empty();
        // Bulk loading. No key comparisons are performed.
        // Loading unsorted keys with this flag will cause data corruption
        w_flags.set(WriteFlags::APPEND, true);

        for mut item in EdjReader::<TYPE>::new("../data/vk/edjlist.bin")? {
            item.v.sort_unstable();
            item.v.dedup();
            let mut buffer = vec![];
            write_vec(&item.v, &mut buffer)?;
            if let Err(e) = writer.put(&item.k.to_le_bytes(), &buffer, w_flags) {
                panic!("{}: {}", e, item.k);
            }
            if item.k as usize % 50000 == 0 {
                eprint!("\r{}", item.k.separate_with_spaces());
            }
        }
        eprintln!();
    }
    txn.commit().unwrap();
    Ok(())
}

// extract degrees of all nodes and save them
pub fn save_degrees() -> Result<()> {
    let mut flags = EnvironmentFlags::empty();
    flags.insert(EnvironmentFlags::READ_ONLY);
    let env = get_environment("../data/vk/graph_db", flags).unwrap();
    let db = env.open_db(Some("social_net")).unwrap();
    let txn = env.begin_ro_txn().unwrap();

    let mut degrees = vec![0; TYPE::max_value() as usize];
    let mut i_max = 0;
    let mut nedges = 0;
    let type_size = std::mem::size_of::<TYPE>();
    {
        let mut cursor = txn.open_ro_cursor(db).unwrap();
        for it in cursor.iter() {
            match it {
                Ok((mut k, v)) => {
                    let deg = v.len() / type_size;
                    let i = TYPE::read_le_bytes(&mut k)? as usize;
                    if i > i_max {
                        i_max = i;
                    }
                    degrees[i] = deg as TYPE;
                    nedges += deg;
                }
                Err(e) => panic!("{:?}", e),
            }
        }
    }

    println!("nedges {:?}", nedges);

    // prune
    degrees.resize(i_max + 1, 0);
    write_vec_file(&degrees, "../data/vk/degrees.bin")?;
    Ok(())
}
