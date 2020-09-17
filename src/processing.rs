use crate::lmdb_client::{get_environment, GraphReader};
use crate::{
    file_size, get_progress_bar, read_vec, read_vec_file, recreate_dir, write_vec_file, EdgeItem,
    EdgeReader, LeBytes, ListItem, TYPE,
};

use lmdb::{EnvironmentFlags, Transaction};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{prelude::*, BufReader, BufWriter, Result, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use thousands::Separable;

fn load_geomap(save_dir: &Path) -> Result<(usize, Vec<Vec<usize>>)> {
    let reader = BufReader::new(File::open(&save_dir.join("geomap.csv"))?);
    let mut geomap = vec![];
    for line in reader.lines() {
        geomap.push(line?.split(',').map(|s| s.parse().unwrap()).collect());
    }
    let fin = File::open(&save_dir.join("geography_splited.csv"))?;
    let nfeatures = BufReader::new(fin).lines().count();

    Ok((nfeatures, geomap))
}

// if user has connections without cities, then neighb counts would be zeros
// or not directly relevant for current user city
pub fn degrees_with_cities<'a, I, T>(reader: I, directed: bool, base_dir: &Path) -> Result<()>
where
    I: Iterator<Item = (TYPE, (T, Option<T>))>,
    T: AsRef<[TYPE]>,
{
    println!("degrees_with_cities");
    let is_city = read_vec_file::<u8>(&base_dir.join("is_city.bin"))?;
    let nnodes = read_vec_file::<TYPE>(&base_dir.join("nodes.bin"))?
        .into_iter()
        .max()
        .unwrap() as usize
        + 1;
    let mut degrees = vec![0f32; nnodes];

    fn count_directed_nodes(nodes: &[TYPE], is_city: &[u8]) -> f32 {
        let mut ncities = 0.;
        for &node in nodes {
            if is_city[node as usize] == 1 {
                ncities += 1.;
            }
        }
        ncities
    }

    for (i, (left_nodes, right_nodes)) in reader {
        let mut ncities = count_directed_nodes(left_nodes.as_ref(), &is_city);
        if directed {
            ncities += count_directed_nodes(right_nodes.unwrap().as_ref(), &is_city);
            ncities /= 2.;
        }
        degrees[i as usize] = ncities;
    }

    write_vec_file(&degrees, &base_dir.join("degrees_with_cities.bin"))?;
    Ok(())
}

pub fn save_neighbor_ftrs_dyn<'a, I, T>(
    reader: I,
    membership: &[i32],
    directed: bool,
    save_dir: &Path,
) -> Result<()>
where
    I: Iterator<Item = (TYPE, (T, Option<T>))>,
    T: AsRef<[TYPE]>,
{
    let (nfeatures, geomap) = load_geomap(&save_dir.parent().unwrap())?;

    // saving matrix in csr format
    let mut writers = vec![];
    for fi in ["data", "indptr", "indices"].iter() {
        let file = File::create(&save_dir.join(&format!("neib_ftrs_{}.bin", fi)))?;
        writers.push(BufWriter::with_capacity(10usize.pow(8), file)); /* 100Mb */
    }

    #[derive(Debug)]
    struct Info<'a> {
        membership: &'a [i32],
        geomap: &'a [Vec<usize>],
        nfeatures: usize,
    };

    let info = Info {
        membership,
        geomap: &geomap,
        nfeatures,
    };

    fn count_directed_nodes(nodes: &[TYPE], counts: &mut [f32], direction: usize, info: &Info) {
        let offset = direction * info.nfeatures;
        for &node in nodes {
            let l = info.membership[node as usize];
            if l > -1 {
                for &j in &info.geomap[l as usize] {
                    counts[offset + j] += 1.;
                }
            }
        }
    }

    let total_features = match directed {
        true => 2 * nfeatures,
        false => nfeatures,
    };
    let mut prev_i = 0;
    let mut ix = 0;
    let mut counts: Vec<f32> = vec![0.; total_features];
    ix.write_le_bytes(&mut writers[1])?;

    println!("saving features {}", save_dir.to_str().unwrap());
    for (i, (left_nodes, right_nodes)) in reader {
        count_directed_nodes(left_nodes.as_ref(), &mut counts, 0, &info);
        if directed {
            count_directed_nodes(right_nodes.unwrap().as_ref(), &mut counts, 1, &info);
        }

        for (j, c) in counts.iter().enumerate().filter(|&(_, &c)| c > 0.) {
            c.write_le_bytes(&mut writers[0])?;
            (j as TYPE).write_le_bytes(&mut writers[2])?;
            for _ in 0..i - prev_i {
                ix.write_le_bytes(&mut writers[1])?;
            }
            prev_i = i;
            ix += 1;
        }
        counts.clear();
        counts.resize(total_features, 0.);

        if i % 50000 == 0 {
            eprint!("\r{}", i.separate_with_spaces());
        }
    }
    for _ in 0..membership.len() - prev_i as usize {
        ix.write_le_bytes(&mut writers[1])?;
    }

    for mut out in writers {
        out.flush()?;
    }
    print!("\n");
    Ok(())
}

pub fn save_neighbor_ftrs(membership: &[i32], sub_nodes: Vec<TYPE>, save_dir: &Path) -> Result<()> {
    let mut flags = EnvironmentFlags::empty();
    flags.insert(EnvironmentFlags::READ_ONLY);
    let env = get_environment("../data/vk/graph_db", flags).unwrap();
    let db = env.open_db(Some("social_net")).unwrap();
    let txn = env.begin_ro_txn().unwrap();

    let fin = File::open("../data/vk/geography_splited.csv")?;
    let nfeatures = BufReader::new(fin).lines().count();
    println!("nfeatures {}", nfeatures);
    let reader = GraphReader::new(db, &txn, sub_nodes.iter().cloned());
    save_neighbor_ftrs_dyn(reader, membership, false, save_dir)?;
    Ok(())
}

// Computes and saves in-degree for target node i using edgelist (j, i) sorted by i
// edgelist should't have duplicate values
pub fn edgelist_degree<I, P: AsRef<Path>>(edgelist: I, nnodes: usize, fname: P) -> Result<()>
where
    I: Iterator<Item = EdgeItem<TYPE>>,
{
    let mut out = BufWriter::new(File::create(fname)?);
    let mut degree: TYPE = 0;
    let mut i_prev = 0;
    for e in edgelist {
        let i = e.1;
        for k in i_prev..i {
            if k == i_prev {
                degree.write_le_bytes(&mut out)?;
                degree = 0;
            } else {
                (0 as TYPE).write_le_bytes(&mut out)?;
            }
        }

        degree += 1;
        i_prev = i;
    }
    if degree > 0 {
        degree.write_le_bytes(&mut out)?;
    }
    for _ in i_prev as usize + 1..nnodes {
        (0 as TYPE).write_le_bytes(&mut out)?;
    }
    out.flush()?;
    Ok(())
}

/// Count number of common nodes
/// Asuming is and js are sorted ascending without duplicates.
#[allow(dead_code)]
fn common_count<I>(iit: &mut I, jit: &mut I) -> TYPE
where
    I: Iterator<Item = TYPE>,
{
    let mut i = iit.next();
    let mut j = jit.next();
    let mut count = 0;

    loop {
        match (i, j) {
            (Some(i_val), Some(j_val)) => {
                if i_val < j_val {
                    i = iit.next();
                } else if i_val > j_val {
                    j = jit.next();
                } else {
                    count += 1;
                    i = iit.next();
                    j = jit.next();
                }
            }
            _ => break,
        }
    }
    count
}

#[allow(dead_code)]
fn select_neibs(
    start_nodes: &[TYPE],
    nsample: &[usize],
    mut out: &mut BufWriter<fs::File>,
    db: lmdb::Database,
    txn: &lmdb::RoTransaction,
    mut rng: &mut rand::rngs::ThreadRng,
) -> Result<()> {
    for &i in start_nodes.iter() {
        let mut counts = vec![];

        let vi: Vec<TYPE> = match &txn.get(db, &i.to_le_bytes()) {
            Ok(blob) => read_vec(*blob)?,
            Err(e) => panic!("LMDB {:?}, {}", e, i),
        };

        let mut vjs = vec![];
        for (ix, &j) in vi.iter().enumerate() {
            let (vj, c);
            if i == j {
                vj = vec![];
                c = 0;
            } else {
                vj = match &txn.get(db, &j.to_le_bytes()) {
                    Ok(blob) => read_vec(*blob)?,
                    Err(e) => panic!("LMDB {:?}, {}", e, j),
                };
                c = common_count(&mut vi.iter().cloned(), &mut vj.iter().cloned());
            }
            counts.push((c, ix));
            vjs.push(vj);
        }

        // sort by counts descending
        counts.sort_unstable_by_key(|e| e.0);

        let nj = nsample[0];
        let nk = nsample[1];
        for &(_, ix) in counts.iter().rev().take(nj) {
            let j = vi[ix];
            //if source->target: j=source, i=target !!!
            if i != j {
                EdgeItem(j, i).encode(&mut out)?;
            }
            let vj = &vjs[ix];
            for &k in vj.choose_multiple(&mut rng, nk) {
                if j != k {
                    EdgeItem(k, j).encode(&mut out)?;
                }
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn walk_depth(
    i: TYPE,
    nsample: &[usize],
    mut out: &mut BufWriter<fs::File>,
    db: lmdb::Database,
    txn: &lmdb::RoTransaction,
    mut rng: &mut rand::rngs::ThreadRng,
) -> Result<()> {
    let vi = match &txn.get(db, &i.to_le_bytes()) {
        Ok(blob) => read_vec(*blob)?,
        Err(e) => panic!("LMDB {:?}, {}", e, i),
    };
    let amount = *nsample.first().unwrap() + 1;
    for &j in vi.choose_multiple(&mut rng, amount) {
        //if source->target: j=source, i=target !!!
        if i != j {
            EdgeItem(j, i).encode(&mut out)?;
        }
        if nsample.len() > 1 {
            walk_depth(j, &nsample[1..], &mut out, db, &txn, &mut rng)?;
        }
    }
    Ok(())
}

pub fn subgraphs(
    chunks: Vec<&[TYPE]>,
    save_dir: PathBuf,
    nsample: &[usize],
) -> Result<Vec<PathBuf>> {
    let dirs = Arc::new(RwLock::new(vec![]));

    let mut flags = EnvironmentFlags::empty();
    flags.insert(EnvironmentFlags::READ_ONLY);
    flags.insert(EnvironmentFlags::NO_READAHEAD); // good for random access
    let env = get_environment("../data/vk/graph_db", flags).unwrap();
    let db = env.open_db(Some("social_net")).unwrap();

    let pb = get_progress_bar(chunks.len() as u64);

    chunks
        .par_iter()
        .enumerate()
        .for_each(|(part_n, start_nodes)| {
            let mut rng = &mut rand::thread_rng();

            let txn = env.begin_ro_txn().unwrap();

            let path = save_dir.join(format!("part{}", part_n));
            recreate_dir(&path).unwrap();
            dirs.write().unwrap().push(path.clone());
            let colrow_path = path.join("colrow.bin");
            write_vec_file(start_nodes, path.join("nodes.bin")).unwrap();

            let out = File::create(colrow_path).unwrap();
            let mut out = BufWriter::with_capacity(10usize.pow(8), out);

            for &i in start_nodes.iter() {
                walk_depth(i, nsample, &mut out, db, &txn, &mut rng).unwrap();
            }
            pb.inc(1);
            out.flush().unwrap();
        });

    pb.finish();

    let ret = dirs.read().unwrap().to_vec();
    Ok(ret)
}

fn unite_parts(save_dir: &Path, part_paths: &Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    let out_dir = save_dir.join("part_all");
    fs::create_dir(out_dir.as_path())?;
    let mut out = File::create(out_dir.join("colrow.bin"))?;

    for in_dir in part_paths.iter() {
        let mut buffer = Vec::new();
        let mut fin = File::open(in_dir.join("colrow.bin"))?;
        fin.read_to_end(&mut buffer)?;
        out.write(&mut buffer)?;
        fs::remove_dir_all(in_dir)?;
    }
    out.flush()?;
    Ok(vec![out_dir])
}

pub fn save_subgraph_edge_index(nsample: &[usize], nparts: usize, train: bool) -> Result<()> {
    let mut start_nodes: Vec<TYPE>;
    let save_dir;
    if train {
        save_dir = Path::new("../data/vk/train");
        start_nodes = read_vec_file(Path::new("../data/vk/train_index.bin"))?;
        start_nodes.append(&mut read_vec_file(Path::new("../data/vk/test_index.bin"))?);
    } else {
        save_dir = Path::new("../data/vk/predict");
        start_nodes = read_vec_file(Path::new("../data/vk/predict_index.bin"))?;
    }
    start_nodes.sort_unstable();
    let chunk_size = start_nodes.len() / nparts + 1;
    recreate_dir(save_dir)?;
    let nodes_total = read_vec_file::<TYPE>(Path::new("../data/vk/nodes.bin"))?
        .into_iter()
        .max()
        .unwrap() as usize
        + 1;

    let mut part_paths;
    {
        println!("building subgraphs");
        let chunks: Vec<_> = start_nodes.chunks(chunk_size).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(chunks.len())
            .build()
            .unwrap();
        part_paths = pool.install(|| subgraphs(chunks, save_dir.to_path_buf(), nsample).unwrap());
        if train {
            part_paths = unite_parts(save_dir, &part_paths)?;
        }
    }

    let mut is_subnode: Vec<bool> = vec![false; nodes_total];

    println!("processing colrows");
    let pb = get_progress_bar(part_paths.len() as u64);
    for dir_name in part_paths.iter() {
        let colrow_path = dir_name.join("colrow.bin");
        let mut edges = Vec::<EdgeItem<TYPE>>::with_capacity(file_size::<TYPE>(&colrow_path)? / 2);
        for e in EdgeReader::new(&colrow_path)? {
            edges.push(e); // e = (j,i)
        }
        edges.par_sort_unstable_by_key(|e| e.1); // (j,i) should be sorted by i !!!
        edges.dedup();
        edges.shrink_to_fit();

        let out = File::create(&colrow_path)?;
        let mut out = BufWriter::with_capacity(10usize.pow(8), out);
        for e in edges {
            e.encode(&mut out)?;
            is_subnode[e.0 as usize] = true;
            is_subnode[e.1 as usize] = true;
        }
        out.flush()?;

        // edgelist_degree(
        //     EdgeReader::new(&colrow_path)?,
        //     nodes_total,
        //     &dir_name.join("degrees.bin"),
        // )?;
        pb.inc(1);
    }
    pb.finish();

    let mut sub_nodes = vec![];
    for (i, _) in is_subnode.iter().enumerate().filter(|e| *e.1) {
        sub_nodes.push(i as TYPE);
    }
    write_vec_file(&sub_nodes, &save_dir.join("sub_graph_nodes.bin"))?;

    let membership: Vec<i32> = read_vec_file(Path::new("../data/vk/labels.bin"))?;
    save_neighbor_ftrs(&membership, sub_nodes, &save_dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn common_count_test() {
        use super::common_count;

        let iit = &mut [2, 5, 7, 9, 33, 78].iter().cloned();
        let jit = &mut [5, 9, 12, 15, 33, 66, 70, 78, 100, 101].iter().cloned();
        assert_eq!(common_count(iit, jit), 4);
    }
}
