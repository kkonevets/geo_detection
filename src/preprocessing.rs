extern crate sys_info;

use crate::{
    file_size, processing::save_neighbor_ftrs_dyn, read_vec_file, recreate_dir, write_vec,
    write_vec_file, EdgeItem, EdgeReader, EdgeValueItem, EdjItem, EdjReader, ExternalSorter,
    LeBytes, ListItem, TYPE,
};

use bit_set::BitSet;
use itertools::Itertools;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::{prelude::*, BufReader, BufWriter, Result, Write};
use std::path::{Path, PathBuf};
use thousands::Separable;

pub fn connections_csv2bin<P: AsRef<Path>>(fin: P, fout: P) -> Result<()> {
    let reader = BufReader::with_capacity(10usize.pow(9), File::open(fin)?); // 1Gb
    let mut out = BufWriter::with_capacity(10usize.pow(9), File::create(fout)?);
    let lines = reader.lines().skip(1); // skip header
    let mut i_prev = None;
    for line in lines {
        let line = line?;
        let splited: Vec<&str> = line.split(',').collect();
        let i: TYPE = splited[0].parse().unwrap();
        let mut js: Vec<TYPE> = splited[1].split(' ').map(|s| s.parse().unwrap()).collect();
        js.sort_unstable();
        js.dedup();
        EdjItem::new(i, js).encode(&mut out)?;
        assert!(Some(i) != i_prev, "duplicate {}", i);
        i_prev = Some(i);
    }
    out.flush()?;
    Ok(())
}

pub fn save_nodes(nodes: &mut BitSet) -> Result<()> {
    let mut nodes: Vec<TYPE> = nodes.iter().map(|x| x as TYPE).collect();
    nodes.sort_unstable();
    write_vec_file(&nodes, "../data/vk/nodes.bin")?;
    println!("{} nodes", nodes.len().separate_with_spaces());
    Ok(())
}

pub fn build_edjlist() -> Result<()> {
    let tempdir = "../data/vk/temp";
    recreate_dir(tempdir)?;

    let mut nodes = BitSet::new();
    let fconn = "../data/vk/connections.bin";
    let reader = EdjReader::<TYPE>::new(fconn)?;
    println!("\nreading {}", fconn);

    let input = reader
        .filter(|el| !el.v.is_empty())
        .enumerate()
        .map(|(linen, item)| {
            let mut buf = Vec::<_>::with_capacity(2 * item.v.len());
            nodes.insert(item.k as usize);
            for &j in item.v.iter().filter(|&j| &item.k != j) {
                nodes.insert(j as usize);
                // make edges undirected
                buf.push(EdgeItem(item.k, j));
                buf.push(EdgeItem(j, item.k));
            }
            if linen % 50000 == 0 {
                eprint!("\r{}", linen.separate_with_spaces());
            }
            buf
        })
        .flatten();

    let mut sorter = ExternalSorter::new(PathBuf::from(tempdir), 45 * 2usize.pow(30)); // 45Gb
    let sorted_iter = sorter
        .sort_unstable_by_key(input, |e| (e.0, e.1))?
        .dedup()
        .group_by(|e| e.0);

    println!("\nsaving nodes.bin");
    save_nodes(&mut nodes)?;

    let mut out = BufWriter::new(File::create("../data/vk/edjlist.bin")?);
    for (k, group) in sorted_iter.into_iter() {
        let v = group.map(|e| e.1).collect();
        EdjItem::new(k, v).encode(&mut out)?;
    }
    out.flush()?;
    fs::remove_dir_all(tempdir)?;

    Ok(())
}

//===================================================================

pub mod directed {
    use super::*;

    pub fn save_nodes<P>(base_dir: &PathBuf, fedjlists: &[P]) -> Result<()>
    where
        P: std::fmt::Debug + AsRef<Path>,
    {
        let fout = &base_dir.join("nodes.bin");
        println!("saving {}", fout.to_str().unwrap());
        let mut set = HashSet::<u64>::new();
        for fin in fedjlists {
            for item in EdjReader::<u64>::new(fin)? {
                set.insert(item.k);
                for vi in item.v {
                    set.insert(vi);
                }
            }
        }

        let mut nodes: Vec<u64> = set.iter().cloned().collect();
        nodes.sort_unstable();
        println!("{:?} nodes", nodes.len());

        write_vec_file(&nodes, fout)?;
        Ok(())
    }

    pub fn unite_graphs<P: AsRef<Path>>(
        base_dir: &PathBuf,
        fedjlists: &[P],
        fweights: &[P],
    ) -> Result<()> {
        println!("unite_graphs");
        let nodes = read_vec_file::<u64>(&base_dir.join("nodes.bin"))?;
        let mut node2ix = HashMap::<u64, i32>::default();
        for (i, node) in nodes.into_iter().enumerate() {
            node2ix.insert(node, i32::try_from(i).unwrap());
        }

        let mut lists = vec![];
        let cl = |(i, item): (usize, EdjItem<u64>)| {
            (
                i,
                EdjItem::new(
                    *node2ix.get(&item.k).unwrap(),
                    item.v.iter().map(|x| *node2ix.get(x).unwrap()).collect(),
                ),
            )
        };

        for (i, fname) in fedjlists.iter().enumerate() {
            let it = EdjReader::<u64>::new(fname)?
                .map(move |item| (i, item))
                .map(cl);
            lists.push(it);
        }
        let mut weights = vec![];
        for fname in fweights {
            weights.push(BufReader::new(File::open(fname)?));
        }

        let mut edgelist_out = BufWriter::new(File::create(&base_dir.join("edgelist.bin"))?);
        let mut weights_out = BufWriter::new(File::create(&base_dir.join("weights.bin"))?);

        let n_edge_types = fedjlists.len();
        let mut ws_buff = vec![0f32; n_edge_types];
        let mut wj_sums = vec![0f32; n_edge_types];
        for (k, group_iter) in lists
            .into_iter()
            .kmerge_by(|a, b| a.1.k < b.1.k)
            .filter(|item| item.1.v.len() > 0)
            .group_by(|item| item.1.k)
            .into_iter()
        {
            let mut edge_info = vec![];
            for (i, item) in group_iter {
                let wit = &mut weights[i];
                let wjsi = &mut wj_sums[i];
                for vj in item.v {
                    let wj = f32::read_le_bytes(wit)?;
                    edge_info.push((vj, wj, i));
                    *wjsi += wj;
                }
            }

            // sort by vj
            edge_info.sort_unstable_by_key(|e| e.0);

            for (vj, group) in edge_info.into_iter().group_by(|e| e.0).into_iter() {
                for (_, wj, i) in group {
                    let wjsi = wj_sums[i];
                    if wjsi == 0. {
                        ws_buff[i] = 0.;
                    } else {
                        ws_buff[i] = wj / wjsi; // normalize incoming weights
                    }
                }
                EdgeItem(vj, k).encode(&mut edgelist_out)?;
                write_vec(&ws_buff, &mut weights_out)?;
                ws_buff.clear();
                ws_buff.resize(n_edge_types, 0.);
            }
            wj_sums.clear();
            wj_sums.resize(n_edge_types, 0.);
        }

        edgelist_out.flush()?;
        weights_out.flush()?;

        Ok(())
    }

    pub fn double_edge_index(base_dir: &PathBuf, wsize: usize) -> Result<()> {
        println!("double_edge_index",);
        let edgelist = base_dir.join("edgelist.bin");
        let ereader = EdgeReader::<i32>::new(&edgelist)?;

        let tempdir = &base_dir.join("temp");
        recreate_dir(tempdir)?;

        let input = ereader
            .enumerate()
            .map(|(i, e)| {
                vec![
                    EdgeValueItem(e.0, e.1, i64::try_from(i).unwrap()),
                    EdgeValueItem(e.1, e.0, -1),
                ]
            })
            .flatten();

        let mut sorter = ExternalSorter::new(PathBuf::from(tempdir), 45 * 2usize.pow(30)); // 45Gb
        let sorted_iter = sorter.sort_unstable_by_key(input, |e| (e.1, e.0))?;

        let weights = read_vec_file::<f32>(&base_dir.join("weights.bin"))?;

        let fedgelist_double = base_dir.join("colrow.bin");
        let mut edgelist_out = BufWriter::new(File::create(fedgelist_double)?);

        // saving weight matrix in csr format
        let mut wwriters = vec![];
        for fi in ["data", "indptr", "indices"].iter() {
            let file = File::create(base_dir.join(&format!("edge_ftrs_{}.bin", fi)))?;
            wwriters.push(BufWriter::with_capacity(10usize.pow(8), file)); /* 100Mb */
        }

        let mut ix: u32 = 0;
        ix.write_le_bytes(&mut wwriters[1])?;

        fn write_one(
            mut edgelist_out: &mut BufWriter<File>,
            wwriters: &mut [BufWriter<File>],
            wsize: usize,
            ix: &mut u32,
            kvi: (i32, i32, i64),
            weights: &Vec<f32>,
        ) -> Result<()> {
            let (k, v, i) = kvi;
            EdgeItem(k, v).encode(&mut edgelist_out)?;
            if i > -1 {
                let start = i as usize * wsize;
                for j in 0..wsize {
                    let w = weights[start + j];
                    if w > 0.0 {
                        (j as u32).write_le_bytes(&mut wwriters[2])?;
                        w.write_le_bytes(&mut wwriters[0])?;
                        *ix += 1;
                    }
                }
            }
            ix.write_le_bytes(&mut wwriters[1])?;
            Ok(())
        }

        let mut prev_k = -1;
        let mut prev_v = -1;
        let mut prev_i = -2;
        for mut e in sorted_iter.into_iter() {
            if (prev_k, prev_v) == (e.0, e.1) {
                e.2 = std::cmp::max(e.2, prev_i);
            } else if prev_k > -1 {
                write_one(
                    &mut edgelist_out,
                    wwriters.as_mut_slice(),
                    wsize,
                    &mut ix,
                    (prev_k, prev_v, prev_i),
                    &weights,
                )?;
            }
            prev_k = e.0;
            prev_v = e.1;
            prev_i = e.2;
        }

        write_one(
            &mut edgelist_out,
            wwriters.as_mut_slice(),
            wsize,
            &mut ix,
            (prev_k, prev_v, prev_i),
            &weights,
        )?;

        edgelist_out.flush()?;
        for mut out in wwriters {
            out.flush()?;
        }

        fs::remove_dir_all(tempdir)?;

        Ok(())
    }

    pub fn build_reverse_edge_map(base_dir: &PathBuf) -> Result<()> {
        println!("build_reverse_edge_map",);
        let fdouble_edgelist = &base_dir.join("colrow.bin");
        let freverse_edge_map = &base_dir.join("reverse_edge_map.bin");

        let ereader = EdgeReader::<i32>::new(fdouble_edgelist)?;

        let mut edgelist: Vec<(u32, i32, i32)> = ereader
            .enumerate()
            .map(|(i, e)| {
                (
                    u32::try_from(i).unwrap(),
                    std::cmp::min(e.0, e.1),
                    std::cmp::max(e.0, e.1),
                )
            })
            .collect();
        edgelist.par_sort_unstable_by_key(|e| (e.1, e.2));

        let mut map = vec![0u32; edgelist.len()];
        for pair in edgelist.chunks(2) {
            let i = pair[0].0;
            let j = pair[1].0;
            map[i as usize] = j;
            map[j as usize] = i;
        }

        write_vec_file(&map, freverse_edge_map)?;

        Ok(())
    }

    pub fn load_in_out_edge_map(base_dir: &PathBuf) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
        println!("load_in_out_edge_map",);
        let nnodes = file_size::<u64>(&base_dir.join("nodes.bin"))?;
        let ereader = EdgeReader::<i32>::new(&base_dir.join("edgelist.bin"))?;

        let mut edjlist = vec![(vec![], vec![]); nnodes];
        for e in ereader {
            edjlist.get_mut(e.0 as usize).unwrap().0.push(e.1 as u32);
            edjlist.get_mut(e.1 as usize).unwrap().1.push(e.0 as u32);
        }
        // println!("available memory {}", sys_info::mem_info().unwrap().avail);

        Ok(edjlist)
    }

    pub fn save_degrees(base_dir: &PathBuf) -> Result<()> {
        println!("save_degrees",);
        let nnodes = file_size::<u64>(&base_dir.join("nodes.bin"))?;
        let ereader = EdgeReader::<i32>::new(&base_dir.join("colrow.bin"))?;

        // degree = in + out
        let mut degrees = vec![0u32; nnodes];
        for e in ereader {
            degrees[e.1 as usize] += 1;
        }

        write_vec_file(&degrees, &base_dir.join("degrees.bin"))?;

        Ok(())
    }

    pub fn save_neighbour_ftrs(
        edjlist: &Vec<(Vec<u32>, Vec<u32>)>,
        base_dir: &PathBuf,
    ) -> Result<()> {
        let save_dir = base_dir.join("train");
        recreate_dir(&save_dir)?;

        let membership = read_vec_file::<i32>(&base_dir.join("labels.bin"))?;
        save_neighbor_ftrs_dyn(
            edjlist
                .iter()
                .enumerate()
                .map(|(i, (l, r))| (u32::try_from(i).unwrap(), (l, Some(r)))),
            &membership,
            true,
            &save_dir,
        )?;

        Ok(())
    }
}
