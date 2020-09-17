use cities::{file_size, read_vec_file, EdgeReader, EdjReader, LeBytes};
use itertools::Itertools;
use std::fs::File;
use std::io::{BufReader, BufWriter, LineWriter, Read, Result, Write};
use std::path::Path;

#[allow(dead_code)]
fn city_coincide(base_dir: &Path) -> Result<()> {
    let fname = &base_dir.join("labels.bin");
    let membership: Vec<i32> = read_vec_file(fname)?;

    let fedgelist = &base_dir.join("edgelist.bin");
    let fweights = &base_dir.join("weights.bin"); // should not be normalized with wj_sums

    let nedges = 2 * file_size::<f32>(fweights)? / file_size::<i32>(fedgelist)?;
    let size = nedges * u32::MAX as usize;
    let mut counts_eq = vec![0u32; size];
    let mut counts_neq = vec![0u32; size];
    let mut wreader = BufReader::new(File::open(fweights)?);

    fn fill_counts<R: Read>(counts: &mut Vec<u32>, mut wreader: R, nedges: usize) -> Result<()> {
        for i in 0..nedges {
            let w = f32::read_le_bytes(&mut wreader)? as usize;
            if w > 0 {
                *counts.get_mut(i * u32::MAX as usize + w).unwrap() += 1;
            }
        }
        Ok(())
    }

    for (j, es) in EdgeReader::<i32>::new(fedgelist)?
        .group_by(|e| e.1)
        .into_iter()
    {
        let lj = membership[j as usize];
        if lj == -1 {
            continue;
        }
        for e in es {
            let li = membership[e.0 as usize];
            if li == -1 {
                continue;
            }
            if lj == li {
                fill_counts(&mut counts_eq, &mut wreader, nedges)?;
            } else {
                fill_counts(&mut counts_neq, &mut wreader, nedges)?;
            }
        }
    }

    fn save_counts(counts: &Vec<u32>, base_dir: &Path, prefix: &str) -> Result<()> {
        for (chn, chunk) in counts
            .iter()
            .chunks(u32::MAX as usize)
            .into_iter()
            .enumerate()
        {
            let fname = format!("counts_{}_{}.bin", prefix, chn);
            let fname = base_dir.join(fname);
            let mut writer = BufWriter::new(File::create(fname)?);
            for (i, v) in chunk.enumerate() {
                if v > &0 {
                    (i as u32).write_le_bytes(&mut writer)?;
                    v.write_le_bytes(&mut writer)?;
                }
            }
            writer.flush()?;
        }
        Ok(())
    }

    save_counts(&counts_eq, &base_dir, "eq")?;
    save_counts(&counts_neq, &base_dir, "neq")?;

    Ok(())
}

#[allow(dead_code)]
pub fn read_weighted_connections(fin: &Path, fout: &Path) -> Result<()> {
    let mut counts: Vec<u32> = vec![0; u32::MAX as usize];
    for item in EdjReader::<u64>::new(&fin)? {
        counts[item.v.len()] += 1;
    }

    let mut out = LineWriter::new(File::create(fout)?);
    let mut total = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 {
            out.write_all(format!("{},{}\n", i, c).as_bytes())?;
            total += 1;
        }
    }
    println!("total {:?}", total);
    Ok(())
}

fn main() -> Result<()> {
    let base_dir = Path::new("../data/facebook");
    // city_coincide(Path::new(&base_dir))?;
    read_weighted_connections(
        &base_dir.join("comment_edjlist.bin"),
        &base_dir.join("in_degree_comment.bin"),
    )?;
    read_weighted_connections(
        &base_dir.join("share_edjlist.bin"),
        &base_dir.join("in_degree_share.bin"),
    )?;

    // let v = [3, 5, 1, 9, 10, 4, 2];
    // println!("{:?}", v.iter().enumerate().map(|(i, v)| (v, i)).max());

    Ok(())
}
