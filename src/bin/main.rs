use cities::preprocessing::directed;
use std::convert::TryFrom;
use std::fs::File;
use std::io::Result;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

#[allow(dead_code)]
fn execute_script(socialnet: &str, args: &str) -> Result<()> {
    println!("saving labels and data from python");
    Command::new("sh")
        .arg("-c")
        .arg(format!("python geometric.py {} {}", socialnet, args))
        .status()?;
    Ok(())
}

#[allow(dead_code)]
fn vk() -> Result<()> {
    cities::preprocessing::connections_csv2bin(
        "../data/vk/connections.csv",
        "../data/vk/connections.bin",
    )?;
    cities::preprocessing::build_edjlist()?;
    cities::lmdb_client::to_embeded_db()?;
    cities::lmdb_client::save_degrees()?;

    execute_script("vk", "--labels")?;
    cities::processing::degrees_with_cities(
        cities::EdjReader::<cities::TYPE>::new("../data/vk/edjlist.bin")?
            .map(|e| (e.k, (e.v, None))),
        false,
        Path::new("../data/vk/"),
    )?;
    execute_script("vk", "--data")?;

    let nsample = [30, 5];
    write!(File::create("../data/vk/nsample.conf")?, "{:?}", nsample)?;
    cities::processing::save_subgraph_edge_index(&nsample, 10, true)?;
    cities::processing::save_subgraph_edge_index(&nsample, 20, false)?;

    Ok(())
}

fn directed_network<P: std::fmt::Debug + AsRef<Path>>(
    base_dir: &PathBuf,
    fedjlists: &[P],
    fweights: &[P],
    socialnet: &str,
) -> Result<()> {
    directed::save_nodes(&base_dir, fedjlists)?;
    directed::unite_graphs(&base_dir, fedjlists, fweights)?;
    directed::double_edge_index(&base_dir, fweights.len())?;
    directed::build_reverse_edge_map(&base_dir)?;
    directed::save_degrees(&base_dir)?;

    execute_script(socialnet, "--labels")?;

    let edjlist = directed::load_in_out_edge_map(&base_dir)?;
    cities::processing::degrees_with_cities(
        &mut edjlist
            .iter()
            .enumerate()
            .map(|(i, (l, r))| (u32::try_from(i).unwrap(), (l, Some(r)))),
        true,
        &base_dir,
    )?;

    execute_script(socialnet, "--data")?;

    directed::save_neighbour_ftrs(&edjlist, &base_dir)?;

    println!();
    Ok(())
}

#[allow(dead_code)]
fn facebook() -> Result<()> {
    let base_dir = PathBuf::from("../data/facebook");

    let fedjlists = &[
        &base_dir.join("comment_edjlist.bin"),
        &base_dir.join("mention_edjlist.bin"),
        &base_dir.join("share_edjlist.bin"),
    ];

    let fweights = &[
        &base_dir.join("comment_weights.bin"),
        &base_dir.join("mention_weights.bin"),
        &base_dir.join("share_weights.bin"),
    ];

    directed_network(&base_dir, fedjlists, fweights, "facebook")?;
    Ok(())
}

#[allow(dead_code)]
fn twitter() -> Result<()> {
    let base_dir = PathBuf::from("../data/twitter");

    let fedjlists = &[
        &base_dir.join("comment_edjlist.bin"),
        &base_dir.join("share_edjlist.bin"),
    ];

    let fweights = &[
        &base_dir.join("comment_weights.bin"),
        &base_dir.join("share_weights.bin"),
    ];

    directed_network(&base_dir, fedjlists, fweights, "twitter")?;
    Ok(())
}

fn main() -> Result<()> {
    vk()?;
    // twitter()?;
    // facebook()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn sorted_strict() {
        extern crate cities;
        use std::collections::HashMap;

        let mut files = HashMap::new();
        files.insert("facebook", vec!["comment", "mention", "share"]);
        files.insert("twitter", vec!["comment", "share"]);

        for (platform, edge_types) in &files {
            for edge_type in edge_types {
                let fname = format!("../data/{}/{}_edjlist.bin", platform, edge_type);
                let reader = cities::EdjReader::<u64>::new(fname).unwrap();

                let mut prev_k = 0;
                for (_i, item) in reader.enumerate() {
                    // println!("{}, {:?}", _i, k);
                    assert!(
                        prev_k < item.k,
                        "{}, {}, prev_k {}, k {}",
                        platform,
                        edge_type,
                        prev_k,
                        item.k
                    );
                    prev_k = item.k;
                }
            }
        }
    }
}
