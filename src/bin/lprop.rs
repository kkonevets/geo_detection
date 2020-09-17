extern crate num_cpus;
extern crate rand;

use crossbeam::crossbeam_channel::{bounded, Receiver};
use rand::seq::SliceRandom;
use rustc_hash::FxHashMap;
use std::io::Result;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;
use thousands::Separable;

#[derive(Debug, Clone)]
struct Graph {
    edjlist: Option<Arc<Vec<Vec<u32>>>>,
    inclist: Option<Arc<Vec<Vec<(u32, u16)>>>>,
    membership: Arc<RwLock<Vec<i32>>>,
    mptr: *mut i32,
}

unsafe impl Send for Graph {}

impl Graph {
    #[allow(dead_code)]
    pub fn uniform(edjlist: Arc<Vec<Vec<u32>>>, membership: Arc<RwLock<Vec<i32>>>) -> Graph {
        let mut g = Graph::init(membership);
        g.edjlist = Some(edjlist);
        g
    }

    #[allow(dead_code)]
    pub fn weighted(
        inclist: Arc<Vec<Vec<(u32, u16)>>>,
        membership: Arc<RwLock<Vec<i32>>>,
    ) -> Graph {
        let mut g = Graph::init(membership);
        g.inclist = Some(inclist);
        g
    }

    fn init(membership: Arc<RwLock<Vec<i32>>>) -> Graph {
        Graph {
            edjlist: None,
            inclist: None,
            membership: membership.clone(),
            mptr: membership.write().unwrap().as_mut_ptr(),
        }
    }

    unsafe fn aggregate(&self, vx: usize) -> FxHashMap<i32, u32> {
        let mut frequency = FxHashMap::<i32, u32>::default();

        if let Some(elist) = &self.edjlist {
            for label in elist[vx]
                .iter()
                .map(|&x| *self.mptr.offset(x as isize))
                .filter(|&l| l > 0)
            {
                *frequency.entry(label).or_insert(0) += 1;
            }
        } else if let Some(ilist) = &self.inclist {
            for (label, w) in ilist[vx]
                .iter()
                .map(|&(x, w)| (*self.mptr.offset(x as isize), w))
                .filter(|&(l, _)| l > 0)
            {
                *frequency.entry(label).or_insert(0) += w as u32;
            }
        } else {
            panic!("graph is empty");
        }

        frequency
    }
}

fn emit_reciever(r: Receiver<usize>, g: Graph) -> thread::JoinHandle<u32> {
    thread::spawn(move || {
        let mut rng = rand::thread_rng();
        r.iter()
            .map(|vx| propagate_one(vx, g.clone(), &mut rng))
            .collect::<Vec<u32>>()
            .into_iter()
            .sum()
    })
}

fn propagate_one(vx: usize, g: Graph, rng: &mut rand::prelude::ThreadRng) -> u32 {
    let frequency;
    let old_label;
    unsafe {
        old_label = *g.mptr.offset(vx as isize);
        frequency = g.aggregate(vx);
    }

    let mut nondominant = 1;
    let mut sorted: Vec<_> = frequency.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    if let Some((_, max_count)) = sorted.first() {
        let dominant_labels: Vec<_> = sorted
            .iter()
            .take_while(|(_, v)| v == max_count)
            .map(|(k, _)| **k)
            .collect();
        let new_label = *dominant_labels.choose(rng).unwrap();
        /* Check if the _current_ label of the node is also dominant */
        if frequency.get(&old_label).unwrap_or(&0) != *max_count {
            /* Nope, it is not */
            nondominant = 0;
        }
        if new_label != old_label {
            unsafe {
                *g.mptr.offset(vx as isize) = new_label;
            }
        }
    }

    nondominant
}

#[allow(dead_code)]
fn label_propagation(g: Graph, num_threads: usize, threshold: u32) {
    /* The implementation uses a trick to avoid negative array indexing:
     * elements of the membership vector are increased by 1 at the start
     * of the algorithm; this allows us to denote unlabeled vertices
     * (if any) by zeroes. The membership vector is shifted back in the end
     */

    let mut membership = g.membership.write().unwrap();

    println!("label propagation started:");
    for label in membership.iter_mut() {
        *label += 1;
    }

    /* Take all non fixed nodes and all unique labels*/
    let mut node_order = Vec::<usize>::new();
    for (i, &label) in membership.iter().enumerate() {
        if label == 0 {
            node_order.push(i);
        }
    }
    let mut rng = rand::thread_rng();
    let mut iteration_n = 0;
    let mut nondominant = node_order.len() as u32;
    let start = Instant::now();

    while nondominant > threshold {
        nondominant = node_order.len() as u32;
        /* Shuffle the node ordering vector */
        node_order.shuffle(&mut rng);

        let mut handles = vec![];
        let (s, r) = bounded::<usize>(num_threads * 1000);
        for _ in 0..num_threads - 1 {
            let handle = emit_reciever(r.clone(), g.clone());
            handles.push(handle);
        }
        let handle = emit_reciever(r, g.clone());
        handles.push(handle);

        for &vx in node_order.iter() {
            s.send(vx).unwrap();
        }
        drop(s);

        for handle in handles {
            nondominant -= handle.join().unwrap();
        }

        iteration_n += 1;
        eprintln!(
            "iter {0}, nondominant {1}, elapsed {2:.2} hours",
            iteration_n,
            nondominant.separate_with_spaces(),
            (start.elapsed().as_secs() as f64) / 3600.
        );
    }

    /* Decrement membership back */
    for label in membership.iter_mut() {
        *label -= 1;
    }
    print!("\n");
}

#[cfg(test)]
mod tests {
    #[test]
    fn lprop() {
        use super::{label_propagation, Graph};
        use cities::edjlist_init;
        use std::sync::{Arc, RwLock};

        let edgelist: Vec<(u32, u32)> = vec![(0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5)];
        let ejlist = Arc::new(edjlist_init(9, edgelist.into_iter(), false));

        for _ in 0..10 {
            let membership: Vec<i32> = vec![0, 0, -1, 1, -1, 0];
            let g = Graph::uniform(ejlist.clone(), Arc::new(RwLock::new(membership)));
            label_propagation(g.clone(), 4, 0);
            println!("{:?}", g.membership.read().unwrap());
        }
    }
}

fn main() -> Result<()> {
    // let nnodes = all_nodes()?.len();
    // println!("number of nodes {}", nnodes);
    // let mut membership = read_vector_csv("../data/vk/labels.csv");
    // println!("membership len {}", membership.len());
    // let test_index = read_vector_csv("../data/vk/test_index.csv");
    // for ix in test_index {
    //     membership[ix as usize] = -1;
    // }

    // let reader = EdjReader::new("../data/vk/edjlist.bin");
    // let elist = Arc::new(reader.map(|(_, row)| row).collect());
    // let g = Graph::uniform(elist, Arc::new(RwLock::new(membership)));
    // let ilist = Arc::new(load_inclist!("../data/vk/inclist.bin", nnodes, u16));
    // let g = Graph::weighted(ilist, Arc::new(RwLock::new(membership)));

    // label_propagation(g.clone(), num_cpus::get(), 10);
    // write_vec!(
    //     &g.membership.read().unwrap(),
    //     "../data/vk/preds_test_weighted_10.bin"
    // );

    Ok(())
}
