use actix_web::middleware::Logger;
use actix_web::{web, App, FromRequest, HttpResponse, HttpServer};
use cities::is_sorted_strict;
use cities::lmdb_client::get_environment;
use cities::messages::{Info, InfoGet};
use lmdb::{Database, EnvironmentFlags, Result, RwTransaction, Transaction, WriteFlags};
use std::convert::TryInto;
use std::path::Path;

// TODO: write tests

static DB_PATH: &'static str = "../data/vk/graph_db_prod";

fn index_get(info: web::Json<InfoGet>) -> HttpResponse {
    let mut flags = EnvironmentFlags::empty();
    flags.insert(EnvironmentFlags::READ_ONLY);
    flags.insert(EnvironmentFlags::NO_READAHEAD); // good for random access

    let env = match get_environment(Path::new(DB_PATH), flags) {
        Ok(env) => env,
        Err(e) => return HttpResponse::InternalServerError().body(format!("{}", e)),
    };
    let db = match env.open_db(Some(info.db_name.as_str())) {
        Ok(db) => db,
        Err(e) => return HttpResponse::BadRequest().body(format!("{}: {}", info.db_name, e)),
    };
    let txn = match env.begin_ro_txn() {
        Ok(txn) => txn,
        Err(e) => return HttpResponse::InternalServerError().body(format!("{}", e)),
    };

    let mut vis = vec![];
    let mut res = vec![];
    for id in &info.keys {
        match &txn.get(db, &id.to_ne_bytes()) {
            Ok(mut blob) => {
                let len = (blob.len() / std::mem::size_of::<u32>()) as u32;
                res.extend_from_slice(&len.to_ne_bytes());
                vis.extend_from_slice(&mut blob);
            }
            Err(_) => {
                res.extend_from_slice(&0u32.to_ne_bytes());
            }
        };
    }
    res.extend_from_slice(&mut vis);

    txn.abort();

    HttpResponse::Ok()
        .content_type("application/octet-stream")
        .body(web::Bytes::from(res))
}

/// Merge/Delete js with/from is and return the result, avoid self loop with ix.
/// Asuming is and js are sorted ascending without duplicates.
fn sorted_merge<I, J>(ix: u32, iit: &mut I, jit: &mut J, delete: bool) -> Vec<u8>
where
    I: Iterator<Item = u32>,
    J: Iterator<Item = u32>,
{
    let mut result = vec![];
    let mut i = iit.next();
    let mut j = jit.next();

    loop {
        match (i, j) {
            (Some(i_val), Some(j_val)) => {
                if i_val < j_val {
                    result.extend_from_slice(&i_val.to_ne_bytes());
                    i = iit.next();
                } else if i_val > j_val {
                    if !delete && ix != j_val {
                        result.extend_from_slice(&j_val.to_ne_bytes());
                    }
                    j = jit.next();
                } else {
                    if !delete {
                        result.extend_from_slice(&j_val.to_ne_bytes());
                    }
                    i = iit.next();
                    j = jit.next();
                }
            }
            (Some(i_val), None) => {
                result.extend_from_slice(&i_val.to_ne_bytes());
                i = iit.next();
            }
            (None, Some(j_val)) => {
                if delete {
                    break;
                }
                if ix != j_val {
                    result.extend_from_slice(&j_val.to_ne_bytes());
                }
                j = jit.next();
            }
            (None, None) => break,
        }
    }

    result
}

fn merge(
    db: Database,
    txn: &mut RwTransaction,
    w_flags: WriteFlags,
    i: u32,
    js: &[u32],
    delete: bool,
) -> Result<()> {
    let i_key = i.to_ne_bytes();

    let blob: Vec<_> = match &txn.get(db, &i_key) {
        Ok(blob) => {
            let iit = &mut blob
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()));
            let jit = &mut js.iter().map(|&j| j);
            sorted_merge(i, iit, jit, delete)
        }
        Err(_) => vec![],
    };

    // put back the blob
    txn.put(db, &i_key, &blob, w_flags)
}

fn index_merge((info, delete): (web::Json<Info>, bool)) -> HttpResponse {
    // TODO: reduce subgraph in (info.keys, info.values) to smallest possible

    let mut flags = EnvironmentFlags::empty();
    flags.insert(EnvironmentFlags::NO_READAHEAD); // good for random access
    flags.insert(EnvironmentFlags::WRITE_MAP);
    flags.insert(EnvironmentFlags::MAP_ASYNC);

    let env = match get_environment(Path::new(DB_PATH), flags) {
        Ok(env) => env,
        Err(e) => return HttpResponse::InternalServerError().body(format!("{}", e)),
    };
    let db = match env.open_db(Some(info.db_name.as_str())) {
        Ok(db) => db,
        Err(e) => return HttpResponse::BadRequest().body(format!("{}: {}", info.db_name, e)),
    };
    let mut txn = match env.begin_rw_txn() {
        Ok(txn) => txn,
        Err(e) => return HttpResponse::InternalServerError().body(format!("{}", e)),
    };
    let w_flags = WriteFlags::empty();

    if info.keys.len() != info.values.len() {
        return HttpResponse::BadRequest().body(format!("keys length != values length\n"));
    }

    for (&i, js) in info.keys.iter().zip(&info.values) {
        if !is_sorted_strict(js) {
            return HttpResponse::BadRequest().body(format!(
                "values should be sorted ascending and have no duplicates"
            ));
        }

        // merging js to i
        if let Err(e) = merge(db, &mut txn, w_flags, i, js, delete) {
            return HttpResponse::InternalServerError().body(format!("key {}: {}", i, e));
        }

        // going the other way - merging i to js
        for &j in js {
            if let Err(e) = merge(db, &mut txn, w_flags, j, &[i], delete) {
                return HttpResponse::InternalServerError().body(format!("key {}: {}", j, e));
            }
        }
    }

    if let Err(e) = txn.commit() {
        return HttpResponse::InternalServerError().body(format!("{}", e));
    }
    HttpResponse::Ok().body(format!("Success\n"))
}

fn index_put(info: web::Json<Info>) -> HttpResponse {
    index_merge((info, false))
}

fn index_delete(info: web::Json<Info>) -> HttpResponse {
    index_merge((info, true))
}

pub fn main() {
    // std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();

    HttpServer::new(move || {
        App::new()
            .route("/get", web::post().to(index_get))
            .data(web::Json::<InfoGet>::configure(|cfg| {
                cfg.limit(2usize.pow(30))
            }))
            .route("/put", web::post().to(index_put))
            .data(web::Json::<Info>::configure(|cfg| {
                cfg.limit(2usize.pow(30))
            }))
            .route("/delete", web::post().to(index_delete))
            .data(web::Json::<Info>::configure(|cfg| {
                cfg.limit(2usize.pow(30))
            }))
            .wrap(Logger::default())
    })
    .bind("0.0.0.0:8088")
    .unwrap()
    .run()
    .unwrap();
}

#[cfg(test)]
mod tests {
    #[test]
    fn sorted_merge_test() {
        extern crate cities;

        use super::sorted_merge;
        use cities::write_vec;

        fn vec2blob(vec: &[u32]) -> std::vec::Vec<u8> {
            let mut buffer = vec![];
            write_vec(vec, &mut buffer).unwrap();
            buffer
        }

        assert_eq!(
            sorted_merge(
                100,
                &mut [1, 3, 5, 7].iter().cloned(),
                &mut [1, 2, 3, 4, 6, 7, 8, 9].iter().cloned(),
                false
            ),
            vec2blob(&[1, 2, 3, 4, 5, 6, 7, 8, 9])
        );

        assert_eq!(
            sorted_merge(
                100,
                &mut [3, 5, 7].iter().cloned(),
                &mut [4, 5, 9].iter().cloned(),
                false
            ),
            vec2blob(&[3, 4, 5, 7, 9])
        );

        assert_eq!(
            sorted_merge(
                100,
                &mut [3, 5, 7].iter().cloned(),
                &mut [0, 2, 5, 9].iter().cloned(),
                false
            ),
            vec2blob(&[0, 2, 3, 5, 7, 9])
        );

        assert_eq!(
            sorted_merge(
                4,
                &mut [1, 3, 5, 7].iter().cloned(),
                &mut [1, 2, 3, 4, 6, 7, 8, 9].iter().cloned(),
                false
            ),
            vec2blob(&[1, 2, 3, 5, 6, 7, 8, 9])
        );

        assert_eq!(
            sorted_merge(
                9,
                &mut [1, 3, 5, 7].iter().cloned(),
                &mut [1, 2, 3, 4, 6, 7, 8, 9].iter().cloned(),
                false
            ),
            vec2blob(&[1, 2, 3, 4, 5, 6, 7, 8])
        );

        assert_eq!(
            sorted_merge(
                100,
                &mut [1, 3, 5, 7].iter().cloned(),
                &mut [1, 2, 3, 4, 6, 7, 8, 9].iter().cloned(),
                true
            ),
            vec2blob(&[5])
        );

        assert_eq!(
            sorted_merge(
                100,
                &mut [3, 5, 7].iter().cloned(),
                &mut [0, 2, 5, 9].iter().cloned(),
                true
            ),
            vec2blob(&[3, 7])
        );
    }
}
