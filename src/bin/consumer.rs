#[macro_use]
extern crate log;

use amiquip::{
    AmqpValue, Channel, Connection, ConsumerMessage, ConsumerOptions, FieldTable, Queue,
    QueueDeclareOptions, Result,
};
use cities::messages::{exchange_declare, AccountEvent, Event, Info};
use cities::models::Geography;
use cities::schema::geography as GeoTable;
use diesel::pg::upsert::*;
use diesel::prelude::*;
use diesel::result::QueryResult;
use rustc_hash::FxHashMap;
use std::convert::TryFrom;
use std::mem;

static MAX_LENGTH: u32 = 100_000;
static LMDB_SAVE_PERIOD: usize = 50_000;
static PSQL_SAVE_PERIOD: usize = 13_107; // 2**16/5

type GraphsType = FxHashMap<(String, String), FxHashMap<u32, Vec<u32>>>;

fn save_graphs(graphs: &mut GraphsType) -> std::result::Result<(), String> {
    for ((ref platform, ref event_type), ref mut graph) in graphs.iter_mut() {
        let method = match event_type.as_str() {
            "event_follow" => "put",
            "event_unfollow" => "delete",
            _ => unimplemented!("unknown event_type"),
        };
        let uri = format!("http://server3:8088/{}", method);
        let db_name = match platform.split('.').nth(0) {
            Some(db_name) => db_name,
            None => return Err(format!("incorrect plafrorm {}", platform)),
        };
        let (mut keys, mut values) = (vec![], vec![]);
        let mut nedges: usize = 0;
        for (&k, v) in graph.iter_mut() {
            keys.push(k);

            let mut v = mem::replace(v, vec![]);
            v.sort_unstable();
            v.dedup();
            nedges += v.len();
            values.push(v);
        }

        if keys.is_empty() {
            continue;
        }

        let new_post = Info {
            db_name: db_name.to_string(),
            keys: keys,
            values: values,
        };

        info!("{}, nedges {:?}", event_type, nedges);
        match reqwest::Client::new()
            .post(uri.as_str())
            .json(&new_post)
            .send()
        {
            Ok(response) => debug!("{}", response.status()),
            Err(e) => return Err(format!("{:?}", e)),
        }
    }
    graphs.clear();
    Ok(())
}

fn dedup_with_merging(values: &mut Vec<Geography>) -> Vec<Geography> {
    values.sort_by(|a, b| (a.extid, &a.platform).cmp(&(b.extid, &b.platform)));

    let mut ret = vec![];
    let mut prev: Option<Geography> = None;
    for v in values {
        if let Some(prev) = prev {
            if (prev.extid, &prev.platform) == (v.extid, &v.platform) {
                if v.location == None && prev.location.is_some() {
                    v.location = prev.location;
                }
                if v.city == None && prev.city.is_some() {
                    v.city = prev.city;
                }
                if v.hometown == None && prev.hometown.is_some() {
                    v.hometown = prev.hometown;
                }
            } else {
                ret.push(prev);
            }
        }
        prev = Some(v.clone());
    }

    if let Some(prev) = prev {
        // always add the last element
        ret.push(prev);
    }

    ret
}

fn upsert_geography(conn: &PgConnection, values: &Vec<Geography>) -> QueryResult<()> {
    diesel::insert_into(GeoTable::table)
        .values(values)
        .on_conflict(on_constraint("geography_pkey"))
        .do_update()
        .set((
            GeoTable::location.eq(diesel::dsl::sql(
                "COALESCE (EXCLUDED.location, geography.location)",
            )),
            GeoTable::city.eq(diesel::dsl::sql("COALESCE (EXCLUDED.city, geography.city)")),
            GeoTable::hometown.eq(diesel::dsl::sql(
                "COALESCE (EXCLUDED.hometown, geography.hometown)",
            )),
        ))
        .execute(conn)?;
    Ok(())
}

fn save_geography(mut values: &mut Vec<Geography>) -> std::result::Result<(), String> {
    if !values.is_empty() && values.len() % PSQL_SAVE_PERIOD == 0 {
        let new_values = dedup_with_merging(&mut values);
        let conn = cities::pgsql_connection();

        upsert_geography(&conn, &new_values).map_err(|e| format!("{}", e))?;
        info! {"pgsql upserted {} items", new_values.len()};
        values.clear();
    }
    Ok(())
}

fn build_queue<'a>(channel: &'a Channel, event_types: &[&'a str]) -> Result<Queue<'a>> {
    let exchanges = event_types
        .iter()
        .map(|&name| exchange_declare(&channel, name));

    let queue = channel.queue_declare(
        "geo_detection", // doesn't matter if queue is Fanout
        QueueDeclareOptions {
            exclusive: true, // exclusive queue can only be used by its declaring connection
            arguments: {
                let mut ft = FieldTable::new();
                ft.insert("x-max-length".to_string(), AmqpValue::LongUInt(MAX_LENGTH));
                ft
            },
            ..QueueDeclareOptions::default()
        },
    )?;

    for exchange in exchanges {
        queue.bind(&exchange?, "geo_detection", FieldTable::new())?;
    }
    Ok(queue)
}

fn parse_account_id<'a>(account_id: &'a str, platforms: &[&'a str]) -> Option<(u32, &'a str)> {
    let id_platform: Vec<&str> = account_id.split('@').collect();
    if id_platform.len() != 2 {
        error!("wrong account_id: {}", account_id);
        return None;
    }

    if !platforms[..].contains(&id_platform[1]) {
        return None;
    }

    let id: u32 = match id_platform[0].parse::<i64>() {
        Ok(id) if id < 0 => return None, // is a social group
        Ok(id) => match u32::try_from(id) {
            Ok(i) => i,
            Err(e) => {
                error!("try_from({}): {}", id, e);
                return None;
            }
        },
        Err(_) => {
            error!("wrong account_id: {}", account_id);
            return None;
        }
    };

    Some((id, id_platform[1]))
}

fn main() -> Result<()> {
    env_logger::init();

    let event_types = [
        "event_follow",
        "event_unfollow",
        "event_profile_field_location",
        "event_profile_field_hometown",
        // "event_profile_field_city",
    ];
    let platforms = ["vk.com"];

    // let mut connection = Connection::insecure_open("amqp://konevec:123@localhost:5672")?;
    let mut connection = Connection::insecure_open("amqp://afdb:afdb01@blabla.company:5672")?;

    // Open a channel - None says let the library choose the channel ID.
    let channel = connection.open_channel(None)?;
    let queue = build_queue(&channel, &event_types)?;

    // Start a consumer. Use no_ack: true so the server doesn't wait for us to ack
    // the messages it sends us.
    let consumer = queue.consume(ConsumerOptions {
        no_ack: true,
        ..ConsumerOptions::default()
    })?;

    let mut graphs = GraphsType::default();
    let mut geography = vec![];
    let mut i: usize = 0;
    for message in consumer.receiver().iter() {
        match message {
            ConsumerMessage::Delivery(delivery) => {
                let body = String::from_utf8_lossy(&delivery.body);
                if let Ok(AccountEvent {
                    account_id,
                    event_type,
                    event,
                    ..
                }) = serde_json::from_str(&body)
                {
                    if !event_types.contains(&event_type.as_str()) {
                        continue;
                    }

                    let (src_id, src_platform) = match parse_account_id(&account_id, &platforms) {
                        Some((id, platform)) => (id, platform),
                        None => continue,
                    };

                    match event {
                        Event::Follow {
                            dst,
                            followers_list_id: _,
                        } => {
                            let (dst_id, dst_platform) = match parse_account_id(&dst, &platforms) {
                                Some((id, platform)) => (id, platform),
                                None => continue,
                            };

                            if src_platform != dst_platform {
                                error!("lmdb: platforms don't match {} and {}", account_id, dst);
                                continue;
                            }

                            &graphs
                                .entry((src_platform.to_string(), event_type.to_string()))
                                .or_default()
                                .entry(src_id)
                                .or_default()
                                .push(dst_id);

                            debug!(
                                "({}) {}: src_id: {}, src_platform {}, \
                                 dst_id: {}, dst_platform {}",
                                i, event_type, src_id, src_platform, dst_id, dst_platform
                            );

                            if (i + 1) % LMDB_SAVE_PERIOD == 0 {
                                if let Err(e) = save_graphs(&mut graphs) {
                                    error!("lmdb: {}", e);
                                }
                                // break;
                            }
                        }
                        Event::Location { location_rubrs, .. } => {
                            debug! {"{}, location {:?}", account_id, location_rubrs}
                            let geo = Geography {
                                extid: src_id as i64,
                                platform: src_platform.to_string(),
                                location: Some(location_rubrs.join(";")),
                                city: None,
                                hometown: None,
                            };
                            geography.push(geo);

                            if let Err(e) = save_geography(&mut geography) {
                                error!("pgsql: {}", e);
                            }
                        }
                        Event::Hometown { hometown_rubrs, .. } => {
                            debug! {"{}, hometown {:?}", account_id, hometown_rubrs}
                            let geo = Geography {
                                extid: src_id as i64,
                                platform: src_platform.to_string(),
                                location: None,
                                city: None,
                                hometown: Some(hometown_rubrs.join(";")),
                            };
                            geography.push(geo);

                            if let Err(e) = save_geography(&mut geography) {
                                error!("pgsql: {}", e);
                            }
                        }
                        // Event::Comment { dst, .. } => {},
                        _ => continue,
                    };
                    i += 1;
                };
                // consumer.ack(delivery)?;
            }
            other => {
                info!("Consumer ended: {:?}", other);
                break;
            }
        }
    }

    connection.close()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cities::schema::geography::dsl::*;

    #[test]
    fn on_conflict() -> QueryResult<()> {
        let conn = cities::pgsql_connection();
        let ext_id = -1i64;

        diesel::delete(GeoTable::table.filter(extid.eq(ext_id))).execute(&conn)?;

        let v_old = vec![Geography {
            extid: ext_id,
            platform: "test".to_string(),
            location: None,
            hometown: Some("Voroneg".to_string()),
            city: None,
        }];

        upsert_geography(&conn, &v_old)?;

        let v_new = vec![Geography {
            extid: ext_id,
            platform: "test".to_string(),
            location: Some("Piter".to_string()),
            city: None,
            hometown: None,
        }];

        upsert_geography(&conn, &v_new)?;

        let v_expected = vec![Geography {
            extid: ext_id,
            platform: "test".to_string(),
            location: Some("Piter".to_string()),
            hometown: Some("Voroneg".to_string()),
            city: None,
        }];

        let v_actual: Vec<Geography> = GeoTable::table
            .filter(extid.eq(ext_id))
            .filter(platform.eq("test".to_string()))
            .load(&conn)?;

        assert_eq!(v_expected, v_actual);

        Ok(())
    }

    #[test]
    fn geography_merging() {
        let mut values = vec![
            Geography {
                extid: 1,
                platform: "vk.com".to_string(),
                location: None,
                city: None,
                hometown: Some("Voroneg".to_string()),
            },
            Geography {
                extid: 1,
                platform: "facebook.com".to_string(),
                location: Some("XXX".to_string()),
                city: None,
                hometown: Some("YYY".to_string()),
            },
            Geography {
                extid: 4,
                platform: "vk.com".to_string(),
                location: Some("ZZZ".to_string()),
                city: None,
                hometown: Some("DDD".to_string()),
            },
            Geography {
                extid: 1,
                platform: "vk.com".to_string(),
                location: Some("Moscow".to_string()),
                city: None,
                hometown: None,
            },
            Geography {
                extid: 0,
                platform: "vk.com".to_string(),
                location: Some("Piter".to_string()),
                city: None,
                hometown: None,
            },
            Geography {
                extid: 1,
                platform: "vk.com".to_string(),
                location: Some("Vologda".to_string()),
                city: None,
                hometown: None,
            },
            Geography {
                extid: 4,
                platform: "vk.com".to_string(),
                location: Some("GGG".to_string()),
                city: None,
                hometown: None,
            },
        ];

        let res = dedup_with_merging(&mut values);
        // println!("{:?}", res);
        assert_eq!(
            res,
            vec![
                Geography {
                    extid: 0,
                    platform: "vk.com".to_string(),
                    location: Some("Piter".to_string()),
                    city: None,
                    hometown: None,
                },
                Geography {
                    extid: 1,
                    platform: "facebook.com".to_string(),
                    location: Some("XXX".to_string()),
                    city: None,
                    hometown: Some("YYY".to_string()),
                },
                Geography {
                    extid: 1,
                    platform: "vk.com".to_string(),
                    location: Some("Vologda".to_string()),
                    city: None,
                    hometown: Some("Voroneg".to_string()),
                },
                Geography {
                    extid: 4,
                    platform: "vk.com".to_string(),
                    location: Some("GGG".to_string()),
                    city: None,
                    hometown: Some("DDD".to_string()),
                }
            ]
        );
    }
}
