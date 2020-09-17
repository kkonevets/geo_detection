use amiquip::{Channel, Exchange, ExchangeDeclareOptions, ExchangeType, Result};
use chrono::offset::Utc;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::BTreeMap;

pub type DateTime = ::chrono::DateTime<Utc>;

#[derive(Deserialize, Serialize)]
pub struct InfoGet {
    pub db_name: String,
    pub keys: Vec<u32>,
}

#[derive(Deserialize, Serialize)]
pub struct Info {
    pub db_name: String,
    pub keys: Vec<u32>,
    pub values: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AccountEvent {
    pub account_id: String,
    pub event_type: String,
    pub event: Event,
    pub ptime: DateTime,
    pub ctime: DateTime,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileRaw {
    pub render: serde_json::Value,
    pub text: Option<String>,
    pub src: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Event {
    Follow {
        dst: String,
        followers_list_id: String,
    },
    Comment {
        dst: String,
        event_url: String,
        parent_url: String,
    },
    Location {
        location_rubrs: Vec<String>,
        #[serde(skip_serializing)]
        raw: ProfileRaw,
    },
    City {
        city: String,
        city_rubrs: Vec<String>,
        #[serde(skip_serializing)]
        city_render: BTreeMap<String, String>,
    },
    Hometown {
        hometown_rubrs: Vec<String>,
        #[serde(skip_serializing)]
        raw: ProfileRaw,
    },
}

pub fn exchange_declare<'a>(channel: &'a Channel, event_name: &'a str) -> Result<Exchange<'a>> {
    channel.exchange_declare(
        ExchangeType::Fanout,
        event_name,
        ExchangeDeclareOptions {
            durable: true,
            ..ExchangeDeclareOptions::default()
        },
    )
}
