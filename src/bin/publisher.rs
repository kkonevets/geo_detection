use amiquip::{AmqpProperties, Channel, Connection, Publish, Result};
use chrono::offset::TimeZone;
use chrono::offset::Utc;

use cities::messages::{exchange_declare, AccountEvent, Event};

fn publish(channel: &Channel, ae: AccountEvent) -> Result<()> {
    let exchange = exchange_declare(&channel, &ae.event_type)?;
    let msg = serde_json::to_string(&ae).unwrap();
    exchange.publish(Publish::with_properties(
        msg.as_bytes(),
        "geo_detection", // doesn't matter if queue is Fanout
        AmqpProperties::default().with_delivery_mode(1), // Non-persistent
    ))?;
    Ok(())
}

fn main() -> Result<()> {
    // Open connection.
    let mut connection = Connection::insecure_open("amqp://konevec:123@localhost:5672")?;

    // Open a channel - None says let the library choose the channel ID.
    let channel = connection.open_channel(None)?;

    let event_follow = AccountEvent {
        account_id: "4764@vk.com".to_string(),
        ptime: Utc.ymd(2019, 8, 9).and_hms(3, 14, 55),
        ctime: Utc.ymd(2019, 8, 9).and_hms(3, 15, 1),
        event_type: "event_follow".to_string(),
        event: Event::Follow {
            dst: "23425@vk.com".to_string(),
            followers_list_id: "100004742708467@vk.com/friends".to_string(),
        },
    };

    let event_unfollow = AccountEvent {
        account_id: "123@vk.com".to_string(),
        ptime: Utc.ymd(2019, 8, 9).and_hms(3, 14, 55),
        ctime: Utc.ymd(2019, 8, 9).and_hms(3, 15, 1),
        event_type: "event_unfollow".to_string(),
        event: Event::Follow {
            dst: "777@vk.com".to_string(),
            followers_list_id: "100004742708467@vk.com/friends".to_string(),
        },
    };

    publish(&channel, event_follow)?;
    publish(&channel, event_unfollow)?;
    // exchange.publish(Publish::new("hello!!!".as_bytes(), "hello"))?;

    connection.close().unwrap();

    Ok(())
}
