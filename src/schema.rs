table! {
    geography (extid, platform) {
        extid -> Int8,
        platform -> Varchar,
        location -> Nullable<Text>,
        hometown -> Nullable<Text>,
        city -> Nullable<Text>,
    }
}
