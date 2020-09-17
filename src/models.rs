use super::schema::geography;

#[derive(Debug, Clone, Insertable, AsChangeset, PartialEq)]
#[table_name = "geography"]
#[derive(Queryable)]
pub struct Geography {
    pub extid: i64,
    pub platform: String,
    pub location: Option<String>,
    pub hometown: Option<String>,
    pub city: Option<String>,
}
