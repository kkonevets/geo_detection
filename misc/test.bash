
curl -sd '{
	"db_name":"vk",
   	"keys": [11]
}' -H "Content-Type: application/json" -X POST http://server3:8088/get --fail --silent --show-error | od -w32 -An -tu4 | head

curl -sd '{
	"db_name":"vk",
    "keys": [11],
    "values": [[9]]
}' -H "Content-Type: application/json" -X POST http://localhost:8088/put

curl -sd '{
	"db_name":"vk",
    "keys": [11],
    "values": [[9]]
}' -H "Content-Type: application/json" -X POST http://localhost:8088/delete