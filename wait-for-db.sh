#!/bin/sh
set -e

host="$DB_HOST"
port="$DB_PORT"

echo "Waiting for database at $host:$port..."

until nc -z "$host" "$port"; do
  echo "Database not ready. Sleeping..."
  sleep 2
done

echo "Database is up!"
exec "$@"
