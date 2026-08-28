-- Verification scaffolding: disposable SQLite used by control-sqlsaber launch.
-- Not part of the product. Cleanup deletes the file with the isolated workdir.

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    state TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers (id),
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL,
    shipped_at TEXT
);

INSERT INTO customers (id, name, state) VALUES
    (1, 'Acme', 'CA'),
    (2, 'Globex', 'NY');

INSERT INTO orders (id, customer_id, amount_cents, status, shipped_at) VALUES
    (1, 1, 1999, 'shipped', '2024-01-15'),
    (2, 2, 5000, 'pending', NULL);
