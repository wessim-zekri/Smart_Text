import pyodbc

# Path to your .mdb file
mdb_path = r'dataset\icdar\train\data.mdb'

# Connection string
conn_str = (
    r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
    r"Dbq=" + mdb_path + ";"
)
# Establish connection
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Get the list of all tables
tables = cursor.tables().fetchall()
for table in tables:
    print(table.table_name)

# Example: Querying a specific table to count images
cursor.execute('SELECT COUNT(*) FROM your_table_name')
result = cursor.fetchone()
print(f"Number of images: {result[0]}")

# Close the connection
conn.close()
