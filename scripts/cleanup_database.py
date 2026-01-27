import sqlite3

print("Fixing invalid centroids in naamse.db...")
conn = sqlite3.connect('D:/Projects/pythonProjects/NAAMSE/src/cluster_engine/data_access/adversarial/naamse.db')
cursor = conn.cursor()

# Delete invalid centroids
cursor.execute("DELETE FROM centroids WHERE prompt_id = 0;")
conn.commit()

print("Verifying fix...")
# Check dimensions and embedding sizes
cursor.execute("""
    SELECT dimensions, LENGTH(embedding_vector), COUNT(*) 
    FROM centroids 
    GROUP BY dimensions, LENGTH(embedding_vector);
""")

results = cursor.fetchall()
for row in results:
    print(f"Dimensions: {row[0]}, Embedding Length: {row[1]} bytes, Count: {row[2]}")

conn.close()
print("Done! All centroids should now be uniform (384 dimensions, 1536 bytes).")