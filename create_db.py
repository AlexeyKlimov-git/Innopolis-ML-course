import sqlite3

conn = sqlite3.connect("database.db")
print("Opened database successfully")

conn.execute("DROP TABLE IF EXISTS students")
conn.execute("""CREATE TABLE students (fixed_acidity REAL, volatile_acidity REAL, citric_acid REAL, residual_sugar REAL, chlorides REAL, free_sulfur_dioxide REAL, total_sulfur_dioxide REAL, density REAL, ph REAL, sulphates REAL, alcohol REAL)""")
print("Table created successfully")
conn.close()
