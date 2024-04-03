import sqlite3

# Connect to the database (creates the file if it doesn't exist)
conn = sqlite3.connect('Restaurant_food_data.db')
c = conn.cursor()

# Create the item table
c.execute('''CREATE TABLE IF NOT EXISTS item
             (item_name TEXT, no_of_customers INTEGER, 
             no_of_positives INTEGER, no_of_negatives INTEGER,
             pos_perc REAL, neg_perc REAL)''')

# Commit changes and close the connection
conn.commit()
conn.close()