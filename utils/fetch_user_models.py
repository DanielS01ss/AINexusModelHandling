import psycopg2

def fetch_user_models(email):
    db_connection = psycopg2.connect(
        host="localhost",
        port="5432",
        user="postgres",
        password="admin",
        database="ai_nexus"
    )

    cursor = db_connection.cursor()
    sql_statement = f"SELECT * FROM models WHERE email=\'{email}\'"
    cursor.execute(sql_statement,(email))
    results = cursor.fetchall()
    # db_connection.commit()
    # # Close the cursor and the connection
    # cursor.close()
    # db_connection.close()

    return results