import psycopg2


def save_model_for_users(email, model_name):
    db_connection = psycopg2.connect(
        host="localhost",
        port="5432",
        user="postgres",
        password="admin",
        database="ai_nexus"
    )
    cursor = db_connection.cursor()
    sql_statement = "INSERT INTO models VALUES(%s,%s)"
    cursor.execute(sql_statement,(email,model_name))
    db_connection.commit()
    # Close the cursor and the connection
    cursor.close()
    db_connection.close()
    