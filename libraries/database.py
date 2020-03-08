import sqlite3


''' Helper library for working with SQLITE3 Database '''
   

# Insert Dictionary into matching table
def dictionary_insert(db, table, dictionary):
    try:
        conn = sqlite3.connect(db)  
        cursor = conn.cursor()
        columns = ', '.join(dictionary.keys())
        qmarks = ', '.join('?' * len(dictionary))
        sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % (table, columns, qmarks)
        cursor.execute(sql, tuple(dictionary.values()))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(e)
    return

    
    
# Execute Statement on database
def database_execute(db, statement):
    conn = sqlite3.connect(db)
    cur = conn.cursor()  
    cur.execute(statement)
    conn.commit()
    cur.close()
    conn.close()
    return

# Execute Statement on database
def database_retrieve(db, statement):
    conn = sqlite3.connect(db)
    #conn.row_factory = sqlite3.Row
    cur = conn.cursor()  
    cur.execute(statement)
    data = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return data


      


