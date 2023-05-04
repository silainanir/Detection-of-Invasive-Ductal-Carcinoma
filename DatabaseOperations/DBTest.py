import DatabaseManager


def create_db():
    # creating database
    db = "BreastCancer"
    dbm = DatabaseManager.DatabaseManager()
    dbm.create_con(db)
    dbm.cursor.close()


def create_table():
    # creating table Patients
    db = "BreastCancer"
    dbm = DatabaseManager.DatabaseManager()
    dbm.create_con(db)
    table_name = "Patients"
    columns = [["id", "TEXT"], ["totalPictures", "INT"], ["min", "INT"], ["max", "INT"], ["avg", "REAL"],
               ["entropy", "REAL"]]
    pk = columns[0][0]
    dbm.create_table(table_name, columns, pk)

    # creating table Pictures0
    table_name = "Pictures0"
    columns = [["pat_ID", "TEXT"], ["xcoord", "TEXT"], ["ycoord", "TEXT"], ["min", "INT"], ["max", "INT"],
               ["avg", "REAL"], ["rgb", "TEXT"]]
    pk = "{}, {}, {}".format(columns[0][0], columns[1][0], columns[2][0])
    fk = columns[0][0]
    fk_table_name = "Patients"
    fk_column_name = "id"
    dbm.create_table(table_name, columns, pk, fk, fk_table_name, fk_column_name)

    # creating table Pictures1
    table_name = "Pictures1"
    dbm.create_table(table_name, columns, pk, fk, fk_table_name, fk_column_name)

    dbm.cursor.close()


def insert(valuesList):
    db = "BreastCancer"
    dbm = DatabaseManager.DatabaseManager()
    dbm.create_con(db)
    table_name = "Patients"

    dbm.insert_into_columns(table_name, valuesList)
    dbm.cursor.close()


def delete(table_name, condition):
    db = "BreastCancer"
    dbm = DatabaseManager.DatabaseManager()
    dbm.create_con(db)
    dbm.delete_from(table_name, condition)
    dbm.cursor.close()


def query(sql):
    db = "BreastCancer"
    dbm = DatabaseManager.DatabaseManager()
    dbm.create_con(db)
    data = dbm.send_query(sql)
    dbm.cursor.close()
    return data


# values = ["1", 20, 10, 40, 27.2, 11.4]
# insert(values)
# values = ["2", 20, 10, 40, 27.2, 11.4, "3", 20, 10, 40, 27.2, 11.4, "4", 20, 10, 40, 27.2, 11.4]
# insert(values)
# delete("Patients", "id > 0")
# data = query("SELECT * FROM Patients")
# for i in data:
#     print(i)
print("DB Test")