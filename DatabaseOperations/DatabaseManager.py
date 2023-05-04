import sqlite3
import os


class DatabaseManager:

    def __init__(self):
        self.database_name = None
        self.con = None
        self.cursor = None
        self.column_number = 0

    def create_con(self, db_name):
        """
        Connection is established and cursor is obtained.
        :param db_name: Name of the database which is to be created
        :return: None
        """
        path = os.path.dirname(os.path.abspath(__file__))[:-10] + "\\" + db_name + ".db"
        self.con = sqlite3.connect(path)
        self.cursor = self.con.cursor()

    def create_table(self, tablename, columns, primary_key="", foreign_key="", fk_table_name="", fk_table_column=""):
        """
        :param tablename: table name to be created
        :param columns: is a 2D list which holds column name and type
        :return:
        """
        self.column_number = len(columns)
        sql = "CREATE TABLE IF NOT EXISTS " + tablename + " ("
        for i in columns:
            sql += i[0] + " " + i[1] + ", "
        sql += "\n"
        if primary_key != "":
            sql += "PRIMARY KEY({})".format(primary_key)

        if foreign_key != "":
            sql += "\nFOREIGN KEY ({}) REFERENCES {}({})".format(foreign_key, fk_table_name, fk_table_column)
        sql += "\n);"
        self.cursor.execute(sql)
        self.con.commit()

    def add_pk(self, table_name, primary_key):
        pass

    def add_fk(self, table_name, foreign_key, fk_table_name, fk_table_column):
        pass

    def insert_into(self, table_name, values):
        """
        Adds an instance to the table
        :param table_name: name of the table in which the values are added
        :param values: 1D list which holds values to be added
        :return: None
        """
        if table_name == "Patients":
            self.column_number = 3
        elif table_name == "Pictures0" or table_name == "Pictures1":
            self.column_number = 16

        if len(values) % int(self.column_number) != 0:
            return "Values should be multiple of column number. There is (a) missing value(s)"

        sql = "INSERT INTO " + table_name + " VALUES\n("
        for i in range(len(values)):
            if i % self.column_number == 0 and i != 0:
                sql = sql.rstrip(', ')
                sql += "),\n("
            sql += str(values[i]) + ", "
        sql = sql.rstrip(', ')
        sql += ");"
        self.cursor.execute(sql)
        self.con.commit()

    def insert_into_columns(self, table_name, columns, values):
        """
        Adds an instance to the table
        :param table_name: name of the table in which the values are added
        :param columns: 1D list which holds columns to be insterted into
        :param values: 1D list which holds values to be added
        :return:
        """

        self.column_number = len(columns)

        sql = "INSERT INTO " + table_name + " ("
        for i in range(len(columns)):
            sql += columns[i] + ", "
        sql = sql.rstrip(', ') + ") VALUES\n("

        for i in range(len(values)):
            if i % self.column_number == 0 and i != 0:
                sql = sql.rstrip(', ')
                sql += "),\n("
            if type(values[i]) == str:
                sql += "'{}', ".format(values[i])
                continue
            sql += str(values[i]) + ", "
        sql = sql.rstrip(', ')
        sql += ");"
        self.cursor.execute(sql)
        self.con.commit()

    def delete_from(self, table_name, condition):
        sql = "DELETE FROM " + table_name + " WHERE " + condition + ";"
        self.cursor.execute(sql)
        self.con.commit()


    def send_query(self, sql):
        """
        :param sql: query to be executed
        :return: a list that includes rows as elements
        """
        self.cursor.execute(sql)
        return self.cursor.fetchall()
