Make sure you have python3-mysqldb
    `sudo apt-get install python3-mysqldb`

run database.sql against your mysql instance
    `mysql -u root < ./database.sql`
run populate.py from within this directory
    `python3 populate.py`

if for some reason you need to cleanup the db, run delete.sql
    `mysql -u root < ./delete.sql`

