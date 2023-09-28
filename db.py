import MySQLdb

conn=MySQLdb.connect("localhost","root","","speech-emotion-recognition")

c = conn.cursor()

def prep():    
    c.execute(
    """
        CREATE TABLE IF NOT EXISTS `speeches` (
        `id` int(11) NOT NULL AUTO_INCREMENT,
        `feature` varchar(255) NOT NULL,
        `data` longblob NOT NULL,
        `is_train` boolean DEFAULT FALSE, 
        PRIMARY KEY (`id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
    """
    )
    conn.commit()

def deleteAllDataSet():    
    c.execute(
        f'DELETE FROM `speeches`'
    )
    conn.commit()

def insertMany(rows):    
    c.executemany(
        f'INSERT INTO `speeches`(`feature`, `data`, `is_train`) VALUES (%s, %s, %s)',
        rows
    )
    conn.commit()

def retrieveDataSet():
    c.execute(
        """
            SELECT * FROM `speeches` ORDER BY `feature`
        """
    )
    return c.fetchall()