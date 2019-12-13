import pymysql

import configmysql



class MysqlOP():
    def __init__(self, config=configmysql.configs['db']):
        self.db_config = config
        try:
            self.conn = pymysql.connect(**self.db_config)
        except:
            print('connect error!')
            print(**self.db_config)
        self.cur = self.conn.cursor(cursor = pymysql.cursors.DictCursor)

    def __enter__(self):
        # 返回游标        
        return self.cur

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 提交数据库并执行        
        self.conn.commit()
        # 关闭游标        
        self.cur.close()
        # 关闭数据库连接        
        self.conn.close()
        
        
if __name__ == '__main__':
    with MysqlOP(configmysql.configs['db']) as db:
        db.execute('select * from test')
