CREATE USER mlflow_user WITH PASSWORD 'SuperSecurePwdHere' CREATEDB;
CREATE DATABASE mlflow_pg_db
    WITH 
    OWNER = mlflow_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

CREATE USER prefect_user WITH PASSWORD 'SuperSecurePwdHere' CREATEDB;
CREATE DATABASE prefect_pg_db
    WITH 
    OWNER = prefect_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;

CREATE USER dlservice_user WITH PASSWORD 'SuperSecurePwdHere' CREATEDB;
CREATE DATABASE dlservice_pg_db
    WITH 
    OWNER = dlservice_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1;