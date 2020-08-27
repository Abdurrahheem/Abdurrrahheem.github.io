---
title: "Notes on SQL Basics"
tag:
 - SQL
---
 > That, which is written stays...

The post was mostly intended as a quick-reminder notes on SQL basics for myself. I hope you also can make use of them. The notes are from [this](https://www.youtube.com/watch?list=PLY4rE9dstrJwFmPYd03vZU90-TvXEpVzD&v=P2Eaf9M4gOU&feature=emb_logo) beautiful crash course on SQL. Unfortunately, the course is available only in Russian language and it consists of 5 short tutorials. Each tutorial is going to be separated by  a heading. These are `sqlite` commands and thus might not work for other RDBMS such `mysql` or `postgreSQL`.


## Basics

{% highlight shell linenos %}
create table TABLE_NAME (FIELD1 INT NOT NULL, FIELD2 VARCHAR(255) NOT NULL, ...); ## create a table
drop table ## to delete a table
insert into TABLE_NAME (FIELD1, FIELD2, ..) values(1, customtext, ..) ## fill in table entries
select * from TABLE_NAME where FIELD =>! 2 ## display table entries
update TABLE_NAME set FIELD1 = 'new_value' where FIELD2 = 'other_value' ## update table entries
delete from TABLE_NAME where FIELD1 = 'old_value' ## delete certain values
select * from TABLE_NAME order by FIELD acs|desc ## display value in a sorted order
{% endhighlight %}

## Normalization
{% highlight shell linenos %}
PRAGMA FOREIGN_KEYS = ON;
DROP TABLE regions;
CREATE TABLE regions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL UNIQUE,
  active BOOLEAN NOT NULL DEFAULT TRUE
);

DROP TABLE cities;
CREATE TABLE cities(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name VARCHAR(255) NOT NULL UNIQUE,
  regions_id INTEGER NOT NULL,
  active BOOLEAN NOT NULL DEFAULT TRUE,
  FOREIGN KEY(regions_id) REFERENCES regions(id)
);
{% endhighlight %}


## Edditing Tables

{% highlight shell linenos %}
alter table TABLE_NAME add column FIELD_NAME BOOLEAN NOT NULL default TRUE ## add a new column of boolean type with default values true
alter table TABLE_NAME drop column FIELD_NAME  ## drop a column from a table. ||| does not work in sqlite
alter table TABLE_NAME change OLD_FIELD_NAME NEW_FIELD_NAME BOOLEAN NOT NULL default TRUE  ## change column name. ||| does not work in sqlite
alter table TABLE_NAME modify FIELD_NAME VARCHAR(255) NOT NULL default 'neft';  ## modif field type. ||| does not work in sqlite  
alter table TABLE_NAME rename to NEW_TABLE_NAME; ## rename table

## row comparison
select * from TABLE_NAME where FIELD_NAME like  'anything'; ## print all row where field equals 'anything'
select * from TABLE_NAME where FIELD_NAME like  '%k'; ## print all row where field ends with 'k'
select * from TABLE_NAME where FIELD_NAME like  'A%'; ## print all row where field starts with 'A'
select * from TABLE_NAME where FIELD_NAME like  '%a%'; ## print all row where field contains with 'a'
select * from TABLE_NAME where FIELD_NAME1 = 1 AND/OR FIELD_NAME2 like '%a%'; ## print all row where field2 contains with 'a' and field1 = 1
select * from TABLE_NAME group by FILED_NAME; ## group by and print
select * from TABLE_NAME group by FILED_NAME1 having FILED_NAME2 > 400; ## group by and print with a condition

select count(FIELD_NAME) from towns; ## counts field entries
select max/min(FIELD_NAME) from towns;  ## min or max out of the field
select round(number) ## round and print a number
select random() ## print a random number

select length(' ')
select trim(' dfsfs ') ## trim empty spaces from begging and the end
select upper(' ')
select lower(' ')
select substr(' ', #)
select replace('a2a2a2a2 ', 2, 1)
select reverse(' ')
select md5(' ') ## hash a string

insert into TABLE_NAME2 select FIELD_NAME1, FIELD_NAME2 from TABLE_NAME1 ## fill in entire of table2 from table1. The entries in the tables must be similar

{% endhighlight %}


## Creating Triggers
{% highlight shell linenos %}

DROP TABLE users;
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username VARCHAR(255) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  created_at DATETIME NOT NULL,
  update_at DATETIME NOT NULL
);
DROP TABLE profiles;
CREATE TABLE IF NOT EXISTS profiles(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  name VARCHAR(255),
  last_name VARCHAR(255),
  photo_path VARCHAR(255),
  about TEXT,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

DROP TRIGGER IF EXISTS new_profile;
CREATE TRIGGER new_profile AFTER INSERT ON users
  FOR EACH ROW
  BEGIN
    INSERT INTO profiles(user_id) VALUES (NEW.id);
  END;

DROP TRIGGER IF EXISTS update_user;
CREATE TRIGGER update_user AFTER UPDATE ON profiles
  FOR EACH ROW
  BEGIN
    UPDATE users SET  update_at = datetime('NOW') WHERE id = OLD.user_id;
  END;

DROP TRIGGER IF EXISTS delete_profile;
CREATE TRIGGER delete_profile BEFORE DELETE ON users
  FOR EACH ROW
  BEGIN
    DELETE FROM profiles WHERE user_id = OLD.id;
  END;
{% endhighlight %}


## Transactions

{% highlight shell linenos %}
begin;
update ...;
insert ...;
delete ...;
commit;


begin;
update ...;
insert ...;
delete ...;
rollback;
commit;

begin;
update ...;
insert ...;
..
..
savepoint POINT_NAME;
delete ...;
rollback to savepoint POINT_NAME;
commit;
{% endhighlight %}
