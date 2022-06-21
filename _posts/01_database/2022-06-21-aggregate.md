---
layout: single
title:  MYSQL> Aggregation function(집계함수)에 대해 알아보자
categories: 01_database
tag: [database, QUERY, aggregate function, group by]
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# MYSQL> Aggregation function(집계함수)

Aggregation function (집계함수): 지금까지 해오던 방식은 집계함수를 where clause에 넣어보고 에러나면 having clause에 넣고 했는데 sql execution order를 알면 당연히 group by 후에 들어가야 하는지 알 수 있다. MYSQL code이지만 oracle, sql server 등 개념은 같다. 각 query 별 결과값을 보여주면 좋겠지만 이 post는 내가 공부한 내용을 정리한거라 그렇게 친절한 설명이 안되는 점 죄송하게 생각합니다. 
- WHERE clause에서는 사용 안된다 (단 subquery 안에서는 결과값만 가져오면 되기 때문에 사용하던 말던 상관없다. )
- HAVING clause에서는 사용 가능 (i.e. HAVING SUM(price * amount) > 1000

1. Average: 아래에서 a와 b는 그 결과값이 다르다. 2번은 null rows도 분모에 포함.
    
    1) NULL rows 무시: 
    
    ```sql
    select AVG(weight) from Players;
    ```
    
    ```sql
    select SUM(weight) / COUNT(weight) from Players;
    ```
    
    2) NULL rows 포함: 
    
    ```sql
    select AVG(CASE
    						WHEN weight IS NULL THEN 0
    						ELSE weight
    						END)
    FROM Players;
    ```
    
2. MIN, MAX: 다른 집계함수들과 달리 문자열이나 날짜에도 사용 가능하다.
3. COUNT, DISTINCT
1) COUNT: *를 parameter로 받을 수 있는 aggregation fun.은 COUNT가 유일
    
    2) DISITNCT: 모든 aggregation function에 사용
    
    ```sql
    select COUNT(DISTINCT NAME) from Players;
    ```
    
    아래와 같이 DISTINCT가 multiple columns가 나오면 이 column들이 합친 것이 하나의 row가 되고 이 것들이 겹치는 값이 없는 값들을 반환한다. 
    
    ```sql
    SELECT DISTINCT column1, column2, ...
    FROM table_name;
    ```
    
4. rollup: groupname 별로 소합계를 내준다.

```sql
select groupname, sum(price * amount) from Atable GROUP BY groupname
with rollup;
```


